"""
Global Bezier Refiner — Pass 2 (gradient-descent polish)
================================
Reads the SVG produced by fit_all.py (which embeds data-atypes),
reparameterizes smooth anchors as (pos, θ_in, l_in, l_out, δθ) so that
C1 continuity is a hard constraint, then polishes every curve with gradient
descent against the true layer boundary.

Loss: for each subpath, the curve is sampled densely and each sample is
scored by its distance to the boundary of that layer's pixel mask (a signed
distance field built from the quantized image). A small penalty keeps the
curve near its starting shape so it cannot slide along the boundary.

Subpaths that still end up more than --max-drift px from their pass-1
curve are reverted, so the polish can only make small corrections.

Usage:
    python global_refine.py --input uz.svg --ref uzumaki.png
    python global_refine.py --quantized uz_quantized.png    # explicit label image
    python global_refine.py --trial                 # 256px wide, coarse only
    python global_refine.py --phase coarse          # one phase only
"""

import argparse, io, math, os, re, time
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import cairosvg

# ── Device selection ─────────────────────────────────────────────────────────
# MPS lacks grid_sampler_2d_backward; CUDA works fine; CPU is the safe default.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'[backend] PyTorch on {DEVICE}')

# ── Tunable defaults ──────────────────────────────────────────────────────────
# Learning rates are in pixels per step (positions and arm lengths are in px).
# The curves start from an already-good pass-1 fit, so the steps are small.
STEPS_COARSE   = 300
STEPS_MID      = 300
STEPS_FINE     = 300
LR_COARSE      = 0.05
LR_MID         = 0.03
LR_FINE        = 0.02
SMOOTH_PRIOR   = 0.01   # λ for δθ² regularisation
ANCHOR_W       = 0.10   # λ for squared drift from the starting curve (px²)
HUBER_DELTA    = 2.0    # px; boundary distances beyond this grow linearly
N_PER_SEG      = 12     # curve samples per bezier segment in the loss
PATIENCE       = 30     # early-stop a phase after this many stalled steps
MIN_STEPS      = 50     # never early-stop before this many steps
MAX_DRIFT      = 1.0    # px; subpaths that end further than this from pass 1 are reverted


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SVG PARSE — read paths + data-atypes, reconstruct A, C, atype arrays
# ═══════════════════════════════════════════════════════════════════════════════

def parse_svg(path):
    """
    Returns (w, h, layers) where layers is a list of:
        (fill_hex,  [region, ...])
    and each region is a list of subpaths, each subpath being a dict:
        { 'A': np.ndarray (N,2) xy anchors,
          'C': np.ndarray (N,2,2) control points  [[cp1x,cp1y],[cp2x,cp2y]],
          'atype': np.ndarray (N,) int }
    """
    with open(path) as f:
        txt = f.read()
    w = int(re.search(r'<svg[^>]+width="(\d+)"',  txt).group(1))
    h = int(re.search(r'<svg[^>]+height="(\d+)"', txt).group(1))

    layers = []
    # Support both flat <path> and <g inkscape:groupmode="layer"> formats
    group_blocks = re.findall(
        r'<g[^>]+inkscape:groupmode="layer"[^>]*>(.*?)</g>', txt, re.DOTALL)

    if group_blocks:
        for block in group_blocks:
            fills = re.findall(r'fill="(#[0-9a-fA-F]{6})"', block)
            if not fills:
                continue
            color = fills[0].lower()
            regions = []
            for pm in re.finditer(r'<path\b([^>]+)/>', block):
                attrs = pm.group(1)
                dm = re.search(r'\bd="([^"]+)"', attrs)
                atm = re.search(r'data-atypes="([^"]+)"', attrs)
                if not dm:
                    continue
                subpaths = _parse_compound_d(dm.group(1), atm.group(1) if atm else None)
                regions.append(subpaths)
            if regions:
                layers.append((color, regions))
    else:
        # Flat format (legacy taos_all.svg without groups)
        for pm in re.finditer(r'<path\b([^>]+)/>', txt):
            attrs = pm.group(1)
            fill = re.search(r'fill="(#[0-9a-fA-F]{6})"', attrs)
            dm   = re.search(r'\bd="([^"]+)"', attrs)
            atm  = re.search(r'data-atypes="([^"]+)"', attrs)
            if not fill or not dm:
                continue
            subpaths = _parse_compound_d(dm.group(1), atm.group(1) if atm else None)
            layers.append((fill.group(1).lower(), [subpaths]))

    return w, h, layers


def _parse_compound_d(d_str, atypes_str):
    """
    Split compound d into subpaths, parse each into A, C.
    atypes_str: '2,0,1,0|1,2' where | separates outer from holes.
    Returns list of subpath dicts.
    """
    raw_sps = re.split(r'(?i)\s*Z\s*', d_str.strip())
    raw_sps = [s.strip() for s in raw_sps if s.strip()]
    at_segs = atypes_str.split('|') if atypes_str else []

    subpaths = []
    for idx, sp in enumerate(raw_sps):
        A, C = _parse_subpath(sp)
        if A is None:
            continue
        N = len(A)
        if idx < len(at_segs):
            vals = [int(x) for x in at_segs[idx].split(',') if x]
            at = np.array(vals[:N] if len(vals) >= N else vals + [0]*(N-len(vals)),
                          dtype=np.int32)
        else:
            at = np.zeros(N, dtype=np.int32)
        subpaths.append({'A': A, 'C': C, 'atype': at})
    return subpaths


def _parse_subpath(sp):
    """Parse 'M x,y C cp1x,cp1y cp2x,cp2y ex,ey C ...' → (A, C)."""
    tokens = sp.split()
    anchors, ctrl_pairs = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == 'M':
            x, y = map(float, tokens[i+1].split(','))
            anchors.append([x, y]); i += 2
        elif t in ('C', 'c'):
            cp1 = list(map(float, tokens[i+1].split(',')))
            cp2 = list(map(float, tokens[i+2].split(',')))
            ep  = list(map(float, tokens[i+3].split(',')))
            ctrl_pairs.append([cp1, cp2])
            anchors.append(ep); i += 4
        else:
            i += 1
    N = len(ctrl_pairs)
    if N < 3:
        return None, None
    # Last anchor duplicates first (closed path) — drop it
    A = np.array(anchors[:N], dtype=np.float64)   # (N, 2)  xy
    C = np.array(ctrl_pairs, dtype=np.float64)    # (N, 2, 2)
    return A, C


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PARAMETERISATION — convert A,C,atype ↔ smooth params
# ═══════════════════════════════════════════════════════════════════════════════

def to_smooth_params(A, C, atype):
    """
    For each anchor:
      corner (atype==2): store pos (fixed), cp1, cp2 (free)
      smooth (atype 0 or 1): store pos, θ_in, l_in, l_out, δθ
    Returns a flat parameter dict suitable for torch optimisation.
    """
    N = len(A)
    params = []
    for i in range(N):
        p0 = A[i]          # xy
        cp2_prev = C[(i-1) % N, 1]   # incoming ctrl pt (end of prev segment)
        cp1_next = C[i, 0]            # outgoing ctrl pt (start of this segment)
        if atype[i] == 2:
            # Corner: pos frozen, both ctrl pts free
            params.append({'type': 'corner',
                           'pos':  p0.copy(),
                           'cp1':  cp1_next.copy(),
                           'cp2':  cp2_prev.copy()})
        else:
            v_in  = p0 - cp2_prev
            v_out = cp1_next - p0
            l_in  = float(np.linalg.norm(v_in))
            l_out = float(np.linalg.norm(v_out))
            th_in = float(np.arctan2(v_in[1], v_in[0]))
            if l_in > 1e-6 and l_out > 1e-6:
                t_in  = v_in  / l_in
                t_out = v_out / l_out
                # angle between them (0 = C1, π = cusp)
                dot   = float(np.clip(np.dot(t_in, t_out), -1, 1))
                cross = float(t_in[0]*t_out[1] - t_in[1]*t_out[0])
                dth   = float(np.arctan2(cross, dot))
            else:
                dth = 0.0
            params.append({'type': 'smooth',
                           'pos':   p0.copy(),
                           'th_in': th_in,
                           'l_in':  max(l_in, 0.5),
                           'l_out': max(l_out, 0.5),
                           'dth':   dth})
    return params


def params_to_AC(params):
    """Reconstruct A (N,2) and C (N,2,2) from param list."""
    N = len(params)
    A = np.zeros((N, 2))
    C = np.zeros((N, 2, 2))
    # First pass: fill positions
    for i, p in enumerate(params):
        A[i] = p['pos']
    # Second pass: fill control points
    for i, p in enumerate(params):
        if p['type'] == 'corner':
            C[i, 0] = p['cp1']                        # outgoing cp1
            C[(i-1) % N, 1] = p['cp2']               # incoming cp2
        else:
            th = p['th_in']
            li, lo, dth = p['l_in'], p['l_out'], p['dth']
            tin  = np.array([math.cos(th),       math.sin(th)])
            tout = np.array([math.cos(th + dth), math.sin(th + dth)])
            C[(i-1) % N, 1] = A[i] - tin  * li   # cp2 of prev segment
            C[i,       0] = A[i] + tout * lo  # cp1 of this segment
    return A, C


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TORCH PARAMETER PACKING — gather all free scalars into tensors
# ═══════════════════════════════════════════════════════════════════════════════

class LayerParams:
    """
    Holds all optimisable tensors for one subpath.
    Frozen values (corner positions) are kept as plain numpy.
    """
    def __init__(self, params):
        self.params = params          # list of per-anchor dicts (numpy)
        N = len(params)

        # Smooth anchors
        smooth_idx = [i for i,p in enumerate(params) if p['type'] == 'smooth']
        corner_idx = [i for i,p in enumerate(params) if p['type'] == 'corner']
        self.smooth_idx = smooth_idx
        self.corner_idx = corner_idx

        if smooth_idx:
            self.s_pos   = torch.tensor([[params[i]['pos'][0],   params[i]['pos'][1]]
                                          for i in smooth_idx], dtype=torch.float32,
                                         device=DEVICE, requires_grad=True)
            self.s_th    = torch.tensor([params[i]['th_in'] for i in smooth_idx],
                                        dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.s_lin   = torch.tensor([params[i]['l_in']  for i in smooth_idx],
                                        dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.s_lout  = torch.tensor([params[i]['l_out'] for i in smooth_idx],
                                        dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.s_dth   = torch.tensor([params[i]['dth']   for i in smooth_idx],
                                        dtype=torch.float32, device=DEVICE, requires_grad=True)
        else:
            self.s_pos = self.s_th = self.s_lin = self.s_lout = self.s_dth = None

        if corner_idx:
            self.c_cp1 = torch.tensor([[params[i]['cp1'][0], params[i]['cp1'][1]]
                                        for i in corner_idx], dtype=torch.float32,
                                       device=DEVICE, requires_grad=True)
            self.c_cp2 = torch.tensor([[params[i]['cp2'][0], params[i]['cp2'][1]]
                                        for i in corner_idx], dtype=torch.float32,
                                       device=DEVICE, requires_grad=True)
        else:
            self.c_cp1 = self.c_cp2 = None

        # Fixed corner positions (not optimised)
        self.corner_pos = {i: torch.tensor(params[i]['pos'], dtype=torch.float32, device=DEVICE)
                           for i in corner_idx}

    def torch_tensors(self):
        """Return list of tensors that need gradients."""
        ts = []
        for t in [self.s_pos, self.s_th, self.s_lin, self.s_lout, self.s_dth,
                  self.c_cp1, self.c_cp2]:
            if t is not None:
                ts.append(t)
        return ts

    def build_AC(self):
        """Build A (N,2) and C (N,2,2) torch tensors from current params."""
        N = len(self.params)
        # Assemble A
        A_list = [None] * N
        for k, i in enumerate(self.smooth_idx):
            A_list[i] = self.s_pos[k]
        for k, i in enumerate(self.corner_idx):
            A_list[i] = self.corner_pos[i]
        A = torch.stack(A_list)   # (N, 2)

        # Assemble C
        C = torch.zeros(N, 2, 2, device=DEVICE)
        # Smooth anchors: derive from (th, l_in, l_out, dth)
        for k, i in enumerate(self.smooth_idx):
            th  = self.s_th[k]
            li  = torch.clamp(self.s_lin[k],  min=0.1)
            lo  = torch.clamp(self.s_lout[k], min=0.1)
            dth = self.s_dth[k]
            tin  = torch.stack([torch.cos(th),        torch.sin(th)])
            tout = torch.stack([torch.cos(th + dth),  torch.sin(th + dth)])
            C[(i-1) % N, 1] = A[i] - tin  * li
            C[i,       0]   = A[i] + tout * lo
        # Corner anchors: direct cp1, cp2
        for k, i in enumerate(self.corner_idx):
            C[i,       0]   = self.c_cp1[k]
            C[(i-1) % N, 1] = self.c_cp2[k]
        return A, C

    def smooth_prior_loss(self):
        """δθ² regularisation — penalises deviation from C1."""
        if self.s_dth is None:
            return torch.tensor(0., device=DEVICE)
        return (self.s_dth ** 2).mean()

    def sync_back(self):
        """Copy current tensor values back into self.params dicts (for saving)."""
        for k, i in enumerate(self.smooth_idx):
            p = self.params[i]
            p['pos']   = self.s_pos[k].detach().cpu().numpy()
            p['th_in'] = float(self.s_th[k].detach().cpu())
            p['l_in']  = float(torch.clamp(self.s_lin[k],  min=0.1).detach().cpu())
            p['l_out'] = float(torch.clamp(self.s_lout[k], min=0.1).detach().cpu())
            p['dth']   = float(self.s_dth[k].detach().cpu())
        for k, i in enumerate(self.corner_idx):
            p = self.params[i]
            p['cp1'] = self.c_cp1[k].detach().cpu().numpy()
            p['cp2'] = self.c_cp2[k].detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CURVE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def bezier_pts(p0, p1, p2, p3, n=20):
    """Sample n+1 points along cubic bezier p0..p3."""
    t = torch.linspace(0, 1, n+1, device=p0.device).unsqueeze(1)
    mt = 1 - t
    return (mt**3*p0 + 3*mt**2*t*p1 + 3*mt*t**2*p2 + t**3*p3)   # (n+1, 2)


def sample_curve_pts(A, C, n_per_seg=8):
    """
    Sample points densely along the closed bezier curve.
    Returns (M, 2) tensor.
    """
    N = len(A)
    parts = []
    for i in range(N):
        p0 = A[i]; p1 = C[i,0]; p2 = C[i,1]; p3 = A[(i+1) % N]
        parts.append(bezier_pts(p0, p1, p2, p3, n_per_seg))
    return torch.cat(parts, dim=0)   # (N*(n+1), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SVG OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def f2(x): return f"{x:.2f}"
def pt(xy): return f"{f2(xy[0])},{f2(xy[1])}"


def subpath_to_d(A, C):
    """A: (N,2) xy,  C: (N,2,2).  Returns d-string with Z."""
    N = len(A)
    d = f"M {pt(A[0])}"
    for i in range(N):
        j = (i + 1) % N
        d += f" C {pt(C[i,0])} {pt(C[i,1])} {pt(A[j])}"
    return d + " Z"


def write_svg(path, w, h, layers_data, stroke_width=0.0):
    """
    layers_data: [(color_hex, [region, ...]), ...]
    region: list of subpath dicts with 'A', 'C', 'atype' (numpy)
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"',
        '     xmlns="http://www.w3.org/2000/svg"',
        '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">',
    ]
    for li, (color, regions) in enumerate(layers_data):
        lines.append(f'  <g id="layer-{li}" inkscape:label="{color}"'
                     f' inkscape:groupmode="layer">')
        sw = (f' stroke="{color}" stroke-width="{stroke_width:.2f}"'
              ' stroke-linejoin="round" stroke-linecap="round"'
              if stroke_width > 0 else ' stroke="none"')
        for ri, subpaths in enumerate(regions):
            d_parts, at_parts = [], []
            for sp in subpaths:
                A, C = params_to_AC(sp['params'])
                d_parts.append(subpath_to_d(A, C))
                at_parts.append(','.join(str(int(x)) for x in sp['atype']))
            combined  = ' '.join(d_parts)
            atypes_attr = '|'.join(at_parts)
            lines.append(f'    <path id="r{li}-{ri}" d="{combined}"')
            lines.append(f'          fill="{color}" fill-rule="evenodd"{sw}')
            lines.append(f'          data-atypes="{atypes_attr}"/>')
        lines.append('  </g>')
    lines.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CURVE GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _bez_np(p0, c0, c1, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return mt**3 * p0 + 3 * mt**2 * t * c0 + 3 * mt * t**2 * c1 + t**3 * p3


def dense_curve_np(params, spacing=0.5):
    """Closed curve sampled about every `spacing` px, so point-to-point
    distances approximate point-to-curve distances to within spacing/2."""
    A, C = params_to_AC(params)
    N = len(A)
    out = []
    for i in range(N):
        p0, c0, c1, p3 = A[i], C[i, 0], C[i, 1], A[(i + 1) % N]
        poly = (np.linalg.norm(c0 - p0) + np.linalg.norm(c1 - c0)
                + np.linalg.norm(p3 - c1))
        out.append(_bez_np(p0, c0, c1, p3, int(np.clip(poly / spacing, 8, 800))))
    return np.vstack(out)


def hausdorff(P, Q, tree_Q=None):
    tq = tree_Q if tree_Q is not None else cKDTree(Q)
    if Q is None:
        Q = tq.data
    return max(float(tq.query(P)[0].max()), float(cKDTree(P).query(Q)[0].max()))


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  BOUNDARY DISTANCE FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

def load_label_map(rgba, layer_colors):
    """Assign each visible pixel to the nearest layer colour (-1 = background)."""
    vis = rgba[:, :, 3] > 128
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in layer_colors],
                   dtype=np.int32)
    rgb = rgba[vis][:, :3].astype(np.int32)
    best = np.full(len(rgb), np.iinfo(np.int32).max, dtype=np.int64)
    lab = np.zeros(len(rgb), dtype=np.int32)
    for k in range(len(pal)):
        dk = ((rgb - pal[k]) ** 2).sum(axis=1)
        better = dk < best
        best[better] = dk[better]
        lab[better] = k
    labels = np.full(rgba.shape[:2], -1, dtype=np.int32)
    labels[vis] = lab
    return labels


def build_boundary_fields(labels, n_layers, dilation=0):
    """
    Signed distance (px) to each layer's mask boundary, positive inside.
    The boundary sits half a pixel from the nearest pixel centres, matching
    the 0.5 level-set contours produced in pass 1.
    """
    fields = []
    for k in range(n_layers):
        m = labels == k
        if dilation > 0:
            m = ndi.binary_dilation(m, iterations=dilation) & (labels >= 0)
        if not m.any() or m.all():
            sd = np.full(m.shape, 1e3 if m.all() else -1e3, dtype=np.float32)
        else:
            sd = np.where(m, ndi.distance_transform_edt(m) - 0.5,
                          -(ndi.distance_transform_edt(~m) - 0.5)).astype(np.float32)
        fields.append(torch.tensor(sd, device=DEVICE).unsqueeze(-1))   # (H, W, 1)
    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  BOUNDARY-FIT LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def bilinear_sample(ref_t, pts):   # noqa: E302
    """
    Manually differentiable bilinear sampling — works on CPU, CUDA and MPS.
    ref_t : (H, W, 3) float32
    pts   : (M, 2)  float32  x=col, y=row  (pixel-space coords)
    Returns (M, 3)
    """
    H, W = ref_t.shape[:2]
    x = pts[:, 0].clamp(0, W - 1 - 1e-4)
    y = pts[:, 1].clamp(0, H - 1 - 1e-4)
    x0 = x.long();  x1 = x0 + 1
    y0 = y.long();  y1 = y0 + 1
    wx1 = x - x0.float();  wx0 = 1.0 - wx1
    wy1 = y - y0.float();  wy0 = 1.0 - wy1
    # clamp indices
    x1 = x1.clamp(max=W - 1)
    y1 = y1.clamp(max=H - 1)
    c00 = ref_t[y0, x0]   # (M, 3)
    c01 = ref_t[y0, x1]
    c10 = ref_t[y1, x0]
    c11 = ref_t[y1, x1]
    return (c00 * (wx0 * wy0).unsqueeze(1)
          + c01 * (wx1 * wy0).unsqueeze(1)
          + c10 * (wx0 * wy1).unsqueeze(1)
          + c11 * (wx1 * wy1).unsqueeze(1))


def snapshot_samples(all_lps, scale, n_per_seg=N_PER_SEG):
    """Detached curve samples for every subpath (the 'stay near here' target)."""
    out = []
    with torch.no_grad():
        for lp in all_lps:
            A, C = lp.build_AC()
            out.append(sample_curve_pts(A * scale, C * scale, n_per_seg).clone())
    return out


def make_loss_fn(all_lps, fields, ref_pts, scale, smooth_w, anchor_w,
                 n_per_seg=N_PER_SEG, delta=HUBER_DELTA):
    """
    Returns loss_fn() -> (total, fit).
      fit   : pseudo-Huber of each curve sample's distance (full-res px) to the
              boundary of its own layer; averaged over samples, then subpaths.
      total : fit + anchor_w * drift from ref_pts + smooth_w * dθ² prior.
    `fields[i]` is the signed boundary distance field for all_lps[i].
    """
    def loss_fn():
        fit = torch.tensor(0., device=DEVICE)
        drift = torch.tensor(0., device=DEVICE)
        prior = torch.tensor(0., device=DEVICE)
        for lp, fld, r0 in zip(all_lps, fields, ref_pts):
            A, C = lp.build_AC()
            pts = sample_curve_pts(A * scale, C * scale, n_per_seg)
            d = bilinear_sample(fld, pts).squeeze(1) / scale
            fit = fit + (delta ** 2 * (torch.sqrt(1 + (d / delta) ** 2) - 1)).mean()
            drift = drift + ((pts - r0) ** 2).sum(1).mean() / scale ** 2
            prior = prior + lp.smooth_prior_loss()
        n = max(len(all_lps), 1)
        fit, drift, prior = fit / n, drift / n, prior / n
        return fit + anchor_w * drift + smooth_w * prior, fit
    return loss_fn


def boundary_stats(all_lps, fields, scale, n_per_seg=16):
    """Distance (full-res px) from curve samples to their layer boundary."""
    ds = []
    with torch.no_grad():
        for lp, fld in zip(all_lps, fields):
            A, C = lp.build_AC()
            pts = sample_curve_pts(A * scale, C * scale, n_per_seg)
            ds.append((bilinear_sample(fld, pts).squeeze(1) / scale).abs())
    d = torch.cat(ds).cpu().numpy()
    return {'mean': float(d.mean()), 'p95': float(np.percentile(d, 95)),
            'max': float(d.max()), 'gt1': float((d > 1.0).mean() * 100),
            'gt2': float((d > 2.0).mean() * 100)}


def print_stats(tag, s):
    print(f"  [{tag}] boundary dist px: mean={s['mean']:.3f} p95={s['p95']:.2f} "
          f"max={s['max']:.2f}  >1px={s['gt1']:.1f}%  >2px={s['gt2']:.1f}%", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  OPTIMISATION PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def _snapshot_state(lp):
    return {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in lp.__dict__.items() if not k.startswith('_')}


def revert_drifters(all_lps, originals, max_drift):
    """Restore any subpath whose curve moved more than max_drift px
    (Hausdorff) from its starting curve. Returns number reverted."""
    n = 0
    for lp, (state, ref_pts) in zip(all_lps, originals):
        lp.sync_back()
        cur = dense_curve_np(lp.params)
        if hausdorff(cur, ref_pts) > max_drift:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    getattr(lp, k).data.copy_(v)
            lp.sync_back()
            n += 1
    return n


def run_phase(all_lps, loss_fn, n_steps, lr, label, verbose,
              patience=PATIENCE, tol=1e-3, min_steps=MIN_STEPS, target_loss=0.0):
    """
    Adam with cosine-annealed lr. Early-stops when the best fit has not
    improved by a relative `tol` for `patience` steps (never before
    `min_steps`), or once it reaches `target_loss`. patience=0 disables
    early stopping. The best state seen is restored at the end.
    """
    all_params = []
    for lp in all_lps:
        all_params += lp.torch_tensors()

    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    best = {'loss': 1e9, 'state': None}
    t0 = time.time()

    log_every = max(1, n_steps // 10)
    stall = 0
    ref_loss = float('inf')   # best fit at the last meaningful improvement
    stop_reason = None
    for step in range(n_steps):
        optimizer.zero_grad()
        total, fit = loss_fn()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        scheduler.step()

        cur = fit.detach().item()
        if cur < best['loss']:
            best['loss'] = cur
            best['state'] = [{k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in lp.__dict__.items()
                              if not k.startswith('_')}
                             for lp in all_lps]

        if cur < ref_loss * (1.0 - tol):
            ref_loss = cur
            stall = 0
        else:
            stall += 1
        if target_loss > 0 and best['loss'] <= target_loss:
            stop_reason = f"target fit {target_loss} reached"
        elif patience > 0 and step + 1 >= min_steps and stall >= patience:
            stop_reason = f"no >{tol:.1%} improvement for {patience} steps"

        if verbose and step % log_every == 0:
            eta = (time.time() - t0) / max(step, 1) * (n_steps - step)
            print(f"  {label} {step:4d}/{n_steps}: fit={cur:.5f} "
                  f"best={best['loss']:.5f}  eta={eta:.0f}s", flush=True)

        if stop_reason:
            print(f"  {label} early stop at step {step + 1}/{n_steps}: {stop_reason}")
            break

    print(f"  {label} best fit={best['loss']:.5f}  ({time.time()-t0:.0f}s)")
    for lp, state in zip(all_lps, best['state']):
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                getattr(lp, k).data.copy_(v)
    for lp in all_lps:
        lp.sync_back()


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',    default='taos_pass1.svg')
    ap.add_argument('--ref',      default='logo-plain-transparent.png')
    ap.add_argument('--output',   default='taos_pass2.svg')
    ap.add_argument('--preview',  default='taos_pass2_preview.png')
    ap.add_argument('--diff',     default='taos_pass2_diff.png')
    ap.add_argument('--trial',    action='store_true',
                    help='Fast trial: 256px, coarse phase only')
    ap.add_argument('--phase',    default='all',
                    choices=['coarse', 'mid', 'fine', 'all'])
    ap.add_argument('--quiet',    action='store_true')
    ap.add_argument('--smooth-w', type=float, default=SMOOTH_PRIOR)
    ap.add_argument('--quantized', default=None,
                    help='Quantized reference PNG (default: <input>_quantized.png if present)')
    ap.add_argument('--mask-dilation', type=int, default=0,
                    help='Must match the --dilation-px used in pass 1 (default 0)')
    ap.add_argument('--max-drift', type=float, default=MAX_DRIFT)
    ap.add_argument('--anchor-w', type=float, default=ANCHOR_W)
    ap.add_argument('--lr-scale', type=float, default=1.0,
                    help='Multiplier applied to every phase learning rate (default 1.0)')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='Early-stop a phase after this many steps without improvement (0 = off)')
    ap.add_argument('--tol',      type=float, default=1e-3,
                    help='Relative loss improvement that counts as progress (default 1e-3)')
    ap.add_argument('--target-loss', type=float, default=0.0,
                    help='Stop a phase once MSE reaches this value (default 0 = off)')
    args = ap.parse_args()
    verbose = not args.quiet

    # ── Load SVG ──────────────────────────────────────────────────────────────
    print(f"Loading {args.input}...")
    full_w, full_h, raw_layers = parse_svg(args.input)
    print(f"  Canvas {full_w}×{full_h}, {len(raw_layers)} layers")

    total_anchors = 0
    # Build per-subpath LayerParams + colour tensors
    all_lps     = []   # flat list of LayerParams (all subpaths across all layers)
    all_colors  = []   # matching colour tensors
    # Also keep structured (color, [[subpath_lp, ...], ...]) for SVG writing
    structured  = []   # [(color, [[lp, ...], ...]), ...]

    for color, regions in raw_layers:
        h = color.lstrip('#')
        rgb = torch.tensor([int(h[0:2],16)/255, int(h[2:4],16)/255,
                            int(h[4:6],16)/255], dtype=torch.float32, device=DEVICE)
        layer_regions = []
        for region_subpaths in regions:
            region_lps = []
            for sp in region_subpaths:
                params = to_smooth_params(sp['A'], sp['C'], sp['atype'])
                lp = LayerParams(params)
                # Attach atype for downstream SVG writing
                lp.atype = sp['atype']
                all_lps.append(lp)
                all_colors.append(rgb)
                region_lps.append(lp)
                total_anchors += len(params)
            layer_regions.append(region_lps)
        structured.append((color, layer_regions))

    print(f"  Total subpaths: {len(all_lps)}, anchors: {total_anchors}")

    # ── Load reference / labels ───────────────────────────────────────────────
    print(f"Loading reference {args.ref}...")
    orig_rgba = np.array(Image.open(args.ref).convert('RGBA'))
    visible   = orig_rgba[:, :, 3] > 128
    qpath = args.quantized
    if not qpath:
        sib = args.input.replace('.svg', '_quantized.png')
        qpath = sib if os.path.exists(sib) else None
    if qpath:
        print(f"  Labels from {qpath}")
        lab_src = np.array(Image.open(qpath).convert('RGBA'))
        lab_src[:, :, 3] = orig_rgba[:, :, 3]
    else:
        print("  Labels from nearest layer colour of reference")
        lab_src = orig_rgba
    labels = load_label_map(lab_src, [c for c, _ in raw_layers])

    if args.trial:
        W = 256
        H = max(1, round(full_h * W / full_w))
        phases = ['coarse']
    else:
        H, W = full_h, full_w
        phases = (['coarse', 'mid', 'fine'] if args.phase == 'all' else [args.phase])
    scale = W / full_w
    if (H, W) != labels.shape:
        labels = np.array(Image.fromarray(labels.astype(np.int16)).resize((W, H), Image.NEAREST)
                          ).astype(np.int32)
    fields_by_layer = build_boundary_fields(labels, len(raw_layers), args.mask_dilation)
    fields = []
    for li, (_, layer_regions) in enumerate(structured):
        for region_lps in layer_regions:
            fields += [fields_by_layer[li]] * len(region_lps)

    # ── Optimisation ─────────────────────────────────────────────────────────
    def make_fn():
        return make_loss_fn(all_lps, fields, snapshot_samples(all_lps, scale),
                            scale, args.smooth_w, args.anchor_w)

    print_stats('start', boundary_stats(all_lps, fields, scale))
    originals = []
    for lp in all_lps:
        lp.sync_back()
        originals.append((_snapshot_state(lp), dense_curve_np(lp.params)))
    phase_cfg = {'coarse': (STEPS_COARSE, LR_COARSE), 'mid': (STEPS_MID, LR_MID),
                 'fine': (STEPS_FINE, LR_FINE)}
    for ph in phases:
        n_steps, lr = phase_cfg[ph]
        lr *= args.lr_scale
        if args.trial:
            n_steps = min(n_steps, 100)
        print(f"\n{'='*60}\nPhase {ph}: {H}x{W} steps={n_steps} lr={lr}")
        run_phase(all_lps, make_fn(), n_steps, lr, ph, verbose,
                  patience=args.patience, tol=args.tol, target_loss=args.target_loss)
    nrev = revert_drifters(all_lps, originals, args.max_drift)
    print(f"  Reverted {nrev}/{len(all_lps)} subpaths that drifted > {args.max_drift} px")
    print_stats('polished', boundary_stats(all_lps, fields, scale))

    # ── Write SVG ─────────────────────────────────────────────────────────────
    # Rebuild structured layer data from LayerParams
    out_layers = []
    lp_iter = iter(all_lps)
    for color, layer_regions in structured:
        out_regions = []
        for region_lps in layer_regions:
            out_sps = []
            for lp in region_lps:
                lp.sync_back()
                A, C = params_to_AC(lp.params)
                out_sps.append({'A': A, 'C': C,
                                'atype': lp.atype,
                                'params': lp.params})
            out_regions.append(out_sps)
        out_layers.append((color, out_regions))

    write_svg(args.output, full_w, full_h, out_layers)
    print(f"\nSaved: {args.output}")

    # ── Preview + diff ────────────────────────────────────────────────────────
    with open(args.output) as f:
        svg_str = f.read()
    png = cairosvg.svg2png(bytestring=svg_str.encode(),
                           output_width=full_w, output_height=full_h)
    preview = Image.open(io.BytesIO(png))
    preview.save(args.preview)
    print(f"Preview: {args.preview}")

    pred_np  = np.array(preview.convert('RGBA'))
    diff_ch  = np.abs(orig_rgba[:,:,:3].astype(np.int16) -
                      pred_np[:,:,:3].astype(np.int16))
    diff_amp = (diff_ch.astype(np.float32) * 4).clip(0, 255).astype(np.uint8)
    diff_amp[~visible] = 0
    Image.fromarray(diff_amp).save(args.diff)
    print(f"Diff: {args.diff}")


if __name__ == '__main__':
    main()
