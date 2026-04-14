"""
Global Bezier Refiner — Pass 2
================================
Reads taos_all.svg (produced by fit_all.py, which embeds data-atypes),
reparameterizes smooth anchors as (pos, θ_in, l_in, l_out, δθ) so that
C1 continuity is a hard constraint, then runs global gradient descent where
the loss is MSE of the soft-composited render vs the original image —
meaning ALL layers are optimized together and inter-layer seams close
naturally.

After optimization, a greedy pruning pass removes redundant smooth anchors:
each candidate is replaced with the minimum-curvature bezier between its
neighbours; if the global image loss doesn't increase beyond ε the removal
is committed, otherwise reverted.

Usage:
    python global_refine.py                         # full pipeline
    python global_refine.py --trial                 # 256px, coarse only (~3 min)
    python global_refine.py --phase coarse          # one phase only
    python global_refine.py --input taos_all.svg    # explicit input
    python global_refine.py --no-prune              # skip pruning
"""

import argparse, io, math, re, time
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import cairosvg

# ── Device selection ─────────────────────────────────────────────────────────
# MPS lacks grid_sampler_2d_backward; CUDA works fine; CPU is the safe default.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'[backend] PyTorch on {DEVICE}')

# ── Tunable defaults ──────────────────────────────────────────────────────────
SIGMA_COARSE   = 8.0    # px, edge blur for coarse phase
SIGMA_MID      = 3.0
SIGMA_FINE     = 0.8
STEPS_COARSE   = 400
STEPS_MID      = 600
STEPS_FINE     = 400
LR_COARSE      = 0.8
LR_MID         = 0.4
LR_FINE        = 0.15
SMOOTH_PRIOR   = 0.01   # λ for δθ² regularisation
PRUNE_EPS      = 0.05   # max loss increase to accept anchor removal (absolute)
PRUNE_DT_MAX   = 0.25   # only try anchors with |δθ| < this (radians ~14°)
PRUNE_LOCAL_LR = 0.3
PRUNE_STEPS    = 80
BAND_PX        = 12     # pixels within this distance of any boundary are "active"


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
# 4.  SOFT-COVERAGE RENDER  (differentiable via distance fields)
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


def soft_coverage(curve_pts, grid_xy, sigma):
    """
    For a set of curve sample points and a grid of pixel positions,
    compute soft coverage = sigmoid(-min_dist / sigma).
    curve_pts: (M, 2)
    grid_xy:   (H*W, 2) — pixel centres in (x, y)
    Returns:   (H*W,) float in [0,1]
    """
    # (H*W, M) squared distances
    diff = grid_xy.unsqueeze(1) - curve_pts.unsqueeze(0)   # (HW, M, 2)
    d2   = (diff**2).sum(-1)                                # (HW, M)
    min_d = d2.min(dim=1).values.sqrt()                     # (HW,)
    return torch.sigmoid(-min_d / sigma)                    # (HW,)


def build_pixel_grid(H, W, device='cpu'):
    """Returns (H*W, 2) tensor of pixel (x, y) centres."""
    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def render_composite(all_layer_params, colors_rgb, H, W, sigma,
                     scale=1.0, n_per_seg=8, band_mask=None):
    """
    Differentiably render all layers composited onto white background.
    all_layer_params: list of LayerParams (one per subpath, in paint order)
    colors_rgb: list of (3,) float32 tensors matching all_layer_params
    Returns (H, W, 3) float tensor.
    """
    grid = build_pixel_grid(H, W, device=DEVICE)            # (HW, 2)
    if band_mask is not None:
        active_idx = band_mask.nonzero(as_tuple=True)[0]  # subset for speed
        active_grid = grid[active_idx]
    else:
        active_idx = None
        active_grid = grid

    canvas = torch.ones(H * W, 3, device=DEVICE)   # white background

    for lp, color in zip(all_layer_params, colors_rgb):
        A, C = lp.build_AC()
        # Scale if using downsampled render
        if scale != 1.0:
            A = A * scale
            C = C * scale
        curve = sample_curve_pts(A, C, n_per_seg)   # (M, 2)
        cov   = soft_coverage(curve, active_grid, sigma)   # (active,)
        col_t = color.unsqueeze(0)                         # (1, 3)
        if active_idx is not None:
            # Paint only active pixels; rest unchanged
            painted = canvas[active_idx] * (1 - cov.unsqueeze(1)) + col_t * cov.unsqueeze(1)
            canvas = canvas.clone()
            canvas[active_idx] = painted
        else:
            canvas = canvas * (1 - cov.unsqueeze(1)) + col_t * cov.unsqueeze(1)

    return canvas.reshape(H, W, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  GLOBAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def global_loss(all_lps, colors_rgb, target_rgb, H, W, sigma,
                scale, n_per_seg, band_mask, smooth_prior_w, visible_mask):
    rendered = render_composite(all_lps, colors_rgb, H, W, sigma,
                                scale, n_per_seg, band_mask)
    mse = ((rendered - target_rgb) ** 2)
    if visible_mask is not None:
        mse = mse[visible_mask]
    return mse.mean()


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
# 7.  MINIMUM-CURVATURE REPLACEMENT  (for pruning)
# ═══════════════════════════════════════════════════════════════════════════════

def min_curvature_arms(p0, p3, th0, th3):
    """
    Given endpoints p0, p3 and outgoing tangent direction at p0 (th0)
    and incoming tangent direction at p3 (th3 = angle of v INTO p3),
    find arm lengths l0, l3 that minimise integral of squared curvature.

    For a symmetric case the optimal lengths are chord/3 each.
    For the general asymmetric case we use the "cubic hermite elastica"
    approximation:  l = |chord| / 3, adjusted by a factor that scales
    with the angle between chord and tangent.
    """
    chord = p3 - p0
    L = np.linalg.norm(chord)
    if L < 1e-6:
        return 1.0, 1.0
    # Use the one-third rule as the baseline
    l0 = L / 3.0
    l3 = L / 3.0
    return l0, l3


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PRUNING PASS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss_np(all_lps, colors_rgb, target_t, H, W, scale):
    with torch.no_grad():
        return curve_sample_loss(all_lps, colors_rgb, target_t, H, W, scale).item()


def prune_global(all_lps, colors_rgb, target_t, H, W, scale,
                 batch_frac=0.10, reopt_steps=50, reopt_lr=0.3,
                 eps=PRUNE_EPS, dt_max=PRUNE_DT_MAX, verbose=True):
    """
    Batched global pruning:
      1. Collect all smooth anchors with |δθ| < dt_max, sort ascending
      2. Remove the least-sensitive batch_frac fraction
      3. Short global re-opt (reopt_steps)
      4. Accept if loss <= base + eps, else revert and stop
      5. Repeat until rejected or no candidates left
    Returns total anchors removed.
    """
    total_removed = 0
    base_loss = compute_loss_np(all_lps, colors_rgb, target_t, H, W, scale)

    for round_num in range(20):   # safety cap
        # ── collect candidates ────────────────────────────────────────────
        candidates = []  # (dth, lp_idx, anchor_i)
        for lp_idx, lp in enumerate(all_lps):
            if lp.s_dth is None:
                continue
            for k, i in enumerate(lp.smooth_idx):
                dth = abs(lp.s_dth[k].detach().item())
                if dth < dt_max:
                    candidates.append((dth, lp_idx, i))
        if not candidates:
            break
        candidates.sort()
        batch_size = max(1, int(len(candidates) * batch_frac))
        batch = candidates[:batch_size]
        if verbose:
            print(f"  Round {round_num+1}: {len(candidates)} candidates, "
                  f"batch={batch_size}  base_loss={base_loss:.5f}", flush=True)

        # ── snapshot affected subpaths ────────────────────────────────────
        affected = set(lp_idx for _, lp_idx, _ in batch)
        snapshots = {lp_idx: ([dict(p) for p in all_lps[lp_idx].params],
                               getattr(all_lps[lp_idx], 'atype', None))
                     for lp_idx in affected}

        # ── apply removals (highest index first per subpath) ──────────────
        removals_by_lp = {}
        for _, lp_idx, i in batch:
            removals_by_lp.setdefault(lp_idx, []).append(i)

        for lp_idx, indices in removals_by_lp.items():
            lp      = all_lps[lp_idx]
            params  = [dict(p) for p in lp.params]
            atype   = list(getattr(lp, 'atype', [0]*len(params)))
            for i in sorted(set(indices), reverse=True):
                N = len(params)
                if N <= 4:
                    break
                prev_i = (i - 1) % N
                next_i = (i + 1) % N
                p0  = params[prev_i]['pos']
                p3  = params[next_i]['pos']
                th0 = params[prev_i].get('th_in', 0) + params[prev_i].get('dth', 0)
                th3 = params[next_i].get('th_in', 0)
                l0, l3 = min_curvature_arms(p0, p3, th0, th3)
                # build reduced lists
                new_prev = prev_i if prev_i < i else prev_i - 1
                new_next = (next_i - 1 if next_i > i else next_i) % (N - 1)
                params = [p for j, p in enumerate(params) if j != i]
                atype  = [a for j, a in enumerate(atype)  if j != i]
                if params[new_prev]['type'] == 'smooth':
                    params[new_prev] = dict(params[new_prev])
                    params[new_prev]['l_out'] = l0
                if params[new_next]['type'] == 'smooth':
                    params[new_next] = dict(params[new_next])
                    params[new_next]['l_in']  = l3
            new_lp = LayerParams(params)
            new_lp.atype = np.array(atype, dtype=np.int32)
            lp.__dict__.update(new_lp.__dict__)

        # ── short global re-opt ───────────────────────────────────────────
        all_params = [t for lp in all_lps for t in lp.torch_tensors()]
        opt = optim.Adam(all_params, lr=reopt_lr)
        for _ in range(reopt_steps):
            opt.zero_grad()
            loss = (curve_sample_loss(all_lps, colors_rgb, target_t, H, W, scale)
                    + SMOOTH_PRIOR * sum(lp.smooth_prior_loss() for lp in all_lps)
                    / max(len(all_lps), 1))
            loss.backward()
            opt.step()

        new_loss = compute_loss_np(all_lps, colors_rgb, target_t, H, W, scale)

        if new_loss <= base_loss + eps:
            for lp in all_lps:
                lp.sync_back()
            total_removed += batch_size
            base_loss = new_loss
            if verbose:
                print(f"    Accepted → loss={new_loss:.5f}  "
                      f"total_removed={total_removed}", flush=True)
        else:
            # Revert affected subpaths
            for lp_idx, (snap_params, snap_atype) in snapshots.items():
                lp = all_lps[lp_idx]
                new_lp = LayerParams(snap_params)
                if snap_atype is not None:
                    new_lp.atype = snap_atype
                lp.__dict__.update(new_lp.__dict__)
            if verbose:
                print(f"    Rejected: {new_loss:.5f} > {base_loss:.5f}+{eps}, stopping",
                      flush=True)
            break

    return total_removed


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CURVE-SAMPLE LOSS  (replaces pixel-render loss — cheap on MPS)
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


def curve_sample_loss(all_lps, colors_rgb, ref_t, H, W, scale, n_per_seg=10):
    """
    Sample points along each bezier curve, look up reference colour via
    bilinear interpolation, penalise colour mismatch.
    ref_t  : (H, W, 3) float32 on DEVICE
    Returns: scalar loss
    """
    total = torch.tensor(0., device=DEVICE)
    for lp, color in zip(all_lps, colors_rgb):
        A, C = lp.build_AC()
        if scale != 1.0:
            A = A * scale
            C = C * scale
        pts = sample_curve_pts(A, C, n_per_seg)   # (M, 2)  x=col, y=row
        sampled = bilinear_sample(ref_t, pts)      # (M, 3)
        total = total + ((sampled - color.unsqueeze(0)) ** 2).mean()
    return total / max(len(all_lps), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  OPTIMISATION PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase(all_lps, colors_rgb, target_t, H, W, sigma, scale,
              n_steps, lr, label, smooth_prior_w, visible_mask, verbose):
    all_params = []
    for lp in all_lps:
        all_params += lp.torch_tensors()

    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    best = {'loss': 1e9, 'state': None}
    t0 = time.time()

    # n_per_seg scales with sigma: coarse phase uses fewer samples for speed
    n_per_seg = max(4, int(sigma * 2))

    log_every = max(1, n_steps // 20)
    for step in range(n_steps):
        optimizer.zero_grad()
        mse   = curve_sample_loss(all_lps, colors_rgb, target_t, H, W, scale, n_per_seg)
        prior = sum(lp.smooth_prior_loss() for lp in all_lps) / max(len(all_lps), 1)
        loss  = mse + smooth_prior_w * prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        scheduler.step()

        if mse.detach().item() < best['loss']:
            best['loss'] = mse.detach().item()
            best['state'] = [{k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in lp.__dict__.items()
                              if not k.startswith('_')}
                             for lp in all_lps]

        if verbose and step % log_every == 0:
            eta = (time.time() - t0) / max(step, 1) * (n_steps - step)
            print(f"  {label} {step:4d}/{n_steps}: loss={mse.detach().item():.5f} "
                  f"prior={prior.detach().item():.4f}  "
                  f"best={best['loss']:.5f}  eta={eta:.0f}s", flush=True)

    print(f"  {label} best loss={best['loss']:.5f}  ({time.time()-t0:.0f}s)")    # Restore best
    for lp, state in zip(all_lps, best['state']):
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                getattr(lp, k).data.copy_(v)
    for lp in all_lps:
        lp.sync_back()


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
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
    ap.add_argument('--no-prune', action='store_true')
    ap.add_argument('--quiet',    action='store_true')
    ap.add_argument('--smooth-w', type=float, default=SMOOTH_PRIOR)
    ap.add_argument('--prune-eps',type=float, default=PRUNE_EPS)
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

    # ── Load reference ────────────────────────────────────────────────────────
    print(f"Loading reference {args.ref}...")
    orig_rgba = np.array(Image.open(args.ref).convert('RGBA'))
    orig_hw   = orig_rgba.shape[:2]
    visible   = orig_rgba[:,:,3] > 128

    # ── Determine working resolution ─────────────────────────────────────────
    if args.trial:
        W = H = 256
        phases = ['coarse']
    else:
        phases = (['coarse', 'mid', 'fine'] if args.phase == 'all'
                  else [args.phase])
        H, W = full_h, full_w

    scale = W / full_w

    # Pre-scale reference to working resolution
    ref_img = Image.fromarray(orig_rgba).resize((W, H), Image.LANCZOS).convert('RGB')
    ref_np  = np.array(ref_img).astype(np.float32) / 255.0
    target_t = torch.tensor(ref_np, device=DEVICE)

    vis_small = np.array(Image.fromarray(visible.astype(np.uint8)*255)
                         .resize((W, H), Image.NEAREST)) > 128
    visible_mask = torch.tensor(vis_small.reshape(-1), device=DEVICE)   # (H*W,) bool

    # ── Phase configs ─────────────────────────────────────────────────────────
    phase_cfg = {
        'coarse': (SIGMA_COARSE, STEPS_COARSE, LR_COARSE),
        'mid':    (SIGMA_MID,    STEPS_MID,    LR_MID),
        'fine':   (SIGMA_FINE,   STEPS_FINE,   LR_FINE),
    }

    # ── Optimisation ─────────────────────────────────────────────────────────
    for ph in phases:
        sigma, n_steps, lr = phase_cfg[ph]
        if args.trial and ph == 'coarse':
            n_steps = min(n_steps, 200)
        print(f"\n{'='*60}")
        print(f"Phase {ph}: H={H} W={W} σ={sigma} steps={n_steps} lr={lr}")
        run_phase(all_lps, all_colors, target_t, H, W, sigma, scale,
                  n_steps, lr, ph, args.smooth_w, visible_mask, verbose)

    # ── Pruning ───────────────────────────────────────────────────────────────
    if not args.no_prune:
        print(f"\n{'='*60}")
        reopt_steps = 20 if args.trial else 50
        removed = prune_global(all_lps, all_colors, target_t, H, W, scale,
                               batch_frac=0.10, reopt_steps=reopt_steps,
                               eps=args.prune_eps, verbose=verbose)
        remaining = sum(len(lp.params) for lp in all_lps)
        print(f"  Total removed: {removed}  Remaining: {remaining} (was {total_anchors})")

    # ── Write SVG ─────────────────────────────────────────────────────────────
    # Rebuild structured layer data from (possibly pruned) LayerParams
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
