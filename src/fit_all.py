"""
png-vectorizer  —  fit_all.py
==============================
Converts a raster PNG into a clean multi-layer SVG using divide-and-conquer
least-squares cubic Bézier fitting.

Algorithm
---------
1. Flood-fill background to transparent (optional --flood-bg).
2. Quantize visible pixels to N colors via k-means in LAB space.
3. For each color region, trace the contour (outer + holes).
4. Detect corners by curvature; coalesce weak corners.
5. Fit each corner-to-corner arc with the minimum number of cubic Bézier
   spans that keep the max per-point error within MAX_BEZIER_ERR_PX (D&C
   least-squares, no gradient descent, no spikes possible).
6. Write SVG with one <path> per region; render preview + diff PNGs.

Usage
-----
    python fit_all.py                                  # default input
    python fit_all.py --input logo.png --colors 6
    python fit_all.py --input logo.png --flood-bg      # remove background
    python fit_all.py --input logo.png --flood-bg --flood-tol 60

License: MIT
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage import measure
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import distance_transform_edt, binary_erosion
from sklearn.cluster import KMeans
import cairosvg, io, time, os

# ── Backend: prefer MLX on Apple Silicon, fall back to PyTorch ────────────────
try:
    import mlx.core as mx
    import mlx.nn as mxnn
    import mlx.optimizers as mxopt
    USE_MLX = True
except ImportError:
    USE_MLX = False

if USE_MLX:
    print("[backend] MLX (Apple GPU)")
else:
    print("[backend] PyTorch CPU")

# ── Tunable parameters ────────────────────────────────────────────────────────
N_PIX                 = 10
MAX_GAP               = np.sqrt(2) * N_PIX
SHARP_THRESH          = 40.0
TGT_SAMPLE            = 150
N_STEPS_SEG           = 200   # optimisation steps per bezier segment (initial fit)
N_STEPS_REFIT         = 50    # gradient descent steps after each anchor removal
LR_SEG                = 1.5   # learning rate for per-segment optimiser
PRUNE_PIXEL_TOLERANCE = 20    # max pixel-wedge area to allow anchor pruning
MAX_BEZIER_ERR_PX     =  1.0  # max per-point error (px) for one bezier span before splitting
C1_WEIGHT             =  1.0
PRUNE_ANGLE_THRESH    =  4.0
MIN_REGION_PX         = 20    # skip regions smaller than this
DILATION_PX           =   0   # expand each mask by this many px to close seams

# Speed options (all exposed on the CLI)
KMEANS_SAMPLE         = 0       # fit k-means on this many pixels (0 = all), then predict all
KMEANS_N_INIT         = 3       # k-means restarts
BEZIER_SEARCH         = 'kdtree'  # nearest-point search for fit error: 'kdtree' or 'brute'
REGION_CROP           = True    # contour each region inside its bounding box, not the full image
CORNER_COALESCE       = 'dc'  # 'dc' (closed-form), 'gradient' (original, slow), or 'off'

# Progress tracking (updated in main)
_progress = {'done': 0, 'total': 0}


# ── Image loading ─────────────────────────────────────────────────────────────
def remove_black_background(arr):
    bg = (arr[:,:,0]<10)&(arr[:,:,1]<10)&(arr[:,:,2]<10)
    labeled,_ = ndi.label(bg)
    cl = set()
    h,w = arr.shape[:2]
    for cy,cx in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
        cl.add(labeled[cy,cx])
    cl.discard(0)
    arr = arr.copy()
    arr[np.isin(labeled,list(cl)), 3] = 0
    return arr


def flood_fill_background(arr, tol=40, shadow_tol=0.08):
    """Flood-fill background to transparent by sampling the corner pixel color.
    Finds all connected components of pixels within `tol` RGB distance of the
    most common corner color, then makes corner-touching components transparent."""
    arr = arr.copy()
    h, w = arr.shape[:2]
    corners = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]
    # Pick the most common corner color (ignoring fully-transparent corners)
    corner_rgbs = [arr[cy,cx,:3].astype(float)
                   for cy,cx in corners if arr[cy,cx,3] > 10]
    if not corner_rgbs:
        return arr
    bg_color = np.mean(corner_rgbs, axis=0)   # average corner color
    # Mask: pixels within `tol` of bg_color
    diff = np.linalg.norm(arr[:,:,:3].astype(float) - bg_color, axis=2)
    bg_mask = diff <= tol
    if shadow_tol > 0:
        # Drop shadows keep the background hue but are darker, so they fall
        # outside the RGB tolerance. Match on colour direction instead, and keep
        # only matches connected to the background so logo pixels are never hit.
        rgb = arr[:,:,:3].astype(float)
        norm = np.linalg.norm(rgb, axis=2)
        unit = rgb / np.maximum(norm, 1e-6)[..., None]
        bg_unit = bg_color / max(np.linalg.norm(bg_color), 1e-6)
        shadow = (np.linalg.norm(unit - bg_unit, axis=2) <= shadow_tol) & (norm > 30)
        lab, _ = ndi.label(shadow | bg_mask)
        touching = np.unique(np.concatenate([lab[0], lab[-1], lab[:,0], lab[:,-1]]))
        bg_mask = bg_mask | (shadow & np.isin(lab, touching[touching > 0]))
    # Make all matching background pixels transparent, including holes that
    # are isolated from the image corners. This avoids filling isolated loops
    # inside letters like the R when the interior is the same background color.
    arr[bg_mask, 3] = 0
    return arr


def extract_color_masks(arr, n_colors=8, min_px=MIN_REGION_PX):
    """
    Cluster visible pixels into n_colors groups using k-means in LAB space
    (perceptually uniform — prevents teal from collapsing into black).
    Returns list of (hex_color, mask) sorted by pixel count, largest first.
    """
    visible = arr[:,:,3] > 128
    if not visible.any():
        return []

    h, w = arr.shape[:2]
    vis_ys, vis_xs = np.where(visible)
    # float64 avoids the divide-by-zero/overflow warnings seen with float32 input
    vis_rgb = arr[vis_ys, vis_xs, :3].astype(np.float64) / 255.0

    # K-means in LAB space for perceptually correct separation
    vis_lab = rgb2lab(vis_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # Normalise so each channel has roughly equal weight
    scale = np.array([100.0, 128.0, 128.0])
    vis_norm = vis_lab / scale

    km = KMeans(n_clusters=n_colors, init='k-means++', n_init=KMEANS_N_INIT,
                max_iter=100, random_state=42)
    if 0 < KMEANS_SAMPLE < len(vis_norm):
        rng = np.random.default_rng(42)
        sample = vis_norm[rng.choice(len(vis_norm), KMEANS_SAMPLE, replace=False)]
        km.fit(sample)
        labels = km.predict(vis_norm)
    else:
        labels = km.fit_predict(vis_norm)
    centroids_norm = km.cluster_centers_

    # Convert centroids back to sRGB hex
    centroids_lab = centroids_norm * scale
    centroids_rgb = (lab2rgb(centroids_lab.reshape(1, -1, 3)).reshape(-1, 3)
                     * 255).clip(0, 255).astype(np.uint8)

    # Map labels back to full-image grid
    q_full = np.full((h, w), -1, dtype=np.int32)
    q_full[vis_ys, vis_xs] = labels

    results = []
    for ci in range(n_colors):
        mask = (q_full == ci)
        px_count = int(mask.sum())
        if px_count < min_px:
            continue
        # Avoid morphological opening/closing here because it can close small
        # interior counters (e.g. the hole in the 'R'). Keep the mask shape
        # as-is and only apply optional dilation for seam closure.
        if DILATION_PX > 0:
            mask = ndi.binary_dilation(mask, iterations=DILATION_PX) & visible
        if not mask.any():
            continue
        color = centroids_rgb[ci]
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        results.append((hex_color, mask, px_count))

    # Sort by containment depth: a layer whose centroid falls inside another
    # layer is more "foreground" and should paint LAST so it appears on top.
    # This fixes cases like yellow (letter fill) being painted before teal
    # (letter outline), which reverses their correct z-order.
    masks_only = [m for _, m, _ in results]
    def containment_depth(mask):
        cy, cx = ndi.center_of_mass(mask)
        cy, cx = int(round(cy)), int(round(cx))
        cy = max(0, min(mask.shape[0]-1, cy))
        cx = max(0, min(mask.shape[1]-1, cx))
        return sum(1 for other in masks_only if other[cy, cx])

    results.sort(key=lambda x: containment_depth(x[1]))

    # Build quantized RGBA: each visible pixel → its centroid RGB, alpha=255.
    # This de-aliases boundaries by mapping blended edge pixels to their
    # nearest color, giving a clean reference for diff scoring.
    quantized = np.zeros((h, w, 4), dtype=np.uint8)
    quantized[vis_ys, vis_xs, :3] = centroids_rgb[labels]
    quantized[vis_ys, vis_xs, 3]  = 255

    return [(hx, m) for hx, m, _ in results], quantized


# ── Contour extraction ────────────────────────────────────────────────────────
def get_contours(mask):
    contours = measure.find_contours(mask.astype(float), level=0.5)
    outer = None; holes = []
    for c in contours:
        x,y = c[:,1],c[:,0]
        sa  = 0.5*np.sum(x[:-1]*y[1:]-x[1:]*y[:-1])
        if sa > 0: outer = c
        else:      holes.append(c)
    return outer, holes


# ── Contour analysis ──────────────────────────────────────────────────────────
def smooth_contour(pts, sigma=2.0):
    m = min(len(pts), 30)
    ext = np.vstack([pts[-m:], pts, pts[:m]])
    out = np.column_stack([gaussian_filter1d(ext[:,0], sigma),
                           gaussian_filter1d(ext[:,1], sigma)])
    return out[m:m+len(pts)]

def curvature_profile(s, window=15):
    n=len(s); curv=np.zeros(n)
    for i in range(n):
        v1=s[i]-s[(i-window)%n]; v2=s[(i+window)%n]-s[i]
        l1=np.linalg.norm(v1); l2=np.linalg.norm(v2)
        if l1>0 and l2>0:
            curv[i]=np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(l1*l2),-1,1)))
    return curv

def find_corners(s, curv, thresh=15.0, min_gap=20, min_dist_px=15.0):
    """Detect curvature-peak corners, deduplicating by both index gap and
    pixel-space distance to prevent spike artifacts at narrow pinch points."""
    n=len(curv); out=[]
    cands=[(curv[i],i) for i in range(n) if curv[i]>thresh and
           curv[i]==curv[max(0,i-min_gap//2):min(n,i+min_gap//2)].max()]
    for a,c in sorted(cands,reverse=True):
        if not out or (all(abs(c-o)>=min_gap for o in out) and
                       all(np.linalg.norm(s[c]-s[o])>=min_dist_px for o in out)):
            out.append(c)
    return sorted(out)

def coalesce_corners_dc(s, curv, corners, max_err_px):
    """
    Closed-form corner pruning. A corner is removed when fitting the merged
    arc (neighbouring corners joined) needs no more bezier spans than the
    two arcs on either side of it need separately. Both fits are bounded by
    max_err_px, so fewer-or-equal spans means no loss in accuracy per anchor.
    Weakest-curvature corners are tried first; no gradient descent involved.
    """
    n = len(s)
    cache = {}

    def arc_of(a, b):
        return s[a:b + 1] if b > a else np.vstack([s[a:], s[:b + 1]])

    def spans(a, b):
        key = (a, b)
        if key not in cache:
            cache[key] = len(_fit_arc_divide(arc_of(a, b), max_err_px))
        return cache[key]

    corners = sorted(corners)
    changed = True
    while changed and len(corners) > 2:
        changed = False
        for ci in sorted(range(len(corners)), key=lambda i: curv[corners[i]]):
            prev_c = corners[(ci - 1) % len(corners)]
            cur_c  = corners[ci]
            next_c = corners[(ci + 1) % len(corners)]
            if spans(prev_c, next_c) <= spans(prev_c, cur_c) + spans(cur_c, next_c):
                corners = corners[:ci] + corners[ci + 1:]
                changed = True
                break
    return corners


def coalesce_corners_by_loss(s, curv, corners, tgt, quick_steps=40, tol=0.05):
    """
    Greedy: remove weakest corners whose absence doesn't hurt the fit.
    For each candidate (ascending curvature), run a quick trial fit without
    it and accept if geom loss stays within tol of the current baseline.
    """
    def quick_loss(clist):
        A_t, at_t = place_anchors(s, curv, clist)
        if len(A_t) < 3:
            return 1e9
        r = run_phase(A_t, init_ctrl(A_t), tgt, at_t, 0, False,
                      C1_WEIGHT, quick_steps, LR_SEG, "", False)
        return r['loss']

    corners = list(corners)
    if len(corners) < 3:
        return corners
    baseline = quick_loss(corners)
    changed = True
    while changed and len(corners) > 2:
        changed = False
        order = sorted(range(len(corners)), key=lambda i: curv[corners[i]])
        for ci in order:
            if len(corners) <= 2:
                break
            trial = corners[:ci] + corners[ci+1:]
            loss = quick_loss(trial)
            if loss <= baseline * (1 + tol):
                corners = trial
                baseline = loss
                changed = True
                break
    return sorted(corners)

def place_anchors(s, curv, corners, max_gap=MAX_GAP):
    n=len(s); nc=len(corners); anchor_idx=[]; atype=[]
    for si in range(nc):
        c0=corners[si]; c1=corners[(si+1)%nc]
        seg_idx=(np.arange(c0,c1+1) if c1>c0
                 else np.concatenate([np.arange(c0,n),np.arange(0,c1+1)]))
        if len(seg_idx)==0: continue
        seg_pts=s[seg_idx%n]; seg_curv=np.abs(curv[seg_idx%n])
        diffs=np.diff(seg_pts,axis=0)
        arclen=np.concatenate([[0],np.cumsum(np.linalg.norm(diffs,axis=1))])
        total=arclen[-1]; n_int=max(1,int(np.ceil(total/max_gap)))

        anchor_idx.append(int(seg_idx[0]))
        atype.append(2 if curv[c0]>=SHARP_THRESH else 1)

        if n_int>1:
            cum=np.cumsum(seg_curv); tot=cum[-1]
            fracs=(list(np.linspace(0,1,n_int+1)[1:-1]) if tot<1e-6
                   else [arclen[np.clip(np.searchsorted(cum,tc),0,len(arclen)-1)]/(total+1e-9)
                         for tc in np.linspace(0,tot,n_int+1)[1:-1]])
            final=[]; prev=0.0
            for frac in sorted(fracs):
                cur=frac*total; gap=cur-prev
                if gap>max_gap:
                    ne=int(np.ceil(gap/max_gap))-1
                    for k in range(1,ne+1): final.append((prev+k*gap/(ne+1))/total)
                final.append(frac); prev=cur
            gap=total-prev
            if gap>max_gap:
                ne=int(np.ceil(gap/max_gap))-1
                for k in range(1,ne+1): final.append((prev+k*gap/(ne+1))/total)
            for frac in final:
                i_s=int(np.argmin(np.abs(arclen-frac*total)))
                anchor_idx.append(int(seg_idx[min(i_s,len(seg_idx)-1)])); atype.append(0)

    return s[np.array(anchor_idx)%n], np.array(atype,dtype=int)

def init_ctrl(A):
    N=len(A); C=np.zeros((N,2,2))
    for i in range(N):
        j=(i+1)%N; C[i,0]=A[i]+(A[j]-A[i])/3; C[i,1]=A[i]+2*(A[j]-A[i])/3
    return C


# ── Loss functions ────────────────────────────────────────────────────────────
if USE_MLX:
    def _make_tgt(s_np):
        return mx.array(s_np.astype(np.float32))

    def _bezier_samples_mx(p0,p1,p2,p3,n=30):
        t=mx.linspace(0,1,n)[:,None]; mt=1-t
        return mt**3*p0+3*mt**2*t*p1+3*mt*t**2*p2+t**3*p3

    def _c1_loss_mx(A, C, atype):
        N=len(atype); loss=mx.array(0.); cnt=0
        for i in range(N):
            if atype[i]==2: continue
            out_dir=A[i]-C[(i-1)%N,1]; in_dir=C[i,0]-A[i]
            cross=out_dir[0]*in_dir[1]-out_dir[1]*in_dir[0]
            loss=loss+cross**2; cnt+=1
        return loss/max(cnt,1)

    def _geom_loss_mx(A, C, tgt_all, n_sample=TGT_SAMPLE):
        N=A.shape[0]
        perm=np.random.permutation(tgt_all.shape[0])[:n_sample]
        tgt=tgt_all[mx.array(perm)]
        segs=[_bezier_samples_mx(A[i],C[i,0],C[i,1],A[(i+1)%N]) for i in range(N)]
        all_pts=mx.concatenate(segs,axis=0)
        sq=(tgt[:,None,:]-all_pts[None,:,:])**2    # (S,M,2)
        sq=sq.sum(-1)                              # (S,M)
        fwd=sq.min(axis=1).mean()   # each target   → nearest bezier pt
        rev=sq.min(axis=0).mean()   # each bezier pt → nearest target (prevents spikes)
        return (fwd + rev) * 0.5

    def run_phase(anp, cnp, tgt_t, atype, corner_penalty, freeze_corners,
                  c1w, n_steps, lr, label, verbose):
        is_corner=(atype>0)
        fc_idx_np=np.where(is_corner)[0]
        cs=mx.array(anp[is_corner].astype(np.float32))
        fc_idx=mx.array(fc_idx_np)
        C=mx.array(cnp.astype(np.float32))

        if freeze_corners:
            fixed=mx.array(anp[is_corner].astype(np.float32))
            fm=~is_corner
            fm_idx=mx.array(np.where(fm)[0])
            free=mx.array(anp[fm].astype(np.float32))
            params={'free':free,'C':C}
            def mkA():
                A=mx.zeros((len(atype),2))
                A[fc_idx]=fixed
                A[fm_idx]=params['free']
                return A
        else:
            params={'A':mx.array(anp.astype(np.float32)),'C':C}
            def mkA(): return params['A']

        optimizer=mxopt.Adam(learning_rate=lr)
        best={'loss':1e9}

        def loss_fn(p):
            if freeze_corners:
                A2=mx.zeros((len(atype),2))
                A2[fc_idx]=fixed
                A2[fm_idx]=p['free']
                anch=A2
            else:
                anch=p['A']
            Cm=p['C']
            lc=_geom_loss_mx(anch,Cm,tgt_t,TGT_SAMPLE)
            lc1=_c1_loss_mx(anch,Cm,atype)
            if not freeze_corners and corner_penalty>0 and len(fc_idx_np)>0:
                lreg=((anch[fc_idx]-cs)**2).mean()
            else:
                lreg=mx.array(0.)
            return lc+c1w*lc1+corner_penalty*lreg, (lc,lc1,lreg)

        loss_and_grad=mx.value_and_grad(loss_fn)

        for step in range(n_steps):
            (loss_val,(lc,lc1,lreg)),grads=loss_and_grad(params)
            mx.eval(loss_val,grads)
            # gradient clipping
            for k,g in grads.items():
                norm=float(mx.sqrt((g**2).sum()))
                if norm>3.0: grads[k]=g*(3.0/norm)
            optimizer.update(params,grads)
            mx.eval(params)
            lv=float(lc)
            if lv<best['loss']:
                A_np=np.array(mkA())
                C_np=np.array(params['C'])
                best.update({'loss':lv,'A':A_np.copy(),'C':C_np.copy()})
            if verbose and step%max(1,n_steps//4)==0:
                print(f"      {label} {step:4d}: geom={float(lc):.4f} c1={float(lc1):.4f} reg={float(lreg):.4f}")
        return best

else:
    def _make_tgt(s_np):
        return torch.tensor(s_np, dtype=torch.float32)

    def bezier_samples(p0,p1,p2,p3,n=30):
        t=torch.linspace(0,1,n).unsqueeze(1); mt=1-t
        return mt**3*p0+3*mt**2*t*p1+3*mt*t**2*p2+t**3*p3

    def c1_loss(A, C, atype):
        N=len(A); loss=torch.tensor(0.); cnt=0
        for i in range(N):
            if atype[i]==2: continue
            out_dir=A[i]-C[(i-1)%N,1]; in_dir=C[i,0]-A[i]
            cross=out_dir[0]*in_dir[1]-out_dir[1]*in_dir[0]
            loss=loss+cross**2; cnt+=1
        return loss/max(cnt,1)

    def geom_loss(A, C, tgt_all, n_sample=TGT_SAMPLE):
        N=len(A)
        idx=torch.randperm(len(tgt_all))[:n_sample]
        tgt=tgt_all[idx]
        all_pts=torch.cat([bezier_samples(A[i],C[i,0],C[i,1],A[(i+1)%N]) for i in range(N)],0)
        sq=(tgt.unsqueeze(1)-all_pts.unsqueeze(0))**2  # (S,M,2)
        sq=sq.sum(-1)                                   # (S,M)
        fwd=sq.min(dim=1).values.mean()   # each target   → nearest bezier pt
        rev=sq.min(dim=0).values.mean()   # each bezier pt → nearest target (prevents spikes)
        return (fwd + rev) * 0.5

    def run_phase(anp, cnp, tgt_t, atype, corner_penalty, freeze_corners,
                  c1w, n_steps, lr, label, verbose):
        is_corner=(atype>0)
        cs=torch.tensor(anp[is_corner],dtype=torch.float32)
        C=torch.tensor(cnp,dtype=torch.float32,requires_grad=True)

        if freeze_corners:
            fixed=torch.tensor(anp[is_corner],dtype=torch.float32)
            fm=~is_corner
            free=torch.tensor(anp[fm],dtype=torch.float32,requires_grad=True)
            params=[free,C]
            def mkA():
                A=torch.zeros(len(atype),2); A[is_corner]=fixed; A[fm]=free; return A
        else:
            A=torch.tensor(anp,dtype=torch.float32,requires_grad=True)
            params=[A,C]
            def mkA(): return A

        opt=optim.Adam(params,lr=lr)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_steps)
        best={'loss':1e9}

        for step in range(n_steps):
            opt.zero_grad()
            anch=mkA()
            lc  =geom_loss(anch,C,tgt_t)
            lc1 =c1_loss(anch,C,atype)
            lreg=((anch[is_corner]-cs)**2).mean() \
                 if (not freeze_corners and corner_penalty>0 and is_corner.any()) \
                 else torch.tensor(0.)
            loss=lc+c1w*lc1+corner_penalty*lreg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,3.0)
            opt.step(); sch.step()
            lv=lc.item()
            if lv<best['loss']:
                best.update({'loss':lv,'A':mkA().detach().numpy().copy(),'C':C.detach().numpy().copy()})
            if verbose and step%max(1,n_steps//4)==0:
                print(f"      {label} {step:4d}: geom={lc:.4f} c1={lc1:.4f} reg={lreg:.4f}")
        return best

def prune_short_chords(A, atype, min_chord=None):
    """Remove anchors that form degenerate sub-threshold chord segments.
    Iterates until stable, always removing the lower-priority anchor
    (atype 0 < 1 < 2) and requiring at least 4 anchors remain."""
    if min_chord is None:
        min_chord = MAX_GAP / 3
    changed = True
    while changed:
        N = len(A)
        if N < 5:
            break
        changed = False
        for i in range(N):
            j = (i + 1) % N
            if np.linalg.norm(A[j] - A[i]) < min_chord:
                victim = j if atype[i] >= atype[j] else i
                keep = np.ones(N, bool); keep[victim] = False
                if keep.sum() >= 4:
                    A = A[keep]; atype = atype[keep]
                    changed = True
                    break
    return A, atype


def _sample_bezier_np(p0, c0, c1, p1, n=30):
    """Sample n+1 points on cubic bezier p0..p1 (numpy, row/col coords)."""
    t = np.linspace(0, 1, n + 1)
    mt = 1 - t
    return ((mt**3)[:,None]*p0 + 3*(mt**2*t)[:,None]*c0 +
            3*(mt*t**2)[:,None]*c1 + (t**3)[:,None]*p1)


def _anchor_removal_score(A, C, i, mask):
    """
    Score for removing anchor i: count sample points where the replacement
    bezier and the original two spans disagree on which side of the mask
    boundary they're on.  O(N) mask lookups — no rasterization needed.
    """
    N    = len(A)
    prev = (i - 1) % N
    nxt  = (i + 1) % N
    n    = 30
    # Original: two spans, concatenated at same parameter density
    b1   = _sample_bezier_np(A[prev], C[prev, 0], C[prev, 1], A[i],  n=n)
    b2   = _sample_bezier_np(A[i],   C[i,   0], C[i,   1], A[nxt], n=n)
    orig = np.vstack([b1, b2[1:]])          # (2n+1, 2)
    # Replacement: single span resampled to the same length
    rep_raw = _sample_bezier_np(A[prev], C[prev, 0], C[i, 1], A[nxt], n=2*n)
    H, W = mask.shape
    def _lookup(pts):
        r = pts[:, 0].clip(0, H - 1).astype(int)
        c = pts[:, 1].clip(0, W - 1).astype(int)
        return mask[r, c]
    orig_in = _lookup(orig)
    rep_in  = _lookup(rep_raw)
    return int((orig_in != rep_in).sum())


def _lstsq_fit_bezier(p0, p3, arc):
    """
    Fit cubic bezier with fixed endpoints p0, p3 to arc via least squares.
    Uses chord-length parameterization. Returns (c0, c1) control points.
    Guaranteed spike-free: control points are the closed-form optimal solution.
    """
    if len(arc) < 2:
        return p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3
    dists  = np.linalg.norm(np.diff(arc, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total  = cumlen[-1]
    if total < 1e-9:
        return p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3
    t   = cumlen / total
    a   = 3.0 * (1 - t)**2 * t      # basis weight for c0
    b   = 3.0 * (1 - t) * t**2      # basis weight for c1
    rhs = arc - (1 - t)[:, None]**3 * p0 - t[:, None]**3 * p3
    A_mat = np.column_stack([a, b])  # (n, 2)
    sol, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
    return sol[0].copy(), sol[1].copy()


def _bezier_max_error(p0, c0, c1, p3, arc):
    """
    Max distance from any arc point to the nearest point on the bezier.
    Used as the splitting criterion in divide-and-conquer fitting.
    """
    return float(_bezier_point_dists(p0, c0, c1, p3, arc).max())


def _bezier_point_dists(p0, c0, c1, p3, arc):
    """Distance from each arc point to the nearest sampled point on the bezier."""
    n_samp = max(len(arc) * 2, 64)
    bpts   = _sample_bezier_np(p0, c0, c1, p3, n=n_samp)
    if BEZIER_SEARCH == 'kdtree':
        return cKDTree(bpts).query(arc)[0]
    diff = arc[:, None, :] - bpts[None, :, :]       # (M, N, 2)
    return np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)


def _fit_arc_divide(arc, max_err_px, min_arc_pts=8):
    """
    Divide-and-conquer: fit arc with the fewest bezier spans such that
    every span has max_error <= max_err_px (measured against arc points).

    Returns list of (endpoint, c0, c1) — one entry per bezier span,
    where endpoint is the span's terminal anchor position and (c0, c1)
    are the control points for the bezier leaving the *previous* anchor.
    """
    p0, p3   = arc[0], arc[-1]
    c0, c1   = _lstsq_fit_bezier(p0, p3, arc)
    dists    = _bezier_point_dists(p0, c0, c1, p3, arc)
    if float(dists.max()) <= max_err_px or len(arc) < min_arc_pts:
        return [(p3, c0, c1)]
    # Find worst-fitting interior arc point
    dists    = dists.copy()
    dists[0] = 0.0; dists[-1] = 0.0
    split  = int(np.argmax(dists))
    if split <= 0 or split >= len(arc) - 1:
        return [(p3, c0, c1)]
    left  = _fit_arc_divide(arc[:split + 1], max_err_px, min_arc_pts)
    right = _fit_arc_divide(arc[split:],     max_err_px, min_arc_pts)
    return left + right


def _fit_seg_range(A, C, at, ai_start, ai_end, s, n_steps):
    """
    Gradient-descent fit for the segment from corner anchor ai_start to ai_end
    (circular indexing).  Uses existing C values as warm start.
    Returns updated (A, C) — only the segment's entries are modified.
    """
    N = len(A)
    if ai_end > ai_start:
        seg_ai = list(range(ai_start, ai_end + 1))
    else:
        seg_ai = list(range(ai_start, N)) + list(range(0, ai_end + 1))
    M = len(seg_ai)
    # Skip if no free anchors between the two corner endpoints
    if not any(at[j] == 0 for j in seg_ai[1:-1]):
        return A, C
    # Contour arc: nearest-neighbour lookup in s for both endpoints
    si_start = int(np.argmin(np.sum((s - A[ai_start]) ** 2, axis=1)))
    si_end   = int(np.argmin(np.sum((s - A[ai_end])   ** 2, axis=1)))
    if si_end > si_start:
        arc = s[si_start:si_end + 1]
    else:
        arc = np.vstack([s[si_start:], s[:si_end + 1]])
    if len(arc) < 2:
        return A, C
    seg_A  = A[np.array(seg_ai)]
    seg_at = at[np.array(seg_ai)]
    seg_C  = C[np.array(seg_ai)]          # warm start from existing control points
    result = run_phase(seg_A, seg_C, _make_tgt(arc),
                       seg_at, 0, True, C1_WEIGHT, n_steps, LR_SEG, "", False)
    for k, ai in enumerate(seg_ai):
        A[ai] = result['A'][k]
    for k in range(M - 1):
        C[seg_ai[k]] = result['C'][k]
    return A, C


def fit_contour(pts, mask, label, verbose=True, is_outer=False):
    if len(pts) < 8:
        return None, None, None
    t0   = time.time()
    H, W = mask.shape

    # Step 1: smoothed edge + curvature
    s    = smooth_contour(pts, 2.0)
    curv = curvature_profile(s, 15)
    n    = len(s)

    # Step 2: corners
    corners  = find_corners(s, curv, 15.0, 50)
    tgt_full = _make_tgt(s)
    if len(corners) >= 2:
        if CORNER_COALESCE == 'gradient':
            corners = coalesce_corners_by_loss(s, curv, corners, tgt_full)
        elif CORNER_COALESCE == 'dc':
            corners = coalesce_corners_dc(s, curv, corners, MAX_BEZIER_ERR_PX)
    if len(corners) < 2:
        corners = [i * n // 4 for i in range(4)]

    # Steps 3-4: D&C least-squares fit per corner-to-corner arc
    # For each arc, _fit_arc_divide returns (endpoint, c0, c1) per bezier span.
    # C[i,0] = first ctrl (near A[i]), C[i,1] = second ctrl (near A[i+1]).
    nc        = len(corners)
    flat_pos  = []   # anchor positions
    flat_type = []   # anchor types (0=smooth, 1=soft-corner, 2=sharp-corner)
    flat_c0   = []   # C[i, 0]: first ctrl of bezier FROM A[i]
    flat_c1   = []   # C[i, 1]: second ctrl of bezier FROM A[i]
    for k in range(nc):
        si_start = corners[k]
        si_end   = corners[(k + 1) % nc]
        atype_k  = 2 if curv[si_start] >= SHARP_THRESH else 1
        if si_end > si_start:
            arc = s[si_start:si_end + 1]
        else:
            arc = np.vstack([s[si_start:], s[:si_end + 1]])
        spans = _fit_arc_divide(arc, MAX_BEZIER_ERR_PX)
        # spans[j] = (endpoint, c0, c1) for bezier j in this arc.
        # Corner anchor at arc[0]; bezier 0 leaves from it.
        flat_pos.append(s[si_start])
        flat_type.append(atype_k)
        flat_c0.append(spans[0][1])
        flat_c1.append(spans[0][2])
        # Intermediate smooth splits: spans[0].endpoint .. spans[ns-2].endpoint.
        # (spans[-1].endpoint == next corner == added by the next k-iteration)
        for j in range(len(spans) - 1):
            flat_pos.append(spans[j][0])
            flat_type.append(0)
            flat_c0.append(spans[j + 1][1])
            flat_c1.append(spans[j + 1][2])

    A  = np.array(flat_pos,  dtype=float)
    at = np.array(flat_type, dtype=int)
    N  = len(A)
    C  = np.zeros((N, 2, 2))
    for i in range(N):
        C[i, 0] = flat_c0[i]
        C[i, 1] = flat_c1[i]

    if verbose:
        pct    = (_progress['done'] / _progress['total'] * 100
                  if _progress['total'] else 0)
        prefix = f"[{_progress['done']}/{_progress['total']} {pct:.0f}%]"
        print(f"  {prefix} {label}: {nc} corners → {N} anchors (D&C)")

    if is_outer:
        _progress['done'] += 1
    if verbose:
        pct = (_progress['done'] / _progress['total'] * 100
               if _progress['total'] else 0)
        print(f"    → done ({time.time()-t0:.1f}s) "
              f"[{_progress['done']}/{_progress['total']} {pct:.0f}%]")
    return A, C, at


# ── SVG export ────────────────────────────────────────────────────────────────
def anchors_to_d(A, C):
    N=len(A); f=lambda rc:f"{rc[1]:.2f},{rc[0]:.2f}"
    d=f"M {f(A[0])}"
    for i in range(N): d+=f" C {f(C[i,0])} {f(C[i,1])} {f(A[(i+1)%N])}"
    return d+" Z"

def atypes_str(at): return ','.join(str(int(x)) for x in at)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global N_STEPS_SEG, MAX_BEZIER_ERR_PX, DILATION_PX
    global KMEANS_SAMPLE, KMEANS_N_INIT, BEZIER_SEARCH, REGION_CROP, CORNER_COALESCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',   default='logo-plain-transparent.png')
    parser.add_argument('--output',  default='taos_pass1.svg')
    parser.add_argument('--preview', default='taos_pass1_preview.png')
    parser.add_argument('--diff',    default='taos_pass1_diff.png')
    parser.add_argument('--quiet',   action='store_true')
    parser.add_argument('--steps',   type=int, default=0,
                        help='Override all phase step counts')
    parser.add_argument('--colors',  type=int, default=5,
                        help='Number of colors to quantize to (default: 5)')
    parser.add_argument('--min-px',  type=int, default=MIN_REGION_PX,
                        help='Minimum pixels for a region to be vectorized')
    parser.add_argument('--max-bezier-err', type=float, default=MAX_BEZIER_ERR_PX,
                        help='Max per-point error (px) for Bézier fitting (default: 1.0)')
    parser.add_argument('--dilation-px', type=int, default=DILATION_PX,
                        help='Mask dilation before contouring (default: 0)')
    parser.add_argument('--flood-bg', action=argparse.BooleanOptionalAction, default=True,
                        help='Flood-fill background to transparent before vectorizing (default: on)')
    parser.add_argument('--flood-tol', type=int, default=40,
                        help='RGB tolerance for --flood-bg (default: 40)')
    parser.add_argument('--shadow-tol', type=float, default=0.08,
                        help='Colour-direction tolerance for removing background-hued drop '
                             'shadows with --flood-bg; 0 = off (default: 0.08)')
    parser.add_argument('--kmeans-sample', type=int, default=KMEANS_SAMPLE,
                        help='Pixels used to fit k-means; 0 = all (default: 0). Sampling can shift the palette')
    parser.add_argument('--kmeans-n-init', type=int, default=KMEANS_N_INIT,
                        help='k-means restarts (default: 3)')
    parser.add_argument('--bezier-search', choices=['kdtree', 'brute'],
                        default=BEZIER_SEARCH,
                        help='Nearest-point search for Bézier fit error (default: kdtree)')
    parser.add_argument('--corner-coalesce', choices=['gradient', 'dc', 'off'],
                        default=CORNER_COALESCE,
                        help="Corner pruning: 'gradient' (original, slow), 'dc' "
                             "(closed-form span count), 'off' (default: dc)")
    parser.add_argument('--no-region-crop', action='store_true',
                        help='Disable bounding-box cropping per region (slow legacy path)')
    args=parser.parse_args()

    N_STEPS_SEG = args.steps if args.steps > 0 else N_STEPS_SEG
    MAX_BEZIER_ERR_PX = args.max_bezier_err
    DILATION_PX = args.dilation_px
    KMEANS_SAMPLE = args.kmeans_sample
    KMEANS_N_INIT = args.kmeans_n_init
    BEZIER_SEARCH = args.bezier_search
    REGION_CROP = not args.no_region_crop
    CORNER_COALESCE = args.corner_coalesce
    verbose=not args.quiet

    print(f"Loading {args.input}...")
    arr=np.array(Image.open(args.input).convert('RGBA'))
    arr=remove_black_background(arr)
    if args.flood_bg:
        arr = flood_fill_background(arr, tol=args.flood_tol, shadow_tol=args.shadow_tol)
        print(f"Background flood-filled (tol={args.flood_tol})")
    h,w=arr.shape[:2]

    print(f"Quantizing to {args.colors} colors...")
    color_masks, quantized_rgba = extract_color_masks(arr, n_colors=args.colors, min_px=args.min_px)
    print(f"Found {len(color_masks)} color layers | MAX_GAP={MAX_GAP:.1f}px | "
          f"SHARP_THRESH={SHARP_THRESH}° | MAX_BEZIER_ERR_PX={MAX_BEZIER_ERR_PX} | "
          f"DILATION_PX={DILATION_PX}\n")

    # Pre-count qualifying regions so progress % is accurate
    layer_data = []
    total_regions = 0
    for li,(hex_color,mask) in enumerate(color_masks):
        cc,n_reg=ndi.label(mask)
        if REGION_CROP:
            counts = np.bincount(cc.ravel(), minlength=n_reg + 1)
            slices = ndi.find_objects(cc)
            sizes = sorted([(int(counts[i]), i) for i in range(1, n_reg + 1)],
                           reverse=True)
        else:
            slices = None
            sizes=sorted([(int(ndi.sum(mask,cc,i)),i)
                          for i in range(1,n_reg+1)],reverse=True)
        qualifying = [(sz,rid) for sz,rid in sizes if sz >= args.min_px]
        total_regions += len(qualifying)
        layer_data.append((hex_color, mask, cc, qualifying, slices))
        print(f"  Layer {li+1}: {hex_color}  ({len(qualifying)} regions)")
    print(f"\nTotal regions to fit: {total_regions}\n")

    _progress['total'] = total_regions
    _progress['done']  = 0

    svg_layers=[]; t0=time.time()

    for li,(hex_color,mask,cc,qualifying,slices) in enumerate(layer_data):
        print(f"\n{'='*60}")
        print(f"Layer {li+1}/{len(layer_data)}: {hex_color} | {len(qualifying)} region(s)")

        layer_ds=[]
        for region_size,region_id in qualifying:
            if slices is not None:
                # Crop to the bounding box (clamped to the image, so contours
                # that touch the image border behave exactly as before).
                sy, sx = slices[region_id - 1]
                y0 = max(sy.start - 2, 0); y1 = min(sy.stop + 2, h)
                x0 = max(sx.start - 2, 0); x1 = min(sx.stop + 2, w)
                region = (cc[y0:y1, x0:x1] == region_id)
                outer,holes = get_contours(region)
                off = np.array([y0, x0], dtype=float)
                if outer is not None:
                    outer = outer + off
                holes = [hc + off for hc in holes]
            else:
                region=(cc==region_id)
                outer,holes=get_contours(region)
            if outer is None:
                _progress['done'] += 1
                continue
            lbl=f"L{li+1}_{hex_color}_R{region_id}"
            print(f"\n  Region {region_id} ({region_size}px)")
            oA,oC,oAT=fit_contour(outer,region,f"{lbl}_outer",verbose,is_outer=True)
            if oA is None:
                continue
            hole_curves=[]
            for hi,hole in enumerate(holes):
                hA,hC,hAT=fit_contour(hole,region,f"{lbl}_h{hi}",verbose,is_outer=False)
                if hA is not None:
                    hole_curves.append((hA,hC,hAT))
            # Each region is one <path> with its own data-atypes
            subpaths=[(oA,oC,oAT)]+hole_curves
            layer_ds.append(subpaths)

        if layer_ds:
            svg_layers.append((hex_color, layer_ds))

    print(f"\nTotal: {time.time()-t0:.0f}s")

    svg =f'<?xml version="1.0" encoding="UTF-8"?>\n'
    svg+=f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
    svg+= 'xmlns="http://www.w3.org/2000/svg"\n'
    svg+= '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">\n'
    for li,(hex_color,regions) in enumerate(svg_layers):
        svg+=f'  <g id="layer-{li}" inkscape:label="{hex_color}"'
        svg+= ' inkscape:groupmode="layer">\n'
        for ri,subpaths in enumerate(regions):
            d_parts=[]; at_parts=[]
            for A,C,AT in subpaths:
                d_parts.append(anchors_to_d(A,C))
                at_parts.append(atypes_str(AT))
            combined=' '.join(d_parts)
            atypes_attr='|'.join(at_parts)  # | separates outer from holes
            svg+=f'    <path id="r{li}-{ri}" d="{combined}"\n'
            svg+=f'          fill="{hex_color}" fill-rule="evenodd" stroke="none"\n'
            svg+=f'          data-atypes="{atypes_attr}"/>\n'
        svg+='  </g>\n'
    svg+='</svg>'

    with open(args.output,'w') as f: f.write(svg)
    print(f"Saved: {args.output}")

    abspath=os.path.abspath(args.output)
    png=cairosvg.svg2png(url=f'file://{abspath}',output_width=w,output_height=h)
    pred_img = Image.open(io.BytesIO(png))
    pred_img.save(args.preview)
    print(f"Preview: {args.preview}")

    # ── Diff analysis ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Diff analysis")
    print(f"{'='*60}")
    # Save quantized reference image for inspection
    quant_preview = args.preview.replace('_preview', '_quantized')
    Image.fromarray(quantized_rgba).save(quant_preview)
    print(f"Quantized ref: {quant_preview}")
    # Compare SVG render against quantized image (not original AA image):
    # a perfect SVG should score zero; AA edges in the original would never score zero.
    orig_img = Image.open(args.input).convert('RGBA')
    quant_img = Image.fromarray(quantized_rgba)
    if pred_img.size != quant_img.size:
        pred_img = pred_img.resize(quant_img.size, Image.LANCZOS)
    quant_np = np.array(quant_img).astype(np.int16)
    orig_np  = np.array(orig_img).astype(np.int16)
    pred_np  = np.array(pred_img.convert('RGBA')).astype(np.int16)
    visible  = orig_np[:,:,3] > 128
    diff_ch  = np.abs(quant_np[:,:,:3] - pred_np[:,:,:3])   # (H,W,3)
    diff_max = diff_ch.max(axis=2)                           # (H,W) per-pixel max channel error

    # Save 4x-amplified diff image
    diff_amp = (diff_ch.clip(0,255).astype(np.float32) * 4).clip(0,255).astype(np.uint8)
    diff_amp[~visible] = 0
    Image.fromarray(diff_amp).save(args.diff)
    print(f"Diff image (4x): {args.diff}")

    # Overall stats
    vis_err = diff_max[visible]
    total_vis = int(visible.sum())
    print(f"\n{'Layer':<22} {'Pixels':>8}  {'Mean err':>9}  {'>10px':>7}  {'>30px':>7}")
    print(f"{'-'*22} {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}")

    # Per-colour-layer stats using quantized labels (already computed)
    layer_rows = []
    for hx, mask in color_masks:
        lmask = mask & visible
        n_px  = int(lmask.sum())
        if n_px == 0:
            continue
        e = diff_max[lmask]
        layer_rows.append((n_px, hx, e.mean(), (e>10).sum(), (e>30).sum()))

    layer_rows.sort(key=lambda x: -x[0])
    for n_px, hx, mean_e, gt10, gt30 in layer_rows:
        pct10 = gt10/n_px*100; pct30 = gt30/n_px*100
        print(f"{hx:<22} {n_px:>8}  {mean_e:>8.2f}  "
              f"{gt10:>5} {pct10:>4.1f}%  {gt30:>5} {pct30:>4.1f}%")

    # Total row
    e_all = diff_max[visible]
    gt10_all = int((e_all>10).sum()); gt30_all = int((e_all>30).sum())
    print(f"{'─'*22} {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}")
    print(f"{'TOTAL':<22} {total_vis:>8}  {e_all.mean():>8.2f}  "
          f"{gt10_all:>5} {gt10_all/total_vis*100:>4.1f}%  "
          f"{gt30_all:>5} {gt30_all/total_vis*100:>4.1f}%")

if __name__=='__main__':
    main()
