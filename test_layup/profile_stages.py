import sys, time
sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from polarbench.stokes import PolarField, polarity_difference
from polarbench.quality import QualityField, BY_ISSUE


def make(H, W, seed=7):
    rng = np.random.default_rng(seed)
    direction = np.where(rng.random((H, W)) > 0.5, 35.0, 85.0).astype(np.float32)
    direction += rng.normal(0, 2.0, (H, W)).astype(np.float32)
    oriented = np.ones((H, W), dtype=bool)
    weak = rng.random((H, W)) < 0.02
    masked = np.zeros((H, W), dtype=bool)
    masked[:, W // 3:W // 3 + 6] = True
    oriented &= ~(weak | masked)
    measured = oriented | weak
    rad = np.deg2rad(direction * 2)
    deg = np.where(oriented, 0.4, np.where(weak, 0.01, 0.0)).astype(np.float32)
    s0 = np.ones((H, W), dtype=np.float32)
    pol = s0 * deg
    q = pol * np.cos(rad).astype(np.float32)
    u = pol * np.sin(rad).astype(np.float32)
    field = PolarField(s0, q, u, pol, deg, direction, np.zeros_like(s0), np.zeros_like(s0),
                       measured | masked, measured, oriented)
    flags = np.zeros((H, W), dtype=np.uint16)
    flags[weak] |= BY_ISSUE['weak-axis'].bit
    flags[masked] |= BY_ISSUE['mask'].bit
    return field, QualityField(flags, measured, oriented, {})


def stages(field, quality):
    t = {}
    t0 = time.time()
    oriented = field.oriented
    weak_axis = quality.reason('weak-axis') & ~oriented
    local_signed = polarity_difference(field.direction, np.float32(35.0))
    anomaly = oriented & (np.abs(local_signed) >= 20.0)
    t['masks'] = time.time() - t0

    t0 = time.time()
    n = int(anomaly.sum())
    idx = np.full(anomaly.shape, -1, dtype=np.int32)
    idx[anomaly] = np.arange(n, dtype=np.int32)
    node_flat = np.flatnonzero(anomaly.ravel())
    fd = field.direction.ravel()
    ea, eb = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[:idx.shape[0] - dy, :idx.shape[1] - dx]
        b = idx[dy:, dx:]
        f, s = np.nonzero((a >= 0) & (b >= 0))
        na, nb = a[f, s], b[f, s]
        agree = np.abs(polarity_difference(fd[node_flat[na]], fd[node_flat[nb]])) <= 25
        ea.append(na[agree]); eb.append(nb[agree])
    left, right = np.concatenate(ea), np.concatenate(eb)
    t['edges'] = time.time() - t0

    t0 = time.time()
    graph = sparse.csr_matrix((np.ones(left.size, dtype=np.int8), (left, right)), shape=(n, n))
    t['csr'] = time.time() - t0

    t0 = time.time()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    t['components'] = time.time() - t0

    t0 = time.time()
    comp_map = np.full(anomaly.shape, -1, dtype=np.int32)
    comp_map[anomaly] = labels
    t['compmap'] = time.time() - t0

    t0 = time.time()
    # vectorized per-component stats like the current _patch_rows
    order = np.argsort(labels, kind='stable')
    sorted_flat = node_flat[order]
    sorted_labels = labels[order]
    bounds = np.flatnonzero(np.r_[True, np.diff(sorted_labels) != 0])
    rows = []
    w = anomaly.shape[1]
    for start, end in zip(bounds, np.r_[bounds[1:], sorted_labels.size]):
        cells = sorted_flat[start:end]
        ys, xs = np.divmod(cells, w)
        rows.append((cells.size, xs.min(), xs.max()))
    t['patchrows'] = time.time() - t0
    print(f'nodes={n:,} edges={left.size:,} components={count:,}')
    for k, v in t.items():
        print(f'  {k:12s} {v:7.3f}s')


for H, W in ((800, 1250),):
    print(f'--- {H}x{W} = {H*W:,} px')
    field, quality = make(H, W)
    stages(field, quality)
