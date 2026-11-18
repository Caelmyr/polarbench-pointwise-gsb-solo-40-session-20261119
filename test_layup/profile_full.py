import sys, time
sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')
import numpy as np
from polarbench.layup import review_layup
from polarbench.quality import QualityField, BY_ISSUE
from polarbench.stokes import PolarField


def make(H, W, seed=7):
    rng = np.random.default_rng(seed)
    direction = np.where(rng.random((H, W)) > 0.5, 35.0, 85.0).astype(np.float32)
    direction += rng.normal(0, 2.0, (H, W)).astype(np.float32)
    oriented = np.ones((H, W), dtype=bool)
    weak = rng.random((H, W)) < 0.02
    masked = np.zeros((H, W), dtype=bool)
    masked[:, W // 3:W // 3 + 6] = True
    masked[3 * H // 5:3 * H // 5 + 3, :] = True
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
    quality = QualityField(flags, measured, oriented, {})

    class R:
        pass
    r = R(); r.field = field; r.quality = quality
    return r


for H, W in ((800, 1250), (1200, 1667), (2000, 2500)):
    r = make(H, W)
    raw = {'reference_box': {'x': 10, 'y': 10, 'width': 80, 'height': 80},
           'deviation': 20.0, 'split': 25.0, 'bridge_gap': 2, 'min_area': 25}
    t0 = time.time()
    rep = review_layup(r, {'scale': None, 'regions': []}, raw)
    dt = time.time() - t0
    print(f'{H*W:>9,} px  {dt:6.2f}s  components={rep["all_components"]:,} '
          f'qualified={rep["qualified_components"]:,} anomaly={rep["anomaly_pixels"]:,} '
          f'weak={rep["weak_pixels"]:,}')
