"""
cross_periodogram_test_cauchy_040126.py

Validates the expected cross-periodogram formula E[I^{qr}(ω)] by comparing:

  Empirical  : mean of N tapered sample cross-periodograms
               DGP = Generalized Cauchy β=1  OR  Matérn ν=0.5 field (FFT circulant embedding)

  Theoretical: Guillaumin et al. (2022) formula with raw covariance of the chosen model
               E[I^{qr}(ω)] = (1/4π²) Σ_u [ C_X(u, t_q-t_r) · c^{qr}_{g,n}(u) ] e^{iω·u}

Design:
  - Fixed obs pattern  : synthetic random missing (seed=42), 70 % cells observed per step
  - Grid               : n1=11 lat, n2=20 lon, p_time=8 time steps
  - Physical spacing   : delta1=0.44° lat, delta2=0.50° lon, delta_t=1.0 h
  - True parameters    : same scale as production fits
  - Taper              : Hamming (same as production DW fits)
  - Output             : saved to outputs/day/cross_periodogram/<model>/

Run locally:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
  python cross_periodogram_test_cauchy_040126.py --model cauchy --n-iter 1000
  python cross_periodogram_test_cauchy_040126.py --model matern --n-iter 1000
"""

import sys
import time
import math
import numpy as np
import torch
import torch.fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer
from pathlib import Path

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import debiased_whittle_2110 as dw_module

DEVICE = torch.device("cpu")   # cross-periodogram computation is CPU-only in DW
DTYPE  = torch.float64

# ── True parameters (shared by both Cauchy and Matérn DGPs) ──────────────────
# sigmasq=13.059, range_lat=0.2°, range_lon=0.25°, range_time=1.5 h
# advec_lat=0.0218 °/h, advec_lon=-0.1689 °/h, nugget=0.247
_true = dict(
    sigmasq=13.059, range_lat=0.2, range_lon=0.25, range_time=1.5,
    advec_lat=0.0218, advec_lon=-0.1689, nugget=0.247,
)
_phi2 = 1.0 / _true["range_lon"]
_phi1 = _true["sigmasq"] * _phi2
_phi3 = (_true["range_lon"] / _true["range_lat"])  ** 2
_phi4 = (_true["range_lon"] / _true["range_time"]) ** 2
TRUE_LOG = [
    math.log(_phi1), math.log(_phi2), math.log(_phi3), math.log(_phi4),
    _true["advec_lat"], _true["advec_lon"], math.log(_true["nugget"]),
]
TRUE_PARAMS = torch.tensor(TRUE_LOG, device=DEVICE, dtype=DTYPE)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Cauchy covariance helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_cauchy_cov_on_grid(lx: torch.Tensor, ly: torch.Tensor,
                            lt: torch.Tensor, params: torch.Tensor,
                            gc_beta: float = 1.0) -> torch.Tensor:
    """
    Generalized Cauchy covariance on a lag grid — used for FFT circulant embedding.

    C(lx, ly, lt) = (φ1/φ2) · (1 + φ2 · sqrt(φ3·lx² + ly² + φ4·lt²))^{−β}

    No nugget added here (nugget is added to observations separately).
    """
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.pow(1.0 + dist * phi2, -gc_beta)


def generate_cauchy_field(n1: int, n2: int, p_time: int,
                           params: torch.Tensor,
                           delta1: float, delta2: float,
                           delta_t: float = 1.0,
                           gc_beta: float = 1.0) -> torch.Tensor:
    """
    FFT circulant embedding — Generalized Cauchy covariance.

    Returns field of shape (n1, n2, p_time) in DTYPE on DEVICE.
    delta1 / delta2 : physical lat / lon spacing of the output grid (degrees).
    delta_t         : physical time step (hours, default 1).
    """
    CPU = torch.device("cpu")
    F32 = torch.float32
    Px, Py, Pt = 2 * n1, 2 * n2, 2 * p_time

    lx = torch.arange(Px, device=CPU, dtype=F32) * delta1
    lx[Px // 2:] -= Px * delta1

    ly = torch.arange(Py, device=CPU, dtype=F32) * delta2
    ly[Py // 2:] -= Py * delta2

    lt = torch.arange(Pt, device=CPU, dtype=F32) * delta_t
    lt[Pt // 2:] -= Pt * delta_t

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing="ij")
    C = get_cauchy_cov_on_grid(Lx, Ly, Lt, params_cpu, gc_beta=gc_beta)

    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0.0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:n1, :n2, :p_time]
    return field.to(dtype=DTYPE, device=DEVICE)


def cov_cauchy_raw(u1: torch.Tensor, u2: torch.Tensor, t,
                    params: torch.Tensor, gc_beta: float = 1.0) -> torch.Tensor:
    """
    Raw Generalized Cauchy covariance — param encoding matches debiased_whittle.py.

    C_X(u1, u2, t) = σ² · (1 + φ2·√(φ3·u1_adv² + u2_adv² + φ4·t²))^{−β}
                   + nugget · 𝟙[u=0]

    u1, u2 : PHYSICAL spatial lags (degrees, matching delta1/delta2 grid)
    t      : PHYSICAL time lag (hours)
    """
    device = params.device
    phi1 = torch.exp(params[0]);  phi2 = torch.exp(params[1])
    phi3 = torch.exp(params[2]);  phi4 = torch.exp(params[3])
    advec_lat = params[4];         advec_lon = params[5]
    nugget    = torch.exp(params[6])
    sigmasq   = phi1 / phi2

    t_dev = t.to(device) if isinstance(t, torch.Tensor) else \
            torch.tensor(float(t), device=device, dtype=torch.float64)

    u1_adv = u1 - advec_lat * t_dev
    u2_adv = u2 - advec_lon * t_dev

    # Dimensionless distance  (same form as cov_x_spatiotemporal_model_kernel)
    dist = torch.sqrt(
        (u1_adv * torch.sqrt(phi3) * phi2).pow(2) +
        (u2_adv * phi2).pow(2) +
        (t_dev  * torch.sqrt(phi4) * phi2).pow(2) + 1e-12
    )
    cov = sigmasq * torch.pow(1.0 + dist, -gc_beta)

    is_zero = (u1.abs() < 1e-9) & (u2.abs() < 1e-9) & (t_dev.abs() < 1e-9)
    return torch.where(is_zero, cov + nugget, cov)


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Matérn ν=0.5 (exponential) covariance helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_matern_cov_on_grid(lx: torch.Tensor, ly: torch.Tensor,
                            lt: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Matérn ν=0.5 (exponential) covariance on a lag grid — for FFT circulant embedding."""
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def generate_matern_field(n1: int, n2: int, p_time: int,
                           params: torch.Tensor,
                           delta1: float, delta2: float,
                           delta_t: float = 1.0) -> torch.Tensor:
    """FFT circulant embedding — Matérn ν=0.5 covariance. Returns (n1, n2, p_time)."""
    CPU = torch.device("cpu")
    F32 = torch.float32
    Px, Py, Pt = 2 * n1, 2 * n2, 2 * p_time

    lx = torch.arange(Px, device=CPU, dtype=F32) * delta1
    lx[Px // 2:] -= Px * delta1

    ly = torch.arange(Py, device=CPU, dtype=F32) * delta2
    ly[Py // 2:] -= Py * delta2

    lt = torch.arange(Pt, device=CPU, dtype=F32) * delta_t
    lt[Pt // 2:] -= Pt * delta_t

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing="ij")
    C = get_matern_cov_on_grid(Lx, Ly, Lt, params_cpu)

    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0.0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:n1, :n2, :p_time]
    return field.to(dtype=DTYPE, device=DEVICE)


def cov_matern_raw(u1: torch.Tensor, u2: torch.Tensor, t,
                    params: torch.Tensor) -> torch.Tensor:
    """
    Raw Matérn ν=0.5 (exponential) covariance — param encoding matches debiased_whittle.py.

    C_X(u1, u2, t) = σ² · exp(−dist) + nugget · 𝟙[u=0]
    where dist = φ2 · √(φ3·u1_adv² + u2_adv² + φ4·t²)
    """
    device = params.device
    phi1 = torch.exp(params[0]);  phi2 = torch.exp(params[1])
    phi3 = torch.exp(params[2]);  phi4 = torch.exp(params[3])
    advec_lat = params[4];         advec_lon = params[5]
    nugget    = torch.exp(params[6])
    sigmasq   = phi1 / phi2

    t_dev = t.to(device) if isinstance(t, torch.Tensor) else \
            torch.tensor(float(t), device=device, dtype=torch.float64)

    u1_adv = u1 - advec_lat * t_dev
    u2_adv = u2 - advec_lon * t_dev

    dist = torch.sqrt(
        (u1_adv * torch.sqrt(phi3) * phi2).pow(2) +
        (u2_adv * phi2).pow(2) +
        (t_dev  * torch.sqrt(phi4) * phi2).pow(2) + 1e-12
    )
    cov = sigmasq * torch.exp(-dist)

    is_zero = (u1.abs() < 1e-9) & (u2.abs() < 1e-9) & (t_dev.abs() < 1e-9)
    return torch.where(is_zero, cov + nugget, cov)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Theoretical expected periodogram (model-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def cn_bar(u1_grid: torch.Tensor, u2_grid: torch.Tensor, t_diff,
            params: torch.Tensor,
            n1: int, n2: int,
            taper_autocorr: torch.Tensor,
            delta1: float, delta2: float,
            model: str = "cauchy",
            gc_beta: float = 1.0,
            q_idx=None, r_idx=None) -> torch.Tensor:
    """
    c̃^{qr}(u) = C_X(u·δ, t_diff) · c^{qr}_{g,n}(u)

    model : "cauchy" uses Generalized Cauchy C_X
            "matern" uses Matérn ν=0.5 (exponential) C_X
    """
    device = params.device
    lag_u1 = u1_grid.to(device) * delta1
    lag_u2 = u2_grid.to(device) * delta2

    if model == "cauchy":
        cov_val = cov_cauchy_raw(lag_u1, lag_u2, t_diff, params, gc_beta)
    else:
        cov_val = cov_matern_raw(lag_u1, lag_u2, t_diff, params)

    u1_idx = u1_grid.long().to(device)
    u2_idx = u2_grid.long().to(device)
    idx1   = torch.clamp(n1 - 1 + u1_idx, 0, 2 * n1 - 2)
    idx2   = torch.clamp(n2 - 1 + u2_idx, 0, 2 * n2 - 2)

    if taper_autocorr.ndim == 4 and q_idx is not None and r_idx is not None:
        taper_val = taper_autocorr[q_idx, r_idx, idx1, idx2]
    else:
        taper_val = taper_autocorr[idx1, idx2]

    return cov_val * taper_val


def expected_periodogram(params: torch.Tensor,
                          n1: int, n2: int, p_time: int,
                          taper_autocorr: torch.Tensor,
                          delta1: float, delta2: float,
                          model: str = "cauchy",
                          gc_beta: float = 1.0,
                          delta_t: float = 1.0) -> torch.Tensor:
    """
    Theoretical E[I^{qr}(ω)] — Guillaumin et al. (2022) Lemma 2.
    Supports model="cauchy" (Generalized Cauchy) and model="matern" (ν=0.5).
    Returns real tensor of shape (n1, n2, p_time, p_time).
    """
    device  = params.device
    u1_lags = torch.arange(n1, dtype=DTYPE, device=device)
    u2_lags = torch.arange(n2, dtype=DTYPE, device=device)
    u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing="ij")

    tilde_cn = torch.zeros((n1, n2, p_time, p_time), dtype=torch.complex128, device=device)

    for q in range(p_time):
        for r in range(p_time):
            t_diff = (q - r) * delta_t
            _q = q if taper_autocorr.ndim == 4 else None
            _r = r if taper_autocorr.ndim == 4 else None

            t1 = cn_bar(u1_mesh,      u2_mesh,      t_diff, params, n1, n2,
                         taper_autocorr, delta1, delta2, model, gc_beta, _q, _r)
            t2 = cn_bar(u1_mesh - n1, u2_mesh,      t_diff, params, n1, n2,
                         taper_autocorr, delta1, delta2, model, gc_beta, _q, _r)
            t3 = cn_bar(u1_mesh,      u2_mesh - n2, t_diff, params, n1, n2,
                         taper_autocorr, delta1, delta2, model, gc_beta, _q, _r)
            t4 = cn_bar(u1_mesh - n1, u2_mesh - n2, t_diff, params, n1, n2,
                         taper_autocorr, delta1, delta2, model, gc_beta, _q, _r)

            tilde_cn[:, :, q, r] = (t1 + t2 + t3 + t4).to(torch.complex128)

    fft_result = torch.fft.fft2(tilde_cn, dim=(0, 1))
    return (fft_result.real / (4.0 * math.pi ** 2))  # (n1, n2, p_time, p_time)


# backward-compat alias
def expected_periodogram_cauchy(params, n1, n2, p_time, taper_autocorr,
                                  delta1, delta2, gc_beta=1.0, delta_t=1.0):
    return expected_periodogram(params, n1, n2, p_time, taper_autocorr,
                                 delta1, delta2, "cauchy", gc_beta, delta_t)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build simulated tensor_list  (for generate_Jvector_tapered_mv input)
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor_list(field: torch.Tensor,
                       lat_flat: torch.Tensor, lon_flat: torch.Tensor,
                       obs_masks: torch.Tensor,
                       nugget_std: float) -> list:
    """
    For each time step t, build a tensor of shape (n1*n2, 3) = [lat, lon, val].
    Unobserved cells get val=NaN so generate_Jvector_tapered_mv marks them missing.
    """
    n1, n2, p_time = field.shape
    tensor_list = []
    for t in range(p_time):
        # Add nugget noise to observed cells
        vals = field[:, :, t].clone().flatten()       # (n1*n2,)
        noise = torch.randn_like(vals) * nugget_std
        obs_t = obs_masks[t].flatten()                # (n1*n2,) bool
        vals[obs_t]  = vals[obs_t] + noise[obs_t]
        vals[~obs_t] = float("nan")
        tensor_t = torch.stack([lat_flat, lon_flat, vals], dim=1)   # (n1*n2, 3)
        tensor_list.append(tensor_t)
    return tensor_list


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI
# ══════════════════════════════════════════════════════════════════════════════

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    model:    str   = typer.Option("cauchy", help="DGP model: 'cauchy' or 'matern'"),
    gc_beta:  float = typer.Option(1.0,  help="Cauchy β (only used when model=cauchy)"),
    n_iter:   int   = typer.Option(50,   help="Number of simulation iterations"),
    n1:       int   = typer.Option(11,   help="Grid lat cells"),
    n2:       int   = typer.Option(20,   help="Grid lon cells"),
    p_time:   int   = typer.Option(8,    help="Time steps per day"),
    delta1:   float = typer.Option(0.44, help="Lat spacing (degrees)"),
    delta2:   float = typer.Option(0.50, help="Lon spacing (degrees)"),
    delta_t:  float = typer.Option(1.0,  help="Time step (hours)"),
    obs_frac: float = typer.Option(0.70, help="Fraction of cells observed per time step"),
    seed:     int   = typer.Option(42,   help="Random seed"),
) -> None:

    if model not in ("cauchy", "matern"):
        raise ValueError(f"model must be 'cauchy' or 'matern', got '{model}'")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model_tag = f"cauchy_b{int(gc_beta * 10):02d}" if model == "cauchy" else "matern_v05"

    print(f"Model   : {model_tag}")
    print(f"Grid    : n1={n1}, n2={n2}, p_time={p_time}")
    print(f"Spacing : Δlat={delta1}°, Δlon={delta2}°, Δt={delta_t} h")
    print(f"obs_frac={obs_frac}  N iter={n_iter}  seed={seed}")

    output_path = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/cross_periodogram") / model_tag
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Instantiate DW helper ──────────────────────────────────────────────────
    dwl          = dw_module.debiased_whittle_likelihood()
    tapering_fn  = dwl.cgn_hamming

    nugget_std = math.sqrt(math.exp(TRUE_LOG[6]))

    # ── Fixed obs pattern ─────────────────────────────────────────────────────
    # obs_masks[t, i, j] = True if cell (i,j) is observed at time step t
    # Use a spatially correlated pattern to mimic real cloud coverage.
    print("\n[1/5] Building fixed obs pattern...")
    obs_masks = torch.zeros((p_time, n1, n2), dtype=torch.bool, device=DEVICE)
    for t in range(p_time):
        # Smooth a random field and threshold → realistic cluster-shaped missing
        raw = torch.tensor(rng.standard_normal((n1, n2)), dtype=DTYPE, device=DEVICE)
        # Simple 3×3 box blur
        blurred = raw.clone()
        blurred[1:-1, 1:-1] = (
            raw[:-2, :-2] + raw[:-2, 1:-1] + raw[:-2, 2:] +
            raw[1:-1, :-2] + raw[1:-1, 1:-1] + raw[1:-1, 2:] +
            raw[2:, :-2]  + raw[2:, 1:-1]  + raw[2:, 2:]
        ) / 9.0
        threshold = float(torch.quantile(blurred.flatten(), 1.0 - obs_frac))
        obs_masks[t] = blurred >= threshold

    n_obs_avg = obs_masks.float().sum(dim=(1, 2)).mean().item()
    print(f"   Average observed cells per step: {n_obs_avg:.1f} / {n1 * n2}")

    # ── Lat / lon grids (uniform, centred on zero for simplicity) ─────────────
    lat_vals = torch.arange(n1, dtype=DTYPE, device=DEVICE) * delta1
    lon_vals = torch.arange(n2, dtype=DTYPE, device=DEVICE) * delta2
    lat_grid, lon_grid = torch.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    # ── Taper autocorrelation c^{qr}_{g,n}  (computed once, fixed obs) ────────
    print("[2/5] Computing fixed taper autocorrelation c^{qr}...")
    # Build a reference tensor_list from the obs pattern (all-zero values, correct structure)
    ref_field = torch.zeros((n1, n2, p_time), dtype=DTYPE, device=DEVICE)
    ref_tensor_list = build_tensor_list(ref_field, lat_flat, lon_flat, obs_masks, nugget_std=0.0)

    _, _, _, _, taper_grid, obs_masks_ref = dwl.generate_Jvector_tapered_mv(
        ref_tensor_list, tapering_func=tapering_fn,
        lat_col=0, lon_col=1, val_col=2, device=DEVICE
    )
    if taper_grid is None:
        raise RuntimeError("generate_Jvector_tapered_mv returned None taper_grid — check tensor format.")

    taper_autocorr = dwl.calculate_taper_autocorrelation_multivariate(
        taper_grid, obs_masks_ref, n1, n2, DEVICE
    )
    print(f"   taper_autocorr shape : {tuple(taper_autocorr.shape)}")

    # ── Theoretical E[I^{qr}(ω)] ─────────────────────────────────────────────
    print(f"[3/5] Computing theoretical expected periodogram ({model_tag})...")
    t0 = time.time()
    E_theory = expected_periodogram(
        TRUE_PARAMS, n1, n2, p_time,
        taper_autocorr, delta1, delta2,
        model=model, gc_beta=gc_beta, delta_t=delta_t
    )   # (n1, n2, p_time, p_time)
    print(f"   Done in {time.time() - t0:.1f} s")
    print(f"   E_theory range: [{E_theory.min():.4f}, {E_theory.max():.4f}]")

    # ── Monte Carlo: average sample periodogram ───────────────────────────────
    print(f"[4/5] Running {n_iter} simulations ({model_tag})...")
    I_accum = torch.zeros((n1, n2, p_time, p_time), dtype=DTYPE, device=DEVICE)

    for it in range(n_iter):
        if model == "cauchy":
            field = generate_cauchy_field(n1, n2, p_time, TRUE_PARAMS,
                                           delta1, delta2, delta_t, gc_beta)
        else:
            field = generate_matern_field(n1, n2, p_time, TRUE_PARAMS,
                                           delta1, delta2, delta_t)
        tensor_list_sim = build_tensor_list(field, lat_flat, lon_flat, obs_masks, nugget_std)

        J_vec, _, _, _, _, _ = dwl.generate_Jvector_tapered_mv(
            tensor_list_sim, tapering_func=tapering_fn,
            lat_col=0, lon_col=1, val_col=2, device=DEVICE
        )
        I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
        I_accum += I_sample.real.to(DTYPE)

        if (it + 1) % 10 == 0:
            print(f"   iter {it + 1}/{n_iter}")

    E_empirical = I_accum / n_iter        # (n1, n2, p_time, p_time)
    print(f"   E_empirical range: [{E_empirical.min():.4f}, {E_empirical.max():.4f}]")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diff = (E_empirical - E_theory).abs()
    rel  = diff / (E_theory.abs() + 1e-8)
    print(f"\n[5/5] Comparison summary:")
    print(f"   Mean abs diff      : {diff.mean():.5f}")
    print(f"   Max  abs diff      : {diff.max():.5f}")
    print(f"   Mean rel diff      : {rel.mean():.4f}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    PAIRS = [(0, 0), (0, 1), (1, 1), (0, p_time // 2)]
    freq_lat = np.fft.fftfreq(n1, d=delta1)
    freq_lon = np.fft.fftfreq(n2, d=delta2)

    # --- 4-panel 2D heatmaps ---
    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(14, 4 * len(PAIRS)))
    for row, (q, r) in enumerate(PAIRS):
        emp = E_empirical[:, :, q, r].numpy()
        thr = E_theory[:, :, q, r].numpy()
        diff_qr = emp - thr
        vmax = max(np.abs(emp).max(), np.abs(thr).max()) + 1e-8
        dmax = np.abs(diff_qr).max() + 1e-8

        im0 = axes[row, 0].imshow(emp, aspect="auto", origin="lower",
                                   extent=[freq_lon[0], freq_lon[-1],
                                           freq_lat[0], freq_lat[-1]],
                                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        axes[row, 0].set_title(f"Empirical  E[I^{{{q}{r}}}(ω)]  (N={n_iter})")
        axes[row, 0].set_xlabel("ω lon"); axes[row, 0].set_ylabel("ω lat")
        fig.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(thr, aspect="auto", origin="lower",
                                   extent=[freq_lon[0], freq_lon[-1],
                                           freq_lat[0], freq_lat[-1]],
                                   vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        axes[row, 1].set_title(f"Theoretical E[I^{{{q}{r}}}(ω)]")
        axes[row, 1].set_xlabel("ω lon")
        fig.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(diff_qr, aspect="auto", origin="lower",
                                   extent=[freq_lon[0], freq_lon[-1],
                                           freq_lat[0], freq_lat[-1]],
                                   vmin=-dmax, vmax=dmax, cmap="RdBu_r")
        axes[row, 2].set_title(f"Empirical − Theoretical  (max|diff|={dmax:.4f})")
        axes[row, 2].set_xlabel("ω lon")
        fig.colorbar(im2, ax=axes[row, 2])

    fig.suptitle(
        f"Cross-periodogram validation: {model_tag}  N={n_iter}  "
        f"grid {n1}×{n2}×{p_time}  obs_frac={obs_frac:.0%}",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    heatmap_path = output_path / "cross_periodogram_heatmaps.png"
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    print(f"   Heatmaps saved → {heatmap_path}")

    # --- Scatter: empirical vs theoretical (all q≤r, all frequencies) ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PAIRS)))
    for ci, (q, r) in enumerate(PAIRS):
        emp_flat = E_empirical[:, :, q, r].numpy().ravel()
        thr_flat = E_theory[:, :, q, r].numpy().ravel()
        ax2.scatter(thr_flat, emp_flat, s=8, alpha=0.5, color=colors[ci],
                    label=f"(q={q}, r={r})")
    lim_min = min(E_theory.min().item(), E_empirical.min().item()) - 0.1
    lim_max = max(E_theory.max().item(), E_empirical.max().item()) + 0.1
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="y=x")
    ax2.set_xlim(lim_min, lim_max); ax2.set_ylim(lim_min, lim_max)
    ax2.set_xlabel("Theoretical E[I^{qr}(ω)]")
    ax2.set_ylabel(f"Empirical E[I^{{qr}}(ω)]  (N={n_iter})")
    ax2.set_title(f"Scatter: {model_tag}  grid {n1}×{n2}×{p_time}")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    scatter_path = output_path / "cross_periodogram_scatter.png"
    fig2.tight_layout()
    fig2.savefig(scatter_path, dpi=120)
    plt.close(fig2)
    print(f"   Scatter saved    → {scatter_path}")

    # --- 1D cross-sections at ω_lat = 0 ---
    fig3, axes3 = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4), sharey=False)
    for ci, (q, r) in enumerate(PAIRS):
        emp_row = E_empirical[0, :, q, r].numpy()
        thr_row = E_theory[0, :, q, r].numpy()
        axes3[ci].plot(freq_lon, thr_row, "b-",  lw=2,   label="Theory")
        axes3[ci].plot(freq_lon, emp_row, "r--", lw=1.5, label=f"Empirical (N={n_iter})")
        axes3[ci].set_title(f"(q={q}, r={r})  ω_lat=0")
        axes3[ci].set_xlabel("ω lon (cycles/degree)")
        axes3[ci].legend(fontsize=7)
    fig3.suptitle(f"1D cross-section at ω_lat=0  {model_tag}", fontsize=11)
    fig3.tight_layout()
    section_path = output_path / "cross_periodogram_1d_sections.png"
    fig3.savefig(section_path, dpi=120)
    plt.close(fig3)
    print(f"   1-D sections saved → {section_path}")

    print("\nDone.")


if __name__ == "__main__":
    app()
