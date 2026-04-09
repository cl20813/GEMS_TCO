"""
debiased_whittle_reduced.py

Four reduced-parameter DW variants for sign ambiguity diagnosis.

All models inherit the raw DW infrastructure (taper, FFT, expected periodogram,
Whittle loss) via parameter expansion to the full 7-param tensor.

Model 1 — DW_iso_lat   : Isotropic (range_lat=range_lon), advec_lat free, advec_lon=0
  params (5): [log_phi1, log_phi2, log_phi4, advec_lat, log_nugget]
  expand  →7: [log_phi1, log_phi2, 0,        log_phi4, advec_lat, 0,         log_nugget]

Model 2 — DW_iso_lon   : Isotropic, advec_lon free, advec_lat=0
  params (5): [log_phi1, log_phi2, log_phi4, advec_lon, log_nugget]
  expand  →7: [log_phi1, log_phi2, 0,        log_phi4, 0,         advec_lon, log_nugget]

Model 3 — DW_aniso_lat : Anisotropic, advec_lat free, advec_lon=0
  params (6): [log_phi1, log_phi2, log_phi3, log_phi4, advec_lat, log_nugget]
  expand  →7: [log_phi1, log_phi2, log_phi3, log_phi4, advec_lat, 0,         log_nugget]

Model 4 — DW_aniso_lon : Anisotropic, advec_lon free, advec_lat=0
  params (6): [log_phi1, log_phi2, log_phi3, log_phi4, advec_lon, log_nugget]
  expand  →7: [log_phi1, log_phi2, log_phi3, log_phi4, 0,         advec_lon, log_nugget]

7-param indexing (raw model):
  [0: log_phi1, 1: log_phi2, 2: log_phi3, 3: log_phi4, 4: advec_lat, 5: advec_lon, 6: log_nugget]
  phi1 = sigmasq*phi2,  phi2 = 1/range_lon,  phi3 = (range_lon/range_lat)^2,
  phi4 = (range_lon/range_time)^2

Natural scale:
  sigmasq    = phi1/phi2
  range_lon  = 1/phi2
  range_lat  = range_lon / sqrt(phi3)
  range_time = range_lon / sqrt(phi4)

Gradient flow:
  params_reduced → expand_to_7param → 7-param tensor (requires_grad=True)
  → _RawDWL.whittle_likelihood_loss_tapered → loss.backward()
  → gradients propagate to params_reduced via torch.cat / autograd ✓
"""

import torch
import numpy as np

from GEMS_TCO.debiased_whittle_raw import debiased_whittle_likelihood as _RawDWL

_DELTA_LAT = 0.044
_DELTA_LON = 0.063


# ─────────────────────────────────────────────────────────────────────────────
# Shared LBFGS loop
# ─────────────────────────────────────────────────────────────────────────────

def _lbfgs_loop(params_list, optimizer, loss_fn, get_params_str, max_steps=5, grad_tol=1e-5):
    """
    Generic L-BFGS loop used by all 4 reduced models.

    Parameters
    ----------
    params_list     : list of 1-element tensors (requires_grad=True)
    optimizer       : torch.optim.LBFGS instance
    loss_fn         : callable () → scalar loss tensor (uses closure over params_list)
    get_params_str  : callable (params_list) → str  (for step logging)
    """
    best_state = [p.detach().clone() for p in params_list]
    best_loss  = float('inf')
    prev_loss  = float('inf')
    loss_tol   = 1e-12
    steps_done = 0

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        if torch.isnan(loss) or torch.isinf(loss):
            return loss
        loss.backward()
        if any(p.grad is not None and
               (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
               for p in params_list):
            optimizer.zero_grad()
        return loss

    for i in range(max_steps):
        steps_done = i + 1
        loss = optimizer.step(closure)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {i+1}/{max_steps}: Loss NaN/Inf. Stopping.")
            break

        cur = loss.item()
        if cur < best_loss and not any(
                torch.isnan(p.data).any() or torch.isinf(p.data).any()
                for p in params_list):
            best_loss  = cur
            best_state = [p.detach().clone() for p in params_list]

        max_grad = max((p.grad.abs().item()
                        for p in params_list if p.grad is not None), default=0.0)
        print(f'--- Step {i+1}/{max_steps} | Loss: {cur:.6f} | MaxGrad: {max_grad:.2e}')
        print(f'    {get_params_str(params_list)}')

        if i > 2:
            if max_grad < grad_tol:
                print(f"--- Converged (max|grad| < {grad_tol}) at step {i+1} ---")
                break
            if abs(cur - prev_loss) < loss_tol:
                print(f"--- Converged (loss_change < {loss_tol}) at step {i+1} ---")
                break
        prev_loss = cur

    print("--- Training Complete ---")
    final_loss = round(best_loss, 3) if best_loss != float('inf') else float('inf')
    return best_state, final_loss, steps_done


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — DW_iso_lat : Isotropic + advec_lat
# ─────────────────────────────────────────────────────────────────────────────

class DW_iso_lat:
    """
    5-param model: [log_phi1, log_phi2, log_phi4, advec_lat, log_nugget]
    range_lat = range_lon = 1/phi2  (isotropic, log_phi3=0)
    advec_lon fixed to 0
    """

    N_PARAMS   = 5
    ADVEC_IDX  = 3          # index of advec in params_list / reduced tensor
    ADVEC_NAME = 'advec_lat'

    @staticmethod
    def expand_to_7param(p5):
        """[lp1, lp2, lp4, advec_lat, lnug] → [lp1, lp2, 0, lp4, advec_lat, 0, lnug]"""
        d  = p5.device; dt = p5.dtype
        z  = torch.zeros(1, device=d, dtype=dt)
        return torch.cat([p5[0:2], z, p5[2:3], p5[3:4], z, p5[4:5]])

    @staticmethod
    def whittle_likelihood_loss_tapered(p5, I_sample, n1, n2, p_time,
                                         taper_autocorr_grid, delta1, delta2):
        p7 = DW_iso_lat.expand_to_7param(p5.to(I_sample.device))
        return _RawDWL.whittle_likelihood_loss_tapered(
            p7, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                           taper_autocorr_grid, max_steps=5, device='cpu', grad_tol=1e-5):
        I_dev = I_sample.to(device)
        t_dev = taper_autocorr_grid.to(device)

        def loss_fn():
            p5 = torch.cat(params_list)
            return DW_iso_lat.whittle_likelihood_loss_tapered(
                p5, I_dev, n1, n2, p_time, t_dev, _DELTA_LAT, _DELTA_LON)

        return _lbfgs_loop(params_list, optimizer, loss_fn,
                           DW_iso_lat.get_raw_log_params, max_steps, grad_tol)

    # ── Param helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def backmap(params_list):
        p = [x.item() if isinstance(x, torch.Tensor) else float(x)
             for x in params_list]
        phi2 = np.exp(p[1]); phi4 = np.exp(p[2])
        r = 1.0 / max(phi2, 1e-12)
        return {
            'sigmasq':    np.exp(p[0]) / max(phi2, 1e-12),
            'range':      r,
            'range_time': r / max(phi4**0.5, 1e-12),
            'advec':      p[3],
            'nugget':     np.exp(p[4]),
        }

    @staticmethod
    def get_printable_params(params_list):
        try:
            m = DW_iso_lat.backmap([p.item() for p in params_list])
            return (f"sigmasq={m['sigmasq']:.4f}  range={m['range']:.4f}  "
                    f"range_t={m['range_time']:.4f}  "
                    f"advec_lat={m['advec']:+.4f}  nugget={m['nugget']:.4f}")
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def get_raw_log_params(params_list):
        try:
            p = torch.cat([x.detach().cpu() for x in params_list])
            return (f"lp1={p[0]:.3f} lp2={p[1]:.3f} lp4={p[2]:.3f} "
                    f"advec_lat={p[3]:+.4f} lnug={p[4]:.3f}")
        except Exception as e:
            return f"[Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — DW_iso_lon : Isotropic + advec_lon
# ─────────────────────────────────────────────────────────────────────────────

class DW_iso_lon:
    """
    5-param model: [log_phi1, log_phi2, log_phi4, advec_lon, log_nugget]
    range_lat = range_lon = 1/phi2  (isotropic, log_phi3=0)
    advec_lat fixed to 0
    """

    N_PARAMS   = 5
    ADVEC_IDX  = 3
    ADVEC_NAME = 'advec_lon'

    @staticmethod
    def expand_to_7param(p5):
        """[lp1, lp2, lp4, advec_lon, lnug] → [lp1, lp2, 0, lp4, 0, advec_lon, lnug]"""
        d  = p5.device; dt = p5.dtype
        z  = torch.zeros(1, device=d, dtype=dt)
        return torch.cat([p5[0:2], z, p5[2:3], z, p5[3:4], p5[4:5]])

    @staticmethod
    def whittle_likelihood_loss_tapered(p5, I_sample, n1, n2, p_time,
                                         taper_autocorr_grid, delta1, delta2):
        p7 = DW_iso_lon.expand_to_7param(p5.to(I_sample.device))
        return _RawDWL.whittle_likelihood_loss_tapered(
            p7, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                           taper_autocorr_grid, max_steps=5, device='cpu', grad_tol=1e-5):
        I_dev = I_sample.to(device)
        t_dev = taper_autocorr_grid.to(device)

        def loss_fn():
            p5 = torch.cat(params_list)
            return DW_iso_lon.whittle_likelihood_loss_tapered(
                p5, I_dev, n1, n2, p_time, t_dev, _DELTA_LAT, _DELTA_LON)

        return _lbfgs_loop(params_list, optimizer, loss_fn,
                           DW_iso_lon.get_raw_log_params, max_steps, grad_tol)

    @staticmethod
    def backmap(params_list):
        p = [x.item() if isinstance(x, torch.Tensor) else float(x)
             for x in params_list]
        phi2 = np.exp(p[1]); phi4 = np.exp(p[2])
        r = 1.0 / max(phi2, 1e-12)
        return {
            'sigmasq':    np.exp(p[0]) / max(phi2, 1e-12),
            'range':      r,
            'range_time': r / max(phi4**0.5, 1e-12),
            'advec':      p[3],
            'nugget':     np.exp(p[4]),
        }

    @staticmethod
    def get_printable_params(params_list):
        try:
            m = DW_iso_lon.backmap([p.item() for p in params_list])
            return (f"sigmasq={m['sigmasq']:.4f}  range={m['range']:.4f}  "
                    f"range_t={m['range_time']:.4f}  "
                    f"advec_lon={m['advec']:+.4f}  nugget={m['nugget']:.4f}")
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def get_raw_log_params(params_list):
        try:
            p = torch.cat([x.detach().cpu() for x in params_list])
            return (f"lp1={p[0]:.3f} lp2={p[1]:.3f} lp4={p[2]:.3f} "
                    f"advec_lon={p[3]:+.4f} lnug={p[4]:.3f}")
        except Exception as e:
            return f"[Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — DW_aniso_lat : Anisotropic + advec_lat
# ─────────────────────────────────────────────────────────────────────────────

class DW_aniso_lat:
    """
    6-param model: [log_phi1, log_phi2, log_phi3, log_phi4, advec_lat, log_nugget]
    advec_lon fixed to 0
    """

    N_PARAMS   = 6
    ADVEC_IDX  = 4
    ADVEC_NAME = 'advec_lat'

    @staticmethod
    def expand_to_7param(p6):
        """[lp1,lp2,lp3,lp4,advec_lat,lnug] → [lp1,lp2,lp3,lp4,advec_lat,0,lnug]"""
        d  = p6.device; dt = p6.dtype
        z  = torch.zeros(1, device=d, dtype=dt)
        return torch.cat([p6[0:5], z, p6[5:6]])

    @staticmethod
    def whittle_likelihood_loss_tapered(p6, I_sample, n1, n2, p_time,
                                         taper_autocorr_grid, delta1, delta2):
        p7 = DW_aniso_lat.expand_to_7param(p6.to(I_sample.device))
        return _RawDWL.whittle_likelihood_loss_tapered(
            p7, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                           taper_autocorr_grid, max_steps=5, device='cpu', grad_tol=1e-5):
        I_dev = I_sample.to(device)
        t_dev = taper_autocorr_grid.to(device)

        def loss_fn():
            p6 = torch.cat(params_list)
            return DW_aniso_lat.whittle_likelihood_loss_tapered(
                p6, I_dev, n1, n2, p_time, t_dev, _DELTA_LAT, _DELTA_LON)

        return _lbfgs_loop(params_list, optimizer, loss_fn,
                           DW_aniso_lat.get_raw_log_params, max_steps, grad_tol)

    @staticmethod
    def backmap(params_list):
        p = [x.item() if isinstance(x, torch.Tensor) else float(x)
             for x in params_list]
        phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
        rlon = 1.0 / max(phi2, 1e-12)
        return {
            'sigmasq':    np.exp(p[0]) / max(phi2, 1e-12),
            'range_lat':  rlon / max(phi3**0.5, 1e-12),
            'range_lon':  rlon,
            'range_time': rlon / max(phi4**0.5, 1e-12),
            'advec':      p[4],
            'nugget':     np.exp(p[5]),
        }

    @staticmethod
    def get_printable_params(params_list):
        try:
            m = DW_aniso_lat.backmap([p.item() for p in params_list])
            return (f"sigmasq={m['sigmasq']:.4f}  rlat={m['range_lat']:.4f}  "
                    f"rlon={m['range_lon']:.4f}  rt={m['range_time']:.4f}  "
                    f"advec_lat={m['advec']:+.4f}  nugget={m['nugget']:.4f}")
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def get_raw_log_params(params_list):
        try:
            p = torch.cat([x.detach().cpu() for x in params_list])
            return (f"lp1={p[0]:.3f} lp2={p[1]:.3f} lp3={p[2]:.3f} lp4={p[3]:.3f} "
                    f"advec_lat={p[4]:+.4f} lnug={p[5]:.3f}")
        except Exception as e:
            return f"[Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — DW_aniso_lon : Anisotropic + advec_lon
# ─────────────────────────────────────────────────────────────────────────────

class DW_aniso_lon:
    """
    6-param model: [log_phi1, log_phi2, log_phi3, log_phi4, advec_lon, log_nugget]
    advec_lat fixed to 0
    """

    N_PARAMS   = 6
    ADVEC_IDX  = 4
    ADVEC_NAME = 'advec_lon'

    @staticmethod
    def expand_to_7param(p6):
        """[lp1,lp2,lp3,lp4,advec_lon,lnug] → [lp1,lp2,lp3,lp4,0,advec_lon,lnug]"""
        d  = p6.device; dt = p6.dtype
        z  = torch.zeros(1, device=d, dtype=dt)
        return torch.cat([p6[0:4], z, p6[4:5], p6[5:6]])

    @staticmethod
    def whittle_likelihood_loss_tapered(p6, I_sample, n1, n2, p_time,
                                         taper_autocorr_grid, delta1, delta2):
        p7 = DW_aniso_lon.expand_to_7param(p6.to(I_sample.device))
        return _RawDWL.whittle_likelihood_loss_tapered(
            p7, I_sample, n1, n2, p_time, taper_autocorr_grid, delta1, delta2)

    @staticmethod
    def run_lbfgs_tapered(params_list, optimizer, I_sample, n1, n2, p_time,
                           taper_autocorr_grid, max_steps=5, device='cpu', grad_tol=1e-5):
        I_dev = I_sample.to(device)
        t_dev = taper_autocorr_grid.to(device)

        def loss_fn():
            p6 = torch.cat(params_list)
            return DW_aniso_lon.whittle_likelihood_loss_tapered(
                p6, I_dev, n1, n2, p_time, t_dev, _DELTA_LAT, _DELTA_LON)

        return _lbfgs_loop(params_list, optimizer, loss_fn,
                           DW_aniso_lon.get_raw_log_params, max_steps, grad_tol)

    @staticmethod
    def backmap(params_list):
        p = [x.item() if isinstance(x, torch.Tensor) else float(x)
             for x in params_list]
        phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
        rlon = 1.0 / max(phi2, 1e-12)
        return {
            'sigmasq':    np.exp(p[0]) / max(phi2, 1e-12),
            'range_lat':  rlon / max(phi3**0.5, 1e-12),
            'range_lon':  rlon,
            'range_time': rlon / max(phi4**0.5, 1e-12),
            'advec':      p[4],
            'nugget':     np.exp(p[5]),
        }

    @staticmethod
    def get_printable_params(params_list):
        try:
            m = DW_aniso_lon.backmap([p.item() for p in params_list])
            return (f"sigmasq={m['sigmasq']:.4f}  rlat={m['range_lat']:.4f}  "
                    f"rlon={m['range_lon']:.4f}  rt={m['range_time']:.4f}  "
                    f"advec_lon={m['advec']:+.4f}  nugget={m['nugget']:.4f}")
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def get_raw_log_params(params_list):
        try:
            p = torch.cat([x.detach().cpu() for x in params_list])
            return (f"lp1={p[0]:.3f} lp2={p[1]:.3f} lp3={p[2]:.3f} lp4={p[3]:.3f} "
                    f"advec_lon={p[4]:+.4f} lnug={p[5]:.3f}")
        except Exception as e:
            return f"[Error: {e}]"
