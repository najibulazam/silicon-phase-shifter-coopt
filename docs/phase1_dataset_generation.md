# Phase 1: Automated Dataset Generation and Physics Sanity Check

## Objective
Generate a clean, versioned, multi-physics dataset of silicon thermo-optic phase shifters covering both In-Distribution (ID) training/testing and Out-of-Distribution (OOD) extrapolation regimes, storing both scalar figures of merit and 2D spatial fields (`T(x, y)` and `|E(x, y)|^2`).

## Sampling Strategy
Deterministic Latin Hypercube Sampling (LHS) was used via `scipy.stats.qmc.LatinHypercube` with resume-safe CSV/HDF5 persistence.

### Parameter Bounds
| Parameter | ID Sweep (N = 1,200, seed = 42) | OOD High-Power (N = 100, seed = 999) | OOD Geom-Extrap (N = 100, seed = 1000) |
| :--- | :--- | :--- | :--- |
| `wg_width_nm` | `[400.0, 600.0]` | `[400.0, 600.0]` | `[350.0, 650.0]` |
| `wg_height_nm` | `[200.0, 260.0]` | `[200.0, 260.0]` | `[200.0, 260.0]` |
| `heater_width_nm` | `[800.0, 2000.0]` | `[800.0, 2000.0]` | `[800.0, 2000.0]` |
| `heater_gap_nm` | `[600.0, 1500.0]` | `[600.0, 1500.0]` | `[400.0, 600.0]` |
| `heater_power_mW` | `[5.0, 60.0]` | `[60.0, 90.0]` | `[5.0, 60.0]` |

## Simulation Pipeline per Sample
1. **HEAT Solver:** Update `wg` and `wire` geometry, set `uniform heat` total power, run steady-state solve, export thermal map `.mat` file, and interpolate `T(x, y)` onto a uniform `64 x 64` grid.
2. **MODE Unperturbed Pass:** Disable temperature perturbation, solve fundamental TE mode (`mode1`), record `neff_0`, `loss_0`, and normalized mode intensity `|E(x, y)|^2` on the `64 x 64` grid.
3. **MODE Perturbed Pass:** Import thermal `.mat` map, re-solve `mode1`, record `neff_hot`, `loss_hot`, and compute `delta_neff = Re(neff_hot) - Re(neff_0)`.

## Data Integrity and Perturbation Verification
Verified via `sim_scripts/check_dataset.py`:
- **ID Dataset (`N = 1,200`):** `0` NaNs in CSV/HDF5, `0` non-positive `delta_neff`.
  - `delta_neff` range: `[6.745e-04, 8.408e-03]`
  - `T_max` range: `[307.3 K, 387.2 K]`
- **OOD Dataset (`N = 200`):** `0` NaNs in CSV/HDF5, `0` non-positive `delta_neff`.
  - `delta_neff` range: `[7.267e-04, 1.266e-02]`
  - `T_max` range: `[308.0 K, 430.6 K]`
- **First-Order Perturbation Integral Check:**
  - Unscaled `sum(dn/dT * dT * |E|^2)` exhibits a strict linear correlation with Lumerical FDE `delta_neff` with a constant Poynting/index-contrast normalization ratio of `1.5876` (ID) and `1.6025` (OOD). This constant is incorporated as a learnable physics parameter (`pert_scale`) in Phase 2.

## Artifacts Generated
- `sim_scripts/run_to_sweep.py`, `sim_scripts/run_to_ood_sweep.py`, `sim_scripts/check_dataset.py`
- `data/raw/to_scalars.csv`, `data/raw/to_fields.h5`
- `data/raw/to_ood_scalars.csv`, `data/raw/to_ood_fields.h5`
- `results/dataset_sanity_check.png`