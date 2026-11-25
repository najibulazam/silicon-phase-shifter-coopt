# Phase 4: Crosstalk Extraction, Device-Driver-Mesh Co-Optimization, and Optical AI Case Study

## Objective
Address **RQ3** by extracting a physics-grounded lateral thermal crosstalk model directly from Ansys Lumerical HEAT, jointly co-optimizing silicon thermo-optic phase shifter geometry, electronic driver resolution (DAC bits), and 4x4 Clements MZI mesh layout, verifying the Pareto front via Lumerical ground-truth re-simulation, and evaluating end-to-end hardware accuracy on an optical neural network (ONN) classification task.

## 1. Lumerical HEAT Lateral Thermal Crosstalk Extraction (`sim_scripts/extract_crosstalk.py`)
- **Setup:** 15 steady-state 2D HEAT runs across `heater_gap_nm in [600, 800, 1000, 1250, 1500]` and `heater_width_nm in [800, 1400, 2000]` over an expanded `90 um` lateral simulation window (`P = 25 mW`).
- **Extracted Crosstalk Ratio:** `XT(pitch) = dT(x=pitch, y=y_wg) / dT(x=0, y=y_wg)` sampled from `5 um` to `40 um`.
  - At `pitch = 15 um`: `XT = 3.91%`
  - At `pitch = 30 um`: `XT = 0.12%`
- **Fitted Model (`models/crosstalk_model.json`):**
  - `XT(pitch_um) = 1.4108 * exp(-pitch_um / 4.167)`
  - Fit quality: `RMSE = 0.00030`, `R2 = 1.00000`

## 2. System Model and NSGA-II Co-Optimization (`system_coopt/nsga2_coopt.py`)
- **Architecture:** 4x4 Clements rectangular MZI mesh (6 MZIs, 12 phase shifters), evaluated across 12 deterministic random target 4x4 unitary matrices.
- **Joint Decision Variables (7):**
  - Device geometry: `wg_width_nm [400, 600]`, `wg_height_nm [200, 260]`, `heater_width_nm [800, 2000]`, `heater_gap_nm [600, 1500]`, `length_um [50, 250]`
  - Mesh layout: `pitch_um [10, 40]`
  - Drive electronics: `dac_bits [6, 12]`
- **Objectives (3):**
  1. Total 4x4 Mesh + Driver Power (mW)
  2. Mean Unitary Infidelity `1 - F` (%) under DAC phase quantization and Lumerical-fitted nearest-neighbor thermal crosstalk
  3. Active Mesh Footprint (`mm^2`)
- **Constraint (1):** Peak heater temperature at full `2*pi` tuning `T_max_2pi_K <= 450.0 K`
- **Optimizer:** NSGA-II (`pop_size = 80`, `n_gen = 60`, total `4,800` system evaluations completed in `5.83 s`).

### Pareto Front Summary (80 Feasible Designs)
- **Min Power Design:** `312.4 mW` total power (`Infidelity = 0.6218%`, `DAC = 7-bit`)
- **Min Infidelity Design:** `0.00000%` infidelity (`Power = 344.3 mW`, `DAC = 12-bit`)
- **Min Footprint Design:** `0.0236 mm^2` (`Power = 720.7 mW`, `Infidelity = 7.886%`)

## 3. Lumerical Ground-Truth Re-Simulation (`sim_scripts/verify_pareto.py`)
10 representative Pareto designs were re-simulated in Ansys Lumerical HEAT + MODE at their surrogate-predicted `P_pi` to verify that the optimizer did not exploit surrogate extrapolation errors:

| Pareto ID | `P_pi` (mW) | Length (um) | Achieved Phase (`dphi / pi`) | Relative Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| `PID=00` | `21.90` | `249.7` | `1.0001` | `0.005%` |
| `PID=07` | `22.12` | `248.8` | `0.9999` | `0.008%` |
| `PID=08` | `22.78` | `240.3` | `0.9999` | `0.010%` |
| `PID=17` | `26.78` | `205.9` | `1.0009` | `0.089%` |
| `PID=26` | `28.12` | `195.9` | `1.0005` | `0.047%` |
| `PID=35` | `31.39` | `176.7` | `0.9999` | `0.007%` |
| `PID=43` | `35.06` | `156.2` | `1.0004` | `0.038%` |
| `PID=52` | `38.70` | `141.5` | `1.0008` | `0.077%` |
| `PID=61` | `41.86` | `132.2` | `1.0000` | `0.001%` |
| `PID=76` | `51.46` | `106.4` | `1.0006` | `0.058%` |

- **Mean Verification Error:** `0.0339%`
- **Max Verification Error:** `0.0892%`

## 4. Optical AI Classification Case Study (`system_coopt/optical_ai_benchmark.py`)
- **Task:** 3-class Iris classification (4 input features mapped directly to a 4-port optical waveguide array; 70/30 stratified train/test split).
- **Architecture:** Two cascaded 4x4 Clements rectangular MZI meshes (`2 x 12 = 24` thermo-optic phase shifters) with photodetector intensity nonlinearity (`|E|^2`) between layers and a `4 -> 3` linear electronic readout.
- **Hardware Impairment Injection:** Trained continuous phases are quantized to each Pareto design's `dac_bits` (`6-12 bit`) and perturbed by Lumerical-fitted nearest-neighbor thermal crosstalk (`xt_coeff`).

### Optical AI Benchmark Results
- **Ideal Software Accuracy:** `88.89%` (400 epochs, `5.67 s`)
- **Best HW Accuracy / Optimal Sweet Spot (`PID=05`):** `88.89%` (`0.00%` drop) at `333.3 mW` total power, `11-bit DAC`, `37.7 um` pitch, `0.1748 mm^2` footprint.
- **Worst HW Accuracy (`PID=59`):** `44.44%` (`44.45%` drop) at `629.8 mW` total power, `7-bit DAC`, `10.0 um` pitch (severe lateral thermal crosstalk).

### Key System-Level Insight
Tightly packed meshes (`pitch < 20 um`, footprint `< 0.08 mm^2`) suffer severe classification accuracy collapse (down to `44.44%`) due to lateral thermal crosstalk extracted from Lumerical HEAT. Increasing heater pitch to `>= 25-30 um` (`footprint >= 0.085 mm^2`) suppresses crosstalk sufficiently to recover full software-equivalent accuracy (`88.89%`) with minimal power overhead (`~21 mW` above the minimum-power design).

## Artifacts Generated
- `sim_scripts/extract_crosstalk.py`, `sim_scripts/verify_pareto.py`
- `system_coopt/nsga2_coopt.py`, `system_coopt/optical_ai_benchmark.py`
- `data/raw/to_crosstalk.csv`, `models/crosstalk_model.json`
- `results/lumerical_crosstalk_fit.png`, `results/pareto_front.csv`, `results/pareto_coopt_tradeoffs.png`
- `results/pareto_verified.csv`, `results/pareto_verification.png`
- `results/optical_ai_results.csv`, `results/optical_ai_accuracy_tradeoff.png`