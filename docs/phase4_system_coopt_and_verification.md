# Phase 4: Device-Driver-Mesh System-Level Co-Optimization and Verification

## Objective
Address **RQ3** by jointly co-optimizing silicon thermo-optic phase shifter geometry, electronic driver resolution (DAC bits), and 4x4 Clements MZI mesh layout using the circuit-ready compact model, followed by full Lumerical ground-truth re-simulation of the Pareto front.

## System Model and Optimization Setup (`system_coopt/nsga2_coopt.py`)
- **Architecture:** 4x4 Clements rectangular MZI mesh (6 MZIs, 12 phase shifters), evaluated across 12 deterministic random target 4x4 unitary matrices.
- **Joint Decision Variables (7):**
  - Device geometry: `wg_width_nm [400, 600]`, `wg_height_nm [200, 260]`, `heater_width_nm [800, 2000]`, `heater_gap_nm [600, 1500]`, `length_um [50, 250]`
  - Mesh layout: `pitch_um [15, 60]`
  - Drive electronics: `dac_bits [6, 12]`
- **Objectives (3):**
  1. Total 4x4 Mesh + Driver Power (mW)
  2. Mean Unitary Infidelity `1 - F` (%) under DAC phase quantization and nearest-neighbor thermal crosstalk
  3. Active Mesh Footprint (`mm^2`)
- **Constraint (1):** Peak heater temperature at full `2*pi` tuning `T_max_2pi_K <= 450.0 K`
- **Optimizer:** NSGA-II (`pop_size = 80`, `n_gen = 60`, total `4,800` system evaluations completed in `5.88 s`).

## Pareto Front Summary
80 feasible Pareto-optimal designs discovered:
- **Min Power Design:** `312.3 mW` total power (`Infidelity = 0.894%`, `DAC = 6-bit`)
- **Min Infidelity Design:** `0.0011%` infidelity (`Power = 330.3 mW`, `DAC = 11-bit`)
- **Min Footprint Design:** `0.0353 mm^2` (`Power = 721.7 mW`, `Infidelity = 1.418%`)

## Lumerical Ground-Truth Re-Simulation (`sim_scripts/verify_pareto.py`)
10 representative Pareto designs (extremes + evenly spaced trade-off points) were re-simulated in Ansys Lumerical HEAT + MODE at their surrogate-predicted `P_pi`:

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

## Artifacts Generated
- `system_coopt/nsga2_coopt.py`, `sim_scripts/verify_pareto.py`
- `results/pareto_front.csv`, `results/pareto_coopt_tradeoffs.png`
- `results/pareto_verified.csv`, `results/pareto_verification.png`