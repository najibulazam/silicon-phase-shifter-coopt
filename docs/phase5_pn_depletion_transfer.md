# Phase 5: Cross-Platform Framework Transfer to PN Carrier-Depletion Shifter (RQ4)

## Objective
Address **RQ4** by transferring the simulation-to-co-optimization pipeline from silicon thermo-optic (TO) phase shifters to a second physical mechanism—**silicon PN carrier-depletion phase shifters**—using Ansys Lumerical CHARGE and MODE (FDE).

## 1. Automated PN LHS Dataset Generation (`sim_scripts/run_pn_sweep.py`)
- **Templates:** `templates/pn_phase_shifter/PN_phase_shifter.ldev` (CHARGE) and `PN_phase_shifter.lms` (MODE FDE).
- **Sampling:** 25 Latin Hypercube Sampling (LHS) geometric/doping cases (`wg_width_nm in [450, 550]`, `p_conc_1e23 in [2.0, 6.0]`, `n_conc_1e23 in [2.0, 6.0]`, `pn_offset_nm in [-60, +60]`), each evaluated across 9 reverse-bias drain voltage steps (`0.0 V` to `-4.0 V`, `0.5 V` step) for a total of **225 samples** (`data/raw/pn_depletion_sweep.csv`).
- **Observed Index Modulation:** `|delta_neff(-4V)|` spans `[9.983e-05, 1.215e-04]`.

## 2. Surrogate Benchmark and Compact Model Export (`models/run_pn_transfer.py`)
Split by `case_id` into 18 training cases (162 samples) and 7 unseen test cases (63 samples):

| Model | Test MAPE (%) | Test $R^2$ | Notes |
| :--- | :--- | :--- | :--- |
| **GP Surrogate (Matern 5/2)** | `0.446%` | `0.99941` | High-accuracy compact model (`0.0045 ms/eval`, `~4.26e6x` speedup vs. ~19 s/pt Lumerical) |
| **GBR** | `0.704%` | `0.99824` | Strong tree baseline |
| **Standard MLP** | `11.013%` | `0.63282` | Higher error on unseen doping/geometry cases |
| **PN_PhysicsNet** | `11.644%` | `-3.47241` | Learned `V_bi = 0.78 V`; follows trend on most points but exhibits outliers near zero |

- **Exported Circuit Model:** `models/silicon_pn_shifter.va` (includes reverse-bias junction capacitance `c_j`, carrier-depletion phase shift, and free-carrier insertion loss).

## 3. PN 4x4 Clements MZI Mesh Co-Optimization
Using NSGA-II (`pop_size = 80`, `n_gen = 60`) over `wg_width_nm`, `p_conc_1e23`, `n_conc_1e23`, `pn_offset_nm`, `length_mm [1.0, 8.0]`, and `dac_bits [6, 12]` under `V_pi <= 4.0 V`:
- **Feasible Pareto Designs:** `36` designs (`results/pn_pareto_front.csv`)
- **Min Drive Voltage ($V_\pi$):** `0.500 V` (`L = 6.37 mm`, `Loss + Quantization Penalty = 6.27 dB`)
- **Min Loss + Quantization Penalty:** `6.128 dB` (`V_pi = 0.500 V`, `L = 6.38 mm`)
- **Min Active Mesh Footprint:** `2.5726 mm^2` (`V_pi = 0.500 V`)

## Artifacts Generated
- `sim_scripts/inspect_pn_template.py`, `sim_scripts/inspect_pn_props.py`, `sim_scripts/run_pn_sweep.py`
- `models/run_pn_transfer.py`, `models/silicon_pn_shifter.va`
- `data/raw/pn_lhs_cases.csv`, `data/raw/pn_depletion_sweep.csv`
- `results/pn_surrogate_metrics.csv`, `results/pn_pareto_front.csv`, `results/pn_transfer_results.png`