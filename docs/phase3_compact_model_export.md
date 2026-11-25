# Phase 3: Circuit-Ready Compact Model Export and Speedup Benchmark

## Objective
Address **RQ2** by packaging the trained surrogates into fast, circuit-ready compact models compatible with both Python system simulators (for Phase 4 MZI mesh co-optimization) and industry-standard SPICE/photonic circuit simulators (Cadence Spectre / Ansys Lumerical INTERCONNECT via Verilog-A).

## Exported Compact Models
Executed via `models/export_compact_model.py`:
1. **Python Compact Model Bundle (`models/compact_model_bundle.pkl`):**
   - Contains fitted `StandardScaler`, `GP` surrogate for `delta_neff`, and `GP` surrogate for peak temperature rise `dT_max` (trained on `N = 800` ID samples).
   - Used directly inside the Phase 4 NSGA-II multi-objective optimizer.
2. **Verilog-A Compact Model (`models/silicon_to_shifter.va`):**
   - Implements electrical heater resistance (`r_heater_ohm`), first-order thermal low-pass dynamics (`tau_th_us` via `laplace_nd`), log-power-law surrogate equations for `delta_neff` and `T_max`, and optical phase output `delta_phi_rad = 2 * pi * L * delta_neff / lambda`.

## Fidelity and Inference Speedup Benchmark
Benchmarked on the `N = 150` ID Test set against average Lumerical HEAT + FDE runtime (`~8.00 s/eval` on Ryzen 5 5500U):

| Surrogate Model | Output Type | ID Test MAPE | ID Test $R^2$ | Inference Time | Speedup vs. Lumerical |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GP Compact (Python)** | Scalar `delta_neff`, `T_max` | `0.032%` | `0.999999` | `0.0155 ms/eval` | `520,986x` |
| **Verilog-A Compact** | Circuit-level `phi`, `T_max` | `0.347%` | `0.999892` | `0.088 us/eval` | `92,261,064x` |
| **Field-PINN 2D** | Full `64 x 64` `T(x, y)` field | `1.680%` | `0.997300` | `0.0981 ms/eval` | `82,602x` |

## Artifacts Generated
- `models/export_compact_model.py`
- `models/compact_model_bundle.pkl`
- `models/silicon_to_shifter.va`
- `results/combined_learning_curves.png`