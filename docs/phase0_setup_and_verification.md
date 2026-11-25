# Phase 0: Environment Setup and Solver Verification

## Objective
Establish a reproducible simulation and machine learning workspace, verify Python-to-Lumerical API (`lumapi`) connectivity, inspect Ansys Lumerical template object hierarchies, and benchmark single-pass solver runtime.

## Hardware and Software Environment
- **Workstation:** AMD Ryzen 5 5500U, 16 GB DDR4 RAM, Windows 11
- **Photonic/Thermal Solver:** Ansys Lumerical v241 (`DEVICE`/HEAT and `MODE`/FDE)
- **Python Environment:** Isolated `venv` (Python 3.12)
- **Core Packages:** `numpy`, `scipy`, `pandas`, `h5py`, `pyyaml`, `matplotlib`, `scikit-learn`, `pymoo`, `torch` (2.15.0+cpu)

## Template Verification
Reference templates from Ansys Lumerical (*Thermally tuned waveguide - FDE*) were placed in `templates/to_phase_shifter/`:
- `waveguide_heat.ldev` (2D steady-state thermal FEM solver)
- `waveguide_fde.lms` (2D Z-normal Finite-Difference Eigenmode solver)

### Inspected Object Hierarchy
- **HEAT (`::model::geometry`):** `substrate`, `wg`, `oxide`, `gas`, `wire`
- **HEAT (`::model::HEAT`):** `wg_monitor`, `boundary conditions`, `uniform heat`, `wg_monitor_1`
- **MODE (`::model`):** `oxide`, `silicon`, `wg`, `FDE`, `temperature`, `mesh`

### Coordinate Convention
- Cross-section plane: **2D X-Y plane (Z-normal)**
- Silicon waveguide (`wg`): centered at `x = 0`, `y min = 0.0`, `y max = wg_height`
- Metal heater (`wire`): centered at `x = 0`, thickness `200 nm`, positioned at `y min = wg_height + heater_gap`
- FDE optical window: `x in [-1.0, 1.0] um`, `y in [-0.89, 1.11] um`, operating wavelength `1.55 um`

## Baseline Timing Benchmark
Executed via `sim_scripts/test_lumapi.py` and `sim_scripts/inspect_props.py`:
- **HEAT single pass:** `12.56 s` (cold launch)
- **MODE (FDE) single pass:** `5.50 s` (cold launch)
- **Total cold-launch runtime:** `18.06 s`
- **Persistent session runtime (in sweep loop):** `~7.9 - 8.1 s` per sample

## Artifacts Generated
- `config.yaml` (centralized project configuration)
- `sim_scripts/test_lumapi.py`
- `sim_scripts/inspect_props.py`
- Execution logs archived in `log/`