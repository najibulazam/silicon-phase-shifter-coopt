# Circuit-Ready Physics-Informed Surrogates for Co-Optimizing Silicon Phase Shifters and Their Drive Electronics

**Target journal:** PhotoniX Synergy (Springer Nature)
**Status:** Idea / pre-proposal. Supersedes `INIT.md`.
**Basis:** A preliminary web search, not a full systematic review. Every novelty claim below must be re-checked against the full texts before submission.

---

## 1. Why the original idea needed to change

The original concept (`INIT.md`) was a physics-informed surrogate for one thermo-optic phase shifter, followed by NSGA-II Pareto optimization. A preliminary search found that each piece already exists:

| Existing work | Relevance |
|---|---|
| [ThermoPhoton, ICCAD 2025](https://personal.hkust-gz.edu.cn/yuzhema/papers/ICCAD2025-PIC-Thermal.pdf) | Operator-learning surrogate for 3D photonic thermal simulation, with PINN-style methods discussed |
| [Teofilovic, Da Ros et al., 2024](https://arxiv.org/html/2404.10589v1) | Physics-based vs. data-driven thermal crosstalk models for programmable PICs |
| [PINN review (arXiv 2606.21945)](https://arxiv.org/pdf/2606.21945) | Cites PINN thermal surrogates (heat-equation residual) and thermo-optic PINN work |
| [Sci. Rep. 2023, MZM design](https://www.nature.com/articles/s41598-023-41558-8) | Deep-learning surrogate plus heuristic optimizer for a silicon modulator |
| [TCAD surrogate + NSGA-II](https://par.nsf.gov/servlets/purl/10440571) | The surrogate-plus-Pareto workflow is standard in device design |

Three further weaknesses applied to the original plan:

1. **The physics loss was nearly trivial.** Δn = (dn/dT)·ΔT and Δφ = 2πLΔn_eff/λ are algebraic, so a plain MLP would probably match the PINN.
2. **The speedup was marginal.** A simulation takes about 15 s, so the data generation costs about as much as running a direct optimizer.
3. **It was device-level and simulation-only.** PhotoniX Synergy [prioritizes](#7-fit-with-photonix-synergy) system-level co-integration, real-world benchmarking and co-optimization, and this plan had none of them.

## 2. Reframed concept

**One-sentence pitch:** a physics-informed, *circuit-ready* surrogate of silicon phase shifters (thermo-optic and carrier-depletion) that is used to **co-optimize the phase shifter and its drive electronics at system level**, demonstrated on a programmable MZI mesh and benchmarked against measured data.

The surrogate is a means, not the result. The contribution is what it lets you do that direct simulation cannot: explore a joint device, driver and circuit design space, and report system-level metrics.

### Research questions

| # | Question | Evidence needed |
|---|---|---|
| RQ1 | Does a *field-level*, parametric physics-informed model beat MLP / ResNet / GP baselines in data efficiency and out-of-distribution accuracy? | Learning curves, extrapolation tests, physics-residual metrics |
| RQ2 | Can the surrogate be exported as a fast compact model that a circuit or system simulator can use? | Verilog-A or Python/INTERCONNECT-compatible model, with fidelity checks |
| RQ3 | How do device, driver and mesh choices trade off power, phase accuracy, thermal crosstalk and speed *at system level*? | A Pareto front verified by re-simulation, plus a mesh-level case study |
| RQ4 | Does the same framework transfer across platforms (silicon TO vs. PN depletion)? | Same pipeline on both devices (thin-film LN is an optional stretch goal) |

## 3. Technical approach

### 3.1 Devices and parameters

| Device | Template | Swept parameters | Outputs |
|---|---|---|---|
| Thermo-optic (primary) | [Thermally tuned waveguide (FDE)](https://optics.ansys.com/hc/en-us/articles/360042833673-Thermally-tuned-waveguide-FDE) | Waveguide width/height, heater width, heater–waveguide gap, cladding thickness, heater power, optionally trench or undercut | Temperature field, Δn_eff, loss, Pπ, thermal time constant τ |
| Carrier depletion (secondary) | [PN depletion phase shifter](https://optics.ansys.com/hc/en-us/articles/360042328674-PN-depletion-phase-shifter) | Junction offset, doping levels, voltage, waveguide geometry | Δn_eff, loss, V_π·L, capacitance |
| Two-heater crosstalk | Same as the TO device with a second heater | Lateral heater spacing, isolation trench | Crosstalk coefficient between neighbouring shifters |

### 3.2 What makes the model genuinely physics-informed

Instead of regressing scalars with a "physics term", use a **parametric field model**:

- A network `T(x, z; θ, P)` predicts the temperature field for geometry `θ` and heater power `P`.
- The loss combines sparse Lumerical HEAT data with the **steady-state heat-equation residual** ∇·(k∇T) + q = 0, plus boundary conditions.
- A **differentiable perturbation layer** converts the temperature field to Δn_eff using first-order perturbation theory, Δn_eff ≈ (dn/dT)·∫ΔT|E|²dA / ∫|E|²dA. The mode profile `E` comes from the unperturbed FDE solve.
- Transient behaviour (τ) is handled either by a first-order fit to HEAT transient runs or as a stretch goal with a time-dependent residual.

This puts real physics inside the model: the field residual and the overlap integral. A scalar regressor with an algebraic penalty does not.

### 3.3 Baselines (required, not optional)

1. Analytic 1D/2D thermal model (the strongest "free" competitor)
2. MLP and ResNet (from the original plan)
3. Gaussian process or Bayesian-optimization surrogate
4. Gradient-boosted trees
5. Optional: a DeepONet/operator-learning baseline, in the spirit of ThermoPhoton

### 3.4 Co-optimization and system-level case study

- **Circuit-ready export:** wrap the trained model as a compact model for circuit/system simulation.
- **Driver co-design variables:** DAC resolution, voltage headroom, power budget and update rate.
- **System case study:** a small MZI mesh (for example 4×4 or 8×8, Clements layout) used for optical computing. Report total tuning power, programmed-matrix fidelity under thermal crosstalk, and the effect on a small classification task.
- **Optimization:** multi-objective (NSGA-II or Bayesian) over device, driver and layout. **Every reported Pareto point is re-simulated in Lumerical**, because surrogates often fail in regions the optimizer proposes (see [EngiBench](https://arxiv.org/pdf/2508.00831)).

## 4. Validation plan (this decides the journal tier)

| Tier | What | Value |
|---|---|---|
| A (minimum) | Compare against published measured values (Pπ, τ, V_π·L, crosstalk) for comparable geometries | Baseline credibility |
| B (strongly recommended) | Use a foundry PDK or collaborator data, or an academia–industry partner | Matches the journal's stated priorities |
| C (best) | Own fabricated and measured devices | Makes it a convincing synergistic/system-level paper |

**Without tier B or C, the paper is simulation-only. In that case choose a lower-barrier venue.**

## 5. Roadmap (indicative, about 6 months)

| Phase | Weeks | Work | Exit criterion |
|---|---|---|---|
| 0 | 1–3 | Full literature review and gap statement; confirm Lumerical license; reproduce both templates | Written gap analysis; templates run |
| 1 | 3–7 | `lumapi` automation, dataset generation (TO first), mesh-level data hooks | Clean, versioned dataset; per-sample runtime logged |
| 2 | 7–12 | Baselines, then field-level PINN with perturbation layer on Colab | PINN vs. baselines on learning curves and OOD tests |
| **Gate** | 12 | **Go/no-go:** does the physics-informed model win in low-data or OOD regimes? | See pivot options below |
| 3 | 12–16 | PN depletion pipeline; compact-model export | Compact model reproduces Lumerical within the target error |
| 4 | 16–21 | Mesh case study, driver co-design, Pareto re-simulation | Verified Pareto front and system-level metrics |
| 5 | 21–26 | Validation (tier A, plus B/C if available); writing | Manuscript draft |

**Pivot options at the gate:** if the PINN does not beat well-tuned baselines, reframe as (a) an open benchmark dataset plus a systematic comparison of surrogate classes for photonic phase shifters, or (b) a system-level co-optimization study using the best surrogate. Both are honest outcomes. Do not claim a PINN advantage that the data don't show.

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PINN offers no gain over MLP/GP | High | Test early at the gate. Use the field-level formulation and low-data/OOD regimes, where physics priors usually help. Prepare the pivot. |
| No experimental or industry data | Medium | Tier A as the floor. Seek a foundry or lab collaborator in Phase 0. |
| Laptop compute limits (Ryzen 5 5500U, 16 GB) | Medium | The 2D FDE/HEAT templates are light. The earlier "~700 samples in 3–4 h" figure is an estimate to verify in Phase 1. Train on Colab. |
| Surrogate-guided optimizer exploits model errors | Medium | Re-simulate all Pareto points. Use ensembles or uncertainty estimates. |
| Novelty challenged by related work | Medium | Full literature review in Phase 0. State the contribution as the *system-level co-optimization with circuit-ready models*, not the surrogate alone. |
| Lumerical licensing | Unknown | Confirm access before starting. |

## 7. Fit with PhotoniX Synergy

Source: the journal's published aims and scope.

| Journal emphasis | How this project addresses it |
|---|---|
| AI-driven design and optimization of photonic and electronic ICs | Physics-informed surrogate plus co-optimization |
| Photonic chip modelling and characterization | Compact models, validated against measurements |
| Optoelectronic system-level co-optimization | Device–driver–mesh co-design |
| PICs and architecture for optical computing and AI | MZI-mesh case study |
| Co-deployment of diverse material platforms | TO vs. PN depletion, optionally TFLN |
| Real-world benchmarking; academia–industry collaboration | Validation tiers B/C (needs partners) |

**Honest assessment:** topic fit is good. Whether it clears the bar depends mostly on the system-level demonstration and on measured or industry data. I could not find any published papers from this journal to calibrate against, so ask the editors or check recent issues before committing.

## 8. Expected contributions (to be claimed only if the results support them)

1. A parametric, field-level physics-informed model for silicon phase shifters, benchmarked against analytic, MLP, ResNet and GP baselines.
2. A circuit-ready compact model derived from it.
3. A device–driver–mesh co-optimization with simulation-verified Pareto fronts.
4. An open dataset and benchmark.
5. Validation against measured or published data.

## 9. Open questions

- Do you have access to a Lumerical license, and which version?
- Is there a collaborator, foundry or lab for measurements or PDK data?
- What is the submission deadline, and does the journal need to be indexed (impact factor)?
- Should thin-film lithium niobate be in scope, or keep it to silicon?

## 10. Immediate next steps

1. Run the full literature review (Phase 0) and write the one-page gap statement.
2. Reproduce both Ansys templates manually.
3. Write the `lumapi` automation script and time a single run.
4. Settle the validation tier, which sets the journal strategy.

## References

- [Ansys Lumerical — Thermally tuned waveguide (FDE)](https://optics.ansys.com/hc/en-us/articles/360042833673-Thermally-tuned-waveguide-FDE)
- [Ansys Lumerical — PN depletion phase shifter](https://optics.ansys.com/hc/en-us/articles/360042328674-PN-depletion-phase-shifter)
- [ThermoPhoton: Fast 3D Thermal Simulation of PICs via Operator Learning (ICCAD 2025)](https://personal.hkust-gz.edu.cn/yuzhema/papers/ICCAD2025-PIC-Thermal.pdf)
- [Thermal Crosstalk Modelling and Compensation Methods for Programmable PICs (arXiv 2404.10589)](https://arxiv.org/html/2404.10589v1)
- [PINN review for multi-physics design (arXiv 2606.21945)](https://arxiv.org/pdf/2606.21945)
- [Silicon MZM design with deep learning and evolutionary algorithms (Sci. Rep. 2023)](https://www.nature.com/articles/s41598-023-41558-8)
- [TCAD surrogate + NSGA-II (NSF PAR 10440571)](https://par.nsf.gov/servlets/purl/10440571)
- [EngiBench (arXiv 2508.00831)](https://arxiv.org/pdf/2508.00831)

*Note: these references come from search snippets. Read each one in full, and verify the details, before citing it.*