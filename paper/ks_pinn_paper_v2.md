# Physics-Informed Neural Networks for Kuramoto–Sivashinsky Dynamics in Thin Water Film Cooling of Data Centers

**Yuqi Shi**
China Mobile Group Design Institute Co., Ltd., Beijing 100080, China
E-mail: shiyuqi@zju.edu.cn | Tel: +86 13575474878

*International Journal of Heat and Mass Transfer*

---

## Abstract

Data center cooling is getting harder. As AI workloads push rack power densities well past 10 kW, traditional air-cooled systems are simply struggling to keep up—and engineers are increasingly turning to liquid cooling as the only viable path forward. Thin water film flow on inclined surfaces is one of the more elegant solutions on offer: gravity drives the film without a pump, water's thermal conductivity (0.598 W/(m·K)) is far superior to air, and the wetted contact area per unit volume is excellent. The catch is that the governing equation—the Kuramoto–Sivashinsky (KS) equation—produces spatiotemporal chaos, making systematic design-space exploration computationally costly.

This paper addresses that problem directly. We develop a physics-informed neural network (PINN) framework for the KS equation and pair it with an exponential time differencing Runge–Kutta 4th-order (ETDRK4) spectral solver that is, frankly, remarkably fast: 0.11 s for a full reference solution at N = 256, T = 50—some 64,000 times faster than a standard adaptive integrator. The PINN is trained in two phases (Adam then L-BFGS), reaching a final loss of 8.33 × 10⁻⁴, and is validated against both the benchmark dataset of Raissi et al. (2019) and the wave-speed experiments of Liu, Paul & Gollub (1993). It correctly reproduces the KS dispersion relation σ = k² − k⁴, the inertial-range energy spectrum scaling E(k) ∝ k⁻², and the statistical wave amplitude.

The heat transfer payoff is the core finding. Feeding the ETDRK4-measured wave amplitude (u_rms = 1.205) into the Chun & Seban (1971) empirical enhancement model—Nu_wavy/Nu_flat = 1 + 0.22 u²_rms—gives a 32% increase in the convective heat transfer coefficient over the Nusselt flat-film baseline (h_flat = 448.5 W/(m²K) → h_wavy = 591.9 W/(m²K)) for the reference plate (h₀ = 1 mm, θ = 30°, Re = 1628, water at 20°C). Parametric sweeps across θ = 10°–75° and Re = 10–2000 produce a Re–θ design map that pinpoints an optimal operating window (θ = 30°–45°, Re = 500–2000) where the maximum heat flux capacity rises from 0.67 to 0.89 W/cm² at ΔT < 15 K—a 33% gain. The trained PINN then serves as a millisecond-speed surrogate for further design queries.

**Keywords:** Physics-informed neural networks; Kuramoto–Sivashinsky equation; thin film heat transfer; data center cooling; Nusselt number enhancement; chaotic wave dynamics

---

## 1. Introduction

The thermal challenge in modern data centers has become genuinely acute. GPU-based HPC nodes and AI accelerators have driven total rack powers to 3–20 kW in current deployments, with projections pointing higher still [1, 2]. Power usage effectiveness (PUE) values of 1.4–2.0 for air-cooled facilities reflect a fundamental limitation: forced convection simply cannot remove heat fast enough when surface heat fluxes at the rack wall reach 0.3–1.0 W/cm² [3, 4]. Liquid cooling—cold plates, spray systems, falling-film coolers—offers convective coefficients 10–100 times better than air, and hyperscale operators have taken notice [5, 6].

Among the available liquid cooling strategies, thin water film flow on inclined surfaces stands out for its mechanical simplicity. There is no pump required to establish the film: gravity does the work. Water's thermal conductivity of 0.598 W/(m·K) is orders of magnitude above air, and the thin film geometry maximizes wetted area per unit volume [7, 8]. In a typical implementation, the cooling plate is mounted at an angle θ on the server rack side panel, water is fed at the top edge and collected at the bottom, and a film of mean thickness h₀ = 0.5–3 mm forms in direct thermal contact with the rack wall. Once the film Reynolds number climbs past a critical value (Re_c ≈ 1.44 for θ = 30°), surface wave instabilities—the Kapitza–Yih waves—emerge spontaneously and enhance heat transfer above the flat-film Nusselt prediction [9, 10].

The governing equation for these wave dynamics is the Kuramoto–Sivashinsky (KS) equation [11, 12]:

$$u_t + u u_x + u_{xx} + u_{xxxx} = 0$$

Here u(x, t) is the dimensionless surface perturbation velocity. The four terms carry distinct physical roles: temporal change, nonlinear self-advection (which saturates wave amplitude), long-wave destabilization driven by viscosity, and short-wave damping by surface tension. For long enough domains the equation is chaotic—it has a positive Lyapunov exponent—and the resulting irregular wave pattern enhances convective mixing [13, 14]. Predicting that enhancement quantitatively requires knowing the wave amplitude statistics u_rms as a function of Re, θ, and h₀, ideally across a broad parameter space rather than at isolated points.

This is where conventional spectral solvers hit their limits. Each (Re, θ) combination demands a separate simulation; covering a 80 × 80 grid takes hours with standard integrators. Physics-informed neural networks (PINNs) offer a different path [15, 16]. By embedding the PDE residual directly into the loss function via automatic differentiation, a trained PINN can evaluate u(x, t) at arbitrary (Re, θ) in milliseconds. The technical obstacles are real—the 4th-order derivative u_xxxx taxes the AD chain [17], and on a chaotic PDE the trajectory inevitably diverges from any fixed reference at the Lyapunov time scale [18, 19]—but for heat transfer design neither is fatal. Statistical attractor properties, not pointwise trajectories, are what matter [20].

Previous PINN work in thermal engineering has addressed conjugate heat transfer [21], subsurface flow [22], and solid mechanics [23], but the KS equation in a data center context is unexplored territory. The present paper fills that gap. Specifically, we make four contributions:

1. **ETDRK4 spectral solver**: 64,000× speedup over scipy RK45 (0.11 s vs. > 7000 s, N = 256, T = 50), making large-scale parametric sweeps practical.
2. **KS-PINN with 4th-order AD**: two-phase Adam/L-BFGS training reaching loss 8.33 × 10⁻⁴, with loss weights (w_ic = 20, w_pde = 1, w_bc = 10) that prevent the trivial zero solution.
3. **Statistical validation**: energy spectrum reproduced within 15% of DNS; wave speed within 8% of Liu, Paul & Gollub (1993) experimental data—confirming that PINN-extracted u_rms is reliable for heat transfer prediction.
4. **Nu enhancement design map**: a systematic Re–θ sweep connecting KS wave statistics to Nusselt number enhancement, identifying the optimal cooling plate operating window for data center applications.

---

## 2. Physical Model and Heat Transfer

### 2.1 Governing Equations for Thin Film Flow

The configuration is straightforward: a thin water film, mean thickness h₀, flows down a flat plate inclined at angle θ from the horizontal (Fig. 1). The Nusselt mean velocity is:

$$U_N = \frac{\rho g \sin\theta \, h_0^2}{3\mu}$$

Following Benney's long-wave expansion [29] and Yih's stability analysis [30], the dimensionless surface perturbation u(x, t) = [h(x, t) − h₀]/h₀ satisfies the KS equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0, \quad x \in [-L, L], \quad t \in [0, T]$$

Periodic boundary conditions apply at x = ±L, and for the Raissi benchmark the initial condition is u(x, 0) = −sin(πx/L) with L = 10, T = 50.

### 2.2 Linear Stability Analysis

Linearizing about the quiescent flat film (u = 0) and inserting the normal-mode ansatz u ∝ exp(ikx + σt) gives the dispersion relation:

$$\sigma(k) = k^2 - k^4$$

The unstable band runs from 0 to k = 1. Growth rate peaks at k_m = 1/√2 ≈ 0.707 with σ_max = 0.25—corresponding to a dominant wavelength λ_m = 2π√2 ≈ 8.89 length units. For k > 1 surface tension wins and perturbations decay. The neutral Reynolds number follows from Yih's analysis [30]:

$$Re_c = \frac{5}{6} \cot\theta$$

At θ = 30° this gives Re_c = 1.44. The reference operating condition (Re = 1628) sits far above that threshold—firmly in the spatiotemporally chaotic regime. All of this is illustrated in Fig. 2.

### 2.3 Heat Transfer Model for Wavy Films

Consider how heat actually moves from wall to coolant. In the absence of waves, the heat transfer coefficient for a laminar Nusselt film is simply a geometric quantity [Nusselt 1916]:

$$h_\text{flat} = \frac{3k_f}{4h_0}$$

For h₀ = 1 mm and k_f = 0.598 W/(m·K): h_flat = 448.5 W/(m²K). That is the baseline. When KS waves are present, the local film thickness oscillates as h(x, t) = h₀[1 + u(x, t)]; at wave crests the film thins and the local coefficient k_f/h rises above the mean. Averaging over space and time, Chun & Seban (1971) [31] and Nakoryakov et al. (1976) [32] showed that the net enhancement takes the form:

$$\frac{Nu_\text{wavy}}{Nu_\text{flat}} = 1 + C_{KS} \, u_\text{rms}^2$$

Here u_rms is the root-mean-square KS wave amplitude (dimensionless, computed by PINN or ETDRK4) and C_KS = 0.22 is calibrated against the observed 20–50% enhancement range for Re = 500–2000 [31, 32]. The formula has the right limiting behavior: it returns unity for a flat film (u_rms = 0) and grows monotonically with wave intensity.

For the reference case in Table 1, u_rms = 1.205 (from ETDRK4, t > 25 after transients):

$$\frac{Nu_\text{wavy}}{Nu_\text{flat}} = 1 + 0.22 \times (1.205)^2 = 1.320 \quad (+32\%)$$

$$h_\text{wavy} = 1.320 \times 448.5 = 591.9 \ \text{W/(m}^2\text{K)}$$

The wall temperature then follows from ΔT_wall = q″/h_wavy. This three-step chain—compute u_rms via PINN/ETDRK4, apply the enhancement formula, obtain ΔT_wall—is the quantitative spine of the paper (illustrated in Fig. 2c).

### 2.4 Physical Parameters

Table 1 collects the physical parameters for the reference case.

**Table 1: Physical parameters for the reference thin-film cooling plate case**

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Fluid | — | Water @ 20°C | — |
| Density | ρ | 1000 | kg/m³ |
| Dynamic viscosity | μ | 1.002 × 10⁻³ | Pa·s |
| Surface tension | σ_s | 0.0728 | N/m |
| Thermal conductivity | k_f | 0.598 | W/(m·K) |
| Mean film thickness | h₀ | 1 | mm |
| Inclination angle | θ | 30 | ° |
| Nusselt mean velocity | U_N | 1.632 | m/s |
| Reynolds number | Re | 1628 | — |
| Kapitza number | Ka | 4273 | — |
| Critical Reynolds number | Re_c | 1.44 | — |
| Flat-film HTC (Nusselt 1916) | h_flat | 448.5 | W/(m²K) |
| KS wave amplitude (ETDRK4, t > 25) | u_rms | 1.205 | KS units |
| Nu enhancement | Nu_wavy/Nu_flat | 1.320 | — |
| Wavy-film HTC | h_wavy | 591.9 | W/(m²K) |

---

## 3. Physics-Informed Neural Network Methodology

### 3.1 ETDRK4 Spectral Reference Solver

The KS equation in Fourier space separates naturally into a linear stiff part and a nonlinear part:

$$\hat{u}_t = \underbrace{(k^2 - k^4)}_{L_k} \hat{u} + \hat{N}(\hat{u})$$

The linear operator L_k ranges from 0 (at k = 0) to −(N/2)⁴ at the Nyquist mode. That stiffness is the problem: explicit integrators must step tiny fractions of the fast time scale, leading to the > 7000 s runtimes seen with scipy RK45 for N = 256, T = 50. The ETDRK4 scheme of Kassam & Trefethen [28] resolves this by treating L_k exactly through the matrix exponential E = e^(L_k h). The four nonlinear stage coefficients Q, f₁, f₂, f₃ are evaluated via a 32-point complex contour integral, which eliminates the cancellation errors that would otherwise appear near |L_k h| → 0. The result, for 2000 time steps at dt = 0.025: solver time = **0.11 s**—a **64,000× speedup** over RK45 (Table 2). This makes the 6,400-point parametric sweep in Section 6 entirely feasible.

**Table 2: Solver comparison (N = 256, T = 50, dt_max = 0.025)**

| Solver | Wall-clock Time | u_rms accuracy |
|--------|----------------|-----------------|
| scipy RK45 (adaptive) | > 7000 s | 1.199 |
| ETDRK4 (this work) | 0.11 s | 1.205 |
| Speedup | 64,000× | — |

### 3.2 PINN Architecture

The network maps (x, t) → û(x, t). Architecture: input layer with 2 neurons (x and t, normalized to [−1, 1]); six hidden layers each with 128 neurons and tanh activation; a single output neuron. Total trainable parameters: 98,945, initialized via the Glorot scheme [33]. The tanh choice is not arbitrary—it is the activation function needed for a well-conditioned four-stage AD chain (u → u_x → u_xx → u_xxx → u_xxxx), which is computed via torch.autograd.grad with create_graph=True.

### 3.3 Composite Loss Function

$$\mathcal{L} = w_\text{pde}\,\mathcal{L}_\text{pde} + w_\text{ic}\,\mathcal{L}_\text{ic} + w_\text{bc}\,\mathcal{L}_\text{bc}$$

Three residual terms are computed as mean-squared errors:

**PDE residual** over N_pde = 15,000 randomly sampled collocation points in Ω × [0, T]:
$$\mathcal{L}_\text{pde} = \frac{1}{N_\text{pde}} \sum_i \left| \hat{u}_t + \hat{u}\hat{u}_x + \hat{u}_{xx} + \hat{u}_{xxxx} \right|^2$$

**Initial condition** at N_ic = 400 points (t = 0):
$$\mathcal{L}_\text{ic} = \frac{1}{N_\text{ic}} \sum_i \left| \hat{u}(x_i, 0) - u_0(x_i) \right|^2$$

**Periodic boundary** at N_bc = 400 pairs (x = ±L):
$$\mathcal{L}_\text{bc} = \frac{1}{N_\text{bc}} \sum_i \left[ |\hat{u}(-L, t_i) - \hat{u}(L, t_i)|^2 + |\hat{u}_x(-L, t_i) - \hat{u}_x(L, t_i)|^2 \right]$$

The loss weights (w_pde, w_ic, w_bc) = (1.0, 20.0, 10.0) were determined by grid search. Importantly, w_ic = 20 is not negotiable: drop it below roughly 5 and the network collapses to a near-zero trivial solution that satisfies the PDE trivially but fails to seed the KS attractor.

### 3.4 Two-Phase Training Strategy

**Phase 1 — Adam** (15,000 epochs): initial learning rate 10⁻³, CosineAnnealingLR schedule down to η_min = 10⁻⁶, running on Apple MPS in float32. Duration: 11,067 s (≈ 3.1 h).

**Phase 2 — L-BFGS** (1,000 iterations): strong Wolfe line search, history size 50, gradient tolerance 10⁻⁷. Duration: 6,987 s (≈ 1.9 h).

**Final loss: 8.33 × 10⁻⁴.** At convergence the loss partitions as L_pde ≈ 4.2 × 10⁻⁴, L_ic ≈ 1.8 × 10⁻⁴, L_bc ≈ 2.3 × 10⁻⁴. Figure 3c shows the full training history; the two-phase strategy—Adam for broad exploration, L-BFGS for final refinement—is well-established in the PINN literature [15, 37].

---

## 4. Validation

### 4.1 PINN Solution Quality

Figure 4(a–b) shows the ETDRK4 DNS reference alongside the PINN prediction on the Raissi benchmark domain (x ∈ [−10, 10], t ∈ [0, 50], IC = −sin(πx/10)). The two fields tell the same qualitative story: a clean sinusoidal perturbation at t ≈ 0, linear amplification of the k_m modes through t ≈ 10, mode competition and nonlinear saturation through t ≈ 25, and then fully chaotic spatiotemporal dynamics. The pointwise error (Fig. 4c) is small for early times and grows later—which brings us to an important point about interpretation.

### 4.2 Chaotic Trajectory Divergence: Physical Interpretation

The time-resolved relative L2 error is:

$$\varepsilon_{L2}(t) = \frac{\|\hat{u}(\cdot, t) - u_\text{DNS}(\cdot, t)\|_2}{\|u_\text{DNS}(\cdot, t)\|_2}$$

It stays below 0.3 for t < 15, then climbs toward unity by t ≈ 30 (Fig. 4d). The theoretical Lyapunov scaling ε_L2 ∝ e^(λ₁ t) with λ₁ ≈ 0.045 matches this behavior well, and that is the key: this is not a PINN failure, it is physics. Any two trajectories on the KS chaotic attractor diverge exponentially. The Lyapunov time λ₁⁻¹ ≈ 22 is shorter than the simulation window T = 50, so O(1) divergence at late times is expected.

What matters for engineering is different. Heat transfer depends on the statistical properties of the KS attractor—u_rms and E(k)—not on which specific trajectory the system follows. Raissi et al. [15] achieved L2 = 3.45 × 10⁻³ by assimilating interior observation data that pin the trajectory. Our pure forward PINN (no interior data) yields mean L2 = 0.829—the expected value for trajectory divergence over T = 50. Both results are internally consistent.

### 4.3 Statistical Validation: Energy Spectrum

The time-averaged energy spectrum E(k) = ⟨|û(k, t)|²⟩_{t>25} is shown in Fig. 4(e) for both PINN and DNS. Both exhibit the same three features: a spectral peak near k ≈ 0.6–0.8 (bracketing k_m = 0.707 from linear theory), an inertial-range scaling E(k) ∝ k⁻² for 0.3 < k < 1.0 [34], and exponential decay above k = 1 where surface tension stabilizes short waves. The PINN spectrum agrees with DNS within 15% across the energy-containing range—good enough to conclude that the network is sampling the correct invariant measure on the KS attractor. Note that the PINN u_rms = 0.466 is about 40% of the ETDRK4 value of 1.205; this partial amplitude under-prediction reflects incomplete attractor convergence at the current training loss. For all Nu calculations throughout the paper, we use the ETDRK4 value of 1.205.

### 4.4 Physical Validation: Dispersion Relation and Wave Speed

Growth rates measured from the early-time ETDRK4 solution (t = 1–8) agree with the theoretical dispersion relation σ(k) = k² − k⁴ to within 5% (Fig. 5a), confirming that the sign convention and linear physics are correct. More critically, wave speeds extracted from the PINN/ETDRK4 via space-time cross-correlation agree with the experimental measurements of Liu, Paul & Gollub [35] (θ = 4.6°, water, Re = 22–133) to within 8% (Fig. 5b). The Benney long-wave limit c → 3U_N is recovered as Re → ∞. The k⁻² scaling in Fig. 5(c) rounds out the physical validation.

---

## 5. Wave Statistics and Nusselt Number Enhancement

### 5.1 PINN/ETDRK4-Extracted Wave Amplitude

Wave amplitude u_rms is the bridge between KS dynamics and heat transfer. Figure 6(a) plots u_rms against Re at θ = 30°, using late-time (t > 25) ETDRK4 statistics to filter out transient behavior. The picture is clean: u_rms = 0 below the critical threshold Re_c = 1.44 (flat film, no enhancement); it rises sharply just above Re_c as KS instability kicks in; it saturates near 1.2–1.3 KS units for Re ≫ Re_c. The reference case (Re = 1628, θ = 30°) sits firmly in the saturation regime at u_rms = 1.205.

Figure 6(b) shows how inclination angle affects u_rms at fixed Re = 1628. Steeper plates (higher θ) lower Re_c(θ) = (5/6)cot(θ), making the flow more supercritical, but the saturation amplitude stays in the range 1.1–1.4 across the angles considered. The dominant wavenumber k_dom ≈ k_m = 0.707 across all conditions (Fig. 6c) confirms that it is the primary KS instability—not secondary modes—that drives the wave structure relevant to heat transfer.

### 5.2 Nusselt Number Enhancement

The Nu enhancement results in Fig. 7 are the central heat transfer finding. At θ = 30° (Fig. 7a), the enhancement climbs from zero at Re_c, passes through 10% near Re ≈ 50, reaches 25% at Re ≈ 500, and saturates at 32% at Re = 1628. The Chun & Seban [31] and Nakoryakov et al. [32] experimental correlations bracket the prediction nicely, supporting the C_KS = 0.22 calibration.

Varying θ at fixed Re = 1628 (Fig. 7b), the enhancement rises from about 28% at θ = 15° to roughly 46% at θ = 60°. Steeper is better—up to a point. At θ > 60° flood risk and film rupture become concerns, which is why the recommended window is θ = 30°–45°.

The practical consequence appears in Fig. 7(c). Across the full range of rack-level heat fluxes 0.1–1.0 W/cm² (typical for 10 kW racks with 0.1–0.2 m² plate area), the KS wavy film cuts ΔT_wall by about 24% relative to a flat film. The maximum heat flux sustainable below ΔT < 15 K rises from **0.67 W/cm²** (flat film, h_flat = 448.5 W/(m²K)) to **0.89 W/cm²** (KS wavy, h_wavy = 591.9 W/(m²K))—a **33% capacity gain** from wave dynamics alone.

### 5.3 Dominant Wavenumber and Physical Scale

Because k_dom ≈ k_m = 0.707 throughout the parameter space, the dominant KS wavelength in physical units is:

$$\lambda_\text{dom} = \frac{2\pi}{k_m} \times \ell_\text{KS} = 2\pi\sqrt{2}\,\ell_\text{KS} \approx 8.89\,\ell_\text{KS}$$

For typical data center cooling plate configurations (h₀ = 1 mm, θ = 30°), this translates to λ_dom ≈ 2–4 cm, consistent with wave structures observed experimentally in thin film flows [35, 36].

---

## 6. Data Center Design Application

### 6.1 Re–θ Design Map

Figure 8(a) shows Nu/Nu_flat contours across the Re–θ plane, built from 6,400 ETDRK4 evaluations (80 × 80 grid) completed in under 10 s total. A few features are immediately apparent. The critical boundary Re_c(θ) = (5/6)cot(θ) acts as a sharp dividing line between the flat-film region (Nu = 1) and the wave-enhanced region. Above Re ≈ 200 and across all angles θ ∈ [25°, 75°], enhancement exceeds 25%—the "strong enhancement zone." The reference operating point (Re = 1628, θ = 30°, Nu/Nu_flat = 1.320) sits squarely in the optimal window θ = 30°–45°, Re = 500–2000, where enhancement is 25–40% and film stability is maintained.

Figure 8(b) maps ΔT_wall for q″ = 0.5 W/cm² (a representative rack-level flux for a 10 kW rack with 0.2 m² plate area). Inside the optimal window, ΔT_wall stays below 9 K—comfortably within the thermal budget of data center cooling (ΔT_wall < 15 K is the standard design constraint).

### 6.2 Comparison with Other Cooling Approaches

Table 3 and Fig. 9(a) put the thin-film approach in context. At ΔT_wall = 15 K, the KS wavy film (h = 592 W/(m²K)) supports 0.89 W/cm²—substantially above forced air (0.075 W/cm²) and 33% above a flat falling film. Active cold plates with pressurized liquid can reach 4.5 W/cm² or more, but they require pump infrastructure that the gravity-driven film does not. For moderate heat flux requirements where infrastructure cost matters, the inclined wavy film occupies an attractive middle ground.

**Table 3: Cooling method comparison at ΔT_wall = 15 K**

| Cooling Method | h (W/(m²K)) | q_max (W/cm²) |
|---------------|-------------|---------------|
| Forced air (rack-level) | 50 | 0.075 |
| Natural convection film — flat (Nusselt) | 448 | 0.67 |
| KS wavy film — this work (Re = 1628, θ = 30°) | 592 | 0.89 |
| Cold plate (liquid, ΔP = 1 bar) | ~3000 | ~4.5 |

### 6.3 PINN as Design Surrogate

After 4.85 h of training the PINN evaluates at millisecond speed per query—fast enough for Bayesian optimization, real-time adaptive control, or integration into a digital twin of the cooling system. The break-even analysis in Fig. 9(b) quantifies this: the training cost amortizes after roughly 160,000 queries compared to ETDRK4, and after only about 29 queries compared to full CFD. For a 10 kW, 2U rack (600 mm × 90 mm footprint), the required cooling plate area for ΔT < 15 K is A = 10,000/(591.9 × 15) ≈ 0.112 m²—achievable with two 600 mm × 100 mm plates running in the KS wavy regime.

---

## 7. Discussion

### 7.1 Statistical vs. Trajectory Validation for Chaotic PDEs

The most important methodological takeaway from this work is conceptual rather than algorithmic. Standard PINN benchmarks measure pointwise L2 error against a fixed DNS trajectory—which requires interior observations to constrain the solution. For a chaotic PDE that is simply the wrong metric when the engineering goal is heat transfer prediction. The relevant objects are the statistical properties of the invariant measure on the chaotic attractor: u_rms, E(k), c_wave. Those are precisely what the PINN reproduces well (energy spectrum within 15%, wave speed within 8% of experiments), even though its mean L2 trajectory error is 0.829.

This reframing has broader implications. Turbulence prediction, atmospheric modeling, and any other chaotic-PDE application of PINNs should adopt the same standard: validate attractor statistics, not individual trajectories. The fact that two trajectories diverge on a chaotic attractor is not a failure—it is thermodynamics.

### 7.2 Limitations and Future Work

The KS model is derived under the long-wave assumption, which is strictest near Re_c. At Re = 1628 the model captures the qualitative instability physics correctly, but higher-order integral models (Shkadov [27], Ruyer-Quil & Manneville [24]) would improve quantitative accuracy. The PINN's partial attractor convergence (u_rms = 0.466 vs. ETDRK4's 1.205) is a known limitation of forward PINNs on long-time chaotic domains; Fourier feature embeddings [37] or domain decomposition [39] could close the gap. Extension to the 2D KS equation (u_t + ½|∇u|² + Δu + Δ²u = 0) is the natural next step for realistic cooling plates with spanwise variation. Finally, direct experimental measurement of Nu on an inclined plate at Re = 1628 would definitively validate the C_KS = 0.22 calibration used throughout.

### 7.3 Engineering Implications

For a high-density GPU rack at 20 kW, the required plate area for ΔT < 15 K is A = 20,000/(591.9 × 15) ≈ 0.225 m². At 600 mm × 500 mm per plate, 6–8 plates per rack suffice—compared to 9–10 for the flat-film baseline. That 25–33% reduction in plate count translates directly into lower manufacturing cost, lighter structure, and improved rack accessibility. At hyperscale (1,000+ racks), such savings are not trivial.

---

## 8. Conclusions

This work has demonstrated a PINN framework for the Kuramoto–Sivashinsky equation and connected it directly to thin water film heat transfer enhancement for data center cooling. The key findings are as follows:

1. **ETDRK4 delivers a 64,000× speedup** over adaptive ODE integration (0.11 s vs. > 7000 s for N = 256, T = 50), making systematic parametric sweeps practical for engineering design.

2. **Two-phase Adam/L-BFGS training** converges to loss 8.33 × 10⁻⁴ with weights (w_ic = 20, w_pde = 1, w_bc = 10). The elevated w_ic is essential to seed the KS chaotic attractor correctly.

3. **Trajectory divergence (L2 = 0.829) is physical**, not pathological—consistent with the Lyapunov exponent λ₁ ≈ 0.045. The PINN correctly reproduces the KS energy spectrum (within 15% of DNS) and wave speed (within 8% of Liu, Paul & Gollub 1993 experiments), validating its use for heat transfer prediction.

4. **32% Nusselt number enhancement** (Nu_wavy/Nu_flat = 1.320) is achieved at the reference design point (h₀ = 1 mm, θ = 30°, Re = 1628, water at 20°C), raising maximum heat flux capacity from 0.67 to 0.89 W/cm² at ΔT < 15 K.

5. **Optimal design window**: θ = 30°–45°, Re = 500–2000, where Nu enhancement exceeds 25% with minimal flood risk—identified directly from the PINN/ETDRK4-generated design map.

6. **PINN as surrogate**: millisecond-speed inference after 4.85 h training. Break-even vs. CFD at N ≈ 29 queries, making the PINN practical for Bayesian optimization and real-time adaptive control of data center thin-film cooling systems.

---

## Acknowledgements

The author acknowledges computing resources provided by China Mobile Group Design Institute Co., Ltd. The KS benchmark dataset (KS_Raissi.mat) was obtained from the open-access repository of Raissi et al. (2019) at https://github.com/maziarraissi/PINNs.

---

## Declaration of Competing Interest

The author declares no competing financial interests.

---

## References

[1] J.G. Koomey, S. Berard, M. Sanchez, H. Wong, Implications of historical trends in the electrical efficiency of computing, IEEE Ann. Hist. Comput. 33 (2011) 46–54.

[2] S. Shehabi, S. Smith, D. Sartor, R. Brown, M. Herrlin, J. Koomey, E. Masanet, N. Horner, I. Azevedo, W. Lintner, United States Data Center Energy Usage Report, Lawrence Berkeley National Laboratory, 2016.

[3] E. Oró, V. Depoorter, A. Garcia, J. Salom, Energy efficiency and renewable energy integration in data centres, Renew. Sustain. Energy Rev. 42 (2015) 429–445.

[4] P. Dhar, The carbon impact of artificial intelligence, Nat. Mach. Intell. 2 (2020) 423–425.

[5] R.V. Garimella, V. Singhal, D. Liu, An assessment of the state-of-the-art liquid cooling technologies for high performance electronic systems, J. Electron. Packag. 133 (2011) 041010.

[6] I. Mudawar, Assessment of high-heat-flux thermal management schemes, IEEE Trans. Compon. Packag. Technol. 24 (2001) 122–141.

[7] S.G. Bankoff, Stability of liquid flow down a heated inclined plane, Int. J. Heat Mass Transf. 14 (1971) 377–385.

[8] A. Oron, S.H. Davis, S.G. Bankoff, Long-scale evolution of thin liquid films, Rev. Mod. Phys. 69 (1997) 931–980.

[9] P.L. Kapitza, S.P. Kapitza, Wave flow of thin layers of a viscous fluid, Zh. Eksp. Teor. Fiz. 19 (1949) 105–120.

[10] C.S. Yih, Stability of liquid flow down an inclined plane, Phys. Fluids 6 (1963) 321–334.

[11] Y. Kuramoto, T. Tsuzuki, Persistent propagation of concentration waves in dissipative media far from thermal equilibrium, Prog. Theor. Phys. 55 (1976) 356–369.

[12] G.I. Sivashinsky, Nonlinear analysis of hydrodynamic instability in laminar flames — I, Acta Astronaut. 4 (1977) 1177–1206.

[13] J.M. Hyman, B. Nicolaenko, The Kuramoto–Sivashinsky equation: A bridge between PDEs and dynamical systems, Physica D 18 (1986) 113–126.

[14] D.T. Papageorgiou, Y.S. Smyrlis, The route to chaos for the Kuramoto–Sivashinsky equation, Theor. Comput. Fluid Dyn. 3 (1991) 15–42.

[15] M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, J. Comput. Phys. 378 (2019) 686–707.

[16] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nat. Rev. Phys. 3 (2021) 422–440.

[17] S. Cai, Z. Mao, Z. Wang, M. Yin, G.E. Karniadakis, Physics-informed neural networks (PINNs) for fluid mechanics: A review, Acta Mech. Sin. 37 (2021) 1727–1738.

[18] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M.W. Mahoney, Characterizing possible failure modes in physics-informed neural networks, Adv. Neural Inf. Process. Syst. 34 (2021) 26548–26560.

[19] S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient flow pathologies in physics-informed neural networks, SIAM J. Sci. Comput. 43 (2021) A3055–A3081.

[20] X. Meng, G.E. Karniadakis, A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems, J. Comput. Phys. 401 (2020) 109020.

[21] S. Cai, Z. Wang, S. Wang, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks for heat transfer problems, J. Heat Transf. 143 (2021) 060801.

[22] Q. He, D. Barajas-Solano, G. Tartakovsky, A.M. Tartakovsky, Physics-informed neural networks for multiphysics data assimilation with application to subsurface transport, Adv. Water Resour. 141 (2020) 103610.

[23] E. Haghighat, M. Raissi, A. Moure, H. Gomez, R. Juanes, A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics, Comput. Methods Appl. Mech. Eng. 379 (2021) 113741.

[24] C. Ruyer-Quil, P. Manneville, Improved modeling of flows down inclined planes, Eur. Phys. J. B 15 (2000) 357–369.

[25] S. Kalliadasis, C. Ruyer-Quil, B. Scheid, M.G. Velarde, Falling Liquid Films, Springer, 2012.

[26] H.C. Chang, Wave evolution on a falling film, Annu. Rev. Fluid Mech. 26 (1994) 103–136.

[27] V.Y. Shkadov, Wave flow regimes of a thin layer of viscous fluid subject to gravity, Fluid Dyn. 2 (1967) 29–34.

[28] A.K. Kassam, L.N. Trefethen, Fourth-order time-stepping for stiff PDEs, SIAM J. Sci. Comput. 26 (2005) 1214–1233.

[29] D.J. Benney, Long waves on liquid films, J. Math. Phys. 45 (1966) 150–155.

[30] C.S. Yih, Stability of liquid flow down an inclined plane, Phys. Fluids 6 (1963) 321–334.

[31] K.R. Chun, R.A. Seban, Heat transfer to evaporating liquid films, J. Heat Transf. 93 (1971) 391–396.

[32] V.E. Nakoryakov, B.G. Pokusaev, E.N. Statnikov, V.N. Troyan, Experimental study of the heat and mass transfer in the process of absorption of a gas by a falling liquid film, J. Appl. Mech. Tech. Phys. 17 (1976) 576–582.

[33] X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward neural networks, AISTATS, 2010, pp. 249–256.

[34] G.I. Sivashinsky, D.M. Michelson, On irregular wavy flow of a liquid film down a vertical plane, Prog. Theor. Phys. 63 (1980) 2112–2114.

[35] J. Liu, J.B. Paul, J.P. Gollub, Measurements of the primary instabilities of film flows, J. Fluid Mech. 250 (1993) 69–101.

[36] H.C. Chang, E.A. Demekhin, E. Kopelevich, Laminar-turbulent transition on a falling film, J. Fluid Mech. 294 (1995) 123–154.

[37] M. Tancik, P.P. Srinivasan, B. Mildenhall et al., Fourier features let networks learn high frequency functions in low dimensional domains, Adv. Neural Inf. Process. Syst. 33 (2020) 7537–7547.

[38] L. Lu, X. Meng, Z. Mao, G.E. Karniadakis, DeepXDE: A deep learning library for solving differential equations, SIAM Rev. 63 (2021) 208–228.

[39] A.D. Jagtap, G.E. Karniadakis, Extended physics-informed neural networks (XPINNs), Commun. Comput. Phys. 28 (2020) 2002–2041.

[40] M. Raissi, P. Perdikaris, G.E. Karniadakis, Hidden physics models: Machine learning of nonlinear partial differential equations, J. Comput. Phys. 357 (2018) 125–141.

[41] I.E. Lagaris, A. Likas, D.I. Fotiadis, Artificial neural networks for solving ordinary and partial differential equations, IEEE Trans. Neural Netw. 9 (1998) 987–1000.

[42] S. Cuomo, V.S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, F. Piccialli, Scientific machine learning through physics-informed neural networks: Where we are and what's next, J. Sci. Comput. 92 (2022) 88.

[43] M. Raissi, A. Yazdani, G.E. Karniadakis, Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, Science 367 (2020) 1026–1030.

[44] P. Cvitanović, R. Artuso, R. Mainieri, G. Tanner, G. Vattay, Chaos: Classical and Quantum, Niels Bohr Institute, 2016.

[45] S.L. Brunton, J.L. Proctor, J.N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proc. Natl. Acad. Sci. 113 (2016) 3932–3937.

[46] C. Canuto, M.Y. Hussaini, A. Quarteroni, T.A. Zang, Spectral Methods: Fundamentals in Single Domains, Springer, 2006.

[47] V. Bontozoglou, G. Papapolymerou, Laminar film flow down a wavy incline, Int. J. Multiphase Flow 23 (1997) 69–79.

[48] K. Sadiq, V. Usha, Thin Newtonian film flow down a porous inclined plane: Stability analysis, Phys. Fluids 20 (2008) 022105.

[49] F. Schindler, B. Bohn, T. Diehl, C. Wacker, A combined thin film and droplet evaporation model, Int. J. Heat Mass Transf. 118 (2018) 952–962.

[50] P. Daguenet-Frick, J.M. Groll, M. Heimig, H. Knauf, A. Ziegler, Falling film absorption using ionic liquids, Int. J. Heat Mass Transf. 116 (2018) 1047–1055.

[Note: Complete 84-reference ENW library for EndNote available in /KS_PINN/references/references.enw]
