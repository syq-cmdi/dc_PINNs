# KS-PINN: Physics-Informed Neural Networks for the Kuramoto-Sivashinsky Equation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syq-cmdi/dc_PINNs/blob/main/KS_PINN_Colab.ipynb)

Physics-informed neural network (PINN) for solving the Kuramoto-Sivashinsky (KS) equation — the governing equation for thin water film flow down an inclined plane. Benchmarked against Raissi et al. (2019) and validated against Benney (1966) analytical theory.

**Application**: Data center cooling via thin-film flow over inclined heat spreaders. The PINN predicts surface wave dynamics and computes the +32% Nusselt enhancement from KS-driven waves.

---

## Quick Start (Google Colab)

Click the badge above, or open `KS_PINN_Colab.ipynb` directly. All dependencies install automatically. Data downloads from Raissi's GitHub repository. No local setup needed.

**Expected runtimes:**

| Hardware | Adam (15k ep) | L-BFGS (1k iter) | Total |
|----------|---------------|-------------------|-------|
| A100     | ~8 min        | ~12 min           | ~20 min |
| T4       | ~25 min       | ~35 min           | ~60 min |
| CPU only | ~2 h          | ~3 h              | ~5 h |

---

## Local Setup

```bash
git clone https://github.com/syq-cmdi/dc_PINNs.git
cd dc_PINNs
pip install torch numpy scipy matplotlib tqdm
```

Download the benchmark data:
```bash
# KS.mat from Raissi et al. (2019)
wget https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/KS.mat \
     -O code/KS_raissi.mat
```

Run the benchmark:
```bash
python code/ks_pinn_benchmark.py
```

To skip training and use saved weights:
```bash
python code/ks_pinn_benchmark.py --vis-only
```

---

## Governing Equation

The dimensionless KS equation:

```
u_t + u·u_x + u_xx + u_xxxx = 0
```

- `u_t` — unsteady evolution
- `u·u_x` — nonlinear wave steepening
- `u_xx` — long-wave destabilisation (gravity along slope)
- `u_xxxx` — surface-tension stabilisation (short-wave damping)

Domain: `x ∈ [−10, 10]`, `t ∈ [0, 50]`, periodic BC, IC: `u(x,0) = −sin(πx/10)`.

---

## PINN Architecture

- 6-layer fully-connected MLP, 128 units/layer, tanh activation
- Two-phase training: Adam (15k epochs, cosine LR) → L-BFGS (1k iterations)
- Loss weights: `w_pde = 1`, `w_ic = 20`, `w_bc = 10`
- GPU: AMP (float16) for Adam, full float32 for L-BFGS

**Results vs benchmark:**

| Metric | This Work | Raissi et al. (2019) |
|--------|-----------|----------------------|
| Relative L2 error | ~3×10⁻³ | 3.45×10⁻³ |
| Nu enhancement | +32% | — |
| Training time (A100) | ~20 min | — |

---

## Project Structure

```
KS_PINN/
├── KS_PINN_Colab.ipynb        # Google Colab notebook (start here)
├── code/
│   ├── ks_pinn_benchmark.py   # Standalone benchmark script
│   ├── generate_paper_figures.py
│   └── build_docx_v2.py       # Paper Word document builder
├── paper/
│   ├── ks_pinn_paper_v2.md    # Paper manuscript (Markdown)
│   └── ks_pinn_ijhmt_v2.docx  # Formatted Word document
├── figures/                   # Generated paper figures
├── references/
│   └── references.enw         # EndNote reference library (84 refs)
└── dashboard/
    └── index.html             # Project progress dashboard
```

---

## Physical Validation

The PINN is validated against three independent sources:

1. **Raissi et al. (2019) JCP** — spectral DNS benchmark (KS.mat)
2. **Benney (1966) long-wave theory** — critical Re, wave speed
3. **Liu, Paul & Gollub (1993) JFM** — experimental wave speed data (θ = 4°, water)

Heat transfer chain:
`u_rms` (PINN) → `Nu_wavy/Nu_flat = 1 + 0.22·u²_rms` → +32% enhancement → `h_wavy = 591.9 W/(m²·K)`

---

## Reference

Yuqi Shi, "Physics-Informed Neural Networks for Kuramoto-Sivashinsky Dynamics in Thin Water Film Cooling of Data Centers", *International Journal of Heat and Mass Transfer* (submitted 2026).

```bibtex
@article{shi2026kspinn,
  author  = {Shi, Yuqi},
  title   = {Physics-Informed Neural Networks for Kuramoto-Sivashinsky Dynamics
             in Thin Water Film Cooling of Data Centers},
  journal = {International Journal of Heat and Mass Transfer},
  year    = {2026}
}
```

---

## License

MIT License. Benchmark data (KS.mat) courtesy of [Raissi et al. (2019)](https://github.com/maziarraissi/PINNs) under their original license.
