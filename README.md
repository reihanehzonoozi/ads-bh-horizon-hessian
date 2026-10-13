# Numerical companion: horizon curvature and the constrained Hessian of a boundary-pinned $\phi^4$ layer

This repository contains the numerical material accompanying the manuscript

**R. Majid Zonoozi and M. Nadjafikhah, _Boundary-pinned $\phi^4$ layers outside topological anti-de Sitter black holes: a no-go theorem and constrained spectral stability_.**

The code implements the static boundary-pinned scalar profiles, the radial fluctuation spectrum, curvature-shift diagnostics, horizon screening, large-box threshold tests, the near-zone consistency calculation, and the independent horizon-regular no-go/offset analysis used in the manuscript.

## Repository contents

The numerical calculations are implemented in two Python scripts, as stated in the manuscript:

- `bh_numerics.py` — main black-hole numerical calculation: a flat-space Bogomolny infrastructure check, profiles, spectra, grid convergence, AdS threshold and large-box checks, curvature screening, fixed-profile theorem test, offset sensitivity, near-zone consistency, and Figures 1–3.
- `horizon_nogo_numerics.py` — independent calculation of the horizon-regular no-go statement, positivity of $\Delta'$, and the stretched-horizon balance/offset scan.

Every figure in the manuscript is produced with Python (Matplotlib) by these scripts. The numerical table values can be reproduced with the stated parameters, up to solver and floating-point precision. No external research dataset is required and no random seeds are involved.

## Scientific scope

The background is the four-dimensional static topological anti-de Sitter black-hole metric

$$
ds^2=-f(r)dt^2+\frac{dr^2}{f(r)}+r^2d\Sigma_k^2,
$$

with

$$
f(r)=k-\frac{2M}{r}+\frac{r^2}{\ell^2},
\qquad k\in\{+1,0,-1\},
$$

corresponding to spherical, planar, and hyperbolic horizon geometries. The scalar potential is

$$
V(\phi)=\frac{\lambda}{4}(\phi^2-v^2)^2.
$$

The constrained static layer is defined on a stretched exterior interval beginning at

$$
a=r_h(1+\epsilon),
$$

with the inner Dirichlet datum used in the manuscript. The horizon-homogeneous fluctuation operator is

$$
\mathcal H
=-\frac{1}{r^2}\frac{d}{dr}\left(r^2f\frac{d}{dr}\right)+V''(\phi_0).
$$

At fixed horizon radius and fixed profile, the manuscript writes the curvature dependence as an affine operator family

$$
\mathcal H^{(k)}=\mathcal H^{(0)}+k\mathcal D,
$$

with nonnegative quadratic form for $\mathcal D$. The code tests the resulting ordering and Hellmann–Feynman/concavity relations numerically, while separately reporting the self-consistent profiles for each topology.

## Numerical method

The static profile is computed with `scipy.integrate.solve_bvp`. The Hessian is discretized in flux-conservative form on a uniform radial grid with $r^2f$ evaluated at cell midpoints. Dirichlet boundary nodes are eliminated exactly. Removal of the diagonal $r^2$ mass matrix gives a real symmetric tridiagonal eigenproblem, solved with `scipy.linalg.eigh_tridiagonal`.

Unless otherwise stated, the benchmark parameters are

```text
N = 4000
exterior span L = 6
ell = 4
epsilon = 3e-3
lambda = v = 1
```

Eigenvalues are reported in units of $\lambda v^2$. In `bh_numerics.py`, the numerical Morse index counts eigenvalues below `-1e-6`; the tested constrained eigenvalues are strictly positive.

For the hyperbolic horizon, the numerical runs reported here use the positive-mass branch $r_h>\ell$.

## Requirements

- Python 3
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4 for figure generation

A minimal environment can be created with

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Equivalently, without using `requirements.txt`:

```bash
python -m pip install "numpy>=1.20" "scipy>=1.7" "matplotlib>=3.4"
```

These are the same minimum versions quoted in the manuscript. The authors' reference runs used NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.8; the original logs in `logs/` record their exact environments. The independent full reproduction on 12 October 2026 used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8.

## Reproducing the main black-hole calculation

Run

```bash
python bh_numerics.py
```

The run opens with a flat-space validation of the discretization against the exact Bogomolny energy $2\sqrt2/3$,

```text
Bogomolny bound: numerical 0.942809, exact 0.942809, relative error 2.7e-09
```

The benchmark spectrum should include

| comparison | horizon | $r_h$ | $\lambda_{\min}$ | $\lambda_2$ | Morse index |
|---|---|---:|---:|---:|---:|
| $M=0.5$ fixed | $S^2$ | 0.947 | 2.405 | 3.236 | 0 |
| $M=0.5$ fixed | $\mathbb R^2$ | 2.520 | 2.178 | 2.794 | 0 |
| $M=0.5$ fixed | $H^2$ | 4.429 | 2.169 | 3.036 | 0 |
| $r_h=6$ fixed | $S^2$ | 6.000 | 2.336 | 3.985 | 0 |
| $r_h=6$ fixed | $\mathbb R^2$ | 6.000 | 2.289 | 3.769 | 0 |
| $r_h=6$ fixed | $H^2$ | 6.000 | 2.243 | 3.553 | 0 |

At fixed $r_h=6$, the curvature ordering is therefore

$$
\lambda_{\min}^{(-1)}<\lambda_{\min}^{(0)}<\lambda_{\min}^{(+1)},
$$

with a self-consistent $k=+1$ to $k=-1$ gap of approximately `0.0925` for the benchmark offset.

### Horizon screening

The code reproduces the reported decrease of the curvature gap with increasing horizon radius:

| $r_h$ | 5 | 6 | 8 | 12 | 20 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| $\lambda_{\min}^{(+1)}-\lambda_{\min}^{(-1)}$ | 0.1111 | 0.0925 | 0.0714 | 0.0521 | 0.0371 | 0.0274 |
| gap $\times r_h$ | 0.556 | 0.555 | 0.571 | 0.625 | 0.741 | 0.876 |

The drift of `gap × r_h` is intentional: the manuscript does not claim a universal $1/r_h$ law or any exact asymptotic power over the computed interval.

### AdS threshold and large-box check

For $\lambda=v=1$ and $\ell=4$, the vacuum continuum threshold is

$$
2\lambda v^2+\frac{9}{4\ell^2}=2.140625.
$$

The large-box calculation resolves a shallow positive layer-bound state below this threshold. At $r_h=6$, exterior span `300`, and `N=40000`, the reference run gives

```text
k=+1: lambda_1 = 2.139179
k= 0: lambda_1 = 2.131150
k=-1: lambda_1 = 2.120864
vacuum control: lambda_1 = 2.172324
```

The vacuum control uses the same grid and inner offset as the wall calculation and distinguishes the wall-induced level from a finite-grid artifact.

### Fixed-profile theorem check

At $r_h=6$, using the same $k=0$ profile in all three operators, the reference values are

```text
lambda_min(k=+1,0,-1) = 2.3707, 2.2893, 2.2000
common-profile gap     = 0.1707
self-consistent gap    = 0.0925
Rayleigh quotient <D>  = 0.0848
concavity bracket      = 0.0815 <= 0.0848 <= 0.0893
```

The distinction between the fixed-profile theorem and the self-consistent numerical comparison is essential: the sign/order survives profile response, whereas the size of the gap does not.

### Offset sensitivity

At $r_h=6$ the reference gap values are

| $\epsilon$ | $3\times10^{-3}$ | $10^{-2}$ | $3\times10^{-2}$ | $10^{-1}$ |
|---:|---:|---:|---:|---:|
| gap | 0.0925 | 0.0996 | 0.1125 | 0.1542 |

Accordingly, the numerical magnitude of the gap is a defining-scale-dependent quantity; it should not be quoted as convention-free.

### Near-zone consistency

The main script also evaluates the near-zone quantities entering the auxiliary near-zone $\mathfrak{sl}(2,\mathbb R)$ construction and verifies

$$
\beta=\frac{1}{2\pi T_H}=\frac{2r_h^2}{\Delta'(r_h)}
$$

throughout the tabulated test range. This check concerns the auxiliary near-zone surrogate; it does not establish an exact symmetry or a Love-number selection rule for the full anti-de Sitter boundary-value problem.

## Regenerating Figures 1–3

Run

```bash
python bh_numerics.py --figures
```

which writes

```text
figs/bh/fig1_profiles.pdf
figs/bh/fig2_spectrum.pdf
figs/bh/fig3_screening.pdf
```

## Reproducing the horizon-regular no-go and stretched-horizon analysis

Run

```bash
python horizon_nogo_numerics.py
```

This script is logically independent of the constrained spectral calculation and performs three checks.

### 1. Horizon-regular shooting

With $\Delta=r^2f$, regularity at the horizon gives

$$
\phi'(r_h)=\frac{V'(\phi_h)}{f'(r_h)}.
$$

The code shoots 123 initial values of $\phi_h$ across $(-v,v)$ for three horizon radii and each $k$. The reference maxima at $r=600$ are

```text
k=+1: max |phi(600)| = 3.57e-3
k= 0: max |phi(600)| = 3.67e-3
k=-1: max |phi(600)| = 3.74e-3
```

No trajectory approaches either vacuum. This numerical result is a check of, not a substitute for, the analytic virial no-go statement.

### 2. Positivity of $\Delta'$ on the tested exterior

The code evaluates the hypothesis $\Delta'(r)>0$ for the benchmark family of admissible horizons. This is the positivity input used in the no-go identity.

In the manuscript the hypothesis is established analytically for every non-extremal horizon: for $k=0,+1$ it is immediate, and for $k=-1$ non-extremality alone forces $r_h>\ell/\sqrt3$, whence $\Delta''(r_h)\ge2>0$ and $\Delta''$ increases with $r$. The no-go therefore covers the regular negative-mass hyperbolic branch as well, even though the numerical runs here stay on $r_h>\ell$.

### 3. Stretched-horizon balance

For $a=r_h(1+\epsilon)$ and the pinned condition $\phi(a)=0$, the finite-interval balance tested numerically is

$$
\frac12\Delta(a)\phi'(a)^2
=
\frac12\Delta(b)\phi'(b)^2
+a^2V(0)
+\int_a^b\left[\frac12\Delta'(\phi')^2+2rV\right]dr.
$$

At the benchmark $k=0$, $r_h=6$, the relative balance error is approximately `1.2e-7` at $\epsilon=3e-3`, and remains small over the tabulated scan. The rigorous consequence is the lower bound

$$
|\phi'(a)|=\Omega(\epsilon^{-1/2}),
$$

not an assertion of an exact $\epsilon^{-1/2}$ asymptotic law. The computed slope grows faster than that lower bound over the reported interval, while the energy decreases monotonically as the offset is reduced.

What the identity establishes is that the boundary slope diverges as $\epsilon\to0$, so no nontrivial horizon-regular configuration satisfying the pinned datum survives the removal of the offset: the constrained family depends essentially on the stretched-horizon prescription. No claim is made that the energy lacks a limiting value, or that no weaker vacuum limit exists; the decreasing energies are equally consistent with a gradual collapse onto the vacuum.

## Reproducibility status

The supplied reference-run logs record the benchmark results reported above. On 12 October 2026, both scripts were independently rerun in full: `python bh_numerics.py --figures` and `python horizon_nogo_numerics.py`. Both exited successfully, the reported numerical results were reproduced within the stated precision, and all three regenerated figures were pixel-identical to the manuscript figures. This documentation update changes no computational code.

The following interpretation should be preserved:

- the tested boundary-pinned configurations have vanishing constrained Morse index;
- at fixed profile and fixed horizon radius the curvature ordering follows the positive quadratic form of the shift operator;
- self-consistent profiles preserve the observed ordering but partially screen the bare fixed-profile shift;
- the size of the curvature gap depends on the defining stretched-horizon offset;
- the horizon-regular no-go result means the constrained layer should not be reinterpreted as a freely existing regular scalar wall outside the horizon.

## Citation

If this code contributes to published work, please cite the associated manuscript:

> Reihaneh Majid Zonoozi and Mehdi Nadjafikhah, _Boundary-pinned $\phi^4$ layers outside topological anti-de Sitter black holes: a no-go theorem and constrained spectral stability_.

## Reference-log provenance

The files in `logs/` are the original reference runs supplied by the authors and are preserved without editing. The latest documentation update changes no research script, algorithm or numerical parameter. Historical labels in the reference logs may therefore differ from the current wording. Logs of the independent full reproduction on 12 October 2026 are included separately under `Final_Check/` in the submission package.
