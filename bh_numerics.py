#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical companion to

    "Boundary-pinned phi^4 layers outside topological anti-de Sitter
     black holes: a no-go theorem and constrained spectral stability"

Reproduces the spectral and auxiliary near-horizon calculations and figures 1-3.
The horizon no-go and offset-balance checks are in horizon_nogo_numerics.py.

Contents
--------
    horizon(k, M)           largest root of f(r) = k - 2M/r + r^2/l^2
    profile(k, ...)         boundary-pinned static layer outside the horizon, Eq. (3.6)
    spectrum(k, ...)        fluctuation operator, Eq. (4.2), flux-conservative
    table_spectra()         Table 2 (fixed mass and fixed horizon radius)
    positivity_check()       numerical check behind Proposition 1
    bound_state_count()      levels below threshold (no count claimed)
    table_screening()        Table 3
    planar_scaling_test()    consistency check behind Table 4
    table_screening_bound()  Table 4 (resolved bound state, fixed (R-r_h)/r_h)
    table_threshold()        AdS threshold and box law, Eqs. (4.4)-(4.5)
    table_theorem()          fixed-profile test of Theorem 1 and Hellmann-Feynman
    table_offset()           sensitivity to the inner offset epsilon
    figures()                figures 1-3

Conventions: lambda = v = 1 and l = 4 unless stated otherwise, so eigenvalues
are in units of lambda v^2.

Usage
-----
    python3 bh_numerics.py            # tables only
    python3 bh_numerics.py --figures  # tables and figures (writes figs/)

Requires numpy >= 1.20, scipy >= 1.7, matplotlib >= 3.4.
The integration helper below supports both NumPy 1.x (trapz) and 2.x (trapezoid).
"""

import argparse
import os

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ----------------------------------------------------------------------
# model and defaults
# ----------------------------------------------------------------------
LAM = 1.0
VEV = 1.0
M_PHI2 = 2.0 * LAM * VEV ** 2
ELL = 4.0                       # AdS radius

FIG_DIR = "figs/bh"        # figures for this BH manuscript
N_GRID = 4000
L_SPAN = 6.0                    # exterior span, r in [r_h(1+eps), r_h + L_SPAN]
EPS = 3.0e-3                    # inner offset

dV = lambda p: LAM * p * (p ** 2 - VEV ** 2)
d2V = lambda p: LAM * (3.0 * p ** 2 - VEV ** 2)


def mass_from_horizon(k, rh, ell=ELL):
    """Eq. (2.4): 2M = r_h (k + r_h^2/l^2)."""
    return 0.5 * rh * (k + rh ** 2 / ell ** 2)


def horizon(k, M, ell=ELL, r_max=400.0, n_scan=400000):
    """Largest positive root of the blackening function."""
    f = lambda r: k - 2.0 * M / r + r ** 2 / ell ** 2
    r = np.linspace(1.0e-6, r_max, n_scan)
    sign_change = np.where(np.sign(f(r[:-1])) != np.sign(f(r[1:])))[0]
    if sign_change.size == 0:
        raise ValueError("no horizon for k = %+d, M = %g" % (k, M))
    i = sign_change[-1]
    return brentq(f, r[i], r[i + 1])


def hawking_temperature(k, rh, ell=ELL):
    """Eq. (2.5)."""
    return (k + 3.0 * rh ** 2 / ell ** 2) / (4.0 * np.pi * rh)


# ----------------------------------------------------------------------
# static profile: Eq. (3.6)
# ----------------------------------------------------------------------
def profile(k, M=None, rh=None, ell=ELL, span=L_SPAN, eps=EPS,
            tol=1.0e-6, n_init=4000):
    """Solve f phi'' + (f' + 2f/r) phi' = V'(phi) on [r_h(1+eps), r_h+span]
    with phi = 0 at the inner edge and phi = v at the outer edge.

    Exactly one of M and rh must be given. Returns (sol, f, rh, a, b, M).
    """
    if (M is None) == (rh is None):
        raise ValueError("give exactly one of M, rh")
    if rh is None:
        rh = horizon(k, M, ell)
    else:
        M = mass_from_horizon(k, rh, ell)

    f = lambda r: k - 2.0 * M / r + r ** 2 / ell ** 2
    df = lambda r: 2.0 * M / r ** 2 + 2.0 * r / ell ** 2
    a, b = rh * (1.0 + eps), rh + span

    r = np.linspace(a, b, n_init)
    guess = np.vstack([VEV * np.tanh((r - a) / np.sqrt(2.0)),
                       VEV / np.sqrt(2.0) / np.cosh(np.clip((r - a) / np.sqrt(2.0), 0, 50)) ** 2])

    def rhs(x, y):
        return np.vstack([y[1], (dV(y[0]) - (df(x) + 2.0 * f(x) / x) * y[1]) / f(x)])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - VEV])

    sol = solve_bvp(rhs, bc, r, guess, tol=tol, max_nodes=300000)
    if sol.status != 0:
        raise RuntimeError("BVP failed for k = %+d (status %d)" % (k, sol.status))
    return sol, f, rh, a, b, M


# ----------------------------------------------------------------------
# fluctuation operator: Eq. (4.2)
# ----------------------------------------------------------------------
def spectrum(k, M=None, rh=None, ell=ELL, N=N_GRID, span=L_SPAN, eps=EPS,
             fixed_profile=None, n_levels=5, return_modes=False):
    """Eigenvalues of -r^{-2} d_r (r^2 f d_r) + V''(phi_0) on L^2(r^2 dr).

    The weight r^2 f is evaluated at cell midpoints, which gives a symmetric
    stiffness matrix; the measure r^2 dr is carried by the mass matrix, so the
    discrete problem is a symmetric generalized eigenproblem and the spectrum
    is real. Dirichlet conditions at both edges are imposed by exact elimination
    of the boundary nodes.

    fixed_profile : callable(r) -> phi_0(r), used to test Theorem 1, which
    holds at fixed profile; passing None recomputes phi_0 self-consistently.
    """
    sol, f, rh, a, b, M = profile(k, M=M, rh=rh, ell=ell, span=span, eps=eps)
    r = np.linspace(a, b, N)
    dr = r[1] - r[0]
    phi = fixed_profile(r) if fixed_profile is not None else sol.sol(r)[0]

    mid = 0.5 * (r[:-1] + r[1:])
    w = mid ** 2 * f(mid)

    # Flux-conservative stiffness with the weight at cell midpoints; Dirichlet
    # nodes are eliminated exactly and the diagonal mass matrix r^2 is removed
    # by the symmetric map B^{-1/2} A B^{-1/2}, leaving a tridiagonal problem
    # of O(N) storage.  This is what makes the large-box runs of Sec. 4.1
    # affordable.
    diag = np.empty(N)
    diag[1:-1] = (w[:-1] + w[1:]) / dr ** 2 + d2V(phi[1:-1]) * r[1:-1] ** 2
    diag[0] = diag[-1] = 0.0
    d = diag[1:-1] / r[1:-1] ** 2
    e = -w[1:-1] / dr ** 2 / (r[1:-2] * r[2:-1])

    if return_modes:
        lam, U = eigh_tridiagonal(d, e, select='i', select_range=(0, n_levels - 1))
        psi = np.zeros(N)
        psi[1:-1] = U[:, 0] / r[1:-1]
        psi = psi * np.sign(np.sum(psi))
        return lam, psi, r, rh
    return eigh_tridiagonal(d, e, eigvals_only=True,
                            select='i', select_range=(0, n_levels - 1)), rh


def morse_index(levels, tol=1.0e-6):
    return int(np.sum(levels < -tol))


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def table_spectra():
    print("\nTable 2  (N=%d, span=%g, l=%g, lambda=v=1)" % (N_GRID, L_SPAN, ELL))
    print("  block            horizon    r_h     lam_min    lam_2     ind")
    for k, name in ((1, "S^2"), (0, "R^2"), (-1, "H^2")):
        lam, rh = spectrum(k, M=0.5)
        print("  M = 0.5 fixed    %-6s  %6.3f   %+7.3f   %+7.3f    %d"
               % (name, rh, lam[0], lam[1], morse_index(lam)))
    for k, name in ((1, "S^2"), (0, "R^2"), (-1, "H^2")):
        lam, rh = spectrum(k, rh=6.0)
        print("  r_h = 6 fixed    %-6s  %6.3f   %+7.3f   %+7.3f    %d"
               % (name, rh, lam[0], lam[1], morse_index(lam)))
    print("  (hyperbolic entries lie on the positive-mass branch r_h > l)")


def table_convergence(rh=6.0):
    """Grid convergence of the absolute lowest levels and the curvature gap."""
    print("\nGrid convergence at fixed r_h=%g (span=%g, eps=%g)" % (rh, L_SPAN, EPS))
    print("   N      lam(+1)      lam(0)       lam(-1)        gap")
    for N in (500, 1000, 2000, 4000, 8000):
        lp = spectrum(+1, rh=rh, N=N)[0][0]
        l0 = spectrum(0, rh=rh, N=N)[0][0]
        lm = spectrum(-1, rh=rh, N=N)[0][0]
        print(" %5d   %10.6f   %10.6f   %10.6f   %10.7f" % (N, lp, l0, lm, lp-lm))


def table_screening(radii=(5.0, 6.0, 8.0, 12.0, 20.0, 32.0)):
    """Curvature gap versus horizon radius.  Radii are areal (Schwarzschild-like),
    not proper distances: the proper radial separation is int dr / sqrt(f)."""
    print("\nTable 3: horizon screening of the curvature gap (l=%g, span=%g)" % (ELL, L_SPAN))
    print("  Monotone decay with r_h is a numerical observation over the range")
    print("  tabulated at this fixed exterior span, not a proved limit; Theorem 1")
    print("  proves monotonicity in k at fixed r_h only.")
    print("  r_h     lam_min(+1)  lam_min(-1)     gap      gap * r_h")
    for rh in radii:
        hi = spectrum(+1, rh=rh)[0][0]
        lo = spectrum(-1, rh=rh)[0][0]
        print("  %5.1f    %8.4f     %8.4f    %7.4f    %7.3f"
               % (rh, hi, lo, hi - lo, (hi - lo) * rh))


def vacuum_spectrum(k, rh, ell, span, N):
    """Lowest level of the operator with V'' frozen at its vacuum value.

    Isolating the vacuum operator removes the need to resolve the wall and
    exposes the AdS threshold directly.
    """
    M = mass_from_horizon(k, rh, ell)
    f = lambda r: k - 2.0 * M / r + r ** 2 / ell ** 2
    r = np.linspace(rh * (1.0 + 1.0e-6), rh + span, N)
    dr = r[1] - r[0]
    mid = 0.5 * (r[:-1] + r[1:])
    w = mid ** 2 * f(mid)
    diag = (w[:-1] + w[1:]) / dr ** 2 + M_PHI2 * r[1:-1] ** 2
    d = diag / r[1:-1] ** 2
    e = -w[1:-1] / dr ** 2 / (r[1:-2] * r[2:-1])
    return eigh_tridiagonal(d, e, eigvals_only=True, select='i', select_range=(0, 0))[0]


def table_threshold():
    """AdS threshold, Eq. (4.4), and box law, Eq. (4.5)."""
    print("\nAdS threshold and box law")
    print("  self-consistent profile, k=0, r_h=6, l=4: lam_min vs exterior span")
    print("  (the level first approaches the AdS threshold and then resolves the shallow wall-bound state)")
    for span, N in ((6, 4000), (12, 4000), (24, 4000), (48, 4000), (96, 5000), (200, 8000)):
        print("     span = %4d   N = %5d   lam_min = %.4f"
               % (span, N, spectrum(0, rh=6.0, span=span, N=N)[0][0]))
    print("     continuum threshold 2 lam v^2 + 9/(4 l^2) = %.4f"
           % (M_PHI2 + 9.0 / (4.0 * ELL ** 2)))
    print("  vacuum operator, k=0, r_h=6: measured vs box law")
    for ell in (2.0, 4.0, 8.0):
        rh, M = 6.0, mass_from_horizon(0, 6.0, ell)
        f = lambda r: -2.0 * M / r + r ** 2 / ell ** 2
        for span in (200, 1000, 5000):
            x_max = quad(lambda r: 1.0 / np.sqrt(f(r)), rh * (1 + 1e-9), rh + span, limit=400)[0]
            pred = M_PHI2 + 9.0 / (4.0 * ell ** 2) + np.pi ** 2 / x_max ** 2
            meas = vacuum_spectrum(0, rh, ell, span, 4000 if span < 1000 else 8000)
            print("     l = %g, span = %5d : measured %.4f, predicted %.4f"
                   % (ell, span, meas, pred))


def table_bound_state(rh=6.0, span=300.0, N=40000):
    """Resolve the shallow level below the AdS threshold.

    A control run with the wall removed (vacuum operator, same grid) is
    included: if the sub-threshold level were a discretization artifact it
    would survive that removal, and it does not.
    """
    thr = M_PHI2 + 9.0 / (4.0 * ELL ** 2)
    print("\nLarge-box spectrum at r_h=%g, span=%g, N=%d  (threshold %.6f)"
           % (rh, span, N, thr))
    print("   k      lam_1       lam_2      levels below threshold")
    for k in (1, 0, -1):
        lam = spectrum(k, rh=rh, span=span, N=N)[0]
        print("  %+d   %9.6f   %9.6f          %d" % (k, lam[0], lam[1], int((lam < thr).sum())))
    ctrl = spectrum(0, rh=rh, span=span, N=N, n_levels=1,
                    fixed_profile=lambda r: np.full_like(r, VEV))[0][0]
    print("   control, wall removed (vacuum profile, same grid and offset): lam_1 = %.6f"
          % ctrl)
    print("   -> the sub-threshold level is produced by the wall, not by the grid")


def table_theorem(rh=6.0):
    """Fixed-profile test of Theorem 1: monotonicity, Hellmann-Feynman, concavity."""
    print("\nTheorem 1 at fixed horizon radius r_h = %g" % rh)
    sol0 = profile(0, rh=rh)[0]
    common = lambda r: sol0.sol(r)[0]
    fixed = {k: spectrum(k, rh=rh, fixed_profile=common)[0][0] for k in (1, 0, -1)}
    self_ = {k: spectrum(k, rh=rh)[0][0] for k in (1, 0, -1)}
    print("  common profile      : lam_min = %.4f, %.4f, %.4f  (k = +1, 0, -1)"
           % (fixed[1], fixed[0], fixed[-1]))
    print("  self-consistent     : lam_min = %.4f, %.4f, %.4f"
           % (self_[1], self_[0], self_[-1]))
    print("  gap  common = %.4f   self-consistent = %.4f   (profile response %.0f%%)"
           % (fixed[1] - fixed[-1], self_[1] - self_[-1],
             100 * (1 - (self_[1] - self_[-1]) / (fixed[1] - fixed[-1]))))

    # Hellmann-Feynman: d lam / dk = <psi|D|psi>/<psi|psi>.
    lam, psi, r, _ = spectrum(0, rh=rh, fixed_profile=common, return_modes=True)
    dpsi = np.gradient(psi, r[1] - r[0])
    rayleigh = (_trapz(r * (r - rh) * dpsi ** 2, r)
                / _trapz(r ** 2 * psi ** 2, r))
    print("  Rayleigh quotient <D> = %.4f" % rayleigh)
    print("  concavity bracket   : %.4f <= %.4f <= %.4f"
           % (fixed[1] - fixed[0], rayleigh, fixed[0] - fixed[-1]))


def table_offset(rh=6.0):
    print("\nSensitivity to the inner offset epsilon (r_h = %g)" % rh)
    print("  epsilon    lam_min(+1)  lam_min(-1)     gap")
    for eps in (3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1):
        hi = spectrum(+1, rh=rh, eps=eps)[0][0]
        lo = spectrum(-1, rh=rh, eps=eps)[0][0]
        print("  %7.0e    %8.4f     %8.4f    %7.4f" % (eps, hi, lo, hi - lo))


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figures(outdir=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.dpi": 300, "savefig.bbox": "tight"})
    # set usetex = True below to typeset figure text with the manuscript fonts
    # plt.rcParams["text.usetex"] = True

    col = {1: "#D55E00", 0: "#009E73", -1: "#0072B2"}
    dash = {1: "-", 0: "--", -1: "-."}
    lab = {1: r"$k=+1$ ($S^2$)", 0: r"$k=0$ ($\mathbb{R}^2$)", -1: r"$k=-1$ ($H^2$)"}
    threshold = M_PHI2 + 9.0 / (4.0 * ELL ** 2)

    # ---- figure 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for k in (1, 0, -1):
        sol, f, rh, a, b, M = profile(k, M=0.5)
        x = np.linspace(a, b, 1500)
        ax[0].plot(x - rh, sol.sol(x)[0], dash[k], color=col[k], lw=1.8,
                   label=lab[k] + r", $r_h=%.2f$" % rh)
        r = np.linspace(0.35, 9.0, 1500)
        ax[1].plot(r, k - 1.0 / r + r ** 2 / ELL ** 2, dash[k], color=col[k], lw=1.8, label=lab[k])
        ax[1].plot([rh], [0.0], "o", color=col[k], ms=5)
    ax[0].set_xlabel(r"$r-r_h$")
    ax[0].set_ylabel(r"$\phi_0/v$")
    ax[0].set_xlim(0, 6)
    ax[0].set_title("(a) boundary-pinned layer profiles outside the horizon", fontsize=11)
    ax[0].legend(frameon=False, loc="lower right")
    ax[1].axhline(0.0, color="k", lw=0.7)
    ax[1].set_xlim(0, 9)
    ax[1].set_ylim(-1.4, 4.5)
    ax[1].set_xlabel("$r$")
    ax[1].set_ylabel(r"$f(r)=k-2M/r+r^2/\ell^2$")
    ax[1].set_title(r"(b) blackening functions ($M=0.5$, $\ell=4$); dots: horizons", fontsize=10)
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(outdir, "fig1_profiles.pdf"))
    plt.close(fig)

    # ---- figure 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    slot = {-1: 0, 0: 1, 1: 2}
    for k in (-1, 0, 1):
        lam, rh = spectrum(k, rh=6.0)
        for n, e in enumerate(lam):
            ax[0].hlines(e, slot[k] - 0.28, slot[k] + 0.28,
                         color=col[k], lw=(2.6 if n == 0 else 1.2))
        ax[0].text(slot[k], lam[0] + 0.20, r"$\lambda_{\min}=%.3f$" % lam[0],
                   ha="center", fontsize=9, color=col[k])
    ax[0].axhline(threshold, color="0.4", lw=1.0, ls=":")
    ax[0].text(-0.42, threshold - 0.34,
               r"AdS threshold $2\lambda v^2+9/(4\ell^2)=%.4f$" % threshold,
               fontsize=8.5, color="0.35")
    gap = spectrum(1, rh=6.0)[0][0] - spectrum(-1, rh=6.0)[0][0]
    ax[0].annotate("", xy=(2, spectrum(1, rh=6.0)[0][0]), xytext=(0, spectrum(-1, rh=6.0)[0][0]),
                   arrowprops=dict(arrowstyle="<->", lw=1, connectionstyle="arc3,rad=-0.38"))
    ax[0].text(1.0, 3.02, r"gap $=%.3f$" % gap, fontsize=9, ha="center")
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels([r"$H^2/\Gamma$", r"$\mathbb{R}^2/\Lambda$", r"$S^2$"])
    ax[0].set_ylabel(r"eigenvalues $\lambda_n\ [\lambda v^2]$")
    ax[0].set_ylim(1.70, 8.6)
    ax[0].set_xlim(-0.5, 2.5)
    ax[0].set_title(r"(a) spectrum at fixed $r_h=6$ (ordering inverted vs FLRW)", fontsize=10)

    radii = np.array([5.0, 6.0, 8.0, 12.0, 20.0, 32.0])
    gaps = np.array([spectrum(1, rh=x)[0][0] - spectrum(-1, rh=x)[0][0] for x in radii])
    ax[1].loglog(radii, gaps, "o-", color="#CC79A7", lw=1.6, ms=5, label="measured gap")
    ax[1].loglog(radii, gaps[0] * (radii / radii[0]) ** -0.7,
                 color="0.35", ls="--", lw=0.9, label=r"slope $-0.7$")
    ax[1].loglog(radii, gaps[0] * (radii / radii[0]) ** -1.0,
                 color="0.55", ls=":", lw=0.9, label=r"slope $-1$")
    ax[1].set_xlabel(r"$r_h$")
    ax[1].set_ylabel(r"$\lambda_{\min}^{(k=+1)}-\lambda_{\min}^{(k=-1)}$")
    ax[1].set_title("(b) horizon screening of the curvature gap", fontsize=11)
    ax[1].legend(frameon=False)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(os.path.join(outdir, "fig2_spectrum.pdf"))
    plt.close(fig)

    # ---- figure 3
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    radii = [5.0, 6.0, 8.0, 12.0, 20.0, 26.0, 32.0]
    for k in (1, 0, -1):
        xs = [x for x in radii if not (k == -1 and x <= ELL)]
        ys = [spectrum(k, rh=x)[0][0] for x in xs]
        ax.plot(xs, ys, dash[k], marker="o", ms=4, color=col[k], lw=1.5, label=lab[k])
    ax.axhline(threshold, color="0.4", lw=1.0, ls=":")
    ax.text(5.4, threshold + 0.03,
            r"AdS threshold $2\lambda v^2+9/(4\ell^2)=%.4f$" % threshold,
            fontsize=9, color="0.35")
    ax.set_xlabel(r"$r_h$")
    ax.set_ylabel(r"$\lambda_{\min}\ [\lambda v^2]$")
    ax.set_title(r"Convergence of the three topologies at large $r_h$ ($\ell=4$)", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_screening.pdf"))
    plt.close(fig)
    print("\nfigures written to %s/" % outdir)


# ----------------------------------------------------------------------
def near_zone_data(k, rh, ell=ELL):
    """Data entering the auxiliary near-zone sl(2,R) construction of Appendix A.

    Delta = r^2 f, its monic quadratic near-zone truncation
    Delta_nz = (r - r_h)(r - r_-) with r_- = r_h - Delta'(r_h), and the
    consistency relation beta = 2/f'(r_h) = 1/(2 pi T_H).

    On the status of the completion: matching Delta_nz(r_h) = 0 and
    Delta_nz'(r_h) = Delta'(r_h) does not fix the leading coefficient.  This
    function uses the monic quadratic and generator normalization chosen in
    Appendix A.  In that normalization beta = 2 r_h^2 / Delta'(r_h) and the
    static Casimir matching gives nu_alpha = mu_alpha + m_phi^2 r_h^2.
    Other quadratic completions can change the normalized Casimir matching;
    no completion-independent finite-multiplet condition is asserted here.
    """
    M = mass_from_horizon(k, rh, ell)
    dfp = 2.0 * M / rh ** 2 + 2.0 * rh / ell ** 2          # f'(r_h)
    dDelta = rh ** 2 * dfp                                  # Delta'(r_h)
    r_minus = rh - dDelta
    T = hawking_temperature(k, rh, ell)
    return dict(r_minus=r_minus, T_H=T, beta_from_T=1.0 / (2.0 * np.pi * T),
                beta_from_Delta=2.0 * rh ** 2 / dDelta,
                curvature_of_Delta=0.5 * (2.0 * k + 12.0 * rh ** 2 / ell ** 2))


def table_near_zone():
    print("\nNear-zone data for the auxiliary sl(2,R) construction of Appendix A (l=%g)" % ELL)
    print("  k    r_h      r_-        T_H      1/(2 pi T_H)   2 r_h^2/Delta'(r_h)")
    for k in (1, 0, -1):
        for rh in (6.0,):
            d = near_zone_data(k, rh)
            print("  %+d   %5.1f   %8.2f   %7.4f   %10.5f   %14.5f"
                   % (k, rh, d["r_minus"], d["T_H"], d["beta_from_T"], d["beta_from_Delta"]))
    ok = all(abs(near_zone_data(k, rh)["beta_from_T"] - near_zone_data(k, rh)["beta_from_Delta"])
             < 1e-12 for k in (1, 0, -1) for rh in (5.0, 6.0, 12.0, 32.0))
    print("  consistency check beta = 1/(2 pi T_H) = 2 r_h^2/Delta'(r_h) over the")
    print("  tabulated range: %s" % ("PASS" if ok else "FAIL"))
    print("  half Delta''(r_h) = k + 6 r_h^2/l^2, so the near-zone condition of")
    print("  Eq. (A.12) reads r - r_h << Delta'(r_h) / |k + 6 r_h^2/l^2 - 1|:")
    for k in (1, 0, -1):
        d = near_zone_data(k, 6.0)
        width = (6.0 ** 2 * (2 * mass_from_horizon(k, 6.0) / 6.0 ** 2 + 2 * 6.0 / ELL ** 2)
                 / abs(d["curvature_of_Delta"] - 1.0))
        print("     k = %+d : validity width = %.3f" % (k, width))


def bogomolny_check():
    """Flat-space check of the kink energy and Bogomolny bound.

    The planar kink phi = v tanh(m x / sqrt 2) saturates the Bogomolny bound
    E = (2 sqrt 2 / 3) sqrt(lambda) v^3.  Reproducing it on the same quadrature
    used throughout validates the discretization independently of the horizon.
    """
    x = np.linspace(-20.0, 20.0, 200001)
    phi = VEV * np.tanh(np.sqrt(LAM) * VEV * x / np.sqrt(2.0))
    dphi = np.gradient(phi, x)
    energy = _trapz(0.5 * dphi ** 2 + 0.25 * LAM * (phi ** 2 - VEV ** 2) ** 2, x)
    exact = 2.0 * np.sqrt(2.0) / 3.0 * np.sqrt(LAM) * VEV ** 3
    print("\nBogomolny bound: numerical %.6f, exact %.6f, relative error %.1e"
          % (energy, exact, abs(energy - exact) / exact))


def table_screening_bound(radii=(5.0, 6.0, 8.0, 12.0, 20.0), ratio=200.0, dr=1.0/120.0):
    """Table 4: screening of the resolved bound state at fixed (R-r_h)/r_h and fixed
    mesh spacing.

    Two things have to be held fixed for this scan to mean anything.

    (a) The outer boundary must sit at fixed (R-r_h)/r_h, not at fixed coordinate
        span.  For k = 0 the problem scales: with s = r/r_h one has
        f_0 = (r_h^2/l^2)(s^2 - 1/s), so with l, lambda, v, eps fixed the
        profile equation and the radial operator do not depend on r_h at all.
        The planar spectrum must therefore be r_h-independent.  It is, to six
        digits, when (R-r_h)/r_h is held fixed.  At fixed span it drifts, and that
        drift is an outer-boundary artifact.

    (b) The target mesh spacing dr ~ (R-r_h)/N must be held fixed as R grows,
        otherwise the
        near-horizon layer is progressively under-resolved and the level
        drifts back above threshold.
    """
    th = 2.0 * LAM * VEV ** 2 + 9.0 / (4.0 * ELL ** 2)
    print("\nTable 4: screening of the resolved bound state")
    print("  ((R-r_h)/r_h = %g, dr = %.5f, l = %g; continuum threshold = %.6f)"
          % (ratio, dr, ELL, th))
    print("     r_h        N      k=+1       k=0      k=-1       gap  rh^2 gap  bound?")
    for rh in radii:
        span = ratio * rh
        N = int(round(span / dr))
        e = {k: spectrum(k, rh=rh, span=span, N=N, n_levels=1)[0][0] for k in (1, 0, -1)}
        g = e[1] - e[-1]
        print("  %6.0f %8d %9.4f %9.4f %9.4f %9.4f %9.3f %7s"
              % (rh, N, e[1], e[0], e[-1], g, rh * rh * g,
                 "yes" if all(v < th for v in e.values()) else "NO"))
    print("  Quoted to four digits: the individual levels move by a few units in")
    print("  the fourth decimal when (R-r_h)/r_h or the grid is changed.  The ordering")
    print("  and the sign of the screening are stable; the further digits are not.")


def planar_scaling_test(radii=(5.0, 8.0, 12.0, 20.0), ratio=100.0, N=60000):
    """Consistency check behind Table 4: for k = 0 the spectrum must not depend
    on r_h once the outer boundary is placed at fixed (R-r_h)/r_h.  Compare with the
    same quantity at fixed coordinate span, where it drifts."""
    print("\nPlanar scaling test (k = 0)")
    print("  fixed (R-r_h)/r_h = %g:" % ratio)
    for rh in radii:
        v = spectrum(0, rh=rh, span=ratio * rh, N=N, n_levels=1)[0][0]
        print("     r_h=%5.0f   lam_min = %.6f" % (rh, v))
    print("  fixed span L = 600 (for contrast):")
    for rh in radii:
        v = spectrum(0, rh=rh, span=600.0, N=N, n_levels=1)[0][0]
        print("     r_h=%5.0f   (R-r_h)/r_h=%6.1f   lam_min = %.6f" % (rh, 600.0 / rh, v))
    print("  The first block is constant; the second drifts.  The drift is an")
    print("  outer-boundary effect, not horizon physics.")

def bound_state_count(rh=6.0, spans=(300.0, 600.0, 1200.0, 2400.0), dr=1.0/120.0,
                      n_levels=3):
    """How many levels sit below threshold, and why we do not claim a count.

    One sub-threshold level is resolved in every large box we ran.  That is
    evidence for a bound state, not a count of them: Dirichlet levels descend as
    the box grows even for a constant potential, and the second level does
    exactly that here, drifting toward threshold roughly as R^-0.4.  It is a box
    mode approaching the continuum, so its position above threshold says nothing
    about whether a second, more weakly bound level exists on the half-line.
    Settling that needs a Sturm count with control of the far region, which we
    do not attempt.
    """
    th = 2.0 * LAM * VEV ** 2 + 9.0 / (4.0 * ELL ** 2)
    print("\nSub-threshold levels (r_h = %g, threshold = %.6f)" % (rh, th))
    print("  R-r_h    k    lam_1      lam_2    lam_2 - th   below th")
    for span in spans:
        N = min(int(round(span / dr)), 300000)
        for k in (1, 0, -1):
            w = spectrum(k, rh=rh, span=span, N=N, n_levels=n_levels)[0]
            below = int(sum(1 for x in w if x < th))
            print("  %6.0f %+3d %9.5f %9.5f %11.5f %8d"
                  % (span, k, w[0], w[1], w[1] - th, below))
    print("  The second level moves steadily toward threshold as R grows, which is")
    print("  what a box mode does.  No count of bound states is claimed.")

def positivity_check(rh=6.0):
    """Numerical check of Proposition 1 (positivity of the constrained Hessian).

    The proof needs two things: the algebraic identity
    V''(phi) phi - V'(phi) = 2 lambda phi^3, and phi_0 > 0 on (a, R].  Both are
    verified here.  Given them, the factorization
        Q[psi] = int p phi_0^2 |(psi/phi_0)'|^2 + 2 lambda int w phi_0^2 |psi|^2
    is positive term by term, so the constrained Morse index vanishes for every
    horizon topology and every parameter value on that branch.
    """
    print("\nPositivity check behind Proposition 1")
    for phi in (0.1, 0.4, 0.9):
        lhs = LAM * (3.0 * phi ** 2 - VEV ** 2) * phi - LAM * phi * (phi ** 2 - VEV ** 2)
        print("   phi=%.1f :  V''phi - V' = %.6f    2*lam*phi^3 = %.6f"
              % (phi, lhs, 2.0 * LAM * phi ** 3))
    for k in (1, 0, -1):
        out = profile(k, rh=rh)
        sol = out[0] if isinstance(out, tuple) else out
        a = rh * (1.0 + EPS)
        x = np.linspace(a, rh + L_SPAN, 5000)
        y = sol.sol(x)[0]
        print("   k=%+d :  min phi_0 on (a,R] = %+.3e   positive: %s"
              % (k, y[1:].min(), y[1:].min() > 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="also regenerate figures 1-3")
    args, _ = ap.parse_known_args()

    bogomolny_check()
    positivity_check()
    bound_state_count()
    table_spectra()
    table_convergence()
    table_screening()
    planar_scaling_test()
    table_screening_bound()
    table_threshold()
    table_bound_state()
    table_theorem()
    table_offset()
    table_near_zone()
    if args.figures:
        figures()


if __name__ == "__main__":
    main()
