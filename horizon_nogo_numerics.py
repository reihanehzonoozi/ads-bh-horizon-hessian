#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizon-regular solutions and the status of the stretched-horizon offset.

Companion calculation to

    "Boundary-pinned phi^4 layers outside topological anti-de Sitter black holes:
     a no-go theorem and constrained spectral stability"

Three checks, corresponding to Proposition 2 of the manuscript:

  (i)   no-go        : with Delta = r^2 f, multiplying the static equation
                       (Delta phi')' = r^2 V'(phi) by phi' and integrating
                       over the exterior gives

                         r_h^2 V(phi(r_h))
                           + int_{r_h}^inf [ Delta'(phi')^2/2 + 2 r V ] dr = 0.

                       Every term is nonnegative when V >= 0 and Delta' > 0,
                       so phi is constant.  Verified here by shooting outward
                       from the horizon with the regularity condition
                       phi'(r_h) = V'(phi_h)/f'(r_h): no trajectory reaches a
                       vacuum, for any horizon curvature.

                       Scope: the static equation used here is the
                       HORIZON-HOMOGENEOUS RADIAL reduction, so the statement
                       is confined to profiles depending on r alone.  Static
                       configurations with angular dependence, including thick
                       domain walls placed around and intersecting a horizon,
                       lie outside this ansatz and are not excluded by it.

  (ii)  Delta' > 0   : the hypothesis of the no-go, checked over the
                       tabulated range of horizon radii.

  (iii) pinning      : repeating the derivation with the inner edge at
                       a = r_h(1+eps), where Delta(a) != 0, gives

                         Delta(a) phi'(a)^2 / 2
                           = a^2 V(0) + int_a^inf [ Delta'(phi')^2/2 + 2rV ],

                       so the layer exists only because of the boundary term,
                       and |phi'(a)| >= C eps^{-1/2} as the offset is removed.
                       This is a rigorous lower bound, not an asserted exact
                       asymptotic power law, and it does not assert that the
                       energy has no limiting value.  The identity is checked against
                       the computed profiles
                       and the offset scan is tabulated.

Conventions: lambda = v = 1, ell = 4, so eigenvalues and energies are in
units of lambda v^2.

Usage
-----
    python3 horizon_nogo_numerics.py

Requires numpy >= 1.20, scipy >= 1.7.
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

LAM = 1.0
VEV = 1.0
ELL = 4.0

dV = lambda p: LAM * p * (p ** 2 - VEV ** 2)
V = lambda p: 0.25 * LAM * (p ** 2 - VEV ** 2) ** 2


def background(k, rh, ell=ELL):
    """f, f', Delta = r^2 f and Delta' for the horizon radius rh."""
    M = 0.5 * rh * (k + rh ** 2 / ell ** 2)
    f = lambda r: k - 2.0 * M / r + r ** 2 / ell ** 2
    fp = lambda r: 2.0 * M / r ** 2 + 2.0 * r / ell ** 2
    D = lambda r: r ** 2 * f(r)
    Dp = lambda r: 2.0 * k * r - 2.0 * M + 4.0 * r ** 3 / ell ** 2
    return f, fp, D, Dp, M


# ----------------------------------------------------------------------
def shoot_from_horizon(phi_h, k, rh, r_max=600.0, delta=1e-9):
    """Horizon regularity fixes phi'(r_h) = V'(phi_h)/f'(r_h)."""
    f, fp, _, _, _ = background(k, rh)
    slope = dV(phi_h) / fp(rh)
    rhs = lambda r, y: [y[1], (dV(y[0]) - (fp(r) + 2.0 * f(r) / r) * y[1]) / f(r)]
    return solve_ivp(rhs, (rh * (1.0 + delta), r_max),
                     [phi_h + slope * rh * delta, slope],
                     rtol=1e-11, atol=1e-13)


def table_nogo(n_shots=41):
    print("\n(i) Shooting outward from a regular horizon (radial, horizon-homogeneous).")
    print("    If a regular layer existed, some phi_h would reach a vacuum.")
    print("    k     radii            max |phi(600)| over %d shots" % (3 * n_shots))
    for k in (1, 0, -1):
        radii = [r for r in (5.0, 6.0, 12.0) if not (k == -1 and r <= ELL)]
        worst = 0.0
        for rh in radii:
            for ph in np.linspace(-0.99 * VEV, 0.99 * VEV, n_shots):
                worst = max(worst, abs(shoot_from_horizon(ph, k, rh).y[0, -1]))
        print("   %+d    %-15s  %.2e" % (k, radii, worst))
    print("    every trajectory relaxes toward phi = 0; none approaches +-v.")
    print("    Scope: profiles depending on r alone.  Static walls with angular")
    print("    dependence lie outside this ansatz.")


def table_hypothesis():
    print("\n(ii) Hypothesis of the no-go: Delta' > 0 on the exterior.")
    print("     k     r_h     min Delta'(r) on [r_h, 400]")
    for k in (1, 0, -1):
        for rh in (5.0, 6.0, 12.0, 32.0):
            if k == -1 and rh <= ELL:
                continue
            _, _, _, Dp, _ = background(k, rh)
            r = np.linspace(rh, 400.0, 200000)
            print("    %+d   %5.1f    %12.4f" % (k, rh, Dp(r).min()))


# ----------------------------------------------------------------------
def pinned_profile(k, rh, span, eps, n=6000):
    f, fp, _, _, _ = background(k, rh)
    a, b = rh * (1.0 + eps), rh + span
    r = np.linspace(a, b, n)
    guess = np.vstack([VEV * np.tanh((r - a) / np.sqrt(2.0)),
                       VEV / np.sqrt(2.0) / np.cosh(np.clip((r - a) / np.sqrt(2.0), 0, 50)) ** 2])
    sol = solve_bvp(lambda x, y: np.vstack(
        [y[1], (dV(y[0]) - (fp(x) + 2.0 * f(x) / x) * y[1]) / f(x)]),
        lambda u, w: np.array([u[0], w[0] - VEV]), r, guess,
        tol=1e-8, max_nodes=600000)
    if sol.status != 0:
        raise RuntimeError("BVP failed at eps = %g" % eps)
    return sol, a, b


def table_pinning(k=0, rh=6.0, span=6.0):
    print("\n(iii) The stretched-horizon offset is a defining scale.")
    print("      On the finite numerical interval [a,b] the exact balance is")
    print("      Delta(a) phi'(a)^2/2 = Delta(b) phi'(b)^2/2 + a^2 V(0)")
    print("                              + int_a^b [Delta'(phi')^2/2 + 2rV].")
    f, _, D, Dp, _ = background(k, rh)
    print("   eps        lhs         rhs        rel.diff    phi'(a)   lower bound   ratio      E")
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        sol, a, b = pinned_profile(k, rh, span, eps)
        r = np.linspace(a, b, 400000)
        p, dp = sol.sol(r)
        slope_a, slope_b = sol.sol(a)[1], sol.sol(b)[1]
        lhs = 0.5 * D(a) * slope_a ** 2
        outer = 0.5 * D(b) * slope_b ** 2
        rhs = outer + a ** 2 * V(0.0) + _trapz(0.5 * Dp(r) * dp ** 2 + 2.0 * r * V(p), r)
        lower = np.sqrt(2.0 * a ** 2 * V(0.0) / D(a))
        energy = _trapz(r ** 2 * (0.5 * f(r) * dp ** 2 + V(p)), r)
        print("  %7.0e  %10.4f  %10.4f   %8.1e  %9.2f   %10.2f  %6.2f   %8.4f"
               % (eps, lhs, rhs, abs(lhs - rhs) / rhs, slope_a, lower, slope_a/lower, energy))
    print("      the rigorous lower bound grows as eps^{-1/2}; the computed slope")
    print("      grows even faster over this range, so no exact power law is claimed.")
    print("      The energy decreases monotonically over this range.  What is")
    print("      established is that the slope diverges, so no nontrivial")
    print("      horizon-regular configuration satisfying the pinned datum")
    print("      survives eps -> 0.  No claim is made that the energy lacks a")
    print("      limiting value; the data are equally consistent with a")
    print("      gradual collapse onto the vacuum.")


def main():
    table_nogo()
    table_hypothesis()
    table_pinning()


if __name__ == "__main__":
    main()