"""
Tests for the induced-metric torus_dist function in torus_pph.py.

torus_dist computes the arc length of the straight (θ, φ) parameter-space
segment under the first fundamental form of the torus embedding in ℝ³:

    ds² = (R + r cos φ)² dθ² + r² dφ²

It is NOT the intrinsic geodesic; see docstring in torus_pph.py for details.
"""

import math
import sys
import os

import numpy as np
import pytest

# Make sure torus_pph is importable regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torus_pph import torus_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arc_length_exact_pure_theta(R: float, r: float, phi: float, dtheta: float) -> float:
    """Exact arc length for a pure-θ displacement at constant φ.
    The integrand is constant: sqrt((R + r*cos(phi))^2 * dtheta^2) = (R + r*cos(phi))*|dtheta|
    """
    return (R + r * math.cos(phi)) * abs(dtheta)


def _arc_length_exact_pure_phi(r: float, dphi: float) -> float:
    """Exact arc length for a pure-φ displacement (any θ, any R).
    The integrand is r*|dphi|, constant.
    """
    return r * abs(dphi)


# ---------------------------------------------------------------------------
# Tests: pure-θ displacement
# ---------------------------------------------------------------------------

class TestPureTheta:
    """Pure-θ displacements at a fixed φ — exact formula is (R + r cos φ)·|Δθ|."""

    def test_outer_equator(self):
        """At φ=0 (outer equator), dist = (R + r)·|Δθ|."""
        R, r = 10.0, 1.0
        dtheta = 0.5
        phi = 0.0
        expected = (R + r) * dtheta
        got = torus_dist(0.0, phi, dtheta, phi, R, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"

    def test_inner_equator(self):
        """At φ=π (inner equator), dist = (R − r)·|Δθ|."""
        R, r = 10.0, 1.0
        dtheta = 0.5
        phi = math.pi
        expected = (R - r) * dtheta
        got = torus_dist(0.0, phi, dtheta, phi, R, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"

    def test_quarter_phi(self):
        """At φ=π/2, dist = R·|Δθ| (cos(π/2) = 0, so R + r·0 = R)."""
        R, r = 5.0, 2.0
        dtheta = 1.0
        phi = math.pi / 2
        expected = R * dtheta  # R + r*0 = R
        got = torus_dist(0.0, phi, dtheta, phi, R, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"

    def test_scales_linearly_with_R(self):
        """For large R, a pure-θ distance scales approximately linearly with R."""
        r = 1.0
        dtheta = 0.3
        phi = math.pi / 3
        d1 = torus_dist(0.0, phi, dtheta, phi, 100.0, r)
        d2 = torus_dist(0.0, phi, dtheta, phi, 200.0, r)
        # ratio should be ≈ 2 when R >> r
        ratio = d2 / d1
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ≈ 2, got {ratio}"

    def test_wraparound_shorter(self):
        """The function takes the shorter direction across the 2π boundary."""
        R, r = 5.0, 1.0
        phi = 0.0
        # Going from θ=0.1 to θ=2π-0.1 the short way is Δθ = 0.2 (backward)
        theta1 = 0.1
        theta2 = 2 * math.pi - 0.1
        expected = (R + r * math.cos(phi)) * 0.2
        got = torus_dist(theta1, phi, theta2, phi, R, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# Tests: pure-φ displacement
# ---------------------------------------------------------------------------

class TestPurePhi:
    """Pure-φ displacements — exact formula is r·|Δφ|, independent of θ and R."""

    def test_basic(self):
        r = 1.5
        dphi = 0.8
        expected = r * dphi
        got = torus_dist(0.0, 0.0, 0.0, dphi, 10.0, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"

    def test_independent_of_R(self):
        """r·|Δφ| should not depend on R."""
        r = 2.0
        dphi = 1.0
        d1 = torus_dist(0.0, 0.0, 0.0, dphi, 1.0, r)
        d2 = torus_dist(0.0, 0.0, 0.0, dphi, 100.0, r)
        assert abs(d1 - d2) < 1e-4, f"Expected same value, got {d1} vs {d2}"

    def test_independent_of_theta(self):
        """r·|Δφ| should not depend on starting θ."""
        r = 1.0
        dphi = 0.5
        R = 3.0
        d1 = torus_dist(0.0, 0.0, 0.0, dphi, R, r)
        d2 = torus_dist(1.0, 0.0, 1.0, dphi, R, r)
        d3 = torus_dist(math.pi, 0.0, math.pi, dphi, R, r)
        assert abs(d1 - d2) < 1e-4
        assert abs(d1 - d3) < 1e-4

    def test_scales_linearly_with_r(self):
        """For a pure-φ displacement, distance scales exactly as r."""
        dphi = 1.0
        d1 = torus_dist(0.0, 0.0, 0.0, dphi, 10.0, 1.0)
        d2 = torus_dist(0.0, 0.0, 0.0, dphi, 10.0, 3.0)
        assert abs(d2 / d1 - 3.0) < 1e-4, f"Expected ratio 3, got {d2/d1}"

    def test_wraparound_shorter_phi(self):
        """Takes the shorter path across the 2π boundary in φ."""
        r = 1.0
        R = 5.0
        phi1 = 0.1
        phi2 = 2 * math.pi - 0.1
        expected = r * 0.2
        got = torus_dist(0.0, phi1, 0.0, phi2, R, r)
        assert abs(got - expected) < 1e-4, f"Expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# Tests: flat-metric limit
# ---------------------------------------------------------------------------

class TestFlatMetricLimit:
    """When R is huge compared to r·|Δφ|, the metric is approximately flat
    (scaled by R in the θ direction and r in the φ direction):
        dist ≈ √((R·Δθ)² + (r·Δφ)²)
    """

    def test_flat_limit_diagonal(self):
        """For large R and small |Δφ|, dist ≈ √((R·Δθ)² + (r·Δφ)²)."""
        R = 1000.0
        r = 1.0
        dtheta = 0.05
        dphi = 0.05
        # At these scales, (R + r*cos(phi)) ≈ R everywhere along the segment
        expected = math.sqrt((R * dtheta)**2 + (r * dphi)**2)
        got = torus_dist(0.0, 0.0, dtheta, dphi, R, r)
        # 0.5% tolerance for the approximation error from phi-variation
        assert abs(got - expected) / expected < 0.005, (
            f"Expected ≈ {expected}, got {got}"
        )

    def test_flat_limit_pure_theta(self):
        """For large R, a pure-θ step ≈ R·|Δθ|."""
        R = 1000.0
        r = 1.0
        dtheta = 0.1
        phi = 0.0
        expected = R * dtheta
        got = torus_dist(0.0, phi, dtheta, phi, R, r)
        assert abs(got - expected) / expected < 1e-3


# ---------------------------------------------------------------------------
# Tests: symmetry and sanity
# ---------------------------------------------------------------------------

class TestSymmetryAndSanity:

    def test_symmetry(self):
        """dist(u, v) == dist(v, u)."""
        R, r = 3.0, 1.0
        d1 = torus_dist(0.5, 1.2, 2.1, 0.3, R, r)
        d2 = torus_dist(2.1, 0.3, 0.5, 1.2, R, r)
        assert abs(d1 - d2) < 1e-8

    def test_zero_distance(self):
        """Distance from a point to itself is 0."""
        R, r = 3.0, 1.0
        d = torus_dist(1.0, 1.0, 1.0, 1.0, R, r)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        """All distances are non-negative."""
        rng = np.random.default_rng(42)
        R, r = 3.0, 1.0
        for _ in range(50):
            t1, p1, t2, p2 = rng.uniform(0, 2 * math.pi, 4)
            d = torus_dist(t1, p1, t2, p2, R, r)
            assert d >= 0.0

    def test_increases_with_dtheta(self):
        """Holding φ fixed and R, r constant, dist should increase with |Δθ|."""
        R, r = 5.0, 1.0
        phi = 0.5
        d1 = torus_dist(0.0, phi, 0.5, phi, R, r)
        d2 = torus_dist(0.0, phi, 1.0, phi, R, r)
        assert d2 > d1

    def test_increases_with_dphi(self):
        """Holding θ fixed and R, r constant, dist should increase with |Δφ|."""
        R, r = 5.0, 1.0
        theta = 0.5
        d1 = torus_dist(theta, 0.0, theta, 0.5, R, r)
        d2 = torus_dist(theta, 0.0, theta, 1.0, R, r)
        assert d2 > d1

    def test_outer_equator_longer_than_inner(self):
        """A θ-step at φ=0 (outer) should be longer than at φ=π (inner)."""
        R, r = 5.0, 1.0
        dtheta = 0.5
        d_outer = torus_dist(0.0, 0.0, dtheta, 0.0, R, r)       # phi=0
        d_inner = torus_dist(0.0, math.pi, dtheta, math.pi, R, r)  # phi=π
        assert d_outer > d_inner, (
            f"Outer dist {d_outer} should exceed inner dist {d_inner}"
        )

    def test_numerical_convergence(self):
        """Increasing n_steps should converge to the same answer."""
        R, r = 3.0, 1.0
        args = (0.3, 0.7, 1.5, 2.1, R, r)
        d16 = torus_dist(*args, n_steps=16)
        d64 = torus_dist(*args, n_steps=64)
        d256 = torus_dist(*args, n_steps=256)
        assert abs(d64 - d256) < abs(d16 - d64), (
            "Simpson quadrature should converge as n_steps increases"
        )
        assert abs(d256 - d64) < 1e-6
