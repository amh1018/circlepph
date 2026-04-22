"""
Unit tests for torus_dist — induced-metric arc length on T² ⊂ ℝ³.

Tests cover:
  1. Pure-θ displacement: should equal (R + r·cos φ)·|Δθ|
  2. Pure-φ displacement: should equal r·|Δφ| (independent of θ)
  3. Flat-metric limit: large R, small displacements → √((R·Δθ)² + (r·Δφ)²)
  4. Symmetry: d(A→B) == d(B→A)
  5. Short-wraparound: distance via the short path < distance via the long path
  6. Dependence on R, r: changing radii changes the distance
  7. Zero distance: identical points → 0
"""

import sys
import os
import numpy as np
import pytest

# Allow running from repo root or tests/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torus_pph import torus_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_close(got, expected, rtol=1e-4, msg=""):
    """Assert relative closeness, with an informative failure message."""
    rel = abs(got - expected) / (abs(expected) + 1e-12)
    assert rel < rtol, (
        f"{msg}  got={got:.8f}, expected={expected:.8f}, rel_err={rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 1 — Pure-θ displacement at fixed φ
# ---------------------------------------------------------------------------

class TestPureThetaDisplacement:
    """d(θ₁,φ; θ₂,φ) == (R + r·cos φ)·|Δθ| (shorter arc)."""

    def test_outer_equator(self):
        """φ = 0  →  (R + r)·|Δθ|."""
        R, r = 10.0, 1.0
        dtheta = 0.5
        expected = (R + r * np.cos(0.0)) * dtheta
        got = torus_dist(0.0, 0.0, dtheta, 0.0, R, r)
        _assert_close(got, expected, rtol=1e-4, msg="outer equator φ=0:")

    def test_inner_equator(self):
        """φ = π  →  (R − r)·|Δθ|."""
        R, r = 10.0, 1.0
        dtheta = 0.5
        expected = (R + r * np.cos(np.pi)) * dtheta  # = (R - r)*dtheta
        got = torus_dist(0.0, np.pi, dtheta, np.pi, R, r)
        _assert_close(got, expected, rtol=1e-4, msg="inner equator φ=π:")

    def test_arbitrary_phi(self):
        """φ = π/3  →  (R + r·cos(π/3))·|Δθ|."""
        R, r = 5.0, 1.5
        phi = np.pi / 3
        dtheta = 0.8
        expected = (R + r * np.cos(phi)) * dtheta
        got = torus_dist(0.0, phi, dtheta, phi, R, r)
        _assert_close(got, expected, rtol=1e-4, msg=f"φ=π/3:")

    def test_wraparound(self):
        """θ near 2π, Δθ crosses the wrap boundary (shorter arc)."""
        R, r = 4.0, 1.0
        phi = np.pi / 4
        # Points at θ=5.9 and θ=0.1; shorter Δθ = 0.2 + 2π - 5.9 = 2π - 5.8 ≈ 0.483
        theta1, theta2 = 5.9, 0.1
        dtheta_short = (theta2 - theta1) % (2 * np.pi)
        if dtheta_short > np.pi:
            dtheta_short -= 2 * np.pi
        expected = (R + r * np.cos(phi)) * abs(dtheta_short)
        got = torus_dist(theta1, phi, theta2, phi, R, r)
        _assert_close(got, expected, rtol=1e-4, msg="wraparound θ:")


# ---------------------------------------------------------------------------
# Test 2 — Pure-φ displacement at fixed θ
# ---------------------------------------------------------------------------

class TestPurePhiDisplacement:
    """d(θ,φ₁; θ,φ₂) == r·|Δφ| (independent of θ and R)."""

    def test_basic(self):
        R, r = 10.0, 1.0
        dphi = 0.7
        expected = r * dphi
        got = torus_dist(1.0, 0.0, 1.0, dphi, R, r)
        _assert_close(got, expected, rtol=1e-4, msg="pure-φ basic:")

    def test_independence_of_theta(self):
        """Same Δφ at two different θ values gives the same distance."""
        R, r = 5.0, 1.5
        dphi = 1.2
        d1 = torus_dist(0.0, 0.5, 0.0, 0.5 + dphi, R, r)
        d2 = torus_dist(2.3, 0.5, 2.3, 0.5 + dphi, R, r)
        _assert_close(d1, d2, rtol=1e-6, msg="φ-dist θ-independence:")

    def test_independence_of_R(self):
        """Pure-φ distance should not depend on R."""
        r = 1.0
        dphi = 1.0
        d1 = torus_dist(0.0, 0.0, 0.0, dphi, R=2.0, r=r)
        d2 = torus_dist(0.0, 0.0, 0.0, dphi, R=100.0, r=r)
        _assert_close(d1, d2, rtol=1e-6, msg="φ-dist R-independence:")

    def test_phi_wraparound(self):
        """φ near 2π crossing the wrap boundary."""
        R, r = 4.0, 1.0
        phi1, phi2 = 5.9, 0.1
        dphi_short = (phi2 - phi1) % (2 * np.pi)
        if dphi_short > np.pi:
            dphi_short -= 2 * np.pi
        expected = r * abs(dphi_short)
        got = torus_dist(0.0, phi1, 0.0, phi2, R, r)
        _assert_close(got, expected, rtol=1e-4, msg="wraparound φ:")


# ---------------------------------------------------------------------------
# Test 3 — Flat-metric limit (large R)
# ---------------------------------------------------------------------------

class TestFlatMetricLimit:
    """For large R and small displacements, dist ≈ √((R+r·cos φ)²·Δθ² + r²·Δφ²)."""

    def test_large_R_diagonal(self):
        """Diagonal step; R large so R + r·cos φ ≈ R (within a few percent)."""
        R, r = 1000.0, 1.0
        dtheta, dphi = 0.1, 0.1
        phi = 0.0
        # Approximate formula: sqrt((R + r*cos(phi))^2 * dtheta^2 + r^2 * dphi^2)
        expected = np.sqrt((R + r * np.cos(phi))**2 * dtheta**2 + r**2 * dphi**2)
        got = torus_dist(0.0, phi, dtheta, dphi, R, r)
        _assert_close(got, expected, rtol=1e-3, msg="flat-metric diagonal large R:")

    def test_theta_scale_with_R(self):
        """Pure-θ distance scales linearly with R for large R."""
        r = 1.0
        dtheta = 0.5
        phi = np.pi / 2  # cos(φ) = 0  →  R + r·cos φ = R exactly
        d_R10  = torus_dist(0.0, phi, dtheta, phi, R=10.0,  r=r)
        d_R100 = torus_dist(0.0, phi, dtheta, phi, R=100.0, r=r)
        # Ratio should be 100/10 = 10
        ratio = d_R100 / d_R10
        _assert_close(ratio, 10.0, rtol=1e-5, msg="θ-dist linear in R (φ=π/2):")


# ---------------------------------------------------------------------------
# Test 4 — Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_symmetry(self):
        R, r = 3.0, 1.0
        d1 = torus_dist(0.3, 0.7, 1.2, 2.1, R, r)
        d2 = torus_dist(1.2, 2.1, 0.3, 0.7, R, r)
        _assert_close(d1, d2, rtol=1e-8, msg="symmetry:")


# ---------------------------------------------------------------------------
# Test 5 — Short-wraparound is shorter than long way around
# ---------------------------------------------------------------------------

class TestWrap:
    def test_short_wrap_is_shorter(self):
        """Crossing the boundary (short path) must be < going the long way."""
        R, r = 4.0, 1.0
        phi = 0.5
        # θ₁ = 0.1, θ₂ = 2π - 0.1 ≈ 6.183
        # Short path: Δθ ≈ 0.2 (wraps around)
        # Long path: Δθ ≈ 6.08 (straight through middle)
        d_short = torus_dist(0.1, phi, 2 * np.pi - 0.1, phi, R, r)
        # Expected: ≈ (R + r*cos(phi)) * 0.2
        expected_short = (R + r * np.cos(phi)) * 0.2
        _assert_close(d_short, expected_short, rtol=1e-3, msg="short wrap θ:")


# ---------------------------------------------------------------------------
# Test 6 — Changing R or r changes the distance
# ---------------------------------------------------------------------------

class TestRadiusDependence:
    def test_R_matters(self):
        """Changing R should change the distance (except for pure-φ displacements)."""
        r = 1.0
        dtheta, dphi = 0.5, 0.5
        phi = 0.0
        d1 = torus_dist(0.0, phi, dtheta, phi + dphi, R=2.0, r=r)
        d2 = torus_dist(0.0, phi, dtheta, phi + dphi, R=5.0, r=r)
        assert abs(d1 - d2) > 0.01, f"R change did not affect distance: {d1=}, {d2=}"

    def test_r_matters(self):
        """Changing r should change the distance."""
        R = 4.0
        dtheta, dphi = 0.3, 0.5
        phi = 0.0
        d1 = torus_dist(0.0, phi, dtheta, phi + dphi, R=R, r=0.5)
        d2 = torus_dist(0.0, phi, dtheta, phi + dphi, R=R, r=2.0)
        assert abs(d1 - d2) > 0.01, f"r change did not affect distance: {d1=}, {d2=}"


# ---------------------------------------------------------------------------
# Test 7 — Zero distance
# ---------------------------------------------------------------------------

class TestZeroDistance:
    def test_same_point(self):
        R, r = 3.0, 1.0
        got = torus_dist(1.23, 4.56, 1.23, 4.56, R, r)
        assert got < 1e-12, f"same point: expected ~0, got {got}"

    def test_same_point_modulo_2pi(self):
        """Points that differ by exactly 2π should have distance 0."""
        R, r = 3.0, 1.0
        got = torus_dist(0.0, 0.0, 2 * np.pi, 0.0, R, r)
        assert got < 1e-10, f"2π-periodic: expected ~0, got {got}"


if __name__ == "__main__":
    # Allow running directly: python tests/test_torus_dist.py
    import pytest
    pytest.main([__file__, "-v"])
