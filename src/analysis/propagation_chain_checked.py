"""Audited finite-depth CHSBM chain; no equilibrium or HNRD claim.

External labelled seed features estimate one shared prototype difference. Graph
nodes and hyperedges are held out. Shared nodes make edge errors dependent, so
the only probability bound used here is Markov on the changed-edge count.
"""

import argparse

import numpy as np
from scipy.special import ndtr


def hom_error(p):
    return 1.0 - (1.0 - p) ** 3 - p**3


def mixed_error(p):
    return p * (1.0 - p)


def operator_from_edges(n, edges, weights):
    """Return sum_e weights[e] g_e g_e^T, g_e=1_e/sqrt(3)."""
    B = np.zeros((n, n))
    for edge, weight in zip(edges, weights):
        if weight:
            B[np.ix_(edge, edge)] += weight / 3.0
    return B


def max_changed_degree(n, edges, delta_weights):
    degrees = np.zeros(n)
    np.add.at(degrees, edges.ravel(), np.repeat(np.abs(delta_weights), 3))
    return float(degrees.max())


def markov_changed_count(m, eta_bar, delta):
    """With probability >=1-delta, changed count <= this (Markov)."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return min(m, int(np.ceil(m * eta_bar / delta)))


def propagation_difference(B, Bhat, z0, h, depth):
    """Contractive diffusion: actual Frobenius error and telescoping bound."""
    if depth < 0 or h < 0:
        raise ValueError("depth and step size must be nonnegative")
    n = B.shape[0]
    identity = np.eye(n)
    M, Mhat = identity - h * B, identity - h * Bhat
    if max(np.linalg.norm(M, 2), np.linalg.norm(Mhat, 2)) > 1 + 1e-10:
        raise ValueError("step size is not contractive for both operators")
    z, zhat = z0.copy(), z0.copy()
    for _ in range(depth):
        z, zhat = M @ z, Mhat @ zhat
    beta = np.linalg.norm(Bhat - B, 2)
    bound = h * depth * beta * np.linalg.norm(z0, "fro")
    return float(np.linalg.norm(zhat - z, "fro")), float(bound)


def all_positive_certificate(ideal_min_signed_margin, frobenius_error_bound):
    return ideal_min_signed_margin > frobenius_error_bound


def flip_count_bound(frobenius_error_bound, min_ideal_margin):
    """Maximum newly wrong nodes among those with ideal margin >= gamma."""
    if min_ideal_margin <= 0:
        raise ValueError("min_ideal_margin must be positive")
    return int(np.floor((frobenius_error_bound / min_ideal_margin) ** 2))


def quantile_margin_certificate(ideal_margin, perturbed_margin, row_error, q, subset=None):
    """Post-hoc, instance-level margin comparison; subset may use oracle labels."""
    ideal_margin = np.asarray(ideal_margin)
    perturbed_margin = np.asarray(perturbed_margin)
    row_error = np.asarray(row_error)
    if not (ideal_margin.shape == perturbed_margin.shape == row_error.shape):
        raise ValueError("margin and error arrays must have the same shape")
    if np.any(row_error < 0) or np.any(np.abs(perturbed_margin - ideal_margin) > row_error + 1e-10):
        raise ValueError("row error must upper-bound every readout margin change")
    if not 0 <= q <= 1:
        raise ValueError("quantile must be in [0, 1]")
    positive = ideal_margin > 0
    if subset is not None:
        positive &= np.asarray(subset, dtype=bool)
    if not positive.any():
        return None
    gamma = float(np.quantile(ideal_margin[positive], q))
    eligible = positive & (ideal_margin >= gamma)
    actual = int(np.count_nonzero(eligible & (perturbed_margin <= 0)))
    risk = int(np.count_nonzero(eligible & (row_error >= ideal_margin)))
    row_bound = int(np.count_nonzero(eligible & (row_error >= gamma)))
    frob_bound = min(int(eligible.sum()), flip_count_bound(np.linalg.norm(row_error[eligible]), gamma))
    if actual > risk or risk > row_bound or row_bound > frob_bound:
        raise AssertionError("margin certificate ordering failed")
    return {
        "q": q, "gamma": gamma, "eligible": int(eligible.sum()),
        "actual_flips": actual, "risk_count": risk,
        "row_bound": row_bound, "frob_bound": frob_bound,
    }


def sample_edges(rng, y, m, rho_hom):
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == -1)
    edges = np.empty((m, 3), dtype=int)
    for j in range(m):
        major, minor = (positive, negative) if rng.random() < 0.5 else (negative, positive)
        if rng.random() < rho_hom:
            edges[j] = rng.choice(major, 3, replace=False)
        else:
            edges[j, :2] = rng.choice(major, 2, replace=False)
            edges[j, 2] = rng.choice(minor)
    return edges


def expected_error_rates(sigma, ell, dimension, rho_hom, samples, rng):
    w = sigma * np.sqrt(2 / ell) * rng.standard_normal((samples, dimension))
    w[:, 0] += 2.0
    p = ndtr(-w[:, 0] / (sigma * np.linalg.norm(w, axis=1)))
    eta_h, eta_m = np.mean(hom_error(p)), np.mean(mixed_error(p))
    return float(rho_hom * eta_h + (1 - rho_hom) * eta_m), float(eta_h), float(eta_m)


def run_demo(seed=700, n=400, m=600, ell=8, sigma=1.0, rho_hom=0.7, depth=5):
    rng = np.random.default_rng(seed)
    d = 8
    labels = np.tile([1, -1], n // 2)
    rng.shuffle(labels)
    edges = sample_edges(rng, labels, m, rho_hom)

    # Independent training seeds: every graph node is a genuine held-out point.
    seeds_pos = np.zeros((ell, d)); seeds_pos[:, 0] = 1.0
    seeds_neg = np.zeros((ell, d)); seeds_neg[:, 0] = -1.0
    seeds_pos += sigma * rng.standard_normal((ell, d))
    seeds_neg += sigma * rng.standard_normal((ell, d))
    w = seeds_pos.mean(axis=0) - seeds_neg.mean(axis=0)
    X = sigma * rng.standard_normal((n, d))
    X[:, 0] += labels
    predicted = np.where(X @ w >= 0, 1, -1)

    edge_labels = labels[edges]
    edge_predictions = predicted[edges]
    a_star = np.all(edge_labels == edge_labels[:, :1], axis=1).astype(float)
    a_hat = np.all(edge_predictions == edge_predictions[:, :1], axis=1).astype(float)
    changed = int(np.count_nonzero(a_hat - a_star))
    B = operator_from_edges(n, edges, a_star)
    Bhat = operator_from_edges(n, edges, a_hat)
    beta = float(np.linalg.norm(Bhat - B, 2))
    degree_bound = max_changed_degree(n, edges, a_hat - a_star)

    eta, eta_h, eta_m = expected_error_rates(
        sigma, ell, d, rho_hom, 100_000, np.random.default_rng(seed + 1)
    )
    high_probability_count = markov_changed_count(m, eta, 0.05)

    # PSD operators and h <= 1/max weighted degree imply both maps contract.
    max_degree = max(
        max_changed_degree(n, edges, a_star),
        max_changed_degree(n, edges, a_hat),
    )
    h = 1.0 / max(1.0, max_degree)
    z0 = X[:, :1].copy()
    actual, certified = propagation_difference(B, Bhat, z0, h, depth)
    z, zhat = z0.copy(), z0.copy()
    for _ in range(depth):
        z = z - h * B @ z
        zhat = zhat - h * Bhat @ zhat
    ideal_margin = labels * z[:, 0]
    perturbed_margin = labels * zhat[:, 0]
    newly_wrong = int(np.count_nonzero((ideal_margin > 0) & (perturbed_margin <= 0)))
    gamma = 0.2
    protected = ideal_margin >= gamma
    protected_flips = int(np.count_nonzero(protected & (perturbed_margin <= 0)))
    return {
        "eta_hom": eta_h, "eta_mix": eta_m, "eta_bar": eta,
        "changed": changed, "markov_95_count": high_probability_count,
        "beta": beta, "degree_bound": degree_bound,
        "actual_fro_error": actual, "telescoping_bound": certified,
        "ideal_min_signed_margin": float(ideal_margin.min()),
        "newly_wrong": newly_wrong, "protected_count": int(protected.sum()),
        "protected_flips": protected_flips,
        "protected_flip_bound": min(int(protected.sum()), flip_count_bound(certified, gamma)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=700)
    args = parser.parse_args()
    for key, value in run_demo(seed=args.seed).items():
        print(f"{key}: {value}")
