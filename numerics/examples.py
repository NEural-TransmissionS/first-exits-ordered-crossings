#!/usr/bin/env python3
"""Reproduce the paper's three numerical illustrations.

The code is organized to parallel the mathematics.  A polynomial is a
dictionary: the key ``(m_1, ..., m_d)`` denotes the monomial
``z_1**m_1 ... z_d**m_d``, and the associated value is its coefficient.  Only
powers up to the threshold vector ``M`` are retained because the inverse
D-transform at ``M`` depends on only those finitely many coefficients.

The discrete calculations use exact rational arithmetic.  Floating-point
arithmetic is used only for quadrature, simulation, and optional figures.
"""

from __future__ import annotations

import argparse
import itertools
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaln


Index = tuple[int, ...]
Poly = dict[Index, F]


# ---------------------------------------------------------------------------
# Truncated multivariate power-series arithmetic
# ---------------------------------------------------------------------------


def poly_add(a: Poly, b: Poly, scale_a: F = F(1), scale_b: F = F(1)) -> Poly:
    """Return ``scale_a*a + scale_b*b`` coefficient by coefficient."""
    out: Poly = {}
    for key in set(a) | set(b):
        value = scale_a * a.get(key, F(0)) + scale_b * b.get(key, F(0))
        if value:
            out[key] = value
    return out


def poly_scale(a: Poly, scalar: F) -> Poly:
    """Multiply every coefficient by ``scalar``."""
    return {key: scalar * value for key, value in a.items() if scalar * value}


def poly_mul(a: Poly, b: Poly, degree: Index) -> Poly:
    """Multiply two series, discarding powers above the threshold ``degree``."""
    out: Poly = {}
    for left, left_value in a.items():
        for right, right_value in b.items():
            key = tuple(left[i] + right[i] for i in range(len(degree)))
            if all(key[i] <= degree[i] for i in range(len(degree))):
                out[key] = out.get(key, F(0)) + left_value * right_value
    return {key: value for key, value in out.items() if value}


def poly_inverse_one_minus(q: Poly, degree: Index) -> Poly:
    """Return the truncated series of ``1/(1-q)``.

    Its coefficients follow recursively from ``(1-q)K=1``.  Increasing total
    degree guarantees that every previously needed coefficient of ``K`` is
    already known.  A constant term of ``q`` is allowed and represents an
    increment that leaves all selected coordinates unchanged.
    """
    zero = (0,) * len(degree)
    q_zero = q.get(zero, F(0))
    inverse: Poly = {zero: F(1) / (F(1) - q_zero)}
    indices = sorted(
        itertools.product(*(range(bound + 1) for bound in degree)),
        key=lambda index: (sum(index), index),
    )
    for index in indices[1:]:
        value = F(0)
        for power, coefficient in q.items():
            if power == zero or not all(power[i] <= index[i] for i in range(len(degree))):
                continue
            remainder = tuple(index[i] - power[i] for i in range(len(degree)))
            value += coefficient * inverse.get(remainder, F(0))
        if value:
            inverse[index] = value / (F(1) - q_zero)
    return inverse


def inverse_d_at(poly: Poly, threshold: Index) -> F:
    """Evaluate the inverse D-transform at ``threshold``.

    The coefficient of ``z^M`` in ``poly/prod_i(1-z_i)`` is the sum of all
    coefficients of ``poly`` indexed coordinatewise below ``M``.
    """
    return sum(
        poly.get(index, F(0))
        for index in itertools.product(*(range(bound + 1) for bound in threshold))
    )


def ordered_partitions(dimension: int) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate weak orders as ordered simultaneous-crossing blocks.

    Coordinates in one block cross together; block order gives the strict
    order between crossing times.  Indices within a block remain increasing,
    so each weak order has exactly one encoding.
    """
    partitions: list[tuple[tuple[int, ...], ...]] = []
    for block_count in range(1, dimension + 1):
        for labels in itertools.product(range(block_count), repeat=dimension):
            if set(labels) != set(range(block_count)):
                continue
            partitions.append(
                tuple(
                    tuple(i for i, label in enumerate(labels) if label == block)
                    for block in range(block_count)
                )
            )
    return partitions


def ordering_label(partition: tuple[tuple[int, ...], ...]) -> str:
    """Convert ``((0, 2), (1,))`` to the human-readable label ``1=3<2``."""
    return "<".join("=".join(str(i + 1) for i in block) for block in partition)


# ---------------------------------------------------------------------------
# Example 1: all weak crossing orders for a dependent d=3 walk
# ---------------------------------------------------------------------------

# Exact joint law of one Bernoulli-vector increment.  Dependence is visible
# because the eight masses do not factor into three Bernoulli marginals.
SIMPLE_SUPPORT: dict[Index, F] = {
    (0, 0, 0): F(10, 100),
    (1, 0, 0): F(12, 100),
    (0, 1, 0): F(10, 100),
    (0, 0, 1): F(8, 100),
    (1, 1, 0): F(12, 100),
    (1, 0, 1): F(10, 100),
    (0, 1, 1): F(13, 100),
    (1, 1, 1): F(25, 100),
}


def projected_pgf(support: dict[Index, F], active: set[int]) -> Poly:
    """Return the PGF obtained by setting coordinates outside ``active`` to 1."""
    dimension = len(next(iter(support)))
    out: Poly = {}
    for increment, probability in support.items():
        power = tuple(increment[i] if i in active else 0 for i in range(dimension))
        out[power] = out.get(power, F(0)) + probability
    return out


def exact_order_probability(
    partition: tuple[tuple[int, ...], ...], threshold: Index
) -> F:
    """Invert the ordered-partition formula at a fixed threshold.

    For a crossing block, ``tail`` is the set of coordinates that have not
    crossed before that block.  The alternating subset sum is its
    inclusion--exclusion numerator; ``1/(1-g_tail)`` is the geometric waiting
    factor.  Their product is exactly the specialization of Theorem 4.
    """
    dimension = len(threshold)
    result: Poly = {(0,) * dimension: F(1)}
    for block_index, block in enumerate(partition):
        tail = set().union(*(set(value) for value in partition[block_index:]))
        next_tail = (
            set().union(*(set(value) for value in partition[block_index + 1 :]))
            if block_index + 1 < len(partition)
            else set()
        )
        # Inclusion--exclusion over subsets of the simultaneous-crossing block.
        numerator: Poly = {}
        for mask in range(1 << len(block)):
            selected = {block[j] for j in range(len(block)) if (mask >> j) & 1}
            numerator = poly_add(
                numerator,
                projected_pgf(SIMPLE_SUPPORT, next_tail | selected),
                scale_b=F((-1) ** len(selected)),
            )
        denominator_inverse = poly_inverse_one_minus(
            projected_pgf(SIMPLE_SUPPORT, tail), threshold
        )
        result = poly_mul(
            result, poly_mul(numerator, denominator_inverse, threshold), threshold
        )
    return inverse_d_at(result, threshold)


def simulate_simple(
    paths: int,
    rng: np.random.Generator,
    threshold: Index = (1, 1, 1),
) -> dict[str, tuple[float, float]]:
    """Estimate each of the 13 weak-order probabilities by simulation."""
    increments = np.asarray(list(SIMPLE_SUPPORT), dtype=np.int16)
    probabilities = np.asarray([float(SIMPLE_SUPPORT[key]) for key in SIMPLE_SUPPORT])
    position = np.zeros((paths, 3), dtype=np.int16)
    crossing = np.full((paths, 3), -1, dtype=np.int16)
    step = 0
    while np.any(crossing < 0):
        step += 1
        unfinished = np.flatnonzero(np.any(crossing < 0, axis=1))
        position[unfinished] += increments[
            rng.choice(len(increments), size=len(unfinished), p=probabilities)
        ]
        # Crossing is strict, so equality with a threshold is not yet exit.
        newly_crossed = ((crossing[unfinished] < 0)
                         & (position[unfinished] > np.asarray(threshold)))
        row, column = np.nonzero(newly_crossed)
        crossing[unfinished[row], column] = step

    # Assign each crossing-index vector to its unique ordered partition.
    estimates: dict[str, tuple[float, float]] = {}
    for partition in ordered_partitions(3):
        mask = np.ones(paths, dtype=bool)
        levels: list[np.ndarray] = []
        for block in partition:
            level = crossing[:, block[0]]
            levels.append(level)
            for coordinate in block[1:]:
                mask &= crossing[:, coordinate] == level
        for left, right in zip(levels, levels[1:]):
            mask &= left < right
        estimate = float(np.mean(mask))
        standard_error = math.sqrt(estimate * (1.0 - estimate) / paths)
        estimates[ordering_label(partition)] = (estimate, standard_error)
    return estimates


# ---------------------------------------------------------------------------
# Example 2: arbitrary-dimensional continuous common-factor model
# ---------------------------------------------------------------------------

CONTINUOUS_COMMON_RATE = 2.0
CONTINUOUS_IDIOSYNCRATIC_RATE = 2.0
CONTINUOUS_THRESHOLD = 4.0
CONTINUOUS_XI = 0.9
CONTINUOUS_DIMENSIONS = (2, 3, 5, 10, 25)


def continuous_survival(dimension: int, step: int) -> float:
    """Return ``P(A_i(step) <= M for every i)``.

    Conditional on the accumulated common factor ``c``, the idiosyncratic
    Gamma sums are independent.  Hence the multivariate inverse reduces to a
    one-dimensional integral: common-factor density times ``[Gamma CDF]^d``.
    """
    if step == 0:
        return 1.0
    common_rate = CONTINUOUS_COMMON_RATE
    idiosyncratic_rate = CONTINUOUS_IDIOSYNCRATIC_RATE
    threshold = CONTINUOUS_THRESHOLD

    def integrand(common_sum: float) -> float:
        residual_cdf = gammainc(
            step, idiosyncratic_rate * (threshold - common_sum)
        )
        if common_sum <= 0.0 or residual_cdf <= 0.0:
            return 0.0
        # The logarithmic form prevents underflow for larger dimensions.
        log_value = (
            step * math.log(common_rate)
            + (step - 1) * math.log(common_sum)
            - common_rate * common_sum
            - gammaln(step)
            + dimension * math.log(residual_cdf)
        )
        return math.exp(log_value)

    value, _ = quad(
        integrand,
        0.0,
        threshold,
        epsabs=2e-13,
        epsrel=2e-12,
        limit=200,
    )
    return float(value)


def exact_continuous_exit(dimension: int) -> tuple[float, float]:
    """Return ``E[rho]`` and ``E[xi^rho]`` from the explicit LC inverse.

    With ``S_n=P(rho>n)``, the identities are ``E[rho]=sum_n S_n`` and
    ``P(rho=n+1)=S_n-S_(n+1)``.
    """
    survival = [1.0]
    for step in range(1, 10_000):
        value = continuous_survival(dimension, step)
        survival.append(value)
        if step >= 6 and value < 1e-14:
            break
    else:
        raise RuntimeError("continuous survival series did not converge")

    mean_exit_index = float(sum(survival))
    pgf = CONTINUOUS_XI * sum(
        CONTINUOUS_XI**step * (survival[step] - survival[step + 1])
        for step in range(len(survival) - 1)
    )
    return mean_exit_index, float(pgf)


def simulate_continuous_exit(
    paths: int, dimension: int, rng: np.random.Generator, batch_size: int = 50_000
) -> dict[str, tuple[float, float]]:
    """Simulate the common-factor model, returning means and standard errors.

    Batching bounds memory use.  At each step, every active path receives one
    common exponential increment and one independent exponential increment per
    coordinate.
    """
    totals = {"rho": 0.0, "rho_sq": 0.0, "pgf": 0.0, "pgf_sq": 0.0}
    completed = 0
    while completed < paths:
        count = min(batch_size, paths - completed)
        position = np.zeros((count, dimension), dtype=np.float64)
        exit_index = np.zeros(count, dtype=np.int16)
        active = np.ones(count, dtype=bool)
        while np.any(active):
            indices = np.flatnonzero(active)
            common = rng.exponential(
                1.0 / CONTINUOUS_COMMON_RATE, size=len(indices)
            )
            idiosyncratic = rng.exponential(
                1.0 / CONTINUOUS_IDIOSYNCRATIC_RATE,
                size=(len(indices), dimension),
            )
            position[indices] += common[:, None] + idiosyncratic
            exit_index[indices] += 1
            crossed = np.any(
                position[indices] > CONTINUOUS_THRESHOLD, axis=1
            )
            active[indices[crossed]] = False

        rho = exit_index.astype(np.float64)
        pgf = CONTINUOUS_XI**rho
        totals["rho"] += float(rho.sum())
        totals["rho_sq"] += float(rho @ rho)
        totals["pgf"] += float(pgf.sum())
        totals["pgf_sq"] += float(pgf @ pgf)
        completed += count

    output: dict[str, tuple[float, float]] = {}
    for name in ("rho", "pgf"):
        mean = totals[name] / paths
        variance = (totals[f"{name}_sq"] - paths * mean**2) / (paths - 1)
        output[name] = (mean, math.sqrt(max(variance, 0.0) / paths))
    return output


# ---------------------------------------------------------------------------
# Example 3: condition monitoring with dependence and a passive signed cost
# ---------------------------------------------------------------------------

# A row contains (name, probability, duration, damage probabilities, credit
# rate).  One shared regime is selected per interval.  Thus the three damage
# indicators are conditionally independent given the regime but dependent
# after the regime is averaged out.
REGIMES = (
    ("normal", F(55, 100), F(2), (F(10, 100), F(6, 100), F(8, 100)), F(3, 2)),
    ("strained", F(30, 100), F(3, 2), (F(28, 100), F(22, 100), F(25, 100)), F(9, 10)),
    ("severe", F(15, 100), F(1), (F(55, 100), F(45, 100), F(50, 100)), F(1, 5)),
)
DAMAGE_COST = (F(2), F(3), F(4))
RELIABILITY_THRESHOLD = (3, 3, 3)


def reliability_increment_polynomials() -> tuple[Poly, Poly, Poly, F, F]:
    """Construct the probability-, time-, and cost-weighted one-step PGFs.

    For example, the coefficient of ``z^x`` in ``time_weighted`` is
    ``E[sigma * 1_{X=x}]``.  These weighted PGFs are derivatives of the joint
    transform and therefore give the required moments after inversion.
    """
    probability: Poly = {}
    time_weighted: Poly = {}
    cost_weighted: Poly = {}
    mean_time = F(0)
    mean_cost = F(0)
    for _, regime_probability, duration, damage_probability, credit_rate in REGIMES:
        for increment in itertools.product((0, 1), repeat=3):
            mass = regime_probability
            for k in range(3):
                mass *= damage_probability[k] if increment[k] else F(1) - damage_probability[k]
            signed_cost = (
                sum(DAMAGE_COST[k] * increment[k] for k in range(3))
                - credit_rate * duration
            )
            probability[increment] = probability.get(increment, F(0)) + mass
            time_weighted[increment] = time_weighted.get(increment, F(0)) + mass * duration
            cost_weighted[increment] = cost_weighted.get(increment, F(0)) + mass * signed_cost
            mean_time += mass * duration
            mean_cost += mass * signed_cost
    return probability, time_weighted, cost_weighted, mean_time, mean_cost


def exact_reliability_moments(
    threshold: Index = RELIABILITY_THRESHOLD,
) -> dict[str, F]:
    """Extract all reported condition-monitoring moments from Theorem 8.

    Each entry of ``specifications`` supplies derivatives of the delayed law,
    ordinary law, terminal factor, and immediate-exit term, followed by the
    derivative with respect to the exit-index variable.  The product rule and
    resolvent derivative are then applied before inversion at the requested
    threshold vector.
    """
    degree = threshold
    zero = (0, 0, 0)
    one: Poly = {zero: F(1)}
    g, g_time, g_cost, mean_time, mean_cost = reliability_increment_polynomials()

    # Delayed initial state: X(0)=(1,0,1), sigma(0)=0, Y(0)=1.
    g0: Poly = {(1, 0, 1): F(1)}
    g0_time: Poly = {}
    g0_cost: Poly = {(1, 0, 1): F(1)}
    initial_time = F(0)
    initial_cost = F(1)
    initial_active = (F(1), F(0), F(1))
    g_active = [
        {increment: mass * increment[k] for increment, mass in g.items() if increment[k]}
        for k in range(3)
    ]
    mean_active = [sum(weighted.values(), F(0)) for weighted in g_active]
    g0_active = [
        poly_scale(g0, initial_active[k]) if initial_active[k] else {}
        for k in range(3)
    ]

    # Neutral specialization of the exit factor: 1-G(z).
    exit_difference = poly_add(one, g, scale_b=F(-1))
    resolvent = poly_inverse_one_minus(g, degree)
    positive_exit = poly_mul(poly_mul(g0, resolvent, degree), exit_difference, degree)
    transform = poly_add(poly_add(one, g0, scale_b=F(-1)), positive_exit)

    specifications = {
        "rho": ({}, {}, {}, {}, F(1)),
        "tau_minus": (
            poly_scale(g0_time, F(-1)),
            poly_scale(g_time, F(-1)),
            {},
            {},
            F(0),
        ),
        "tau_plus": (
            poly_scale(g0_time, F(-1)),
            poly_scale(g_time, F(-1)),
            poly_add({zero: -mean_time}, g_time),
            poly_add({zero: -initial_time}, g0_time),
            F(0),
        ),
        "cost_minus": (g0_cost, g_cost, {}, {}, F(0)),
        "cost_plus": (
            g0_cost,
            g_cost,
            poly_add({zero: mean_cost}, g_cost, scale_b=F(-1)),
            poly_add({zero: initial_cost}, g0_cost, scale_b=F(-1)),
            F(0),
        ),
    }
    for k in range(3):
        specifications[f"active_minus_{k + 1}"] = (
            poly_scale(g0_active[k], F(-1)),
            poly_scale(g_active[k], F(-1)),
            {},
            {},
            F(0),
        )
        specifications[f"active_plus_{k + 1}"] = (
            poly_scale(g0_active[k], F(-1)),
            poly_scale(g_active[k], F(-1)),
            poly_add({zero: -mean_active[k]}, g_active[k]),
            poly_add({zero: -initial_active[k]}, g0_active[k]),
            F(0),
        )

    moments = {"normalization": inverse_d_at(transform, degree)}
    for name, (dq0, dq, dl, initial_difference, dxi) in specifications.items():
        # d(1-G)^(-1)=(1-G)^(-1)(dG)(1-G)^(-1).
        d_resolvent = poly_mul(
            poly_mul(
                resolvent,
                poly_add(poly_scale(g, dxi), dq),
                degree,
            ),
            resolvent,
            degree,
        )
        derivative: Poly = {}
        if dxi:
            derivative = poly_add(derivative, poly_scale(positive_exit, dxi))
        derivative = poly_add(
            derivative,
            poly_mul(poly_mul(dq0, resolvent, degree), exit_difference, degree),
        )
        derivative = poly_add(
            derivative,
            poly_mul(poly_mul(g0, d_resolvent, degree), exit_difference, degree),
        )
        derivative = poly_add(
            derivative, poly_mul(poly_mul(g0, resolvent, degree), dl, degree)
        )
        derivative = poly_add(initial_difference, derivative)
        value = inverse_d_at(derivative, degree)
        moments[name] = -value if name.startswith(("tau_", "active_")) else value
    return moments


def simulate_reliability(
    paths: int,
    rng: np.random.Generator,
    threshold: Index = RELIABILITY_THRESHOLD,
) -> dict[str, tuple[float, float]]:
    """Simulate the condition-monitoring process through its first exit."""
    regime_probability = np.asarray([float(row[1]) for row in REGIMES])
    duration = np.asarray([float(row[2]) for row in REGIMES])
    damage_probability = np.asarray(
        [[float(value) for value in row[3]] for row in REGIMES]
    )
    credit_rate = np.asarray([float(row[4]) for row in REGIMES])
    damage_cost = np.asarray([float(value) for value in DAMAGE_COST])

    position = np.tile(np.asarray([1, 0, 1], dtype=np.int16), (paths, 1))
    signed_cost = np.ones(paths)
    time = np.zeros(paths)
    exit_index = np.zeros(paths, dtype=np.int16)
    pre_cost = np.empty(paths)
    exit_cost = np.empty(paths)
    pre_time = np.empty(paths)
    exit_time = np.empty(paths)
    pre_position = np.empty((paths, 3), dtype=np.int16)
    exit_position = np.empty((paths, 3), dtype=np.int16)
    active = np.ones(paths, dtype=bool)

    while np.any(active):
        indices = np.flatnonzero(active)
        exit_index[indices] += 1
        regime = rng.choice(3, size=len(indices), p=regime_probability)
        increment = (
            rng.random((len(indices), 3)) < damage_probability[regime]
        ).astype(np.int16)
        cost_increment = increment @ damage_cost - credit_rate[regime] * duration[regime]

        # Save left-limit quantities before applying this interval's increment.
        pre_cost[indices] = signed_cost[indices]
        pre_time[indices] = time[indices]
        pre_position[indices] = position[indices]
        position[indices] += increment
        signed_cost[indices] += cost_increment
        time[indices] += duration[regime]

        crossed = np.any(position[indices] > np.asarray(threshold), axis=1)
        completed = indices[crossed]
        exit_cost[completed] = signed_cost[completed]
        exit_time[completed] = time[completed]
        exit_position[completed] = position[completed]
        active[completed] = False

    samples = {
        "rho": exit_index.astype(float),
        "tau_minus": pre_time,
        "tau_plus": exit_time,
        "cost_minus": pre_cost,
        "cost_plus": exit_cost,
    }
    for k in range(3):
        samples[f"active_minus_{k + 1}"] = pre_position[:, k].astype(float)
        samples[f"active_plus_{k + 1}"] = exit_position[:, k].astype(float)
    return {
        name: (float(values.mean()), float(values.std(ddof=1) / math.sqrt(paths)))
        for name, values in samples.items()
    }


RELIABILITY_NAMES = (
    "rho",
    "tau_minus",
    "active_minus_1",
    "active_minus_2",
    "active_minus_3",
    "cost_minus",
    "tau_plus",
    "active_plus_1",
    "active_plus_2",
    "active_plus_3",
    "cost_plus",
)


def save_figures(
    output_directory: Path,
    simulation_paths: int,
    seed: int,
) -> None:
    """Generate parameter-sweep curves with Monte Carlo dots.

    This follows the numerical design of the 2022 paper: formulas are evaluated
    over a range of model parameters, while simulation is used at a modest
    number of points to confirm the predicted curves.
    """
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ("#4C78A8", "#E45756", "#54A24B")

    def write_figure(figure: object, stem: str) -> None:
        figure.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
        figure.savefig(output_directory / f"{stem}.png", dpi=220,
                       bbox_inches="tight")
        plt.close(figure)

    partitions = ordered_partitions(3)

    def exact_order_groups(threshold: Index, grouping: str) -> np.ndarray:
        """Aggregate the 13 exact weak-order probabilities into three groups."""
        totals = np.zeros(3)
        for partition in partitions:
            probability = float(exact_order_probability(partition, threshold))
            if grouping == "epochs":
                group = len(partition) - 1
            elif partition[0] == (0,):
                group = 0                 # coordinate 1 crosses strictly first
            elif 0 in partition[0]:
                group = 1                 # coordinate 1 ties for first
            else:
                group = 2                 # another coordinate crosses first
            totals[group] += probability
        return totals

    def simulated_order_groups(threshold: Index, grouping: str,
                               local_seed: int) -> np.ndarray:
        estimates = simulate_simple(
            simulation_paths, np.random.default_rng(local_seed), threshold
        )
        totals = np.zeros(3)
        for partition in partitions:
            if grouping == "epochs":
                group = len(partition) - 1
            elif partition[0] == (0,):
                group = 0
            elif 0 in partition[0]:
                group = 1
            else:
                group = 2
            totals[group] += estimates[ordering_label(partition)][0]
        return totals

    # Weak-order sweep, panel (a): as a common threshold increases, crossings
    # are spread across more epochs.  Panel (b): delaying only coordinate 1
    # shifts mass from "coordinate 1 first" to "another coordinate first."
    common_thresholds = np.arange(1, 11)
    common_exact = np.vstack([
        exact_order_groups((m, m, m), "epochs") for m in common_thresholds
    ])
    common_simulated = np.vstack([
        simulated_order_groups((m, m, m), "epochs", seed + 100 + m)
        for m in common_thresholds
    ])
    first_thresholds = np.arange(1, 13)
    first_exact = np.vstack([
        exact_order_groups((m, 6, 6), "rank") for m in first_thresholds
    ])
    first_simulated = np.vstack([
        simulated_order_groups((m, 6, 6), "rank", seed + 200 + m)
        for m in first_thresholds
    ])
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for group, label in enumerate(("one epoch", "two epochs", "three epochs")):
        axes[0].plot(common_thresholds, common_exact[:, group], color=colors[group],
                     label=label)
        axes[0].scatter(common_thresholds, common_simulated[:, group],
                        color=colors[group], s=18, zorder=3)
    for group, label in enumerate(("coordinate 1 first", "ties for first",
                                   "another coordinate first")):
        axes[1].plot(first_thresholds, first_exact[:, group], color=colors[group],
                     label=label)
        axes[1].scatter(first_thresholds, first_simulated[:, group],
                        color=colors[group], s=18, zorder=3)
    axes[0].set(xlabel="Common threshold $m$", ylabel="Probability")
    axes[1].set(xlabel="$M_1$ with $M_2=M_3=6$", ylabel="Probability")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    write_figure(figure, "weak_order_threshold_sweeps")

    # Evaluate the continuous formula at every dimension; simulation dots at a
    # smaller subset are enough to show agreement without obscuring the trend.
    dimensions = np.arange(2, 51)
    continuous_exact = np.asarray([exact_continuous_exit(int(d)) for d in dimensions])
    simulated_dimensions = np.asarray((2, 3, 5, 10, 20, 35, 50))
    continuous_simulated = []
    for d in simulated_dimensions:
        result = simulate_continuous_exit(
            simulation_paths, int(d), np.random.default_rng(seed + 300 + int(d))
        )
        continuous_simulated.append((result["rho"][0], result["pgf"][0]))
    continuous_simulated = np.asarray(continuous_simulated)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for axis, column, ylabel in (
        (axes[0], 0, r"$\mathbb{E}[\rho]$"),
        (axes[1], 1, r"$\mathbb{E}[\xi^\rho]$"),
    ):
        axis.plot(dimensions, continuous_exact[:, column], color=colors[0],
                  label="Exact")
        axis.scatter(simulated_dimensions, continuous_simulated[:, column],
                     color="black", s=22, zorder=3, label="Monte Carlo")
        axis.set(xlabel="Dimension $d$", ylabel=ylabel)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    write_figure(figure, "continuous_dimension_sweep")

    # The full-functional sweep varies all active thresholds together.  It
    # displays an exit-index moment, an observation-time moment, and the signed
    # passive cost, hence exercising distinct pieces of the master transform.
    reliability_thresholds = np.arange(1, 9)
    reliability_exact = []
    reliability_simulated = []
    for m in reliability_thresholds:
        threshold = (int(m), int(m), int(m))
        exact = exact_reliability_moments(threshold)
        simulated = simulate_reliability(
            simulation_paths, np.random.default_rng(seed + 400 + int(m)), threshold
        )
        reliability_exact.append(tuple(float(exact[name])
                                       for name in ("rho", "tau_plus", "cost_plus")))
        reliability_simulated.append(tuple(simulated[name][0]
                                           for name in ("rho", "tau_plus", "cost_plus")))
    reliability_exact = np.asarray(reliability_exact)
    reliability_simulated = np.asarray(reliability_simulated)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    for axis, column, ylabel in (
        (axes[0], 0, r"$\mathbb{E}[\rho]$"),
        (axes[1], 1, r"$\mathbb{E}[\tau_\rho]$"),
        (axes[2], 2, r"$\mathbb{E}[P(\rho)]$"),
    ):
        axis.plot(reliability_thresholds, reliability_exact[:, column],
                  color=colors[0], label="Exact")
        axis.scatter(reliability_thresholds, reliability_simulated[:, column],
                     color="black", s=20, zorder=3, label="Monte Carlo")
        axis.set(xlabel="Common threshold $m$", ylabel=ylabel)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    write_figure(figure, "reliability_threshold_sweep")

    print("\nPARAMETER-SWEEP ENDPOINTS")
    print("weak-order epochs at m=1:", common_exact[0])
    print("weak-order epochs at m=10:", common_exact[-1])
    print("coordinate-1 rank at M1=1:", first_exact[0])
    print("coordinate-1 rank at M1=12:", first_exact[-1])
    print("continuous d=2:", continuous_exact[0])
    print("continuous d=50:", continuous_exact[-1])
    print("reliability m=1:", reliability_exact[0])
    print("reliability m=8:", reliability_exact[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the exact calculations and Monte Carlo checks."
    )
    parser.add_argument("--paths", type=int, default=500_000,
                        help="number of Monte Carlo paths (default: 500000)")
    parser.add_argument("--seed", type=int, default=20_260_924,
                        help="base random seed")
    parser.add_argument("--figures", action="store_true",
                        help="save PDF and PNG parameter-sweep figures")
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"),
                        help="figure output directory (default: figures)")
    parser.add_argument("--sweep-paths", type=int, default=10_000,
                        help="Monte Carlo paths per point in parameter sweeps "
                             "(default: 10000)")
    args = parser.parse_args()
    print("FINITE-SUPPORT WEAK-ORDER EXAMPLE")
    simple_simulation = simulate_simple(args.paths, np.random.default_rng(args.seed))
    simple_exact: dict[str, float] = {}
    exact_sum = F(0)
    for partition in ordered_partitions(3):
        label = ordering_label(partition)
        exact = exact_order_probability(partition, (1, 1, 1))
        simple_exact[label] = float(exact)
        exact_sum += exact
        estimate, standard_error = simple_simulation[label]
        print(
            f"{label:9s} exact={float(exact):.8f} "
            f"mc={estimate:.8f} se={standard_error:.8f}"
        )
    print(f"exact sum={exact_sum}")

    print("\nARBITRARY-D CONTINUOUS EXAMPLE")
    continuous_rng = np.random.default_rng(args.seed + 1)
    continuous_rows: list[tuple[int, float, float, float, float, float, float]] = []
    for dimension in CONTINUOUS_DIMENSIONS:
        exact_mean, exact_pgf = exact_continuous_exit(dimension)
        simulation = simulate_continuous_exit(
            args.paths, dimension, continuous_rng
        )
        mean_estimate, mean_se = simulation["rho"]
        pgf_estimate, pgf_se = simulation["pgf"]
        continuous_rows.append(
            (dimension, exact_mean, mean_estimate, mean_se,
             exact_pgf, pgf_estimate, pgf_se)
        )
        print(
            f"d={dimension:2d} "
            f"E[rho]={exact_mean:.8f} mc={mean_estimate:.8f} se={mean_se:.8f} "
            f"E[xi^rho]={exact_pgf:.8f} mc={pgf_estimate:.8f} se={pgf_se:.8f}"
        )

    print("\nRELIABILITY EXAMPLE")
    reliability_exact = exact_reliability_moments()
    reliability_simulation = simulate_reliability(
        args.paths, np.random.default_rng(args.seed)
    )
    print(f"normalization={reliability_exact['normalization']}")
    for name in RELIABILITY_NAMES:
        estimate, standard_error = reliability_simulation[name]
        print(
            f"{name:10s} exact={float(reliability_exact[name]):.8f} "
            f"mc={estimate:.8f} se={standard_error:.8f}"
        )

    if args.figures:
        save_figures(args.figure_dir, args.sweep_paths, args.seed)
        print(f"\nFigures written to {args.figure_dir.resolve()}")


if __name__ == "__main__":
    main()
