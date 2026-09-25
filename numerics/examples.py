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


def continuous_survival(
    dimension: int,
    step: int,
    threshold: float = CONTINUOUS_THRESHOLD,
) -> float:
    """Return ``P(A_i(step) <= M for every i)``.

    Conditional on the accumulated common factor ``c``, the idiosyncratic
    Gamma sums are independent.  Hence the multivariate inverse reduces to a
    one-dimensional integral: common-factor density times ``[Gamma CDF]^d``.
    """
    if step == 0:
        return 1.0
    common_rate = CONTINUOUS_COMMON_RATE
    idiosyncratic_rate = CONTINUOUS_IDIOSYNCRATIC_RATE

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


def exact_continuous_exit(
    dimension: int,
    threshold: float = CONTINUOUS_THRESHOLD,
) -> tuple[float, float]:
    """Return ``E[rho]`` and ``E[xi^rho]`` from the explicit LC inverse.

    With ``S_n=P(rho>n)``, the identities are ``E[rho]=sum_n S_n`` and
    ``P(rho=n+1)=S_n-S_(n+1)``.
    """
    survival = [1.0]
    for step in range(1, 10_000):
        value = continuous_survival(dimension, step, threshold)
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
    totals = {
        "rho": 0.0,
        "rho_sq": 0.0,
        "pgf": 0.0,
        "pgf_sq": 0.0,
        "exit_by_two": 0.0,
        "exit_by_two_sq": 0.0,
    }
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
        exit_by_two = (rho <= 2).astype(np.float64)
        totals["rho"] += float(rho.sum())
        totals["rho_sq"] += float(rho @ rho)
        totals["pgf"] += float(pgf.sum())
        totals["pgf_sq"] += float(pgf @ pgf)
        totals["exit_by_two"] += float(exit_by_two.sum())
        totals["exit_by_two_sq"] += float(exit_by_two @ exit_by_two)
        completed += count

    output: dict[str, tuple[float, float]] = {}
    for name in ("rho", "pgf", "exit_by_two"):
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


def exact_reliability_profile(
    threshold: Index,
    tolerance: float = 1e-13,
) -> dict[str, np.ndarray]:
    """Return survival and joint cause/regime exit probabilities.

    The finite-state recursion retains every sub-threshold active position.
    Absorbing transitions are classified by the coordinates crossing on that
    transition and by the shared operating regime that generated it.
    """
    regime_laws = []
    for _, regime_probability, _, damage_probability, _ in REGIMES:
        increments = []
        for increment in itertools.product((0, 1), repeat=3):
            conditional_mass = F(1)
            for k in range(3):
                conditional_mass *= (
                    damage_probability[k]
                    if increment[k]
                    else F(1) - damage_probability[k]
                )
            increments.append((increment, float(conditional_mass)))
        regime_laws.append((float(regime_probability), increments))

    states: dict[Index, float] = {(1, 0, 1): 1.0}
    survival = [1.0]
    causes = np.zeros(4)       # coordinates 1, 2, 3 alone; then a tie
    terminal_regimes = np.zeros(3)
    joint_cause_regime = np.zeros((4, 3))
    cumulative_causes = [np.zeros(4)]
    for _ in range(10_000):
        next_states: dict[Index, float] = {}
        causes_this_step = np.zeros(4)
        for position, state_mass in states.items():
            for regime_index, (regime_mass, increments) in enumerate(regime_laws):
                for increment, conditional_mass in increments:
                    mass = state_mass * regime_mass * conditional_mass
                    new_position = tuple(
                        position[k] + increment[k] for k in range(3)
                    )
                    crossed = tuple(
                        k for k in range(3) if new_position[k] > threshold[k]
                    )
                    if crossed:
                        cause = crossed[0] if len(crossed) == 1 else 3
                        causes[cause] += mass
                        causes_this_step[cause] += mass
                        terminal_regimes[regime_index] += mass
                        joint_cause_regime[cause, regime_index] += mass
                    else:
                        next_states[new_position] = (
                            next_states.get(new_position, 0.0) + mass
                        )
        states = next_states
        remaining = sum(states.values())
        survival.append(remaining)
        cumulative_causes.append(cumulative_causes[-1] + causes_this_step)
        if remaining < tolerance:
            break
    else:
        raise RuntimeError("reliability survival recursion did not converge")
    return {
        "survival": np.asarray(survival),
        "causes": causes,
        "terminal_regimes": terminal_regimes,
        "joint_cause_regime": joint_cause_regime,
        "cumulative_causes": np.vstack(cumulative_causes),
    }


def simulate_reliability_samples(
    paths: int,
    rng: np.random.Generator,
    threshold: Index = RELIABILITY_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Simulate paths and retain the exit elements needed for diagnostics."""
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
    exit_regime = np.empty(paths, dtype=np.int8)
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
        exit_regime[completed] = regime[crossed]
        active[completed] = False

    samples = {
        "rho": exit_index.astype(float),
        "tau_minus": pre_time,
        "tau_plus": exit_time,
        "cost_minus": pre_cost,
        "cost_plus": exit_cost,
        "exit_regime": exit_regime,
    }
    for k in range(3):
        samples[f"active_minus_{k + 1}"] = pre_position[:, k].astype(float)
        samples[f"active_plus_{k + 1}"] = exit_position[:, k].astype(float)
    return samples


def simulate_reliability(
    paths: int,
    rng: np.random.Generator,
    threshold: Index = RELIABILITY_THRESHOLD,
) -> dict[str, tuple[float, float]]:
    """Return Monte Carlo means and standard errors of the exit elements."""
    samples = simulate_reliability_samples(paths, rng, threshold)
    return {
        name: (float(values.mean()), float(values.std(ddof=1) / math.sqrt(paths)))
        for name, values in samples.items()
        if name != "exit_regime"
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
    """Generate exact parameter landscapes and a Monte Carlo-checked sweep.

    The first two figures expose genuinely multivariate structure over two
    parameter axes.  The final figure follows the numerical design of the 2022
    paper, with Monte Carlo points checking the exact threshold curves.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

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

    def exact_order_metrics(threshold: Index) -> np.ndarray:
        """Summarize the exact weak-order law by three interpretable events."""
        coordinate_one_first = 0.0
        tied_first = 0.0
        all_distinct = 0.0
        for partition in partitions:
            probability = float(exact_order_probability(partition, threshold))
            if partition[0] == (0,):
                coordinate_one_first += probability
            if len(partition[0]) > 1:
                tied_first += probability
            if len(partition) == 3:
                all_distinct += probability
        return np.asarray((coordinate_one_first, tied_first, all_distinct))

    # Weak-order landscape over two thresholds, with the third held fixed.
    # Each cell is an exact aggregation of the 13 canonical weak orderings.
    order_thresholds = np.arange(1, 13)
    order_landscape = np.empty((3, len(order_thresholds), len(order_thresholds)))
    for row, threshold_two in enumerate(order_thresholds):
        for column, threshold_one in enumerate(order_thresholds):
            order_landscape[:, row, column] = exact_order_metrics(
                (int(threshold_one), int(threshold_two), 6)
            )
    order_titles = (
        r"$\mathbb{P}(\rho_1<\min\{\rho_2,\rho_3\})$",
        r"$\mathbb{P}(|S_1|>1)$",
        r"$\mathbb{P}(\rho_1,\rho_2,\rho_3\ \mathrm{all\ distinct})$",
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=True, sharey=True)
    for axis, values, title in zip(axes, order_landscape, order_titles):
        image = axis.imshow(
            values, origin="lower", extent=(0.5, 12.5, 0.5, 12.5),
            cmap="viridis", aspect="equal",
        )
        contours = axis.contour(
            order_thresholds, order_thresholds, values, colors="white",
            linewidths=0.6, alpha=0.8,
        )
        axis.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        axis.plot(6, 6, marker="x", color="white", markersize=6,
                  markeredgewidth=1.4)
        axis.set(title=title, xlabel="$M_1$", ylabel="$M_2$")
        axis.tick_params(axis="y", labelleft=True)
        figure.colorbar(image, ax=axis, shrink=0.82, fraction=0.045, pad=0.018)
    figure.suptitle("Weak-order landscape with $M_3=6$", y=0.99)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.15,
                           top=0.82, wspace=0.12)
    write_figure(figure, "weak_order_threshold_landscape")

    # Continuous-model landscape: dimension and threshold are both varied.
    continuous_dimensions = np.arange(2, 51, 2)
    continuous_thresholds = np.linspace(1.0, 8.0, 15)
    mean_exit = np.empty((len(continuous_thresholds), len(continuous_dimensions)))
    exit_by_two = np.empty_like(mean_exit)
    for row, threshold in enumerate(continuous_thresholds):
        for column, dimension in enumerate(continuous_dimensions):
            mean_exit[row, column] = exact_continuous_exit(
                int(dimension), float(threshold)
            )[0]
            exit_by_two[row, column] = 1.0 - continuous_survival(
                int(dimension), 2, float(threshold)
            )
    dimension_grid, threshold_grid = np.meshgrid(
        continuous_dimensions, continuous_thresholds
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
    for axis, values, title, color_map in (
        (axes[0], mean_exit, r"$\mathbb{E}[\rho]$", "viridis"),
        (axes[1], exit_by_two, r"$\mathbb{P}(\rho\leq2)$", "magma"),
    ):
        image = axis.pcolormesh(
            dimension_grid, threshold_grid, values, shading="auto", cmap=color_map
        )
        contours = axis.contour(
            dimension_grid, threshold_grid, values, colors="white",
            linewidths=0.6, alpha=0.8,
        )
        axis.clabel(contours, inline=True, fontsize=7, fmt="%.2g")
        axis.set(title=title, xlabel="Dimension $d$")
        figure.colorbar(image, ax=axis, shrink=0.9)
    axes[0].set_ylabel("Common threshold $M$")
    figure.suptitle("Continuous common-factor exit landscape", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    write_figure(figure, "continuous_exit_landscape")

    # Standard reliability diagnostics.  Exact finite-state recursions supply
    # survival, competing-risk, and terminal-regime probabilities.  Simulation
    # points check those curves and supply the passive-cost quantiles.
    reliability_thresholds = np.arange(1, 9)
    reliability_profiles = []
    reliability_samples = []
    for threshold_value in reliability_thresholds:
        threshold = (int(threshold_value),) * 3
        reliability_profiles.append(exact_reliability_profile(threshold))
        reliability_samples.append(simulate_reliability_samples(
            simulation_paths,
            np.random.default_rng(seed + 400 + int(threshold_value)),
            threshold,
        ))

    cause_exact = np.vstack([profile["causes"] for profile in reliability_profiles])
    regime_exact = np.vstack([
        profile["terminal_regimes"] for profile in reliability_profiles
    ])
    cause_simulated = np.zeros_like(cause_exact)
    regime_simulated = np.zeros_like(regime_exact)
    cost_quantiles = np.empty((len(reliability_thresholds), 5))
    for row, (threshold_value, samples) in enumerate(zip(
        reliability_thresholds, reliability_samples
    )):
        crossed = np.column_stack([
            samples[f"active_plus_{k}"] > threshold_value for k in range(1, 4)
        ])
        crossing_count = crossed.sum(axis=1)
        for coordinate in range(3):
            cause_simulated[row, coordinate] = np.mean(
                crossed[:, coordinate] & (crossing_count == 1)
            )
        cause_simulated[row, 3] = np.mean(crossing_count > 1)
        regime_simulated[row] = np.bincount(
            samples["exit_regime"], minlength=3
        ) / simulation_paths
        cost_quantiles[row] = np.quantile(
            samples["cost_plus"], (0.05, 0.25, 0.50, 0.75, 0.95)
        )

    reliability_colors = ("#4C78A8", "#E45756", "#54A24B", "#B279A2")
    cause_labels = ("coordinate 1", "coordinate 2", "coordinate 3", "tie")
    baseline_row = 2                         # common threshold M=3
    baseline_profile = reliability_profiles[baseline_row]
    baseline_samples = reliability_samples[baseline_row]
    baseline_crossed = np.column_stack([
        baseline_samples[f"active_plus_{k}"] > 3 for k in range(1, 4)
    ])
    baseline_count = baseline_crossed.sum(axis=1)
    simulated_cause = np.full(simulation_paths, 3, dtype=np.int8)
    for coordinate in range(3):
        simulated_cause[
            baseline_crossed[:, coordinate] & (baseline_count == 1)
        ] = coordinate

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))

    # Survival through successive inspection epochs.
    for threshold_value in (2, 4, 6):
        row = threshold_value - 1
        survival = reliability_profiles[row]["survival"]
        epochs = np.arange(len(survival))
        axes[0, 0].step(
            epochs, survival, where="post", color=reliability_colors[row // 2],
            label=rf"$M={threshold_value}$",
        )
        marker_epochs = np.arange(0, min(len(survival), 81), 4)
        simulated_survival = np.asarray([
            np.mean(reliability_samples[row]["rho"] > epoch)
            for epoch in marker_epochs
        ])
        axes[0, 0].scatter(
            marker_epochs, simulated_survival,
            color=reliability_colors[row // 2], s=12, zorder=3,
        )
    axes[0, 0].set(
        xlabel="Inspection epoch $n$", ylabel=r"$\mathbb{P}(\rho>n)$",
        title="System survival",
    )
    axes[0, 0].set_xlim(0, 80)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(frameon=False)

    # Cause-specific cumulative incidence combines failure timing and cause.
    cumulative_exact = baseline_profile["cumulative_causes"]
    cumulative_epochs = np.arange(len(cumulative_exact))
    marker_epochs = np.arange(0, min(len(cumulative_exact), 61), 4)
    for cause, (label, color) in enumerate(zip(cause_labels, reliability_colors)):
        axes[0, 1].step(
            cumulative_epochs, cumulative_exact[:, cause], where="post",
            color=color, label=label,
        )
        cumulative_simulated = np.asarray([
            np.mean((baseline_samples["rho"] <= epoch)
                    & (simulated_cause == cause))
            for epoch in marker_epochs
        ])
        axes[0, 1].scatter(
            marker_epochs, cumulative_simulated, color=color, s=12, zorder=3
        )
    axes[0, 1].set(
        xlabel="Inspection epoch $n$", ylabel=r"$\mathbb{P}(\rho\leq n,J=j)$",
        title="Cause-specific cumulative incidence ($M=3$)",
    )
    axes[0, 1].set_xlim(0, 60)
    axes[0, 1].set_ylim(-0.02, 0.55)
    axes[0, 1].legend(frameon=False, ncol=2)

    # A single enlarged heat map carries the joint probabilities and their row
    # conditionals.  Marginal cause probabilities and prior regime weights are
    # included in the tick labels for context.
    joint = baseline_profile["joint_cause_regime"]
    conditional = joint / joint.sum(axis=1, keepdims=True)
    light_joint_map = LinearSegmentedColormap.from_list(
        "light_joint", ("#fffaf0", "#fee8b0", "#fdbb84", "#ef8a62")
    )
    joint_image = axes[1, 0].imshow(
        joint, cmap=light_joint_map, aspect="auto", vmin=0.0
    )
    for row in range(joint.shape[0]):
        for column in range(joint.shape[1]):
            axes[1, 0].text(
                column, row,
                f"{joint[row, column]:.3f}\n({100 * conditional[row, column]:.0f}%)",
                ha="center", va="center", fontsize=9,
                color="#202020",
            )
    prior_regime = np.asarray([float(regime[1]) for regime in REGIMES])
    axes[1, 0].set_xticks(
        np.arange(3),
        [f"{regime[0]}\nprior {prior_regime[k]:.2f}"
         for k, regime in enumerate(REGIMES)],
    )
    axes[1, 0].set_yticks(
        np.arange(4),
        [f"{label}\n{baseline_profile['causes'][k]:.3f}"
         for k, label in enumerate(cause_labels)],
    )
    axes[1, 0].set(
        xlabel="Terminal regime", ylabel="Cause of exit",
        title="Joint probability (row-conditional percentage), $M=3$",
    )
    axes[1, 0].set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axes[1, 0].set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axes[1, 0].grid(which="minor", color="white", linewidth=1.0)
    axes[1, 0].tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(joint_image, ax=axes[1, 0], shrink=0.84,
                    label="Joint probability")

    # The signed passive component is nonmonotone pathwise, so its distribution
    # is more revealing than its Wald-governed mean.
    axes[1, 1].fill_between(
        reliability_thresholds, cost_quantiles[:, 0], cost_quantiles[:, 4],
        color=colors[0], alpha=0.16, label="5th--95th percentiles",
    )
    axes[1, 1].fill_between(
        reliability_thresholds, cost_quantiles[:, 1], cost_quantiles[:, 3],
        color=colors[0], alpha=0.34, label="25th--75th percentiles",
    )
    axes[1, 1].plot(
        reliability_thresholds, cost_quantiles[:, 2], color=colors[0],
        marker="o", markersize=3, label="median",
    )
    axes[1, 1].axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        xlabel="Common threshold $M$", ylabel=r"Exit cost $P_\rho$",
        title="Passive-cost risk (Monte Carlo)",
    )
    axes[1, 1].legend(frameon=False)
    figure.tight_layout()
    write_figure(figure, "reliability_diagnostics")

    # Comparison sheet for selecting the most informative cause/regime panels.
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.0))

    # Alternative A: cause probabilities as the common threshold varies.
    for column, (label, color) in enumerate(zip(cause_labels, reliability_colors)):
        axes[0, 0].plot(
            reliability_thresholds, cause_exact[:, column], color=color, label=label
        )
        axes[0, 0].scatter(
            reliability_thresholds, cause_simulated[:, column],
            color=color, s=12, zorder=3,
        )
    axes[0, 0].set(
        title="Cause versus threshold", xlabel="Common threshold $M$",
        ylabel="Probability", ylim=(-0.02, 1.02),
    )
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)

    # Alternative B: standard cause-specific cumulative incidence.
    for cause, (label, color) in enumerate(zip(cause_labels, reliability_colors)):
        axes[0, 1].step(
            cumulative_epochs, cumulative_exact[:, cause], where="post",
            color=color, label=label,
        )
        cumulative_simulated = np.asarray([
            np.mean((baseline_samples["rho"] <= epoch)
                    & (simulated_cause == cause))
            for epoch in marker_epochs
        ])
        axes[0, 1].scatter(
            marker_epochs, cumulative_simulated, color=color, s=12, zorder=3
        )
    axes[0, 1].set(
        title="Cause-specific cumulative incidence ($M=3$)",
        xlabel="Inspection epoch $n$", ylabel=r"$\mathbb{P}(\rho\leq n,J=j)$",
        xlim=(0, 60), ylim=(-0.02, 0.55),
    )

    # Alternative C: terminal-regime curves over the threshold.
    for column, (regime, color) in enumerate(zip(REGIMES, reliability_colors)):
        axes[0, 2].plot(
            reliability_thresholds, regime_exact[:, column],
            color=color, label=regime[0],
        )
        axes[0, 2].scatter(
            reliability_thresholds, regime_simulated[:, column],
            color=color, s=12, zorder=3,
        )
        axes[0, 2].axhline(
            float(regime[1]), color=color, linestyle=":", linewidth=1.0
        )
    axes[0, 2].set(
        title="Terminal regime versus threshold",
        xlabel="Common threshold $M$", ylabel="Probability", ylim=(-0.02, 1.02),
    )
    axes[0, 2].legend(frameon=False, fontsize=8)

    # Alternative D: a direct prior-versus-terminal comparison.
    regime_positions = np.arange(3)
    bar_width = 0.36
    axes[1, 0].bar(
        regime_positions - bar_width / 2, prior_regime, bar_width,
        color="#BBBBBB", label="Prior",
    )
    axes[1, 0].bar(
        regime_positions + bar_width / 2,
        baseline_profile["terminal_regimes"], bar_width,
        color=colors[0], label="At exit",
    )
    axes[1, 0].set_xticks(
        regime_positions, [regime[0] for regime in REGIMES]
    )
    axes[1, 0].set(
        title="Prior versus terminal regime ($M=3$)", ylabel="Probability",
        ylim=(0.0, 0.65),
    )
    axes[1, 0].legend(frameon=False)

    def annotated_heatmap(axis: object, values: np.ndarray, title: str,
                          color_map: str = "magma") -> None:
        """Draw a small probability heat map with numerical cell labels."""
        image = axis.imshow(values, cmap=color_map, aspect="auto", vmin=0.0)
        midpoint = 0.55 * float(values.max())
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column, row, f"{values[row, column]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if values[row, column] > midpoint else "black",
                )
        axis.set_xticks(np.arange(3), [regime[0] for regime in REGIMES])
        axis.set_yticks(np.arange(4), cause_labels)
        axis.set(title=title, xlabel="Terminal regime", ylabel="Cause of exit")
        figure.colorbar(image, ax=axis, shrink=0.82)

    joint = baseline_profile["joint_cause_regime"]
    annotated_heatmap(
        axes[1, 1], joint,
        r"Joint $\mathbb{P}(J=j,R=r)$ ($M=3$)",
    )
    conditional = joint / joint.sum(axis=1, keepdims=True)
    annotated_heatmap(
        axes[1, 2], conditional,
        r"Conditional $\mathbb{P}(R=r\mid J=j)$ ($M=3$)",
        color_map="viridis",
    )
    figure.tight_layout()
    write_figure(figure, "reliability_panel_alternatives")

    print("\nPARAMETER-SWEEP ENDPOINTS")
    print("weak-order landscape ranges:",
          [(float(values.min()), float(values.max())) for values in order_landscape])
    print("continuous E[rho] range:", float(mean_exit.min()), float(mean_exit.max()))
    print("continuous P(rho<=2) range:",
          float(exit_by_two.min()), float(exit_by_two.max()))
    print("reliability cause probabilities at M=1:", cause_exact[0])
    print("reliability cause probabilities at M=8:", cause_exact[-1])
    print("terminal regime probabilities at M=1:", regime_exact[0])
    print("terminal regime probabilities at M=8:", regime_exact[-1])


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
