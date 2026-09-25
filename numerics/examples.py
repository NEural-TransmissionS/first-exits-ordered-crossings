#!/usr/bin/env python3
"""Reproduce the three numerical illustrations in main.tex.

The exact calculations use truncated multivariate power-series arithmetic with
Fraction coefficients.  This is precisely the inverse D-operator: after a
threshold transform is divided by prod_i(1-z_i), the requested fixed-threshold
value is its z^M coefficient.
"""

from __future__ import annotations

import argparse
import itertools
import math
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaln


Index = tuple[int, ...]
Poly = dict[Index, F]


def poly_add(a: Poly, b: Poly, scale_a: F = F(1), scale_b: F = F(1)) -> Poly:
    out: Poly = {}
    for key in set(a) | set(b):
        value = scale_a * a.get(key, F(0)) + scale_b * b.get(key, F(0))
        if value:
            out[key] = value
    return out


def poly_scale(a: Poly, scalar: F) -> Poly:
    return {key: scalar * value for key, value in a.items() if scalar * value}


def poly_mul(a: Poly, b: Poly, degree: Index) -> Poly:
    out: Poly = {}
    for left, left_value in a.items():
        for right, right_value in b.items():
            key = tuple(left[i] + right[i] for i in range(len(degree)))
            if all(key[i] <= degree[i] for i in range(len(degree))):
                out[key] = out.get(key, F(0)) + left_value * right_value
    return {key: value for key, value in out.items() if value}


def poly_inverse_one_minus(q: Poly, degree: Index) -> Poly:
    """Return the truncated series of 1/(1-q)."""
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
    """Coefficient of z^threshold in poly/prod_i(1-z_i)."""
    return sum(
        poly.get(index, F(0))
        for index in itertools.product(*(range(bound + 1) for bound in threshold))
    )


def ordered_partitions(dimension: int) -> list[tuple[tuple[int, ...], ...]]:
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
    return "<".join("=".join(str(i + 1) for i in block) for block in partition)


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
    dimension = len(next(iter(support)))
    out: Poly = {}
    for increment, probability in support.items():
        power = tuple(increment[i] if i in active else 0 for i in range(dimension))
        out[power] = out.get(power, F(0)) + probability
    return out


def exact_order_probability(
    partition: tuple[tuple[int, ...], ...], threshold: Index
) -> F:
    dimension = len(threshold)
    result: Poly = {(0,) * dimension: F(1)}
    for block_index, block in enumerate(partition):
        tail = set().union(*(set(value) for value in partition[block_index:]))
        next_tail = (
            set().union(*(set(value) for value in partition[block_index + 1 :]))
            if block_index + 1 < len(partition)
            else set()
        )
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


def simulate_simple(paths: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
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
        newly_crossed = (crossing[unfinished] < 0) & (position[unfinished] > 1)
        row, column = np.nonzero(newly_crossed)
        crossing[unfinished[row], column] = step

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


CONTINUOUS_COMMON_RATE = 2.0
CONTINUOUS_IDIOSYNCRATIC_RATE = 2.0
CONTINUOUS_THRESHOLD = 4.0
CONTINUOUS_XI = 0.9
CONTINUOUS_DIMENSIONS = (2, 3, 5, 10, 25)


def continuous_survival(dimension: int, step: int) -> float:
    """Return P(A_i(step) <= M for every i) in the common-factor model."""
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
    """Return E[rho] and E[xi^rho] from the explicit LC inverse."""
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
    """Simulate the arbitrary-dimensional continuous common-factor model."""
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


REGIMES = (
    ("normal", F(55, 100), F(2), (F(10, 100), F(6, 100), F(8, 100)), F(3, 2)),
    ("strained", F(30, 100), F(3, 2), (F(28, 100), F(22, 100), F(25, 100)), F(9, 10)),
    ("severe", F(15, 100), F(1), (F(55, 100), F(45, 100), F(50, 100)), F(1, 5)),
)
DAMAGE_COST = (F(2), F(3), F(4))
RELIABILITY_THRESHOLD = (3, 3, 3)


def reliability_increment_polynomials() -> tuple[Poly, Poly, Poly, F, F]:
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


def exact_reliability_moments() -> dict[str, F]:
    degree = RELIABILITY_THRESHOLD
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
    paths: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
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

        pre_cost[indices] = signed_cost[indices]
        pre_time[indices] = time[indices]
        pre_position[indices] = position[indices]
        position[indices] += increment
        signed_cost[indices] += cost_increment
        time[indices] += duration[regime]

        crossed = np.any(position[indices] > np.asarray(RELIABILITY_THRESHOLD), axis=1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20_260_924)
    args = parser.parse_args()
    print("FINITE-SUPPORT WEAK-ORDER EXAMPLE")
    simple_simulation = simulate_simple(args.paths, np.random.default_rng(args.seed))
    exact_values: list[F] = []
    for partition in ordered_partitions(3):
        label = ordering_label(partition)
        exact = exact_order_probability(partition, (1, 1, 1))
        exact_values.append(exact)
        estimate, standard_error = simple_simulation[label]
        print(
            f"{label:9s} exact={float(exact):.8f} "
            f"mc={estimate:.8f} se={standard_error:.8f}"
        )
    print(f"exact sum={sum(exact_values, F(0))}")

    print("\nARBITRARY-D CONTINUOUS EXAMPLE")
    continuous_rng = np.random.default_rng(args.seed + 1)
    for dimension in CONTINUOUS_DIMENSIONS:
        exact_mean, exact_pgf = exact_continuous_exit(dimension)
        simulation = simulate_continuous_exit(
            args.paths, dimension, continuous_rng
        )
        mean_estimate, mean_se = simulation["rho"]
        pgf_estimate, pgf_se = simulation["pgf"]
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
    for name in (
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
    ):
        estimate, standard_error = reliability_simulation[name]
        print(
            f"{name:10s} exact={float(reliability_exact[name]):.8f} "
            f"mc={estimate:.8f} se={standard_error:.8f}"
        )


if __name__ == "__main__":
    main()
