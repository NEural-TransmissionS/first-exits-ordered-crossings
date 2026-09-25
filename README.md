# Numerical Code for First Exits and Ordered Crossings

This repository contains the accompanying numerical code for the paper *First Exits and Ordered Crossings in Multivariate Renewal--Reward Processes*.

The paper derives a joint transform for first exit of a multivariate renewal--reward process, resolves the transform by the weak ordering of coordinate crossing indices, and proves the compact unrestricted formula in arbitrary finite dimension. The numerical illustrations include exact weak-order probabilities, a continuous common-factor example, and a condition-monitoring example with dependent active coordinates and a signed passive component.

## Repository contents

- `numerics/examples.py`: accompanying script reproducing the numerical tables
- `requirements.txt`: Python dependencies used for the accompanying script

The manuscript, drafts, editorial notes, backups, and broader research workspace are intentionally not included.

## Reproduce the numerical examples

Create a Python environment and install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all three examples with the paper's default of 500,000 Monte Carlo paths:

```bash
python numerics/examples.py
```

To reproduce the candidate figures in both vector PDF and PNG formats:

```bash
python numerics/examples.py --figures
```

The figures are written to `figures/` by default. They use exact curves and
Monte Carlo points across ranges of model parameters, following the numerical
design of the 2022 paper. They show:

1. how the number of distinct crossing epochs changes with a common threshold,
   and how delaying one coordinate changes its probability of crossing first;
2. the dependence of the mean exit index and the probability of exit by the
   second observation on dimension from $d=2$ through $d=50$;
3. how the exit index, terminal-interval duration and signed cost, and active
   positions change as the thresholds vary in the condition-monitoring example.

Each plotted Monte Carlo point uses 10,000 paths by default. Change this
independently of the numerical tables with, for example,

```bash
python numerics/examples.py --figures --sweep-paths 50000
```

For a faster check, reduce the number of paths:

```bash
python numerics/examples.py --paths 10000
```

The script uses fixed random seeds by default. Exact values are computed independently of the Monte Carlo sample size.
