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

The figures are written to `figures/` by default. Exact heat maps expose the
two-parameter structure of the first two examples, while exact curves and
Monte Carlo points check the condition-monitoring example. They show:

1. how strict-first, tied-first, and fully distinct crossing probabilities vary
   over $(M_1,M_2)$ when $M_3=6$;
2. how the mean exit index and probability of exit by the second observation
   vary jointly with dimension $d$ and common threshold $M$;
3. system survival, competing causes of exit, terminal-regime selection, and
   passive-cost risk in the condition-monitoring example.

The script also writes a reliability-panel comparison sheet containing both
marginal and joint cause--regime views for selecting the final manuscript plot.

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
