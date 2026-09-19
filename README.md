# How Snowball Sampling Distorts Inequality

Code and aggregated data for the paper

> **How Snowball Sampling Distorts Inequality: Calibrating Attention-Concentration Estimates on the Bluesky Follow Graph**
> Fernando Broncano, Aurora Cuartero, Alberto López-Trigo, Jesús A. Torrecilla-Pinero and Andrés Caro.
> *Electronics* (MDPI), Special Issue "Deep Learning and Data Analytics Applications in Social Networks".

Inequality measures such as the Gini coefficient are routinely computed on social graphs
reconstructed by sampling, because most platforms do not expose the full network. This
repository contains the code that asks how faithfully snowball sampling followed by
induced-subgraph densification recovers the concentration of attention, and that answers it
on synthetic populations with known ground truth, on a complete early snapshot of Bluesky,
and on the present-day platform enumerated in full.

The headline result is that the bias changes sign around a true Gini of about 0.3, saturates
near 0.63, is governed by the sampled fraction `f = N/N_pop`, and is not repaired by changing
the sampler. On Bluesky a snowball crawl gives an in-degree Gini of 0.62 where a uniform
sample of the 19,952,290-account relay census gives 0.93.

## Requirements

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or later. Scripts are run from the repository root and locate `data/`, `results/`
and `runs/` there regardless of the working directory.

## Reproducing the paper

### Collection from Bluesky (public API, no authentication)

```bash
# Enumerate the relay census once (~12 min, 16 workers). Writes data/relay_dids.txt.
python scripts/uniform_sample.py --enumerate --workers 16

# Five snowball replications: seeds + collect + densify + analyse.
# Writes per-replication node tables and reports into runs/run_{0..4}/, which is local
# output and is not published; its aggregate is in results/snowball_replications.json.
python scripts/run_study.py --runs 5 --n 3000 --k 1000 --seeds-per-run 10

# The uniform reference: draw from the census and read per-node follower counters
python scripts/uniform_sample.py --big --n 100000   # headline estimate
python scripts/uniform_sample.py --n 5000 --reps 5  # five draws, for the scatter
```

### Synthetic experiments, analysis and figures

```bash
python scripts/bias_study.py         # bias curve on populations of known true Gini
python scripts/bias_full.py          # multi-metric bias, design sensitivity, calibration
python scripts/recalibrate.py        # sampled-fraction dependence and scale invariance
python scripts/scale_up.py           # scale invariance up to N=100k, small f
python scripts/alt_samplers.py       # snowball vs MHRW vs uniform, two estimands
python scripts/null_models.py        # from-scratch null comparison (the invalid one)
python scripts/decisive_null.py      # like-for-like null comparison (the valid one)
python scripts/extra_checks.py       # out-degree model sensitivity, prescribed vs realised, tail fit
python scripts/validate_fullgraph.py # ground-truth validation on the 2023 Bluesky snapshot
python scripts/sensitivity.py        # sensitivity to the follow cap K and to N
python scripts/descriptive.py        # dataset descriptives + robustness figure
python scripts/plot_lorenz.py        # the three Lorenz curves
```

JSON summaries and figures are written to `results/`.

## Where each result in the paper comes from

| Paper | Script | Output |
| --- | --- | --- |
| Table 1 (bias curve) | `bias_study.py` | `results/bias_study.json` |
| Table 2 (null models) | `null_models.py`, `decisive_null.py` | `results/null_models_agg.json`, `results/decisive_null.json` |
| Table 3 (samplers × estimands) | `alt_samplers.py` | `results/alt_samplers.json` |
| Table 4 (concentration on Bluesky) | `run_study.py`, `uniform_sample.py` | `results/snowball_replications.json`, `results/uniform_sample.json`, `results/uniform_big.json` |
| Table 5 (size of each replication) | `run_study.py` | `results/snowball_replications.json` |
| Figure 1 (the collected sample) | `descriptive.py` | `results/fig_dataset.png` |
| Figure 2 (sampled vs true Gini) | `bias_study.py` | `results/fig_bias.png` |
| Figure 3 (three measures) | `bias_full.py` | `results/fig_bias_metrics.png` |
| Figure 4 (sampling design) | `bias_full.py` | `results/fig_design.png` |
| Figure 5 (sampled fraction) | `recalibrate.py`, `scale_up.py` | `results/fig_fraction.png` |
| Figure 6 (Lorenz curves) | `plot_lorenz.py` | `results/fig_lorenz.png` |
| Figure 7 (real-graph validation) | `validate_fullgraph.py` | `results/fig_realvalidation.png` |
| Figure 8 (robustness) | `sensitivity.py`, `descriptive.py` | `results/fig_robustness.png` |

## Layout

- `scripts/` — all Python code.
  - Collection: `sample_seeds.py`, `collect.py`, `densify.py`, `run_study.py`, `uniform_sample.py`.
  - Analysis: `analyze.py`, `descriptive.py`, `sensitivity.py`, `plot_lorenz.py`.
  - Experiments: `bias_study.py`, `bias_full.py`, `recalibrate.py`, `scale_up.py`,
    `alt_samplers.py`, `null_models.py`, `decisive_null.py`, `extra_checks.py`,
    `validate_fullgraph.py`.
  - Release: `pseudonymise.py`, the filter applied to every artefact before publication.
- `results/` — JSON summary of every experiment and the figures of the paper. Also
  `snowball_replications.json`, the aggregate of the five Bluesky replications that backs
  the snowball rows of Table 4 and the sizes of Table 5, and the raw follower and post
  counters of the uniform samples (`uniform_*_raw.npz`), which carry no identifiers and
  allow any statistic to be recomputed without re-hydrating profiles.

## What is not redistributed, and why

The study processes only data the platform makes publicly available through its read
interface, under the legitimate-interest basis for scientific research of GDPR Articles
6(1)(f) and 89(1). Consistent with that basis, no individual records are published:

- **The relay census** (`data/relay_dids.txt`, 19,952,290 account identifiers) and the
  **snowball node and edge tables** (`*.parquet`) are not included. Publishing either would
  amount to redistributing individual records. Both are reproducible from the code: the
  enumeration rebuilds the census in about twelve minutes and `run_study.py` repeats the
  crawl.
- **Account identifiers** in the released JSON are replaced by opaque labels. `pseudonymise.py`
  maps each handle or DID to `acct_<12 hex>`, the truncated SHA-256 of the identifier salted
  with 32 random bytes that are generated per invocation and never written out, so the mapping
  cannot be inverted by dictionary attack over the public account list.
- **The per-replication working directory** `runs/` is not published. Its reports carry the
  seed accounts and the node tables of each crawl. Everything the paper draws from it is in
  `results/snowball_replications.json`, which holds only aggregates and sizes.
- **Two ranked per-account listings** produced by `analyze.py` (`top_10_hubs`,
  `top_10_by_power`) are dropped outright rather than relabelled, since a pseudonym does not
  hide the account at the head of a follower ranking. Neither feeds any figure or table.

Because the platform changes continuously, a census rebuilt later will not be identical to
the one used here. The figures should be read as a measurement of Bluesky on its collection
date, 10 September 2026, rather than as a fixed dataset.

The early Bluesky snapshot used for the ground-truth validation in `validate_fullgraph.py` is
a third-party dataset (`andrewconner/bluesky_profiles`, snapshot of 23 April 2023) and is
obtainable from its original source.

## A note on the schema

The analysis keys that `analyze.py` writes into each replication's `report.json` are
organised by the research questions of the original study design: `RQ1_production` (inequality of posting), `RQ2_attention`
(inequality of in-degree within the sample), `RQ2b_follower_distribution` (the global
follower counters), `RQ3_overlap` (whether producers and receivers coincide), `RQ4_concentration`
(the top-percentile shares) and `RQ_power` (PageRank and betweenness). The paper reports a
subset of these.

## Citation

If you use this code, please cite the paper:

> Broncano, F.; Cuartero, A.; López-Trigo, A.; Torrecilla-Pinero, J.A.; Caro, A.
> How Snowball Sampling Distorts Inequality: Calibrating Attention-Concentration Estimates
> on the Bluesky Follow Graph. *Electronics* **2026**.

The archived release carries a DOI from Zenodo.

## License

MIT, see `LICENSE`.
