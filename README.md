# AFSGAW — Apache-Spark Feature-Selection Genetic-Algorithm Wrapper

A distributed **GA-wrapper framework for automatic feature selection** that parallelizes the
costly population-evaluation stage over **Apache Spark RDDs**.

> Master's Thesis (*Trabajo Fin de Máster*, TFM) — Máster en Data Science.
> Code and inline documentation are in Spanish; this README is the English entry point.

---

## Overview

Feature selection seeks the subset of predictor variables that is most informative to a
classification problem — improving model performance while reducing computational cost and
easing interpretation. It is a **combinatorial problem**: for `D` features there are `2^D`
possible subsets. **Wrapper** methods, which evaluate each subset by training the classifier
itself, offer high predictive power but at a high computational cost.

AFSGAW combines a **binary-encoded genetic algorithm** with manually implemented evolutionary
operators and a **wrapper-based fitness function** that jointly optimizes classification
accuracy and feature-subset size. Population evaluation — the stage that dominates the cost of
the evolutionary cycle — is **parallelized with Apache Spark on its Resilient Distributed
Datasets (RDD) model**.

The wrapper fitness is:

```
Fitness(c) = α · Acc_CV(GNB) − β · (|S| / D)
```

where `Acc_CV(GNB)` is the 5-fold cross-validated accuracy of a **Gaussian Naive Bayes**
classifier over the features active in chromosome `c`, `|S|` is the number of selected
features and `D` the total. The final selection is **validated with a linear SVM** — a
classifier different from the one used during the search — to confirm that it generalizes.

---

## Key features

- **Binary GA-wrapper** with hand-implemented operators: tournament selection, one-point
  crossover, flip mutation and elitism.
- **Distributed fitness evaluation over Spark RDDs.** The evaluator is *injectable*, so the
  GA core is Spark-independent and unit-testable (serial GNB by default, or the Spark RDD
  evaluator supplied from a notebook).
- **Reproducible by design**: fixed seeds (`numpy = 42`, `random = 42`), pinned dependencies
  and per-experiment logging of the run context.
- **Evaluated across the dimensionality spectrum**: **WDBC** (`D = 30`),
  **Arrhythmia** (`D = 279`) and **Gisette** (`D = 5000`).
- **Analysis toolkit**: genetic-diversity tracking per generation, selection-stability
  metrics (Jaccard, Kuncheva) and speed-up / weak-scaling experiments.

---

## Results highlights

| Aspect | Result |
|---|---|
| Dimensionality reduction | **60 %–90 %** while maintaining — often improving — accuracy vs. the full feature set |
| Parallel speed-up | **~2×** with 4 partitions; up to **~4×** as population size grows (weak scaling) |
| Datasets | WDBC (30) · Arrhythmia (279) · Gisette (5000) features |
| Selection stability | Dataset-dependent: robust core (WDBC), bimodal (Arrhythmia), massive redundancy (Gisette) |
| Validation | LinearSVC on a held-out test partition confirms the selection generalizes |

Full write-up in [`documentacion/`](documentacion/); raw artifacts (CSV / JSON / PNG) in
[`results/`](results/).

---

## Repository layout

```
AFSGAW/
├── src/                 # Reusable, Spark-independent Python modules
├── notebooks/           # Experimental workflow (run 00 → 07)
├── data/                # WDBC on disk; Arrhythmia & Gisette fetched from OpenML at runtime
├── results/             # Experiment outputs: CSV, JSON records, PNG plots
├── documentacion/       # Thesis documentation (Markdown + PDF), monthly milestone reports
├── tests/               # pytest unit tests (local only)
├── requirements.txt     # Pinned dependencies
├── setup_project.sh     # One-shot environment + directory scaffolding
├── test_spark_setup.py  # Spark environment verification script
└── REPRODUCIBILIDAD.md  # Reproducibility notes
```

The reusable logic lives in `src/`:

| Module | Purpose |
|---|---|
| `src/spark_config.py` | Spark session helpers for Jupyter (`crear_sesion_spark`, `mostrar_info_spark`, `parar_spark`). |
| `src/datasets_loader.py` | Centralized dataset loading via sklearn / OpenML (`cargar_wdbc`, `cargar_arrhythmia`, `cargar_gisette`, `cargar_dataset`), documenting all preprocessing decisions. |
| `src/ga_core.py` | GA-wrapper core: the fitness function, genetic operators and the main entry point `run_ga(...)` with an injectable population evaluator. |
| `src/ga_metrics.py` | Pure diversity and stability metrics (normalized Hamming, Shannon entropy, unique-genotype fraction, Jaccard, Kuncheva index, selection frequency). |

---

## Requirements

- **Python 3.9** (the thesis environment; `REPRODUCIBILIDAD.md` mentions 3.10+).
- **Java (JRE)** — required by Spark.
- **Apache Spark / PySpark 3.5.0** and the pinned libraries in
  [`requirements.txt`](requirements.txt): `scikit-learn 1.5.0`, `pandas`, `numpy`,
  `findspark`, `jupyterlab`, `matplotlib`, `seaborn`, `plotly`.

**Datasets.** WDBC ships in the repository (`data/wdbc/wdbc.csv`). **Arrhythmia** and
**Gisette** are downloaded on first use via `sklearn.datasets.fetch_openml` and cached
locally, so an internet connection is needed the first time you run notebooks 05 and 06.

---

## Setup

```bash
git clone <repo-url>
cd AFSGAW

python3 -m venv spark_env
source spark_env/bin/activate

pip install -r requirements.txt
```

Alternatively, run the one-shot script, which also creates the `data/` and `results/`
subdirectories:

```bash
./setup_project.sh
```

Optionally verify the Spark environment before running the notebooks:

```bash
python test_spark_setup.py
```

---

## Running the notebooks

The notebooks are the primary way to run the framework. Launch Jupyter from the repository
root (with the virtual environment activated):

```bash
source spark_env/bin/activate
jupyter lab
```

Then open the `notebooks/` folder and **run the notebooks in numeric order**. Each notebook
is self-contained and adds `../src` to `sys.path`, so keeping the repository-root working
directory ensures that relative path resolves correctly.

| Notebook | Purpose |
|---|---|
| `00_spark_setup_template.ipynb` | Spark session smoke test / boilerplate. |
| `01_parallel_fitness_prototype.ipynb` | Validate parallel RDD fitness evaluation with a dummy function (sequential vs. parallel). |
| `02_gnb_fitness_evaluation.ipynb` | Real Gaussian Naive Bayes fitness (5-fold CV) on WDBC. |
| `03_speedup_experiment.ipynb` | Speed-up of parallel fitness vs. population size and Spark partitions. |
| `04_ga_full_version.ipynb` | Full GA cycle (selection, crossover, mutation, elitism) on WDBC. |
| `05_ga_full_arrhythmia.ipynb` | Full GA on Arrhythmia (`D = 279`). |
| `06_ga_full_gisette.ipynb` | Full GA on Gisette (`D = 5000`). |
| `07_diversidad_escalabilidad.ipynb` | Diversity tracking, LinearSVC held-out validation, stability (R = 10 seeds) and scalability. |

While a notebook runs, the **Spark Web UI** is available at
[`http://localhost:4040`](http://localhost:4040) for observability. Outputs (CSV, JSON and
PNG artifacts) are written to `results/`.

---

## Testing

Unit tests for `ga_core` and `ga_metrics` live in `tests/` (kept local) and run with pytest:

```bash
pytest                  # full suite
pytest -m "not slow"    # skip tests that load datasets
```

---

## Reproducibility

All experiments use fixed seeds (`numpy = 42`, `random = 42`), pinned dependencies and
4 Spark partitions by default. See [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md) for details.

---

## About / Academic context

This project is the **Trabajo Final de Máster (TFM)** of **Francisco Mateos** for a Máster en Ingeniería de Software.

Main contributions:

- A **distributed, modular and reproducible GA-wrapper framework** for feature selection.
- A **rigorous evaluation methodology** combining a held-out partition, diversity tracking
  and stability analysis.
- An **empirical characterization of the trade-offs** between dimensionality reduction,
  accuracy and computational cost, with practical guidelines for parallelization with
  Spark RDDs.

See [`documentacion/`](documentacion/) for the full thesis write-up.
