import csv
import math
import pathlib
import sys

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from ga_core import evaluar_fitness, run_ga


def test_evaluar_fitness_coincide_con_calculo_directo():
    X, y = load_breast_cancer(return_X_y=True)
    cromosoma = [1, 0] * 15  # 15 de 30 características activas
    indices = [i for i, g in enumerate(cromosoma) if g == 1]
    scores = cross_val_score(GaussianNB(), X[:, indices], y, cv=5, scoring="accuracy")
    esperado = round(0.9 * scores.mean() - 0.1 * (len(indices) / 30), 6)
    obtenido = evaluar_fitness(cromosoma, X, y, cv_folds=5, alpha=0.9, beta=0.1)
    assert math.isclose(obtenido, esperado, abs_tol=1e-9)


def test_evaluar_fitness_cromosoma_nulo_es_cero():
    X, y = load_breast_cancer(return_X_y=True)
    assert evaluar_fitness([0] * 30, X, y, cv_folds=5, alpha=0.9, beta=0.1) == 0.0


PARAMS_TEST = dict(
    P=20, G_max=8, p_cross=0.8, p_mut=0.2, p_mut_bit=1 / 30,
    k_tourn=5, elite_size=2, p_init=0.5, eps_stop=0.001, N_stop=10,
    alpha=0.9, beta=0.1, cv_folds=5,
)


def test_run_ga_invariantes_y_series():
    X, y = load_breast_cancer(return_X_y=True)
    res = run_ga(X, y, PARAMS_TEST, seed=42)
    g = res["generaciones_ejecutadas"]
    # longitudes de series coherentes con nº de generaciones
    assert len(res["series"]["mejor_fitness"]) == g
    assert len(res["series"]["hamming_media_norm"]) == g
    # el mejor global no decrece (elitismo) y |S*| ≤ D
    assert res["mejor_fitness"] == max(res["series"]["mejor_fitness"])
    assert 0 < res["mejor_n_caracteristicas"] <= 30
    # diversidad en rango válido
    assert all(0.0 <= h <= 0.5 for h in res["series"]["hamming_media_norm"])


def test_run_ga_determinista():
    X, y = load_breast_cancer(return_X_y=True)
    r1 = run_ga(X, y, PARAMS_TEST, seed=7)
    r2 = run_ga(X, y, PARAMS_TEST, seed=7)
    assert r1["mejor_cromosoma"] == r2["mejor_cromosoma"]
    assert r1["series"]["mejor_fitness"] == r2["series"]["mejor_fitness"]


def test_run_ga_evaluador_inyectado_equivale_a_serie():
    X, y = load_breast_cancer(return_X_y=True)

    def evaluador_serie(pop):
        return [evaluar_fitness(c, X, y, 5, 0.9, 0.1) for c in pop]

    r_def = run_ga(X, y, PARAMS_TEST, seed=1)
    r_iny = run_ga(X, y, PARAMS_TEST, seed=1, evaluador_poblacion=evaluador_serie)
    assert r_def["series"]["mejor_fitness"] == r_iny["series"]["mejor_fitness"]


ARRH_PARAMS = dict(
    P=100, G_max=5, p_cross=0.8, p_mut=0.2, p_mut_bit=1 / 279,
    k_tourn=5, elite_size=5, p_init=0.5, eps_stop=0.001, N_stop=15,
    alpha=0.9, beta=0.1, cv_folds=3,
)


@pytest.mark.slow
def test_repro_arrhythmia_gen1():
    """Reproduce la generación 1 de results/ga_full_arrhythmia_series.csv."""
    raiz = pathlib.Path(__file__).resolve().parents[1]
    sys_path_src = str(raiz / "src")
    if sys_path_src not in sys.path:
        sys.path.insert(0, sys_path_src)
    from datasets_loader import cargar_dataset

    X, y, _, _ = cargar_dataset("arrhythmia")
    X = np.ascontiguousarray(X)
    res = run_ga(X, y, ARRH_PARAMS, seed=42)

    with open(raiz / "results" / "ga_full_arrhythmia_series.csv") as f:
        fila1 = next(csv.DictReader(f))

    # genes_medio de la gen 1: solo depende del np.random.seed(42) -> exacto.
    assert math.isclose(res["series"]["genes_medio"][0],
                        float(fila1["genes_medio"]), abs_tol=1e-9)
    # fitness: tolerancia por posible drift entre versiones de scikit-learn.
    assert math.isclose(res["series"]["mejor_fitness"][0],
                        float(fila1["mejor_fitness"]), abs_tol=2e-2)
