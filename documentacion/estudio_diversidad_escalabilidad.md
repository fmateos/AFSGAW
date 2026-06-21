# Diversidad genética, escalabilidad y validación con LinearSVC — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Fecha:** 2026-06-14

**Goal:** Añadir al proyecto el análisis de diversidad genética (dentro del GA), la validación del subconjunto final con LinearSVC sobre test held-out, la estabilidad de selección (R=10 semillas) y la escalabilidad (algorítmica + speed-up) para los tres datasets (WDBC, Arrhythmia, Gisette).

**Architecture:** Se extrae el GA hoy triplicado en los notebooks 04/05/06 a `src/ga_core.py` con un `run_ga()` **independiente de Spark** (la evaluación de fitness se inyecta como callable; en serie por defecto, con Spark RDD en el notebook). Las métricas de diversidad y estabilidad viven en `src/ga_metrics.py` como funciones puras de NumPy, testeables sin Spark. Un notebook orquestador `07_diversidad_escalabilidad.ipynb` produce todos los artefactos.

**Tech Stack:** Python 3.9, NumPy 1.24, scikit-learn (GaussianNB, LinearSVC, cross_val_score, train_test_split), PySpark 3.5 (RDD broadcast/map), matplotlib, pytest 7.4.

**Spec de referencia:** `documentacion/plan_diversidad_escalabilidad.md`

---

## Parámetros del GA por dataset (fijados desde los experimentos existentes)

| Param | WDBC | Arrhythmia | Gisette |
|---|---|---|---|
| P | 50 | 100 | 100 |
| G_max | 50 | 100 | 100 |
| p_cross | 0.8 | 0.8 | 0.8 |
| p_mut | 0.2 | 0.2 | 0.2 |
| p_mut_bit | 1/30 | 1/279 | 1/5000 |
| k_tourn | 5 | 5 | 5 |
| elite_size | 2 | 5 | 5 |
| p_init | 0.5 | 0.5 | 0.1 |
| eps_stop | 0.001 | 0.001 | 0.001 |
| N_stop | 10 | 15 | 15 |
| alpha / beta | 0.9 / 0.1 | 0.9 / 0.1 | 0.9 / 0.1 |
| cv_folds | 5 | 3 | 5 |

`seed` base = 42. `p_mut_bit` se calcula como `1.0/D` tras cargar cada dataset.

## Decisión metodológica registrada (held-out)

Los experimentos actuales (nb 04/05/06) buscan sobre el dataset **completo**. El nuevo notebook 07 introduce un **split train/test estratificado antes del GA**: el GA busca solo sobre *train*, y `S*` se valida con GNB y LinearSVC sobre *test* no visto. Esto **cambia los resultados** respecto a los artefactos `ga_full_*` existentes (que se conservan intactos). Los artefactos nuevos llevan prefijos distintos y son los "definitivos" del capítulo experimental. La reproducción del GA antiguo (para validar que el refactor es fiel) se hace sobre datos completos, por separado.

## Mapa de ficheros

| Fichero | Responsabilidad |
|---|---|
| `src/ga_metrics.py` | **Crear.** Funciones puras: `diversity_metrics`, `jaccard_mean`, `kuncheva_index`, `selection_frequency`. |
| `src/ga_core.py` | **Crear.** `evaluar_fitness`, operadores genéticos, `run_ga()` (sin dependencia de Spark). |
| `tests/conftest.py` | **Crear.** Añade `src/` al `sys.path` para pytest. |
| `tests/test_ga_metrics.py` | **Crear.** Tests unitarios de las métricas (sin Spark). |
| `tests/test_ga_core.py` | **Crear.** Tests del GA: fitness, invariantes, determinismo, diversidad. |
| `notebooks/07_diversidad_escalabilidad.ipynb` | **Crear.** Orquesta diversidad + validación + estabilidad + speed-up + consolidación. |
| `results/<ds>_holdout_series.csv` | Series por generación con columnas de diversidad. |
| `results/<ds>_diversidad.png` | Curva de diversidad por dataset + panel comparativo. |
| `results/<ds>_estabilidad.csv` / `.png` | Frecuencias de selección + Jaccard/Kuncheva. |
| `results/<ds>_speedup.csv` / `.png` | T(n), speed-up, eficiencia por dataset. |
| `results/escalabilidad_algoritmica.csv` | Tabla cross-dataset consolidada. |

`<ds>` ∈ {`wdbc`, `arrhythmia`, `gisette`}.

---

# Fase 0 — Plumbing de tests

### Task 0: Configurar pytest para encontrar `src/`

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Crear el conftest**

```python
# tests/conftest.py
import sys
from pathlib import Path

# Permite `import ga_core, ga_metrics` desde los tests sin instalar el paquete.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
```

- [ ] **Step 2: Verificar que pytest arranca (sin tests aún)**

Run: `cd "$(git rev-parse --show-toplevel)" && python -m pytest tests/ -q`
Expected: `no tests ran` (exit 5) — confirma que pytest funciona y el conftest carga.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: configurar pytest para resolver imports desde src/"
```

---

# Fase 1 — `ga_metrics.py` (funciones puras, TDD)

### Task 1: Métricas de diversidad de la población

**Files:**
- Create: `src/ga_metrics.py`
- Test: `tests/test_ga_metrics.py`

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_ga_metrics.py
import math
import numpy as np
from ga_metrics import diversity_metrics


def test_diversity_poblacion_maxima():
    # Mitad todo-ceros, mitad todo-unos -> p_j = 0.5 en todos los loci.
    pop = [[1, 1, 0, 0], [0, 0, 1, 1]]
    m = diversity_metrics(pop)
    assert math.isclose(m["hamming_media_norm"], 0.5, abs_tol=1e-9)
    assert math.isclose(m["entropia_media"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["n_unicos_frac"], 1.0, abs_tol=1e-9)


def test_diversity_poblacion_nula():
    # Todos idénticos -> sin diversidad.
    pop = [[1, 0, 1], [1, 0, 1]]
    m = diversity_metrics(pop)
    assert math.isclose(m["hamming_media_norm"], 0.0, abs_tol=1e-9)
    assert math.isclose(m["entropia_media"], 0.0, abs_tol=1e-9)
    assert math.isclose(m["n_unicos_frac"], 0.5, abs_tol=1e-9)
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `python -m pytest tests/test_ga_metrics.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'ga_metrics'`.

- [ ] **Step 3: Implementar `diversity_metrics`**

```python
# src/ga_metrics.py
"""Métricas de diversidad genética y estabilidad de selección (funciones puras).

Independientes de Spark. Se separan del bucle del GA (ga_core) para poder
testearlas de forma aislada con pytest.
"""
import numpy as np


def diversity_metrics(poblacion):
    """Diversidad genotípica de una población binaria.

    Parameters
    ----------
    poblacion : array-like, shape (P, D)
        Matriz población × características; cada fila es un cromosoma binario.

    Returns
    -------
    dict
        hamming_media_norm : distancia de Hamming media normalizada entre pares,
            (1/D)·Σ_j 2·p_j·(1−p_j), con p_j la frecuencia alélica del locus j.
            Rango [0, 0.5]; 0 = población idéntica.
        entropia_media : entropía de Shannon media por locus (bits), en [0, 1].
        n_unicos_frac : nº de genotipos distintos / P, en (0, 1].
    """
    pop = np.asarray(poblacion, dtype=np.int8)
    P, D = pop.shape
    p = pop.mean(axis=0)                       # frecuencia alélica por locus
    pq = p * (1.0 - p)

    hamming_media_norm = float(2.0 * pq.sum() / D)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            (p > 0.0) & (p < 1.0),
            -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p),
            0.0,
        )
    entropia_media = float(term.sum() / D)

    n_unicos = len({tuple(row) for row in pop.tolist()})
    n_unicos_frac = float(n_unicos / P)

    return {
        "hamming_media_norm": hamming_media_norm,
        "entropia_media": entropia_media,
        "n_unicos_frac": n_unicos_frac,
    }
```

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `python -m pytest tests/test_ga_metrics.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ga_metrics.py tests/test_ga_metrics.py
git commit -m "feat(metrics): diversidad genotípica de la población (Hamming/entropía/únicos)"
```

### Task 2: Métricas de estabilidad de selección

**Files:**
- Modify: `src/ga_metrics.py` (añadir funciones)
- Test: `tests/test_ga_metrics.py` (añadir tests)

- [ ] **Step 1: Añadir los tests que fallan**

```python
# añadir a tests/test_ga_metrics.py
from ga_metrics import jaccard_mean, kuncheva_index, selection_frequency


def test_jaccard_identicos_y_disjuntos():
    assert math.isclose(jaccard_mean([[1, 1, 0, 0], [1, 1, 0, 0]]), 1.0, abs_tol=1e-9)
    assert math.isclose(jaccard_mean([[1, 1, 0, 0], [0, 0, 1, 1]]), 0.0, abs_tol=1e-9)


def test_jaccard_solapamiento_parcial():
    # inter {0,1}=2, union {0,1,2}=3 -> 2/3
    assert math.isclose(jaccard_mean([[1, 1, 1, 0], [1, 1, 0, 0]]), 2 / 3, abs_tol=1e-9)


def test_kuncheva_identicos_y_disjuntos():
    # idénticos -> 1.0
    assert math.isclose(kuncheva_index([[1, 1, 0, 0], [1, 1, 0, 0]], D=4), 1.0, abs_tol=1e-9)
    # disjuntos de tamaño k=2 en D=4: (0 - 1) / (2 - 1) = -1.0
    assert math.isclose(kuncheva_index([[1, 1, 0, 0], [0, 0, 1, 1]], D=4), -1.0, abs_tol=1e-9)


def test_selection_frequency():
    f = selection_frequency([[1, 0, 1], [1, 1, 0]])
    assert np.allclose(f, [1.0, 0.5, 0.5])
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `python -m pytest tests/test_ga_metrics.py -q`
Expected: FAIL con `ImportError: cannot import name 'jaccard_mean'`.

- [ ] **Step 3: Implementar las tres funciones**

```python
# añadir a src/ga_metrics.py
def _as_index_sets(subsets):
    """Convierte una matriz binaria (R, D) en lista de conjuntos de índices."""
    arr = np.asarray(subsets, dtype=np.int8)
    return [set(np.flatnonzero(row).tolist()) for row in arr]


def jaccard_mean(subsets):
    """Índice de Jaccard medio entre todos los pares de subconjuntos.

    subsets : array-like (R, D) binaria. Devuelve la media de
    |S_i ∩ S_j| / |S_i ∪ S_j| sobre los R(R−1)/2 pares. NaN si R < 2.
    Tolera subconjuntos de distinto tamaño.
    """
    sets = _as_index_sets(subsets)
    R = len(sets)
    if R < 2:
        return float("nan")
    vals = []
    for i in range(R):
        for j in range(i + 1, R):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            vals.append(len(inter) / len(union) if union else 1.0)
    return float(np.mean(vals))


def kuncheva_index(subsets, D):
    """Índice de consistencia de Kuncheva medio entre pares (corrige por azar).

    Para dos subconjuntos de tamaños k_i, k_j de D características con r comunes,
    se usa la forma generalizada (Lustgarten 2009), que coincide con la fórmula
    clásica de Kuncheva (2007) cuando k_i = k_j:

        I_C(i, j) = (r − k_i·k_j/D) / (min(k_i, k_j) − k_i·k_j/D)

    Rango ≈ [−1, 1]; 0 = solapamiento esperado por azar. NaN si R < 2.
    """
    sets = _as_index_sets(subsets)
    R = len(sets)
    if R < 2:
        return float("nan")
    vals = []
    for i in range(R):
        for j in range(i + 1, R):
            ki, kj = len(sets[i]), len(sets[j])
            r = len(sets[i] & sets[j])
            expected = ki * kj / D
            denom = min(ki, kj) - expected
            if denom == 0:
                continue
            vals.append((r - expected) / denom)
    return float(np.mean(vals)) if vals else float("nan")


def selection_frequency(subsets):
    """Frecuencia de selección por característica a lo largo de R ejecuciones.

    subsets : array-like (R, D) binaria. Devuelve array (D,) con la fracción
    de runs que seleccionó cada característica.
    """
    arr = np.asarray(subsets, dtype=np.float64)
    return arr.mean(axis=0)
```

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `python -m pytest tests/test_ga_metrics.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ga_metrics.py tests/test_ga_metrics.py
git commit -m "feat(metrics): estabilidad de selección (Jaccard, Kuncheva, frecuencia)"
```

---

# Fase 2 — `ga_core.py` (GA reutilizable, TDD)

### Task 3: Función de fitness wrapper

**Files:**
- Create: `src/ga_core.py`
- Test: `tests/test_ga_core.py`

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_ga_core.py
import math
import numpy as np
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
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `python -m pytest tests/test_ga_core.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'ga_core'`.

- [ ] **Step 3: Implementar fitness + operadores (cabecera del módulo)**

```python
# src/ga_core.py
"""GA-wrapper binario para selección de características (núcleo reutilizable).

Refactor del bucle evolutivo replicado en los notebooks 04/05/06. Es
independiente de Spark: la evaluación de fitness se inyecta como callable
(`evaluador_poblacion`); por defecto evalúa en serie con GNB. Esto permite:
  - testear el GA sin levantar Spark,
  - inyectar el evaluador Spark RDD desde el notebook,
  - reutilizar el mismo evaluador con distinto paralelismo (speed-up).
"""
import random
import time
import warnings

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from ga_metrics import diversity_metrics


def evaluar_fitness(cromosoma, X, y, cv_folds, alpha, beta):
    """Fitness wrapper: alpha·Acc_CV(GNB) − beta·(|S|/D). 0.0 si |S| = 0."""
    indices = [i for i, g in enumerate(cromosoma) if g == 1]
    D = len(cromosoma)
    if not indices:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            GaussianNB(), X[:, indices], y, cv=cv_folds, scoring="accuracy"
        )
    return round(alpha * scores.mean() - beta * (len(indices) / D), 6)


# ----- operadores genéticos (mismos que nb 04/05/06) -----
def seleccion_torneo(pop, fit, k):
    cand = random.sample(range(len(pop)), k)
    return list(pop[max(cand, key=lambda i: fit[i])])


def cruce_un_punto(p1, p2):
    punto = random.randint(1, len(p1) - 1)
    return p1[:punto] + p2[punto:], p2[:punto] + p1[punto:]


def mutacion_flip(ind, p_mut_bit):
    return [1 - g if random.random() < p_mut_bit else g for g in ind]


def elitismo(pop, fit, e):
    idx = sorted(range(len(fit)), key=lambda i: fit[i], reverse=True)[:e]
    return [list(pop[i]) for i in idx]


def siguiente_generacion(pop, fit, params):
    P, k = params["P"], params["k_tourn"]
    nueva = elitismo(pop, fit, params["elite_size"])
    while len(nueva) < P:
        p1 = seleccion_torneo(pop, fit, k)
        p2 = seleccion_torneo(pop, fit, k)
        if random.random() < params["p_cross"]:
            h1, h2 = cruce_un_punto(p1, p2)
        else:
            h1, h2 = list(p1), list(p2)
        if random.random() < params["p_mut"]:
            h1 = mutacion_flip(h1, params["p_mut_bit"])
        if random.random() < params["p_mut"]:
            h2 = mutacion_flip(h2, params["p_mut_bit"])
        nueva.append(h1)
        if len(nueva) < P:
            nueva.append(h2)
    return nueva
```

- [ ] **Step 4: Ejecutar y ver pasar (fitness)**

Run: `python -m pytest tests/test_ga_core.py::test_evaluar_fitness_coincide_con_calculo_directo tests/test_ga_core.py::test_evaluar_fitness_cromosoma_nulo_es_cero -q`
Expected: 2 passed. (El test de `run_ga` aún falla porque la función no existe — se implementa en Task 4.)

- [ ] **Step 5: Commit**

```bash
git add src/ga_core.py tests/test_ga_core.py
git commit -m "feat(ga): fitness wrapper y operadores genéticos en src/ga_core"
```

### Task 4: `run_ga()` con registro de diversidad

**Files:**
- Modify: `src/ga_core.py` (añadir `run_ga`)
- Test: `tests/test_ga_core.py` (añadir tests)

- [ ] **Step 1: Añadir los tests que fallan**

```python
# añadir a tests/test_ga_core.py
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
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `python -m pytest tests/test_ga_core.py -q`
Expected: FAIL con `ImportError: cannot import name 'run_ga'` o `AttributeError`.

- [ ] **Step 3: Implementar `run_ga`**

```python
# añadir a src/ga_core.py
SERIES_KEYS = (
    "mejor_fitness", "fitness_medio", "peor_fitness", "genes_medio",
    "t_eval_s", "t_ops_s",
    "hamming_media_norm", "entropia_media", "n_unicos_frac", "std_fitness",
)


def run_ga(X, y, params, seed=42, evaluador_poblacion=None,
           collect_diversity=True, verbose=False):
    """Ejecuta el GA-wrapper completo.

    params keys: P, G_max, p_cross, p_mut, p_mut_bit, k_tourn, elite_size,
                 p_init, eps_stop, N_stop, alpha, beta, cv_folds.
    evaluador_poblacion : callable(pop) -> list[float]. Si None, evalúa en serie
        con GNB (mismo resultado que el evaluador Spark, que es determinista).

    Devuelve dict con mejor_cromosoma, mejor_fitness, mejor_n_caracteristicas,
    generaciones_ejecutadas, motivo_parada, series (dict de listas) y tiempos.
    """
    D = X.shape[1]
    P, G_max = params["P"], params["G_max"]
    alpha, beta, cv = params["alpha"], params["beta"], params["cv_folds"]

    if evaluador_poblacion is None:
        def evaluador_poblacion(pop):
            return [evaluar_fitness(c, X, y, cv, alpha, beta) for c in pop]

    # Semillas en el mismo orden que los notebooks: np.random para la población
    # inicial, random para los operadores.
    np.random.seed(seed)
    pop = [list(np.random.binomial(1, params["p_init"], D)) for _ in range(P)]
    random.seed(seed)

    series = {k: [] for k in SERIES_KEYS}
    hist_mejor = []
    mejor_global, mejor_fit_global = None, -1.0
    gen_parada, motivo = None, None
    t0_total = time.time()

    for gen in range(1, G_max + 1):
        t0 = time.time()
        scores = evaluador_poblacion(pop)
        t_eval = time.time() - t0

        mejor, medio, peor = max(scores), float(np.mean(scores)), min(scores)
        genes_medio = float(np.mean([sum(ind) for ind in pop]))
        hist_mejor.append(mejor)

        series["mejor_fitness"].append(mejor)
        series["fitness_medio"].append(medio)
        series["peor_fitness"].append(peor)
        series["genes_medio"].append(genes_medio)
        series["t_eval_s"].append(t_eval)
        series["std_fitness"].append(float(np.std(scores)))
        if collect_diversity:
            dm = diversity_metrics(pop)
            series["hamming_media_norm"].append(dm["hamming_media_norm"])
            series["entropia_media"].append(dm["entropia_media"])
            series["n_unicos_frac"].append(dm["n_unicos_frac"])
        else:
            for k in ("hamming_media_norm", "entropia_media", "n_unicos_frac"):
                series[k].append(float("nan"))

        idx = scores.index(mejor)
        if mejor > mejor_fit_global:
            mejor_fit_global, mejor_global = mejor, list(pop[idx])

        if gen >= params["N_stop"]:
            ventana = hist_mejor[-params["N_stop"]:]
            delta = max(ventana) - min(ventana)
            if delta < params["eps_stop"]:
                gen_parada = gen
                motivo = (f"convergencia (Δ={delta:.6f} < {params['eps_stop']} "
                          f"en {params['N_stop']} gens)")
                series["t_ops_s"].append(0.0)
                if verbose:
                    print(f"  Gen {gen}: mejor={mejor:.6f} [PARADA]")
                break

        t0 = time.time()
        pop = siguiente_generacion(pop, scores, params)
        series["t_ops_s"].append(time.time() - t0)
        if verbose:
            print(f"  Gen {gen}: mejor={mejor:.6f} media={medio:.6f}")

    t_total = time.time() - t0_total
    if gen_parada is None:
        gen_parada, motivo = G_max, f"G_max alcanzado ({G_max} generaciones)"

    return {
        "mejor_cromosoma": [int(g) for g in mejor_global],
        "mejor_fitness": float(mejor_fit_global),
        "mejor_n_caracteristicas": int(sum(mejor_global)),
        "generaciones_ejecutadas": int(gen_parada),
        "motivo_parada": motivo,
        "series": series,
        "tiempo_total_s": round(t_total, 3),
        "tiempo_eval_total_s": round(sum(series["t_eval_s"]), 3),
        "tiempo_ops_total_s": round(sum(series["t_ops_s"]), 3),
        "tiempo_medio_gen_s": round(t_total / gen_parada, 3),
    }
```

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `python -m pytest tests/test_ga_core.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ga_core.py tests/test_ga_core.py
git commit -m "feat(ga): run_ga con registro de diversidad e inyección de evaluador"
```

### Task 5: Verificación de fidelidad del refactor (reproducción de Arrhythmia, informativa)

**Files:**
- Test: `tests/test_ga_core.py` (añadir test marcado lento)
- Create: `pytest.ini`

> El refactor debe reproducir el comportamiento del nb 05. La 1ª generación depende solo de `np.random.seed(42)` (independiente de la versión de sklearn) y se compara exactamente; el fitness se compara con tolerancia porque puede variar levemente entre versiones de scikit-learn.

- [ ] **Step 1: Añadir el test de reproducción**

```python
# añadir a tests/test_ga_core.py
import csv
import pathlib
import sys
import pytest

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
```

- [ ] **Step 2: Registrar el marker `slow`**

Create: `pytest.ini`
```ini
[pytest]
markers =
    slow: tests que cargan datasets / tardan (excluir con -m "not slow")
```

- [ ] **Step 3: Ejecutar el test de reproducción**

Run: `python -m pytest tests/test_ga_core.py::test_repro_arrhythmia_gen1 -q`
Expected: PASS (requiere Arrhythmia en caché de OpenML; la 1ª vez la descarga).
Si `genes_medio` no coincide exactamente, el refactor alteró el orden de las llamadas a `np.random` → revisar Task 4 Step 3.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ga_core.py pytest.ini
git commit -m "test(ga): verificación de fidelidad del refactor frente a nb 05 (Arrhythmia gen1)"
```

---

# Fase 3 — Notebook 07: scaffolding

> Las celdas se crean con la herramienta NotebookEdit. El notebook trabaja desde `notebooks/`, por lo que las rutas a resultados son `../results/` y a `src/` es `../src`.

### Task 6: Crear el notebook con setup, Spark y configuración de datasets

**Files:**
- Create: `notebooks/07_diversidad_escalabilidad.ipynb`

- [ ] **Step 1: Celda 1 — imports y path**

```python
import sys, os, time, csv, json, warnings
from datetime import datetime
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from spark_config import crear_sesion_spark, mostrar_info_spark, parar_spark
from datasets_loader import cargar_dataset
from ga_core import run_ga, evaluar_fitness
from ga_metrics import jaccard_mean, kuncheva_index, selection_frequency

warnings.filterwarnings('ignore', message='The least populated class in y')
os.makedirs('../results', exist_ok=True)
print('Notebook 07 iniciado:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
```

- [ ] **Step 2: Celda 2 — sesión Spark**

```python
spark = crear_sesion_spark(app_name='TFM_Diversidad_Escalabilidad',
                           master='local[*]', driver_memory='4g',
                           executor_memory='4g', ui_port=4040, verbose=False)
sc = spark.sparkContext
NUM_CORES = sc.defaultParallelism
mostrar_info_spark(spark, ui_port=4040)
print('Cores:', NUM_CORES)
```

- [ ] **Step 3: Celda 3 — configuración de parámetros por dataset**

```python
DATASETS = {
    'wdbc':       dict(P=50,  G_max=50,  k_tourn=5, elite_size=2, p_init=0.5,
                       N_stop=10, cv_folds=5),
    'arrhythmia': dict(P=100, G_max=100, k_tourn=5, elite_size=5, p_init=0.5,
                       N_stop=15, cv_folds=3),
    'gisette':    dict(P=100, G_max=100, k_tourn=5, elite_size=5, p_init=0.1,
                       N_stop=15, cv_folds=5),
}
COMUNES = dict(p_cross=0.8, p_mut=0.2, eps_stop=0.001, alpha=0.9, beta=0.1)
SEED = 42

def construir_params(nombre, D):
    p = dict(COMUNES, **DATASETS[nombre])
    p['p_mut_bit'] = 1.0 / D
    return p
```

- [ ] **Step 4: Celda 4 — helpers de split y evaluador Spark**

```python
def split_holdout(X, y, test_size=0.3, seed=SEED):
    """Split estratificado; cae a no-estratificado si hay clases con 1 muestra."""
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y,
                                random_state=seed)
    except ValueError as e:
        print(f'  [aviso] estratificación imposible ({e}); split sin estratificar')
        return train_test_split(X, y, test_size=test_size, random_state=seed)

def make_evaluador_spark(X_train, y_train, cv, alpha, beta, num_slices):
    """Devuelve un callable(pop)->list[fitness] que evalúa con Spark RDD."""
    X_bc = sc.broadcast(X_train)
    y_bc = sc.broadcast(y_train)
    def _eval(pop):
        rdd = sc.parallelize(pop, numSlices=num_slices)
        return rdd.map(lambda c: evaluar_fitness(c, X_bc.value, y_bc.value,
                                                 cv, alpha, beta)).collect()
    _eval.broadcasts = (X_bc, y_bc)  # para destruir luego
    return _eval
```

- [ ] **Step 5: Verificar ejecución del scaffolding**

Run (manual en Jupyter): ejecutar celdas 1–4.
Expected: Spark arranca, `DATASETS` y helpers definidos sin error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/07_diversidad_escalabilidad.ipynb
git commit -m "feat(nb07): scaffolding — Spark, config por dataset y helpers"
```

---

# Fase 4 — Diversidad (B) + validación held-out con LinearSVC (F)

### Task 7: Ejecutar GA con diversidad y validar S* en test, por dataset

**Files:**
- Modify: `notebooks/07_diversidad_escalabilidad.ipynb` (añadir celdas)

- [ ] **Step 1: Celda 5 — bucle principal de experimentos (diversidad + validación)**

```python
RESULTADOS = {}   # nombre -> dict con métricas para la tabla consolidada

for nombre in ['wdbc', 'arrhythmia', 'gisette']:
    print('=' * 60, f'\n{nombre.upper()}')
    X, y, feat, info = cargar_dataset(nombre)
    X = np.ascontiguousarray(X)
    D = X.shape[1]
    params = construir_params(nombre, D)

    X_tr, X_te, y_tr, y_te = split_holdout(X, y)
    print(f'  D={D}  train={X_tr.shape[0]}  test={X_te.shape[0]}')

    evaluador = make_evaluador_spark(X_tr, y_tr, params['cv_folds'],
                                     params['alpha'], params['beta'], NUM_CORES)
    t0 = time.time()
    res = run_ga(X_tr, y_tr, params, seed=SEED,
                 evaluador_poblacion=evaluador, collect_diversity=True, verbose=True)
    for bc in evaluador.broadcasts:
        bc.destroy()
    print(f'  GA: {res["generaciones_ejecutadas"]} gen, fitness={res["mejor_fitness"]:.6f}, '
          f'|S*|={res["mejor_n_caracteristicas"]}/{D}, {time.time()-t0:.1f}s')

    # --- Validación de S* en test held-out ---
    S = [i for i, g in enumerate(res['mejor_cromosoma']) if g == 1]
    acc_gnb = GaussianNB().fit(X_tr[:, S], y_tr).score(X_te[:, S], y_te)
    svc = make_pipeline(StandardScaler(),
                        LinearSVC(dual='auto', max_iter=5000))
    acc_svc = svc.fit(X_tr[:, S], y_tr).score(X_te[:, S], y_te)
    print(f'  Validación test:  GNB={acc_gnb:.4f}   LinearSVC={acc_svc:.4f}')

    RESULTADOS[nombre] = dict(
        D=D, gen=res['generaciones_ejecutadas'], motivo=res['motivo_parada'],
        n_sel=res['mejor_n_caracteristicas'],
        reduccion=1 - res['mejor_n_caracteristicas'] / D,
        fitness=res['mejor_fitness'], acc_gnb_test=acc_gnb, acc_svc_test=acc_svc,
        t_total=res['tiempo_total_s'], t_gen=res['tiempo_medio_gen_s'],
        cromosoma=res['mejor_cromosoma'], series=res['series'],
    )

    # --- Guardar series con columnas de diversidad ---
    s = res['series']
    with open(f'../results/{nombre}_holdout_series.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['generacion'] + list(s.keys())
        w.writerow(cols)
        for i in range(res['generaciones_ejecutadas']):
            w.writerow([i + 1] + [s[k][i] for k in s.keys()])
    print(f'  Series -> ../results/{nombre}_holdout_series.csv')
```

- [ ] **Step 2: Celda 6 — gráfico de diversidad por dataset**

```python
def plot_diversidad(nombre, series):
    g = list(range(1, len(series['mejor_fitness']) + 1))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(g, series['hamming_media_norm'], 'o-', color='teal', label='Hamming media norm.')
    ax[0].plot(g, series['n_unicos_frac'], 's--', color='darkorange', label='Frac. individuos únicos')
    ax[0].set_xlabel('Generación'); ax[0].set_ylabel('Diversidad'); ax[0].legend()
    ax[0].set_title(f'Diversidad genotípica — {nombre}'); ax[0].grid(alpha=0.3)
    ax[1].plot(g, series['std_fitness'], 'D-', color='purple')
    ax[1].set_xlabel('Generación'); ax[1].set_ylabel('std(fitness)')
    ax[1].set_title(f'Dispersión fenotípica — {nombre}'); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../results/{nombre}_diversidad.png', dpi=150, bbox_inches='tight')
    plt.show()

for nombre in RESULTADOS:
    plot_diversidad(nombre, RESULTADOS[nombre]['series'])
```

- [ ] **Step 3: Celda 7 — panel comparativo de diversidad (3 datasets)**

```python
fig, ax = plt.subplots(figsize=(9, 5))
for nombre, c in zip(['wdbc', 'arrhythmia', 'gisette'], ['steelblue', 'green', 'crimson']):
    s = RESULTADOS[nombre]['series']['hamming_media_norm']
    ax.plot(np.linspace(0, 1, len(s)), s, '-', color=c, linewidth=2,
            label=f"{nombre} (D={RESULTADOS[nombre]['D']})")
ax.set_xlabel('Progreso de generaciones (normalizado)')
ax.set_ylabel('Hamming media normalizada')
ax.set_title('Decaimiento de la diversidad por dataset\n(Gisette parte de p_init=0.1)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../results/diversidad_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 4: Ejecutar las celdas 5–7**

Run (manual en Jupyter): ejecutar 5, 6, 7.
Expected: 3 ejecuciones del GA (WDBC y Arrhythmia en segundos, Gisette ~2 min con held-out), 3 `*_diversidad.png`, 1 `diversidad_comparativa.png`, 3 `*_holdout_series.csv`, y `RESULTADOS` con accuracies de validación.

- [ ] **Step 5: Commit**

```bash
git add notebooks/07_diversidad_escalabilidad.ipynb results/*_diversidad.png results/diversidad_comparativa.png results/*_holdout_series.csv
git commit -m "feat(nb07): diversidad por generación + validación LinearSVC en test held-out"
```

---

# Fase 5 — Estabilidad de selección (C), R = 10

### Task 8: R ejecuciones por dataset y métricas de estabilidad

**Files:**
- Modify: `notebooks/07_diversidad_escalabilidad.ipynb` (añadir celdas)

- [ ] **Step 1: Celda 8 — bucle de estabilidad (R=10 semillas)**

```python
R = 10
ESTABILIDAD = {}

for nombre in ['wdbc', 'arrhythmia', 'gisette']:
    print('=' * 60, f'\nEstabilidad — {nombre} (R={R})')
    X, y, feat, info = cargar_dataset(nombre)
    X = np.ascontiguousarray(X)
    D = X.shape[1]
    params = construir_params(nombre, D)
    X_tr, X_te, y_tr, y_te = split_holdout(X, y)

    subsets = []
    t0 = time.time()
    for s in range(R):
        evaluador = make_evaluador_spark(X_tr, y_tr, params['cv_folds'],
                                         params['alpha'], params['beta'], NUM_CORES)
        res = run_ga(X_tr, y_tr, params, seed=SEED + s,
                     evaluador_poblacion=evaluador, collect_diversity=False)
        for bc in evaluador.broadcasts:
            bc.destroy()
        subsets.append(res['mejor_cromosoma'])
        print(f'  semilla {SEED+s}: |S*|={sum(res["mejor_cromosoma"])}, '
              f'fitness={res["mejor_fitness"]:.6f}')
    print(f'  {R} ejecuciones en {time.time()-t0:.1f}s')

    subsets = np.array(subsets)
    jac = jaccard_mean(subsets)
    kun = kuncheva_index(subsets, D)
    frec = selection_frequency(subsets)
    nucleo = int(np.sum(frec >= 0.8))
    ESTABILIDAD[nombre] = dict(jaccard=jac, kuncheva=kun, frecuencias=frec,
                               nucleo=nucleo, subsets=subsets, feat=feat)
    print(f'  Jaccard medio={jac:.4f}  Kuncheva={kun:.4f}  núcleo(≥8/10)={nucleo}')

    # Guardar CSV de estabilidad (frecuencias por característica)
    with open(f'../results/{nombre}_estabilidad.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['caracteristica', 'frecuencia_seleccion'])
        for i, fr in enumerate(frec):
            w.writerow([feat[i] if i < len(feat) else f'V{i}', round(float(fr), 4)])
```

- [ ] **Step 2: Celda 9 — gráfico de estabilidad (histograma de frecuencias)**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, nombre in zip(axes, ['wdbc', 'arrhythmia', 'gisette']):
    frec = ESTABILIDAD[nombre]['frecuencias']
    ax.hist(frec, bins=np.linspace(0, 1, 11), color='slateblue', edgecolor='black')
    ax.axvline(0.8, color='red', linestyle='--', label='umbral núcleo (0.8)')
    ax.set_xlabel('Frecuencia de selección'); ax.set_ylabel('Nº de características')
    ax.set_title(f"{nombre}\nJaccard={ESTABILIDAD[nombre]['jaccard']:.3f}, "
                 f"Kuncheva={ESTABILIDAD[nombre]['kuncheva']:.3f}")
    ax.legend()
plt.tight_layout()
plt.savefig('../results/estabilidad_frecuencias.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Ejecutar celdas 8–9**

Run (manual en Jupyter).
Expected: WDBC y Arrhythmia en segundos; **Gisette ~22 min**. Genera `*_estabilidad.csv` (3) y `estabilidad_frecuencias.png`. Valores de Jaccard/Kuncheva impresos por dataset.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_diversidad_escalabilidad.ipynb results/*_estabilidad.csv results/estabilidad_frecuencias.png
git commit -m "feat(nb07): estabilidad de selección (R=10) — Jaccard, Kuncheva, frecuencias"
```

---

# Fase 6 — Escalabilidad paralela (E): speed-up por dataset

### Task 9: Medir speed-up serie vs Spark por dataset

**Files:**
- Modify: `notebooks/07_diversidad_escalabilidad.ipynb` (añadir celdas)

- [ ] **Step 1: Celda 10 — harness de speed-up**

```python
def medir_speedup(X_tr, y_tr, params, poblacion, slices_list, repeticiones=3):
    """T_serie y T_spark(n) evaluando una población fija. Devuelve lista de filas."""
    cv, a, b = params['cv_folds'], params['alpha'], params['beta']
    # Baseline en serie (sin Spark)
    ts = []
    for _ in range(repeticiones):
        t0 = time.time()
        [evaluar_fitness(c, X_tr, y_tr, cv, a, b) for c in poblacion]
        ts.append(time.time() - t0)
    t_serie = float(np.median(ts))

    X_bc, y_bc = sc.broadcast(X_tr), sc.broadcast(y_tr)
    filas = [dict(modo='serie', n=1, t=t_serie, speedup=1.0, eficiencia=1.0)]
    for n in slices_list:
        tn = []
        for _ in range(repeticiones):
            t0 = time.time()
            sc.parallelize(poblacion, numSlices=n).map(
                lambda c: evaluar_fitness(c, X_bc.value, y_bc.value, cv, a, b)
            ).collect()
            tn.append(time.time() - t0)
        tmed = float(np.median(tn))
        filas.append(dict(modo='spark', n=n, t=tmed,
                          speedup=t_serie / tmed, eficiencia=(t_serie / tmed) / n))
    X_bc.destroy(); y_bc.destroy()
    return filas

SPEEDUP = {}
for nombre in ['wdbc', 'arrhythmia', 'gisette']:
    X, y, feat, info = cargar_dataset(nombre)
    X = np.ascontiguousarray(X)
    D = X.shape[1]
    params = construir_params(nombre, D)
    X_tr, X_te, y_tr, y_te = split_holdout(X, y)
    np.random.seed(SEED)
    poblacion = [list(np.random.binomial(1, params['p_init'], D))
                 for _ in range(params['P'])]
    filas = medir_speedup(X_tr, y_tr, params, poblacion, slices_list=[1, 2, 4, 8])
    SPEEDUP[nombre] = filas
    print(f'{nombre}: ' + '  '.join(
        f"{f['modo']}{f['n']}={f['t']:.2f}s(S={f['speedup']:.2f})" for f in filas))
    with open(f'../results/{nombre}_speedup.csv', 'w', newline='') as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=['modo', 'n', 't', 'speedup', 'eficiencia'])
        w.writeheader()
        for f in filas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in f.items()})
```

- [ ] **Step 2: Celda 11 — gráfico de speed-up**

```python
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for nombre, c in zip(['wdbc', 'arrhythmia', 'gisette'], ['steelblue', 'green', 'crimson']):
    spark_rows = [f for f in SPEEDUP[nombre] if f['modo'] == 'spark']
    ns = [f['n'] for f in spark_rows]
    ax[0].plot(ns, [f['speedup'] for f in spark_rows], 'o-', color=c, label=nombre)
    ax[1].plot(ns, [f['eficiencia'] for f in spark_rows], 's-', color=c, label=nombre)
ax[0].plot([1, 8], [1, 8], 'k:', alpha=0.5, label='ideal')
ax[0].set_xlabel('nº particiones (n)'); ax[0].set_ylabel('Speed-up vs serie'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax[1].set_xlabel('nº particiones (n)'); ax[1].set_ylabel('Eficiencia'); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[0].set_title('Speed-up de la evaluación de fitness')
ax[1].set_title('Eficiencia (speed-up / n)')
plt.tight_layout()
plt.savefig('../results/speedup_comparativo.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Ejecutar celdas 10–11**

Run (manual en Jupyter).
Expected: por dataset, una fila `serie` + filas `spark` n∈{1,2,4,8}. Honesto que WDBC no acelere (overhead de Spark > beneficio) y que Gisette sí. Genera `*_speedup.csv` (3) y `speedup_comparativo.png`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_diversidad_escalabilidad.ipynb results/*_speedup.csv results/speedup_comparativo.png
git commit -m "feat(nb07): speed-up por dataset (serie vs Spark RDD), eficiencia y Amdahl"
```

---

# Fase 7 — Consolidación (D): tabla de escalabilidad algorítmica

### Task 10: Generar la tabla cross-dataset

**Files:**
- Modify: `notebooks/07_diversidad_escalabilidad.ipynb` (añadir celda)

- [ ] **Step 1: Celda 12 — tabla consolidada**

```python
campos = ['dataset', 'D', 'gen', 'n_sel', 'reduccion_pct',
          'acc_gnb_test', 'acc_svc_test', 'jaccard', 'kuncheva',
          't_total_s', 't_gen_s']
filas = []
for nombre in ['wdbc', 'arrhythmia', 'gisette']:
    r = RESULTADOS[nombre]
    e = ESTABILIDAD.get(nombre, {})
    filas.append([
        nombre, r['D'], r['gen'], r['n_sel'], round(r['reduccion'] * 100, 1),
        round(r['acc_gnb_test'], 4), round(r['acc_svc_test'], 4),
        round(e.get('jaccard', float('nan')), 4), round(e.get('kuncheva', float('nan')), 4),
        r['t_total'], r['t_gen'],
    ])

with open('../results/escalabilidad_algoritmica.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(campos); w.writerows(filas)

print('TABLA DE ESCALABILIDAD ALGORÍTMICA')
print('  '.join(campos))
for fila in filas:
    print('  '.join(str(x) for x in fila))
print('\nGuardada en ../results/escalabilidad_algoritmica.csv')
```

- [ ] **Step 2: Ejecutar celda 12**

Run (manual en Jupyter).
Expected: tabla con las 3 filas y `results/escalabilidad_algoritmica.csv`. Verificar que `reduccion_pct` ≈ valores esperados (WDBC alto, Gisette ~90%).

- [ ] **Step 3: Celda 13 — limpieza de Spark**

```python
parar_spark(spark)
print('Spark detenido. Experimento completo.')
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_diversidad_escalabilidad.ipynb results/escalabilidad_algoritmica.csv
git commit -m "feat(nb07): tabla consolidada de escalabilidad algorítmica cross-dataset"
```

---

# Fase 8 — Cierre

### Task 11: Suite de tests completa y limpieza

- [ ] **Step 1: Ejecutar toda la suite rápida (sin Spark, sin datasets pesados)**

Run: `python -m pytest tests/ -q -m "not slow"`
Expected: todos los tests de `ga_metrics` y `ga_core` (synthetic) en verde.

- [ ] **Step 2: Ejecutar la suite completa (incluida la reproducción)**

Run: `python -m pytest tests/ -q`
Expected: todo verde, incluida `test_repro_arrhythmia_gen1`.

- [ ] **Step 3: Actualizar el documento de diseño con el resultado**

Modify: `documentacion/plan_diversidad_escalabilidad.md` — rellenar la tabla de la Parte III (Componente D) con los valores reales de `results/escalabilidad_algoritmica.csv`.

- [ ] **Step 4: Commit final**

```bash
git add documentacion/plan_diversidad_escalabilidad.md
git commit -m "docs: completar tabla de escalabilidad con resultados de nb07"
```

---

## Notas de ejecución

- **Coste de reloj** dominado por Gisette: Fase 4 ~2 min, Fase 5 ~22 min, Fase 6 ~varios min. WDBC y Arrhythmia, segundos. Total < 1 h.
- **Orden recomendado de fases:** 0 → 1 → 2 (todo testeable sin Spark) → 3 → 4 → 5 → 6 → 7 → 8. Las fases 1–2 son seguras y rápidas; las 4–7 requieren Spark y datasets.
- **Reproducibilidad:** todas las ejecuciones fijan `seed`. La estabilidad usa `SEED+s` para s∈[0,R).
- **Artefactos antiguos** (`results/ga_full_*`) se conservan intactos; los nuevos llevan prefijos `*_holdout_*`, `*_estabilidad*`, `*_speedup*`, `escalabilidad_algoritmica.csv`.

## Cobertura del spec (self-review)

| Requisito del spec | Tarea(s) |
|---|---|
| Diversidad genética dentro del GA (Hamming/entropía/únicos) | Task 1, 4, 7 |
| Dispersión fenotípica (std fitness) | Task 4, 7 |
| Validación con LinearSVC en test held-out | Task 7 |
| Estabilidad de selección R=10 (Jaccard, Kuncheva, frecuencia) | Task 2, 8 |
| Escalabilidad algorítmica (tabla cross-dataset) | Task 10 |
| Escalabilidad paralela (speed-up por dataset) | Task 9 |
| Refactor a `src/ga_core.py` | Task 3, 4 |
| Caveat Arrhythmia (clases raras en el split) | Task 6 (`split_holdout`) |
| Caveat Gisette p_init=0.1 en diversidad | Task 7 (panel comparativo normalizado) |
