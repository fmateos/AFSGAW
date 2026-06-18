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
