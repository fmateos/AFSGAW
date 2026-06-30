import math
import numpy as np
from ga_metrics import (
    diversity_metrics,
    jaccard_mean,
    kuncheva_index,
    selection_frequency,
)


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
