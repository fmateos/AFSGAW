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
