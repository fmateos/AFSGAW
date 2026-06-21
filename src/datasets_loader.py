"""
Carga de datasets del TFM via OpenML (sklearn.datasets.fetch_openml).

Centraliza la descarga, el cacheo y el preprocesado minimo de los tres
conjuntos de datos del estudio para que los notebooks 04-06 compartan
exactamente el mismo punto de entrada y sean reproducibles:

    - WDBC       (D=30,   sklearn.load_breast_cancer)  -> ya usado en nb 04
    - Arrhythmia (D=279,  OpenML 'arrhythmia', v1)      -> nb 05
    - Gisette    (D=5000, OpenML 'gisette')             -> nb 06

Decisiones de preprocesado (documentadas para el TFM):

  * Imputacion de faltantes (solo Arrhythmia): OpenML codifica los valores
    ausentes como NaN. GaussianNB no admite NaN, asi que se imputa por la
    MEDIA de cada columna (SimpleImputer). Es la opcion mas conservadora y
    no introduce hiperparametros. La columna 'J' de Arrhythmia (~83% de
    faltantes) queda practicamente constante tras imputar; el GA debería
    descartarla a través de la penalizacion de parsimonia.

  * Codificacion de la clase (Arrhythmia): el target es multiclase (16
    clases). Se mantiene tal cual (entero) porque GNB es multiclase de
    forma nativa; cross_val_score con StratifiedKFold preserva las
    proporciones. Las clases con muy pocas instancias pueden hacer que
    algun fold no las contenga: se usa cv=CV_FOLDS estandar y, si scikit
    avisa, es un efecto conocido del desbalanceo de Arrhythmia.

  * Gisette ya viene escalado (enteros 0-999, sin faltantes); no requiere
    preprocesado adicional para GNB.

Cache: fetch_openml usa ~/scikit_learn_data por defecto. La primera llamada
descarga de internet; las siguientes leen de cache local.
"""
import numpy as np


def cargar_wdbc():
    """WDBC (Wisconsin Diagnostic Breast Cancer) desde sklearn. D=30."""
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    X = ds.data.astype(np.float64)
    y = ds.target.astype(np.int64)
    feature_names = list(ds.feature_names)
    info = {
        "nombre": "WDBC",
        "fuente": "sklearn.datasets.load_breast_cancer",
        "n_clases": int(len(np.unique(y))),
        "tiene_faltantes": False,
        "preprocesado": "ninguno",
    }
    return X, y, feature_names, info


def cargar_arrhythmia():
    """Arrhythmia desde OpenML (v1). D=279, multiclase, con faltantes.

    Imputa los NaN por la media de cada columna (necesario para GaussianNB).
    """
    from sklearn.datasets import fetch_openml
    from sklearn.impute import SimpleImputer

    ds = fetch_openml("arrhythmia", version=1, as_frame=True, parser="auto")
    df = ds.data
    feature_names = list(df.columns)

    # A numerico (los faltantes vienen como NaN); imputacion por media.
    X_raw = df.apply(lambda c: c.astype(np.float64)).to_numpy()
    n_nan = int(np.isnan(X_raw).sum())
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X_raw).astype(np.float64)

    y = ds.target.astype(np.int64).to_numpy()

    info = {
        "nombre": "Arrhythmia",
        "fuente": "OpenML 'arrhythmia' v1",
        "n_clases": int(len(np.unique(y))),
        "tiene_faltantes": True,
        "n_valores_imputados": n_nan,
        "preprocesado": "imputacion por media (SimpleImputer)",
    }
    return X, y, feature_names, info


def cargar_gisette():
    """Gisette desde OpenML. D=5000, binario, ya escalado, sin faltantes."""
    from sklearn.datasets import fetch_openml

    ds = fetch_openml("gisette", version=1, as_frame=False, parser="auto")
    # OpenML puede devolver Gisette como matriz dispersa (csr_matrix); GaussianNB
    # necesita un array denso, asi que se materializa con toarray() (~280 MB).
    data = ds.data
    if hasattr(data, "toarray"):
        data = data.toarray()
    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(ds.target).astype(np.int64)
    feature_names = (
        list(ds.feature_names)
        if getattr(ds, "feature_names", None) is not None
        else [f"V{i+1}" for i in range(X.shape[1])]
    )
    info = {
        "nombre": "Gisette",
        "fuente": "OpenML 'gisette'",
        "n_clases": int(len(np.unique(y))),
        "tiene_faltantes": bool(np.isnan(X).any()),
        "preprocesado": "ninguno (ya escalado por origen)",
    }
    return X, y, feature_names, info


CARGADORES = {
    "wdbc": cargar_wdbc,
    "arrhythmia": cargar_arrhythmia,
    "gisette": cargar_gisette,
}


def cargar_dataset(nombre):
    """Devuelve (X, y, feature_names, info) para 'wdbc'|'arrhythmia'|'gisette'."""
    clave = nombre.strip().lower()
    if clave not in CARGADORES:
        raise ValueError(
            f"Dataset desconocido: {nombre!r}. Opciones: {list(CARGADORES)}"
        )
    return CARGADORES[clave]()
