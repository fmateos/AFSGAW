---
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{Francisco Mateos Díaz}
  - \fancyhead[R]{TFM, Curso 2025-2026}
  - \fancyfoot[C]{\thepage}
  - \usepackage{tikz}
  - \usetikzlibrary{decorations.pathreplacing,positioning,arrows.meta}
  - \usepackage{listings}
  - \lstset{basicstyle=\ttfamily\small, breaklines=true, frame=single, numbers=left, numberstyle=\tiny}
geometry: margin=2.5cm
---
# Documento hito #4 (de 5) de mes 2

\begin{center}
\begin{tikzpicture}[scale=0.72]
    \draw[thick,->] (0,0) -- (14,0) node[right]{\small Semanas};
    \foreach \x/\lab in {0/0,2.8/4,5.6/8,8.4/12,11.2/16,14/20} {
        \draw (\x,0.1) -- (\x,-0.1) node[below]{\tiny \lab};
    }
    % Mes 1 - debil
    \fill[yellow!15] (0,0.8) rectangle (2.8,1.5);
    \node[font=\scriptsize, text=black!40] at (1.4,1.15) {Estado del Arte};
    % Mes 2 - destacado
    \fill[blue!60] (2.8,0.8) rectangle (5.6,1.5);
    \draw[blue!80, thick] (2.8,0.8) rectangle (5.6,1.5);
    \node[font=\scriptsize\bfseries] at (4.2,1.15) {Dise\~no};
    % Mes 3 - debil
    \fill[green!15] (5.6,0.8) rectangle (8.4,1.5);
    \node[font=\scriptsize, text=black!40] at (7,1.15) {Implementaci\'on};
    % Mes 4 - debil
    \fill[orange!15] (8.4,0.8) rectangle (11.2,1.5);
    \node[font=\scriptsize, text=black!40] at (9.8,1.15) {Experimentaci\'on};
    % Mes 5 - debil
    \fill[red!15] (11.2,0.8) rectangle (14,1.5);
    \node[font=\scriptsize, text=black!40] at (12.6,1.15) {Redacci\'on};
    % Hitos
    \foreach \x in {2.8,5.6,8.4,11.2,14} {
        \node[draw,circle,fill=gray!30,inner sep=1pt,font=\tiny,text=black!40] at (\x,2) {H};
    }
    % Hito mes 2 destacado
    \node[draw,circle,fill=blue!70,inner sep=1.5pt,font=\tiny\bfseries,text=white] at (5.6,2) {H};
    % Flecha vertical apuntando al mes 2
    \draw[blue!80, very thick, ->, >=stealth] (4.2,2.8) -- (4.2,1.6);
    \node[font=\footnotesize\bfseries, blue!80] at (4.2,3.1) {};
\end{tikzpicture}
\end{center}

# Mini-Prototipo de Evaluación Paralela con Spark RDD

**TFM:** Selección Automática de Características en Algoritmos de Clasificación Mediante Algoritmos Genéticos y Métodos de Envoltorio con Apache Spark RDD

**Notebook:** `notebooks/01_parallel_fitness_prototype.ipynb`

---

## 1. Objetivo

Validar experimentalmente que la arquitectura de paralelización diseñada en el documento de arquitectura Spark RDD funciona correctamente antes de implementar el GA completo. En concreto, se verifica que:

1. Una población de cromosomas puede distribuirse como RDD y evaluarse en paralelo con `rdd.map()`.
2. El paralelismo produce un **speedup real** medible frente a la ejecución secuencial.
3. Los resultados paralelos son **idénticos** a los secuenciales (correctitud).
4. La ejecución es **observable** en la Spark Web UI (`http://localhost:4040`).
5. El patrón `parallelize → map → collect` funciona dentro de un bucle generacional del GA.

## 2. Relación con la Arquitectura del Proyecto

Este prototipo implementa de forma simplificada el patrón descrito en los documentos de diseño del TFM:

- **Documento de diseño del GA (`diseño_GA.md`)**: Define el cromosoma binario de longitud $D$, la función de fitness multiobjetivo $\text{Fitness}(\mathbf{c}) = \alpha \cdot \text{Accuracy}(\mathbf{c}) - \beta \cdot |S|/D$, y el ciclo evolutivo con evaluación paralela via `RDD.map()`.

- **Documento de arquitectura Spark (`arquitectura_spark_rdd_ga.md`)**: Establece la separación Driver/Workers, donde la evaluación de fitness se ejecuta en los Workers (paralelizable, patrón *embarrassingly parallel*) y los operadores genéticos se ejecutan secuencialmente en el Driver.

El prototipo replica exactamente este patrón:

\begin{center}
\begin{tikzpicture}[
    node distance=1.1cm,
    box/.style={rectangle, draw, rounded corners=3pt, minimum width=4.2cm, minimum height=0.7cm, align=center, font=\small},
    driver/.style={box, fill=blue!10, draw=blue!50},
    spark/.style={box, fill=orange!12, draw=orange!50},
    action/.style={box, fill=green!12, draw=green!50, minimum width=5.5cm},
    worker/.style={rectangle, draw, rounded corners=2pt, fill=yellow!15, draw=yellow!60, minimum width=3.8cm, minimum height=0.55cm, align=center, font=\scriptsize},
    arr/.style={->, thick, >=stealth},
    lbl/.style={font=\scriptsize\itshape, text=gray!70},
]

% --- Columna Driver (izquierda) ---
\node[driver] (pop) {poblaci\'on (Python list)};

\node[spark, below=of pop] (par) {sc.parallelize(poblaci\'on, N\_CORES)};

\node[spark, below=of par] (map) {rdd.map(fitness\_costoso)};

\node[action, below=of map] (col) {rdd.collect()};

\node[driver, below=1.6cm of col] (fit) {fitness\_scores (Python list)};

\node[driver, below=of fit] (ops) {operadores\_gen\'eticos(...)};

\node[driver, below=of ops] (new) {nueva\_poblaci\'on $\rightarrow$ sig.\ generaci\'on};

% --- Workers (derecha de collect) ---
\node[worker, right=2.2cm of col, yshift=0.9cm]  (w0) {Core 0: eval cromosomas 0--24};
\node[worker, below=0.18cm of w0]                 (w1) {Core 1: eval cromosomas 25--49};
\node[worker, below=0.18cm of w1]                 (w2) {Core 2: eval cromosomas 50--74};
\node[worker, below=0.18cm of w2]                 (w3) {Core 3: eval cromosomas 75--99};

% --- Flechas verticales ---
\draw[arr] (pop) -- (par);
\draw[arr] (par) -- (map);
\draw[arr] (map) -- (col);
\draw[arr] (col) -- (fit);
\draw[arr] (fit) -- (ops);
\draw[arr] (ops) -- (new);

% --- Llave de workers ---
\draw[decorate, decoration={brace, amplitude=5pt, mirror}, thick, gray]
    ([xshift=-4pt]w0.north west) -- ([xshift=-4pt]w3.south west)
    node[midway, left=8pt, font=\scriptsize, align=right, text=gray!80] {Ejecuci\'on\\paralela};

% --- Flecha collect -> workers ---
\draw[arr, dashed, orange!70] (col.east) -- ++(0.6,0) |- (w0.west);
\draw[arr, dashed, orange!70] (col.east) -- ++(0.6,0) |- (w3.west);

% --- Etiquetas de tipo de operación ---
\node[lbl, right=0.15cm of par.east] {Transformation (lazy)};
\node[lbl, right=0.15cm of map.east] {Transformation (lazy)};
\node[lbl, left=0.15cm of col.west, align=right] {ACTION};
\node[lbl, right=0.15cm of ops.east] {Secuencial en Driver};

% --- Títulos de columna ---
\node[font=\small\bfseries, above=0.4cm of pop] {Driver};
\node[font=\small\bfseries, above=0.4cm of w0] {Workers (cores)};

\end{tikzpicture}
\end{center}

## 3. Descripción del Código

### 3.1 Población ficticia

Se genera una población de $P = 100$ individuos con cromosomas binarios de longitud $D = 30$ (equivalente al dataset WDBC del TFM). Cada gen se inicializa con probabilidad $p_{init} = 0.5$:

```python
poblacion = [list(np.random.binomial(1, P_INIT, D)) for _ in range(P)]
```

Esto produce cromosomas como `[1, 0, 1, 1, 0, 0, 1, 0, ...]` donde cada posición indica si la característica correspondiente está incluida en el subconjunto candidato.

### 3.2 Función fitness dummy

Se implementan dos variantes:

- **`fitness_rapido`**: Usa la función Rastrigin invertida como pseudo-accuracy. No introduce retardo artificial. Sirve para validar correctitud pero el overhead de serialización de Spark puede enmascarar el beneficio del paralelismo.

- **`fitness_costoso`**: Añade `time.sleep(sleep_ms)` para simular el coste real de entrenar un clasificador con 5-fold cross-validation. Con `sleep_ms=50` se simula un coste de 50ms por evaluación, representativo de Naive Bayes sobre WDBC. Este es el test clave para demostrar el speedup.

Ambas funciones siguen la fórmula de fitness del diseño con $\alpha = 0.9$ y $\beta = 0.1$:

```python
fitness = ALPHA * score - BETA * (genes_activos / D)
```

El caso especial de cromosoma nulo ($|S| = 0$) retorna fitness $= 0$.

### 3.3 Evaluación secuencial (baseline)

Evalúa los 100 individuos con un bucle Python estándar:

```python
fitness_seq = [fitness_costoso(ind, sleep_ms=50) for ind in poblacion]
```

Con $P = 100$ y $\text{sleep} = 50\text{ms}$, el tiempo esperado es $\sim 5.0\text{s}$.

### 3.4 Evaluación paralela (Spark RDD)

Distribuye la población como RDD con tantas particiones como cores disponibles y evalúa en paralelo:

```python
rdd_poblacion = sc.parallelize(poblacion, numSlices=NUM_CORES)
fitness_par = rdd_poblacion.map(lambda ind: fitness_costoso(ind, sleep_ms=50)).collect()
```

Este es el patrón `parallelize → map → collect` descrito en la arquitectura. Según el documento de arquitectura, con `local[N]` se procesan $N$ individuos simultáneamente, completando la población en `ceil(P/N)` rondas.

**Tiempo esperado** con 4 cores: $\sim 100 \times 50\text{ms} / 4 = 1.25\text{s}$ (speedup teórico $\approx 4\text{x}$).

### 3.5 Validación de correctitud

Se compara elemento a elemento que los resultados secuenciales y paralelos coinciden:

```python
resultados_iguales = all(abs(s - p) < 1e-9 for s, p in zip(fitness_seq, fitness_par))
```

Dado que la función fitness es determinista y no depende de estado compartido, los resultados deben ser idénticos. Esto confirma que la serialización/deserialización de Spark no altera los cálculos.

### 3.6 Experimento de escalabilidad

Se varía el número de particiones del RDD (1, 2, 4, N, 2N) para observar cómo escala el speedup. Se generan dos gráficos:

1. **Tiempo vs. Particiones**: Muestra la reducción de tiempo al aumentar particiones, con la línea baseline secuencial como referencia.
2. **Speedup vs. Particiones**: Compara el speedup obtenido con el speedup ideal (lineal).

El speedup sub-lineal es esperado por el overhead de serialización, scheduling de tareas y recolección de resultados por parte del Driver.

### 3.7 Simulación de generaciones del GA

Se ejecutan 5 generaciones completas del ciclo evolutivo para validar el patrón en un bucle:

1. **Evaluación paralela** (Spark): `sc.parallelize → map(fitness) → collect()`
2. **Selección** (Driver): Torneo binario de tamaño 2
3. **Cruce** (Driver): Un punto con $p_{cross} = 0.8$
4. **Mutación** (Driver): Flip de bit con $p_{mut} = 0.2$, $p_{mut\_bit} = 1/D$
5. **Elitismo** (Driver): Top 2 individuos pasan directamente

Los operadores genéticos se implementan siguiendo las especificaciones del documento de diseño del GA y se ejecutan secuencialmente en el Driver, como establece la arquitectura.

## 4. Métricas Capturadas

| Métrica | Descripción |
|---|---|
| Tiempo secuencial | Baseline: tiempo total de evaluar $P$ individuos con Python puro |
| Tiempo paralelo | Tiempo con Spark RDD (`map` + `collect`) |
| Speedup | $t_{seq} / t_{par}$ — factor de aceleración |
| Eficiencia | $\text{speedup} / N_{cores} \times 100\%$ |
| Correctitud | `fitness_seq[i] == fitness_par[i]` para todo $i$ |

## 5. Qué Observar en la Spark Web UI

Al ejecutar el notebook, acceder a `http://localhost:4040` para verificar:

| Pestaña | Qué observar |
|---|---|
| **Jobs** | Un job por cada `collect()`. En el experimento de escalabilidad aparecerán varios jobs consecutivos. En la simulación de generaciones, 5 jobs (uno por generación). |
| **Stages** | Cada job tiene un stage con tantas tareas como particiones. Verificar que las tareas se ejecutan en paralelo (columna de tiempos solapados). |
| **Executors** | Verificar que los cores están activos durante la evaluación. La columna "Active Tasks" debe mostrar $N_{cores}$ tareas simultáneas. |
| **Environment** | Confirmar la configuración: `spark.default.parallelism`, `spark.driver.memory`, etc. |

## 6. Criterios de Validación

El prototipo se considera exitoso si cumple todos los siguientes criterios:

| # | Criterio | Umbral |
|---|---|---|
| 1 | Población distribuida como RDD | Sin errores |
| 2 | Evaluación paralela con `rdd.map()` | Sin errores |
| 3 | Correctitud (seq == paralelo) | 100% coincidencia |
| 4 | Speedup significativo | > 1.5x |
| 5 | Ciclo evolutivo completo | 5 generaciones sin error |
| 6 | Observable en Web UI | Jobs visibles en localhost:4040 |

## 7. Limitaciones del Prototipo

- **Fitness simulado**: Se usa `time.sleep` en lugar de un clasificador real. Esto aísla el efecto del paralelismo pero no captura el overhead de scikit-learn ni el broadcasting de datos.
- **Datos no distribuidos**: El dataset de entrenamiento no se incluye; en la implementación real se usarán *broadcast variables* para enviar los datos a los Workers.
- **Escala local**: `local[*]` ejecuta todo en una JVM. Un cluster real tendría overhead de red adicional, pero también más recursos.
- **Pocas generaciones**: 5 generaciones bastan para validar el patrón, pero no para observar convergencia del GA.

## 8. Siguiente Paso

Una vez validado el paralelismo con este prototipo, el siguiente paso es sustituir la función fitness dummy por la evaluación real con Naive Bayes gaussiano y validación cruzada de 5 pliegues, incorporando:

1. Broadcasting del dataset de entrenamiento a los Workers.
2. Integración con DEAP para los operadores genéticos.
3. Criterio de parada dual (generaciones máximas + convergencia).
4. Evaluación sobre el dataset WDBC ($D = 30$) como primer caso de prueba.
