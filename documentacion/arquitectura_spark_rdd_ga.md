---
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{Francisco Mateos Díaz}
  - \fancyhead[R]{TFM, Curso 2025-2026}
  - \fancyfoot[C]{\thepage}
  - \usepackage{tikz}
  - \usepackage{listings}
  - \lstset{basicstyle=\ttfamily\small, breaklines=true, frame=single, numbers=left, numberstyle=\tiny}
geometry: margin=2.5cm
---
# Documento hito #2 (de 5) de mes 2 

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

# Arquitectura de Paralelización con Apache Spark RDDs en un Algoritmo Genético para Selección Automática de Características

**TFM:** Selección Automática de Características en Algoritmos de Clasificación Mediante Algoritmos Genéticos y Métodos de Envoltorio usando Apache Spark RDDs

**Entorno de ejecución para TFM:** MacBook Pro M5 · Spark `local[8]` · Python + PySpark

## 1. Paralelización en Spark

En cada generación del Algoritmo Genético (GA), se dispone de una población de N individuos, cada uno representado como un cromosoma binario, $\mathbf{c}$, de longitud igual al número de características del dataset usado para entrenar el clasificador.

$$\mathbf{c} = [c_1, c_2, \dots, c_D], \quad c_j \in \{0, 1\}$$

donde D es el número de características del dataset estudiado. 

La evaluación de cada individuo implica entrenar el clasificador con el subconjunto de *features* definido por su cromosoma y calcular la métrica de *fitness* que se haya definido (e.g., *accuracy* en validación cruzada más penalización por número de características). Esto supone la operación computacionalmente más costosa del proceso. Cabe destacar que esta evaluación de *fitness* es independiente entre individuos, lo que la hace susceptible a la paralelización.

Apache Spark explota esta independencia asignando cada evaluación a un *worker* distinto. Para las primeras iteraciones del TFM usaremos un clasificador Naive Bayes, pudiendo evaluar, con `local[8]`, 8 individuos simultáneamente si disponemos de 8 núcleos, como es el caso en el contexto de ejecución que hemos definido. Esto permite procesar la población completa en aproximadamente `ceil(N/8)` rondas de ejecución paralela.

Las operaciones genéticas, en cambio, no se paralelizan: la selección por torneo, el *crossover* uniforme y la mutación se ejecutan de forma secuencial en el Driver, al ser operaciones con dependencias lógicas entre individuos (el resultado de la selección afecta al *crossover*, y este a la mutación).

| Operación | Ejecutada en | ¿Paralelizada? | Motivo |
|---|---|---|---|
| Entrenamiento del clasificador | Workers | Sí (8 simultáneamente) | Sin dependencias entre individuos |
| Cálculo del *fitness* | Workers | Sí (8 simultáneamente) | Evaluación independiente |
| Selección de padres (torneo) | Driver | No | Requiere comparar *fitnesses* del conjunto |
| *Crossover* uniforme | Driver | No | Depende de los padres seleccionados |
| Mutación | Driver | No | Opera sobre individuos ya cruzados |
| Elitismo / nueva generación | Driver | No | Gestión global de la población |

**Nota:** La paralelización de las operaciones genéticas es teóricamente posible, pero su coste computacional es despreciable frente a la evaluación, lo que no justifica la complejidad añadida.


## 2. Estructura de RDDs: Driver vs. Workers

### 2.1 Definición de RDD

Un **RDD** (*Resilient Distributed Dataset*) es la abstracción fundamental de Apache Spark para representar colecciones de datos distribuidas de forma inmutable y tolerante a fallos (Apache Spark Documentation, 2024). Un RDD divide los datos en **particiones** que se distribuyen entre los *workers* del clúster, permitiendo operaciones paralelas sobre cada partición sin necesidad de comunicación entre nodos durante la ejecución. La resiliencia proviene del **linaje** (*lineage*): si una partición se pierde, Spark puede reconstruirla automáticamente reejecutando las transformaciones que la generaron desde los datos de origen.

En el contexto de este TFM, el RDD representa la **población de la generación actual del GA**, donde cada elemento es un cromosoma (vector binario).

### 2.2 Roles: Driver y Workers

El **Driver** es el proceso principal que ejecuta el script de Python. Actúa como orquestador del clúster Spark: crea el contexto de Spark, define transformaciones, dispara acciones y recopila resultados. En el GA, el Driver gestiona la población, aplica los operadores genéticos y coordina la ejecución distribuida. Nunca entrena modelos directamente; su rol es dirigir y orquestar.

Los **Workers** son los procesos que ejecutan el trabajo real; son la unidad de ejecución distribuida que procesa particiones de datos. En modo `local[8]`, corresponden a 8 *threads* del JVM local gestionados por Spark. Los Workers reciben las particiones del RDD y el código del Driver (funciones *lambda* serializadas), ejecutan las transformaciones asignadas en paralelo y devuelven resultados. En el GA, cada Worker evalúa sus cromosomas asignados, entrenando modelos de clasificación y calculando *fitness*.

La población se instancia inicialmente en el Driver como una estructura local (e.g., array NumPy). Para distribuirla, se serializa y paraleliza mediante `sc.parallelize()`, dividiéndola en 8 particiones, una por *worker*:

```python
# Pseudocódigo – estructura básica del bucle generacional
# Ejemplo con una población de 50 individuos y 30 características cada uno 
N = 50
K = 30
población = inicializar_población(n=N, n_features=K)  # Driver

for generación in range(N_GENERACIONES):
    rdd_población   = sc.parallelize(población, numPartitions=8)  # Transformation
    rdd_evaluado    = rdd_población.map(evaluar_individuo)         # Transformation
    fitnesses       = rdd_evaluado.collect()                       # Action → ejecución real

    población = operadores_genéticos(población, fitnesses)        # Driver (secuencial)
```

Cada generación del GA reconstruye el RDD a partir de la nueva población modificada por los operadores genéticos. Aunque esto implica serialización repetida, el coste es despreciable frente al entrenamiento de los clasificadores en los *workers* [típicamente <1% del tiempo total de generación].

### 2.3 Distribución física de una población ejemplo (N=50 cromosomas en 8 particiones)

Spark reparte los individuos de forma aproximadamente uniforme (`floor(50/8) = 6`, con 2 particiones de 7 elementos):

```
┌──────────────────────────────────────────────────────────────────┐
│                   RDD: rdd_población (50 cromosomas)             │
├─────────────────────────┬────────────────────────────────────────┤
│  Partición 0 → Core 0   │  Partición 4 → Core 4                 │
│  cromosomas 0 – 6       │  cromosomas 28 – 34                   │
├─────────────────────────┼────────────────────────────────────────┤
│  Partición 1 → Core 1   │  Partición 5 → Core 5                 │
│  cromosomas 7 – 13      │  cromosomas 35 – 41                   │
├─────────────────────────┼────────────────────────────────────────┤
│  Partición 2 → Core 2   │  Partición 6 → Core 6                 │
│  cromosomas 14 – 20     │  cromosomas 42 – 48                   │
├─────────────────────────┼────────────────────────────────────────┤
│  Partición 3 → Core 3   │  Partición 7 → Core 7                 │
│  cromosomas 21 – 27     │  cromosoma  49                        │
└─────────────────────────┴────────────────────────────────────────┘
```

Cada partición reside en la **RAM local del *worker*** asignado. Durante la fase de evaluación, cada *worker* opera exclusivamente sobre sus cromosomas, sin comunicación con los demás [patrón *embarrassingly parallel*].


## 3. Ejecución del Clasificador

### 3.1 Dónde se entrena el modelo

El clasificador se entrena exclusivamente en los Workers, nunca en el Driver. La función `evaluar_individuo` —que recibe un cromosoma, filtra las columnas del dataset, entrena el clasificador y retorna el *fitness*— se serializa y envía a cada *worker* como una *closure* de Python. La ejecución remota es forzada por el patrón `map()` + `collect()`.

En las primeras iteraciones del TFM se empleará **Naive Bayes** (Gaussian o Multinomial, según la naturaleza de los *features* seleccionados) como clasificador base dentro del método *wrapper*. Su bajo coste de entrenamiento —proporcional a $O(n \cdot d)$, donde $n$ es el número de instancias y $d$ el número de *features* del subconjunto activo— lo hace especialmente adecuado para evaluar con rapidez las numerosas combinaciones que el GA explora generación tras generación. Esta elección permite validar la arquitectura de paralelización antes de sustituirlo por clasificadores más complejos en fases posteriores.

### 3.2 Cuándo se ejecuta: el papel de `collect()`

La ejecución no ocurre en el momento de definir el `map()`, sino cuando se invoca `collect()` (ver Sección 4). En ese instante, Spark materializa el plan de ejecución y cada *worker* lleva a cabo el entrenamiento. El clasificador tiene acceso a los datos de entrenamiento, que se difunden a los *workers* mediante *broadcast variables* para evitar transferencias repetidas en cada generación.

### 3.3 Ejemplo de *timeline* de ejecución por generación

```
t =   0 ms  │ Driver invoca collect() → Spark compila el DAG y envía tareas
t =     5 ms   │ Spark serializa la función y el cromosoma; los Workers reciben particiones
t =   5 ms  │ ┌─ Worker 0: NaiveBayes.fit(X[:, cromosoma_0]) → fitness = 0.924
            │ ├─ Worker 1: NaiveBayes.fit(X[:, cromosoma_7]) → fitness = 0.881
            │ ├─ Worker 2: NaiveBayes.fit(X[:, cromosoma_14])→ fitness = 0.956
            │ ├─ ...  (Workers 3–7 en paralelo)
t = 490 ms  │ └─ Worker 7: NaiveBayes.fit(X[:, cromosoma_49])→ fitness = 0.903
t = 495 ms  │ Workers envían resultados (lista de fitnesses) al Driver
t = 497 ms  │ Driver recibe [0.924, 0.881, 0.956, ..., 0.903] → operadores genéticos
```

**Nota:** Naive Bayes en scikit-learn es un algoritmo de coste reducido que no requiere aceleración GPU. En el chip Apple M5, su ejecución aprovecha las instrucciones vectoriales SIMD de los P-cores, resultando en tiempos de entrenamiento muy bajos por individuo, lo que maximiza el beneficio de la paralelización frente a clasificadores más pesados.


## 4. Transformations vs. Actions: Ejecución Lazy vs. Eager

Spark distingue dos tipos de operaciones sobre RDDs con semánticas de ejecución radicalmente distintas.

Las **Transformations** son operaciones ***lazy***: definen *qué* debe hacerse sobre los datos, pero no lo ejecutan. Spark las anota en un grafo acíclico dirigido (DAG) y espera hasta que sea estrictamente necesario computarlas. `sc.parallelize()` y `rdd.map()` son Transformations.

Las **Actions** son operaciones ***eager***: desencadenan la ejecución de todas las Transformations acumuladas en el DAG, materializan los resultados y los traen al Driver. `rdd.collect()` es la Action central del GA.

```
Paso 1: sc.parallelize(población, 8)
        → TRANSFORMATION (Lazy): Spark anota "dividir en 8 particiones"
        → [NO] Ningún dato se mueve aún

Paso 2: rdd.map(evaluar_individuo)
        → TRANSFORMATION (Lazy): Spark anota "aplicar función a cada elemento"
        → [NO] Ningún clasificador se entrena aún

Paso 3: rdd.collect()
        → ACTION (Eager): ¡Spark ejecuta todo el plan!
        → [SI] Datos se distribuyen, clasificadores se entrenan, fitnesses retornan al Driver
```

| Operación Spark | Tipo | Ejecución | Rol en el GA |
|---|---|---|---|
| `sc.parallelize (población, 8)` | Transformation | Lazy | Distribuye los N cromosomas |
| `rdd.map (evaluar_individuo)` | Transformation | Lazy | Define la evaluación *wrapper* |
| `rdd.collect()` | **Action** | **Eager** | **Punto de ejecución real** |
| `rdd.take(n)` | Action | Eager | Alternativa para depuración parcial |

`collect()` constituye el **cuello de botella** de cada generación: su tiempo de ejecución domina completamente el coste total, ya que los operadores genéticos en el Driver consumen menos del 5% del tiempo generacional. En el caso de Naive Bayes, este cuello de botella es notablemente menor que con clasificadores más complejos, lo que acorta el tiempo total del GA y facilita la exploración de un mayor número de generaciones en el mismo tiempo de cómputo. La optimización futura puede pasar por ajustar el número de particiones o sustituir el clasificador base por alternativas con mayor capacidad discriminativa.


## Referencias

- Apache Spark Documentation. (2024). *RDD Programming Guide*. https://spark.apache.org/docs/latest/rdd-programming-guide.html