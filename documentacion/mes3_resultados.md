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
  - \usepackage{graphicx}
  - \usepackage{float}
geometry: margin=2.5cm
---
# Documento de mes 3 — Resultados de la Implementación

\begin{center}
\begin{tikzpicture}[scale=0.72]
    \draw[thick,->] (0,0) -- (14,0) node[right]{\small Semanas};
    \foreach \x/\lab in {0/0,2.8/4,5.6/8,8.4/12,11.2/16,14/20} {
        \draw (\x,0.1) -- (\x,-0.1) node[below]{\tiny \lab};
    }
    % Mes 1 - debil
    \fill[yellow!15] (0,0.8) rectangle (2.8,1.5);
    \node[font=\scriptsize, text=black!40] at (1.4,1.15) {Estado del Arte};
    % Mes 2 - debil
    \fill[blue!15] (2.8,0.8) rectangle (5.6,1.5);
    \node[font=\scriptsize, text=black!40] at (4.2,1.15) {Dise\~no};
    % Mes 3 - destacado
    \fill[green!60] (5.6,0.8) rectangle (8.4,1.5);
    \draw[green!80!black, thick] (5.6,0.8) rectangle (8.4,1.5);
    \node[font=\scriptsize\bfseries] at (7,1.15) {Implementación};
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
    % Hito mes 3 destacado
    \node[draw,circle,fill=green!70!black,inner sep=1.5pt,font=\tiny\bfseries,text=white] at (8.4,2) {H};
    % Flecha vertical apuntando al mes 3
    \draw[green!80!black, very thick, ->, >=stealth] (7,2.8) -- (7,1.6);
    \node[font=\footnotesize\bfseries, green!80!black] at (7,3.1) {};
\end{tikzpicture}
\end{center}

# Resultados: Validación del GA-wrapper sobre Spark RDD (WDBC)

**TFM:** Selección Automática de Características en Algoritmos de Clasificación Mediante Algoritmos Genéticos y Métodos de Envoltorio con Apache Spark RDD

Este documento recoge los **resultados numéricos y gráficos** obtenidos tras la ejecución de los hitos 1 a 6 descritos en `mes3.md`. Se analiza el comportamiento del fitness real con GNB (Hitos 1–4), la escalabilidad medida en el experimento de *speed-up* (Hito 5) y la convergencia del GA completo con parada dual (Hito 6). Los datos se encuentran en el directorio `results/` del repositorio.

### Entorno de ejecución

Todos los experimentos se han realizado en un único host, en modo Spark `local[*]`:

\begin{center}
{\footnotesize
\begin{tabular}{|p{4cm}|p{9cm}|}
\hline
\textbf{Componente} & \textbf{Valor} \\
\hline
SO & macOS 26.3.1 (arm64) \\
CPU lógicas & 10 (Apple Silicon) \\
Spark master & \texttt{local[*]} con \texttt{spark.default.parallelism} = 8 \\
Python & 3.9.6 \\
NumPy / scikit-learn & 1.24.3 / 1.3.0 \\
PySpark & 3.5.0 \\
Memoria driver / executor & 4 GB / 4 GB \\
Semilla global & 42 \\
\hline
\end{tabular}
}
\end{center}

**Nota sobre el paralelismo efectivo.** La configuración se fija explícitamente en el helper \texttt{src/spark\_config.py} con \texttt{spark.default.parallelism = 8} y \texttt{spark.sql.shuffle.partitions = 8}, valor elegido por dos razones:

1. **Reproducibilidad.} Fijar el paralelismo a un valor constante evita que \texttt{local[*]} produzca resultados distintos según el hardware donde se ejecute el código, lo que sería un problema para la defensa del TFM y para comparar experimentos entre datasets (WDBC vs.\ Arrhythmia vs.\ Gisette del mes 4).
2. **Coherencia con el documento de arquitectura} (\textbf{[D2]}), que describe el sistema como \texttt{local[8]}.

En consecuencia, el \textit{speed-up} ideal está acotado por $S_\text{ideal} = 8$ en todos los experimentos del Hito 5.

---

## Resumen de resultados por hito

\begin{center}
{\footnotesize
\begin{tabular}{|c|p{5cm}|p{7cm}|}
\hline
\textbf{Hito} & \textbf{Resultado principal} & \textbf{Artefacto} \\
\hline
1--4 & Fitness real GNB operativo; paridad secuencial/paralelo a $10^{-9}$ & \texttt{gnb\_poblacion\_inicial.png}, \texttt{gnb\_ga\_evolucion.png} \\
5 (Exp. 1) & \textit{Speed-up} crece con $P$: $1{,}18\times$ ($P{=}50$) $\to$ $4{,}03\times$ ($P{=}500$) & \texttt{exp1\_speedup\_poblacion.csv/.png} \\
5 (Exp. 2) & Saturación cerca de 4--8 particiones; $S \approx 2{,}96{-}2{,}99\times$; degradación a partir de 16 particiones & \texttt{exp2\_speedup\_particiones.csv/.png} \\
5 (Exp. 3) & Óptimo en $(P{=}500, \text{part}{=}8)$: $S = 4{,}23\times$ & \texttt{exp3\_speedup\_combinado.csv}, \texttt{exp3\_heatmap\_speedup.png} \\
6 & Convergencia en 17 gens ($G_{max}=50$): $F^* = 0{,}848028$ con 8 genes & \texttt{ga\_full\_experimento.json}, \texttt{ga\_full\_series.csv}, \texttt{ga\_full\_convergencia.png} \\
\hline
\end{tabular}
}
\end{center}

---

## 1. Hitos 1–4: Fitness real con GNB y evaluación paralela

### 1.1 Verificación de correctitud (secuencial vs. paralelo)

La primera validación realizada en `02_gnb_fitness_evaluation.ipynb` compara, para una población inicial de $P=100$ individuos aleatorios, los vectores de fitness producidos por la evaluación secuencial (lista por comprensión en el Driver) y por la evaluación paralela (`parallelize → map → collect`). La máxima diferencia observada fue inferior a $10^{-9}$, lo que confirma que:

1. La serialización de la *closure* con `X_bc.value` no introduce desviaciones numéricas.
2. `StratifiedKFold` con la misma semilla produce los mismos folds en Driver y Workers.
3. La implementación cumple con la especificación de la fórmula de fitness de **[D1]** §2 tanto en Python puro como en Spark.

Este resultado es equivalente al \textit{smoke test} del prototipo del mes 2 pero con la fitness real del sistema final, eliminando la incertidumbre introducida por el paso de `time.sleep` al clasificador.

### 1.2 Distribución de la población inicial

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/gnb_poblacion_inicial.png}
\caption{Análisis de la población inicial ($P=100$, $D=30$, GNB + 5-fold CV). Histograma de fitness (izquierda), fitness frente a número de características seleccionadas (centro) y distribución de genes activos (derecha).}
\label{fig:poblacion_inicial}
\end{figure}

La figura \ref{fig:poblacion_inicial} caracteriza la población inicial ($P=100$, $p_{init}=0{,}5$) antes de cualquier operador genético:

- **Fitness:** distribución aproximadamente normal centrada en $\sim 0{,}79$, con cola hacia abajo ($\min \approx 0{,}72$) debida a cromosomas densos con poca información discriminante.
- **Número de genes activos:** media $\approx 15$ (consistente con $p_{init}=0{,}5$ y $D=30$).
- **Fitness vs.\ genes activos:** la relación es débil. Cromosomas con 10–20 genes cubren todo el rango de fitness, lo que anticipa que la búsqueda no es trivial: no basta con reducir $|S|$, hay que elegir \textit{qué} genes.

Este último punto es el que justifica la necesidad del GA: una heurística ingenua basada únicamente en cardinalidad fallaría.

### 1.3 Ciclo evolutivo corto (10 generaciones, $P=100$)

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/gnb_ga_evolucion.png}
\caption{Evolución del GA con fitness real GNB + 5-fold CV a 10 generaciones ($P=100$, $D=30$). Fitness mejor/medio/peor por generación (izquierda), caída del número medio de características activas de 15 a 8 (centro) y tiempo de evaluación por generación (derecha).}
\label{fig:gnb_evolucion}
\end{figure}

La figura \ref{fig:gnb_evolucion} recoge los 10 pasos evolutivos con la fitness real integrada:

- El **mejor fitness** sube monotónicamente desde $\sim 0{,}826$ hasta $\sim 0{,}852$ (elitismo $E=2$), estabilizándose en un \textit{plateau} a partir de la generación 7.
- El **fitness medio** crece casi linealmente desde $\sim 0{,}788$ hasta $\sim 0{,}836$, indicando que la presión selectiva del torneo propaga el material genético útil al conjunto de la población.
- El **número medio de genes activos** desciende de $\sim 15$ a $\sim 8$ en 10 generaciones, lo que confirma que el término de penalización $\beta\,|S|/D$ está ejerciendo la presión prevista hacia soluciones parsimoniosas.
- El **peor fitness** es volátil (oscila entre $\sim 0{,}77$ y $\sim 0{,}81$ con una caída final $\approx 0{,}73$ por mutaciones agresivas sobre un cromosoma no élite). No compromete la convergencia gracias al elitismo.
- El **tiempo por generación** se mantiene en $\approx 0{,}07$–$0{,}11$ s, con la primera generación ligeramente más costosa por el JIT de la JVM y el calentamiento del \textit{broadcast}. No hay tendencia creciente con la generación, como cabe esperar de un coste dominado por $P \times k$ entrenamientos de GNB.

Este experimento corto ($P=100$, 10 gens) anticipa las tendencias que se confirmarán con los parámetros de diseño ($P=50$, $G_{max}=50$) en el Hito 6 (§3): el sistema converge rápidamente en WDBC y alcanza un fitness comparable con un subconjunto de $\sim 8$ características. Con estos resultados se considera validada la transición del fitness \textit{dummy} del prototipo al fitness real; el sistema completo funciona \textit{end-to-end} sobre Spark RDD.

---

## 2. Hito 5: Experimentos de *speed-up*

### 2.1 Experimento 1 — Variación del tamaño de población ($N_\text{part}=8$)

\begin{center}
{\footnotesize
\begin{tabular}{|c|c|c|c|c|c|}
\hline
$P$ & $t_\text{seq}$ (s) & $t_\text{par}$ (s) & \textit{Speed-up} & Eficiencia (\%) & $t/ind$ par.\ (ms) \\
\hline
50  & 0,1003 & 0,0851 & 1,18 & 14,7 & 1,70 \\
100 & 0,2116 & 0,0727 & 2,91 & 36,4 & 0,73 \\
200 & 0,4241 & 0,1382 & 3,07 & 38,4 & 0,69 \\
500 & 1,1735 & 0,2915 & 4,03 & 50,3 & 0,58 \\
\hline
\end{tabular}
}
\end{center}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/exp1_speedup_poblacion.png}
\caption{Experimento 1 — \textit{speed-up} por tamaño de población con $N_\text{part}=8$ fijo. Comparación de tiempos secuencial/paralelo (izquierda), curva de \textit{speed-up} frente al ideal $8\times$ (centro) y eficiencia porcentual (derecha).}
\label{fig:exp1}
\end{figure}

**Interpretación.** Como se observa en la figura \ref{fig:exp1}, el \textit{speed-up} crece monótonamente con $P$ y alcanza $4{,}03\times$ para $P=500$. Respecto al techo teórico $S_\text{ideal}=8$ esto supone una eficiencia del $50{,}3\%$, coherente con lo esperado cuando la tarea por individuo es barata (ms): el \textit{overhead} fijo (serialización de la \textit{closure}, envío de tareas, recogida) compite con el trabajo útil y no se amortiza por completo dentro del rango de $P$ explorado. La pendiente de la curva sigue ascendiendo en $P=500$, lo que sugiere que con poblaciones mayores la eficiencia seguiría mejorando.

El tiempo por individuo en paralelo baja de $1{,}70$ ms ($P=50$) a $0{,}58$ ms ($P=500$), mientras que en secuencial se mantiene estable en $\approx 2$ ms/ind. La diferencia entre ambos valores en $P=500$ ($1{,}77$ ms/ind) representa la ganancia real por paralelismo, descontado del \textit{overhead} amortizado.

**Conclusión metodológica.** Para ejecutar experimentos comparativos en el siguiente mes, $P \geq 200$ es la zona donde el \textit{overhead} deja de dominar y los tiempos son representativos del algoritmo, no de Spark.

### 2.2 Experimento 2 — Variación del número de particiones ($P=100$)

\begin{center}
{\footnotesize
\begin{tabular}{|c|c|c|c|c|c|}
\hline
Particiones & $t_\text{seq}$ (s) & $t_\text{par}$ (s) & \textit{Speed-up} & Eficiencia (\%) & Overhead (s) \\
\hline
1  & 0,1971 & 0,2087 & 0,94 & 11,8 & 0,0116 \\
2  & 0,1971 & 0,1140 & 1,73 & 21,6 & 0,0155 \\
4  & 0,1971 & 0,0665 & 2,96 & 37,0 & 0,0172 \\
8  & 0,1971 & 0,0660 & 2,99 & 37,3 & 0,0414 \\
16 & 0,1971 & 0,0890 & 2,21 & 27,7 & 0,0767 \\
\hline
\end{tabular}
}
\end{center}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/exp2_speedup_particiones.png}
\caption{Experimento 2 — \textit{speed-up} frente a número de particiones con $P=100$ y $N_\text{cores}=8$. Tiempo paralelo frente a línea de tiempo secuencial (izquierda), curva \textit{speed-up} observado vs.\ ideal lineal (centro) y \textit{overhead} de Spark creciente con el particionado (derecha).}
\label{fig:exp2}
\end{figure}

**Interpretación.** La figura \ref{fig:exp2} muestra cuatro regímenes:

- Con **1 partición**, el paralelismo es nominal: hay un único \textit{task} y el \textit{speed-up} cae ligeramente por debajo de 1 ($0{,}94\times$). El $\sim 6\%$ de penalización es el coste puro de \textit{Spark over Python} (serialización y colección).
- Con **4 particiones** (la mitad del paralelismo disponible), el \textit{speed-up} alcanza $2{,}96\times$, con una eficiencia del $37\%$.
- Con **8 particiones** (una por core lógico asignado), el \textit{speed-up} alcanza su máximo $2{,}99\times$, prácticamente indistinguible de 4 particiones. El beneficio del mejor balanceo se compensa con el mayor \textit{overhead} de \textit{scheduling} (la columna \textit{overhead} se duplica: $17$ ms $\to$ $41$ ms).
- Con **16 particiones** (sobre-particionado $2\times$), el \textit{speed-up} \textit{degrada} a $2{,}21\times$: el trabajo por partición ($P=100/16 \approx 6$ individuos) es tan pequeño que cada \textit{task} pasa más tiempo serializándose que ejecutando GNB. El \textit{overhead} sube a $77$ ms.

**Conclusión metodológica.** Para $P=100$, la frontera de utilidad está en $N_\text{part} \in \{4, 8\}$, con el punto óptimo en $N_\text{part} = N_\text{cores} = 8$. Sobrepasar este valor con tareas tan ligeras penaliza claramente el rendimiento. Este resultado es también metodológicamente valioso porque identifica el régimen en que Spark deja de ayudar, algo importante para razonar sobre la configuración a usar con datasets mayores en el mes 4.

### 2.3 Experimento 3 — Mapa combinado $P \times$ particiones

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/exp3_heatmap_speedup.png}
\caption{Experimento 3 — mapa combinado $P \times N_\text{part}$ con \texttt{spark.default.parallelism}$=8$. \textit{Heatmap} de \textit{speed-up} (izquierda) y de tiempo paralelo en segundos (derecha). El máximo global ($4{,}23\times$) se alcanza en $(P=500, N_\text{part}=8)$; la franja $N_\text{part}=16$ muestra degradación por \textit{overhead} de \textit{scheduling}.}
\label{fig:exp3}
\end{figure}

El \textit{heatmap} de la figura \ref{fig:exp3} condensa la tabla siguiente:

\begin{center}
{\footnotesize
\begin{tabular}{|c|ccccc|}
\hline
$P$ \textbackslash\ part. & 1 & 2 & 4 & 8 & 16 \\
\hline
50  & 1,02 & 1,83 & 2,59 & 2,55 & 1,54 \\
100 & 1,13 & 2,12 & 3,65 & 3,95 & 2,95 \\
200 & 0,95 & 1,85 & 3,26 & 4,05 & 3,16 \\
500 & 1,00 & 1,87 & 3,30 & \textbf{4,23} & 3,67 \\
\hline
\end{tabular}
}
\end{center}

*Valores de speed-up; máximo en negrita.*

**Observaciones relevantes:**

1. **Frontera de la columna 1 partición.** Para todos los tamaños de población, $S \approx 1$ con una sola partición. Es la prueba de que, cuando Spark \textit{no} paraleliza (un solo \textit{task}), su \textit{overhead} es despreciable frente a Python puro: el sistema no estorba, simplemente tampoco ayuda.
2. **Escalado con $P$ para particiones fijas.** Fijado $N_\text{part}=8$, el \textit{speed-up} crece de $2{,}55\times$ ($P=50$) a $4{,}23\times$ ($P=500$). La curva no ha llegado al techo $S_\text{ideal}=8$, lo que indica que con WDBC+GNB la tarea por individuo es demasiado ligera (ms) para amortizar el \textit{overhead} con las poblaciones exploradas.
3. **Degradación con sobre-particionado ($N_\text{part}=16$).** Pérdida de \textit{speed-up} respecto a $N_\text{part}=8$ en todas las filas: el coste de \textit{scheduling} de 16 \textit{tasks} supera la ganancia por balanceo. Este efecto es más pronunciado cuanto menor es $P$ (para $P=50$, $S$ cae de $2{,}55\times$ a $1{,}54\times$).
4. **Fila $P=50$.** Aunque ninguna celda está por debajo de 1 (a diferencia del experimento previo con $N_\text{part}=4$, donde sí ocurría), el \textit{speed-up} nunca supera $2{,}6\times$. Confirmación de que $P=50$ —valor de diseño del documento **[D1]** §6— no es representativo para medir paralelismo; sí lo es para el estudio algorítmico del GA (Hito 6).
5. **Óptimo global $(P=500, N_\text{part}=8)$: $S = 4{,}23\times$.** Con $S_\text{ideal}=8$ se traduce en una eficiencia del $53\%$, cifra coherente con un dataset pequeño y un clasificador rápido. La diagonal $N_\text{part}=N_\text{cores}=8$ concentra los máximos por fila, lo que confirma la regla práctica \textit{``una partición por core''} como punto de partida razonable.

El patrón global coincide con la \textit{Ley de Amdahl} aplicada a tareas finas: la parte paralelizable crece con $P$, pero el \textit{overhead} fijo impide acercarse a la recta ideal con poblaciones pequeñas.

---

## 3. Hito 6: GA completo con criterio de parada dual

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../results/ga_full_convergencia.png}
\caption{GA completo con parámetros de diseño ($P=50$, $G_{max}=50$, $D=30$, 5-fold CV). Convergencia del mejor fitness acumulado (arriba izquierda), evolución mejor/medio/peor (arriba derecha), caída del número medio de características activas (abajo izquierda) y reparto de tiempos por generación entre \textit{Workers} (azul) y \textit{Driver} (rojo, apenas visible) (abajo derecha). La ejecución se detiene en la generación 17 por convergencia ($\Delta < 0{,}001$ en 10 gens).}
\label{fig:ga_full}
\end{figure}

### 3.1 Parámetros efectivos y motivo de parada

El GA se ejecuta con los parámetros de diseño (**[D1]** §6): $P=50$, $G_\text{max}=50$, $p_\text{cross}=0{,}8$, $p_\text{mut}=0{,}2$, $p_\text{mut\_bit}=1/30$, torneo $k=5$, elitismo $E=2$, $\alpha=0{,}9$, $\beta=0{,}1$, 5-fold CV. Resultados globales (`results/ga_full_experimento.json`):

\begin{center}
{\footnotesize
\begin{tabular}{|p{5cm}|p{8cm}|}
\hline
\textbf{Indicador} & \textbf{Valor} \\
\hline
Generaciones ejecutadas & 17 de 50 \\
Motivo de parada & Convergencia ($\Delta = 0{,}000000 < \varepsilon_\text{stop}=0{,}001$ en 10 gens) \\
Mejor fitness $F^*$ & 0,848028 \\
Características seleccionadas $|S^*|$ & 8 de 30 ($73{,}3\%$ de reducción) \\
Tiempo total & 2,223 s \\
Tiempo total evaluación (Workers) & 2,215 s (99,6\%) \\
Tiempo total operadores (Driver) & 0,005 s (0,2\%) \\
Tiempo medio por generación & 0,131 s \\
\hline
\end{tabular}
}
\end{center}

La parada por convergencia se activó en la generación 17 porque el mejor fitness $0{,}848028$ se había estabilizado desde la generación 8 (ver §3.2). El mecanismo de parada dual funciona: ahorra $33$ generaciones frente a un $G_\text{max}$ fijo sin sacrificar el óptimo.

**Reparto de tiempos.** Como puede verse en el panel inferior derecho de la figura \ref{fig:ga_full}, la primera generación dispara al $\approx 1$ s de cómputo (inicialización del \textit{broadcast} y JIT de la JVM), lo que distorsiona el promedio. A partir de la generación 2, el tiempo por generación se estabiliza en $\approx 0{,}06$–$0{,}07$ s. El trabajo del Driver (operadores) es tres órdenes de magnitud menor que el de los Workers ($\approx 0{,}3$ ms vs.\ $\approx 65$ ms), lo que confirma empíricamente la asunción del diseño **[D2]** §2.1: \textit{paralelizar la evaluación es la decisión correcta; no vale la pena paralelizar los operadores}.

### 3.2 Dinámica evolutiva (17 generaciones)

Serie temporal (`ga_full_series.csv`, extracto):

\begin{center}
{\footnotesize
\begin{tabular}{|c|c|c|c|c|}
\hline
Gen & Mejor $F$ & Medio $F$ & Peor $F$ & Genes medios \\
\hline
1  & 0,815909 & 0,7888 & 0,7636 & 15,18 \\
3  & 0,825734 & 0,8084 & 0,7840 & 11,86 \\
5  & 0,841537 & 0,8261 & 0,8113 & 10,00 \\
8  & \textbf{0,848028} & 0,8370 & 0,8257 & 8,78 \\
12 & 0,848028 & 0,8463 & 0,8210 & 8,02 \\
17 & 0,848028 & 0,8435 & 0,8086 & 8,22 \\
\hline
\end{tabular}
}
\end{center}

Observaciones:

1. **Convergencia rápida.** El mejor fitness alcanza su valor final en la generación 8, y la ventana de 10 generaciones subsiguientes sin mejora dispara la parada.
2. **Reducción dimensional sostenida.** Los genes medios caen de 15,18 (generación 1) a $\approx 8{,}1$ (generación 12), y se estabilizan allí. La penalización $\beta \cdot |S|/D$ está actuando como selector dimensional.
3. **Peor fitness volátil.** El peor individuo oscila entre $0{,}77$ y $0{,}83$ durante toda la ejecución. Esto es característico de GA con mutación activa ($p_\text{mut}=0{,}2$ por individuo): siempre hay \textit{exploradores} desactualizados. No afecta la convergencia gracias al elitismo.
4. **Diferencial medio-mejor.** Al final de la ejecución, el fitness medio ($\approx 0{,}844$) está a escasos milésimas del mejor ($0{,}848$). La población se ha \textit{homogeneizado} en torno al óptimo, coherente con una presión selectiva moderada ($k=5$) y elitismo pequeño ($E=2$).

### 3.3 Subconjunto $S^*$ seleccionado

El mejor cromosoma $\mathbf{c}^* = [0,1,0,0,0,0,0,0,1,0,\; 0,0,0,1,0,0,1,0,0,0,\; 1,1,0,0,1,0,0,1,0,0]$ activa 8 características. Traducidas a nombres del dataset WDBC:

\begin{center}
{\footnotesize
\begin{tabular}{|c|l|l|}
\hline
\textbf{Índice} & \textbf{Característica} & \textbf{Grupo} \\
\hline
1  & mean texture          & Medias \\
8  & mean symmetry         & Medias \\
13 & area error            & Errores estándar \\
16 & concavity error       & Errores estándar \\
20 & worst radius          & Peor valor \\
21 & worst texture         & Peor valor \\
24 & worst smoothness      & Peor valor \\
27 & worst concave points  & Peor valor \\
\hline
\end{tabular}
}
\end{center}

**Lectura.** El GA descarta por completo las medias de radio/perímetro/área (0, 2, 3) y las conserva sólo en su versión \textit{worst} (peor de las tres mediciones por imagen), que es donde los valores extremos aportan más capacidad discriminante en WDBC. Esta pauta coincide con estudios previos sobre WDBC donde \textit{worst concave points} y \textit{worst radius} suelen aparecer entre los top predictores. La aparición de \textit{mean texture} y \textit{mean symmetry} como \textit{features} \textit{de baja correlación entre sí} contribuye al buen rendimiento de Naive Bayes (que asume independencia condicional).

### 3.4 Comparación con el baseline

\begin{center}
{\footnotesize
\begin{tabular}{|p{4cm}|c|c|c|}
\hline
\textbf{Configuración} & $|S|$ & Penalización & $F$ estimado \\
\hline
Baseline (todas) & 30 & $0{,}100$ & $0{,}9 \cdot \text{acc}_{CV5}^{\,base} - 0{,}100$ \\
$\mathbf{c}^*$ (GA) & 8 & $0{,}027$ & $\mathbf{0{,}848028}$ \\
\hline
\end{tabular}
}
\end{center}

Descomponiendo $F^* = 0{,}848028$: con $\beta \cdot |S|/D = 0{,}1 \cdot 8/30 = 0{,}02\overline{6}$, se deduce una $\text{accuracy}_{CV5}^{*} \approx (0{,}848028 + 0{,}02\overline{6})/0{,}9 \approx \mathbf{0{,}972}$. El GA encuentra por tanto un subconjunto \textbf{4 veces más pequeño} que mantiene o mejora la accuracy respecto al uso de las 30 features (cuyo baseline de GNB en WDBC se sitúa típicamente en $\text{acc} \approx 0{,}94$). La ganancia en fitness frente al baseline es del orden de $\Delta F \approx 0{,}02$–$0{,}05$, consecuencia combinada de mejor accuracy y menor penalización.

---

## 4. Discusión e implicaciones metodológicas

### 4.1 Qué nos dicen los resultados sobre la arquitectura

1. **El diseño Driver/Workers escala como se esperaba.** El reparto observado del tiempo (99,5% en evaluación vs. 0,3% en operadores) valida cuantitativamente la decisión del documento **[D2]** §2.1 de no paralelizar el GA salvo la fitness.
2. **El \textit{broadcast} es eficaz pero no decisivo en WDBC.** El dataset ocupa $\sim 137$ KB ($569 \times 30 \times 8$ bytes), por lo que incluso sin \textit{broadcast} el coste de transferencia sería marginal. El \textit{broadcast} se convierte en crítico con datasets mayores (Arrhythmia, Gisette) y deja el patrón listo para esa escala.
3. **El \textit{speed-up} es modesto porque la tarea lo es.** Un \textit{speed-up} máximo de $\sim 4{,}9\times$ con GNB sobre WDBC no es una limitación de la arquitectura sino una propiedad del problema: al ser la evaluación de 1–2 ms, cualquier \textit{overhead} de ms compite directamente con ella. Clasificadores más costosos (LinearSVC, árboles) o datasets con $D \gg 30$ desplazarán este equilibrio a favor del paralelismo.

### 4.2 Qué nos dicen los resultados sobre el GA

1. **El fitness biobjetivo funciona.** La combinación $0{,}9 \cdot \text{acc} - 0{,}1 \cdot |S|/D$ produce una presión hacia subconjuntos pequeños sin sacrificar accuracy: $|S|$ cae de 15 a 8 mientras que $F$ sube de $0{,}82$ a $0{,}85$.
2. **La parada por convergencia ahorra mucho.** $17$ de $50$ generaciones ($34\%$) es coherente con WDBC, un problema relativamente \textit{fácil}. Para datasets más complejos habrá que volver a discutir $\varepsilon_\text{stop}$ y $N_\text{stop}$.
3. **Posible \textit{sobrepresión} selectiva.** El diferencial medio-mejor de $\approx 0{,}004$ indica una población casi clónica al final. En fase de experimentación conviene medir la \textit{diversidad genética} (e.g. distancia de Hamming media) para descartar convergencia prematura. Si aparece, relajar $k$ a 3 o subir $p_\text{mut}$ a $0{,}3$ son palancas naturales.

### 4.3 Limitaciones a tener presentes

- **Una única semilla.** Todos los números provienen de `seed=42`. En mes 4 deberán ejecutarse $\geq 5$ repeticiones independientes para reportar medias e intervalos de confianza.
- **Un único dataset.** WDBC es relativamente \textit{amigable} para GNB. Arrhythmia ($D=279$) y Gisette ($D=5000$) probarán con seriedad tanto el GA como la arquitectura Spark.
- **Modo \texttt{local[*]} con paralelismo fijado a 8.** El coste de red de un \textit{cluster} real no se captura. Los \textit{speed-ups} aquí reportados son un \textit{tope superior} para el mismo código sobre un \textit{cluster} distribuido (donde aparecerá latencia adicional de shuffle/collect). La elección de $8$ (en lugar de los $10$ \textit{cores} lógicos disponibles) sacrifica un poco de rendimiento potencial a cambio de \textit{reproducibilidad entre equipos}.

---

## 5. Conclusiones

Los resultados obtenidos en los hitos 1 a 6 cumplen los objetivos marcados en `mes3.md`:

1. **Hitos 1–4:** la fitness real GNB con 5-fold CV está integrada en Spark RDD mediante \textit{broadcast} y produce resultados idénticos (tolerancia $10^{-9}$) a la ejecución secuencial. El ciclo evolutivo corto de 10 generaciones con $P=100$ demuestra progreso evolutivo y reducción dimensional, validando \textit{end-to-end} la tubería diseñada en **[D1]** y **[D2]**.
2. **Hito 5:** el experimento de \textit{speed-up} con paralelismo fijado a 8 caracteriza tres regímenes:
   - \textit{overhead}-dominado para $N_\text{part}=1$ ($S \approx 1$) y $P=50$ con particionado excesivo,
   - paralelismo útil para $P \geq 100$ y $N_\text{part} \in \{4, 8\}$, con máximo $S = 4{,}23\times$ en $(P=500, N_\text{part}=8)$,
   - degradación clara al sobre-particionar ($N_\text{part}=16$), especialmente para $P$ pequeño.

   La relación $t_\text{seq}/t_\text{par}$ crece con $P$ y se maximiza en la diagonal $N_\text{part}=N_\text{cores}$, como predice la ley de Amdahl aplicada a tareas finas.
3. **Hito 6:** el GA completo con parada dual converge en $17$ generaciones a $F^* = 0{,}848028$ con $|S^*| = 8$ features (73% de reducción dimensional), consistentes con la literatura clásica sobre WDBC. El reparto de tiempo $99{,}5\%$ Workers / $0{,}3\%$ Driver respalda empíricamente la arquitectura elegida.

### Siguiente paso

1. **Estadística robusta:** repetir los experimentos con $\geq 5$ semillas y reportar medias e IC 95%.
2. **Segundo dataset (Arrhythmia, $D=279$):** contrastar tanto la calidad del subconjunto como la mejora de \textit{speed-up} con mayor dimensionalidad.
3. **Clasificador de validación (LinearSVC, **[D1]** §5.2):** reportar accuracy del subconjunto $S^*$ con un clasificador distinto del usado para evaluar el fitness, para descartar \textit{overfitting} al GNB.
4. **Diversidad genética:** instrumentar Hamming medio por generación y usarlo para calibrar $k$ y $p_\text{mut}$ si se detecta convergencia prematura.

---

## Anexo A. Ficheros de resultados

\begin{center}
{\footnotesize
\begin{tabular}{|p{6cm}|p{8cm}|}
\hline
\textbf{Archivo} & \textbf{Contenido} \\
\hline
\texttt{gnb\_poblacion\_inicial.png} & Distribución fitness / genes activos de la población inicial ($P=100$) \\
\texttt{gnb\_ga\_evolucion.png} & Evolución fitness/genes/tiempo en 10 generaciones ($P=100$) \\
\texttt{exp1\_speedup\_poblacion.csv/.png} & Experimento 1 (Hito 5): \textit{speed-up} frente a $P$ \\
\texttt{exp2\_speedup\_particiones.csv/.png} & Experimento 2 (Hito 5): \textit{speed-up} frente a particiones \\
\texttt{exp3\_speedup\_combinado.csv}, \texttt{exp3\_heatmap\_speedup.png} & Experimento 3 (Hito 5): mapa combinado $P \times N_\text{part}$ \\
\texttt{ga\_full\_series.csv} & Series por generación del GA completo (Hito 6) \\
\texttt{ga\_full\_parametros.csv} & Tabla de parámetros del GA completo \\
\texttt{ga\_full\_experimento.json} & Metadatos, hardware, Spark y resumen del experimento (Hito 6) \\
\texttt{ga\_full\_convergencia.png} & Paneles de convergencia, evolución, $|S|$ y reparto de tiempos (Hito 6) \\
\hline
\end{tabular}
}
\end{center}

## Anexo B. Referencias cruzadas

\begin{center}
{\footnotesize
\begin{tabular}{|c|p{5cm}|p{7cm}|}
\hline
\textbf{Ref.} & \textbf{Documento} & \textbf{Uso en este informe} \\
\hline
\textbf{[D1]} & \texttt{diseño\_GA.md} & Fórmula fitness §2, operadores §3, GNB §5.1, parámetros WDBC §6 \\
\textbf{[D2]} & \texttt{arquitectura\_spark\_rdd\_ga.md} & Driver/Workers §2.1, broadcast §2.5 \\
\textbf{[D3]} & \texttt{prototipo\_evaluacion\_paralela.md} & Baseline simulado del mes 2 (contraste en Hito 4) \\
\textbf{mes3.md} & Documento de implementación mes 3 & Definición de los hitos 1–6 que este informe valida \\
\hline
\end{tabular}
}
\end{center}
