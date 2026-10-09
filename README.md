# Análisis de Burnout en Desarrolladores — Limpieza de datos + Regresión

**Documentación del notebook `burnout_analysis_limpieza_sin_statsmodels.ipynb`**

---

## Índice

1. [¿De qué se trata este trabajo?](#1-de-qué-se-trata-este-trabajo)
2. [Glosario rápido](#2-glosario-rápido)
3. [Los datos](#3-los-datos)
4. [Librerías utilizadas y para qué sirve cada una](#4-librerías-utilizadas-y-para-qué-sirve-cada-una)
5. [Diagnóstico: ¿qué tan sucio estaba el dataset?](#5-diagnóstico-qué-tan-sucio-estaba-el-dataset)
6. [Limpieza: qué hicimos y por qué](#6-limpieza-qué-hicimos-y-por-qué)
7. [El modelado paso a paso](#7-el-modelado-paso-a-paso)
8. [Resultados](#8-resultados)
9. [Diferencias con la versión con StatsModels](#9-diferencias-con-la-versión-con-statsmodels)
10. [Conclusiones](#10-conclusiones)
11. [Limitaciones (lo que el análisis NO dice)](#11-limitaciones-lo-que-el-análisis-no-dice)
12. [Cómo ejecutar el notebook](#12-cómo-ejecutar-el-notebook)
13. [Guía para la defensa: preguntas probables y cómo responderlas](#13-guía-para-la-defensa-preguntas-probables-y-cómo-responderlas)

---

## 1. ¿De qué se trata este trabajo?

El **burnout** es el agotamiento que aparece por el estrés laboral sostenido. En este trabajo nos hicimos dos preguntas:

1. **¿Se puede predecir el nivel de estrés de un desarrollador** (de 0 a 100) a partir de sus hábitos, como cuántas horas trabaja, cuánto duerme, cuánto café toma o cuántas reuniones tiene?
2. **¿Qué hábitos influyen más** en ese estrés?

Partimos de los **datos crudos**, con valores faltantes e inconsistencias. Por eso el trabajo tiene dos grandes partes:

- **Parte A — Limpieza:** detectar los problemas de los datos y corregirlos con criterio.
- **Parte B — Modelado:** entrenar modelos que predicen el estrés y analizar qué variables pesan más.

> **En una frase:** revisamos y limpiamos los datos de 7.000 desarrolladores y nos quedamos con 5.606 registros completos y reales. Le mostramos a la computadora 4.484 de ellos con su nivel de estrés, aprendió el patrón y después adivinó el estrés de otros 1.122 que nunca había visto. Se equivocó en promedio **8 puntos sobre 100**.

---

## 2. Glosario rápido

| Término | Qué significa, en simple |
|---|---|
| **Dataset** | La tabla de datos. Cada fila es un desarrollador y cada columna una característica. |
| **Dato "sucio"** | Un dato con problemas: vacío, duplicado, imposible (ej.: dormir 30 horas), mal escrito o contradictorio con otro dato. |
| **Valor faltante (nulo / NaN)** | Una celda vacía: no se registró ese dato. |
| **`dropna()`** | La función de Pandas que **elimina las filas que tienen algún valor vacío**. Es la estrategia que elegimos. |
| **Imputar** | Rellenar un valor faltante con una estimación (por ejemplo, la mediana). Es la alternativa que **comparamos y descartamos**. |
| **Mediana** | El valor del medio cuando se ordenan los datos de menor a mayor. |
| **Outlier (valor atípico)** | Un valor muy alejado del resto (ej.: alguien con 25 reuniones por día cuando la mayoría tiene 5). |
| **Variable objetivo (y)** | Lo que queremos predecir: el **nivel de estrés**. |
| **Variables predictoras (X)** | Los datos que usamos para predecir: horas de trabajo, sueño, cafeína, etc. |
| **Modelo** | La "fórmula" o "receta" que la computadora aprende a partir de los datos. |
| **Train / Test** | Dividimos los datos en una parte para **enseñar** (train, 80%) y otra para **tomar examen** (test, 20%). El examen se hace con datos que el modelo nunca vio. |
| **Correlación** | Mide si dos variables se mueven juntas. Va de -1 a +1: **+1** = cuando una sube la otra también; **-1** = cuando una sube la otra baja; **0** = sin relación. |
| **R² (R cuadrado)** | Qué porcentaje del estrés logra explicar el modelo. **0.82 = 82%**. Cuanto más cerca de 1, mejor. |
| **MAE** | En promedio, por cuántos puntos se equivoca el modelo. |
| **RMSE** | Parecido al MAE, pero castiga más los errores grandes. |
| **Coeficiente** | En la regresión lineal, cuánto cambia el estrés si una variable sube 1 unidad (dejando las demás fijas). |
| **p-valor** | En estadística clásica, la probabilidad de ver un efecto así "por casualidad". Lo da StatsModels; en esta versión **no lo usamos** (ver sección 9). |
| **Residuo** | Error de una predicción: valor real − valor predicho. |
| **Data leakage (fuga de datos)** | Hacer trampa sin querer: darle al modelo una pista que contiene la respuesta, o dejar que "espíe" el examen. |
| **Multicolinealidad** | Cuando dos variables predictoras miden casi lo mismo y el modelo no sabe a cuál atribuirle el efecto. |

---

## 3. Los datos

**Archivo:** `developer_burnout_dataset_7000.csv`, con **7.000 filas** (desarrolladores) y **12 columnas**.

| Columna | Qué mide | Rango real en los datos |
|---|---|---|
| `age` | Edad | 20 a 44 años |
| `experience_years` | Años de experiencia | 0 a 19 |
| `daily_work_hours` | Horas de trabajo por día | 4 a 14 |
| `sleep_hours` | Horas de sueño | 4 a 9 |
| `caffeine_intake` | Consumo de cafeína (unidades por día) | 0 a 7 |
| `bugs_per_day` | Errores de código (bugs) por día | 0 a 19 |
| `commits_per_day` | Commits (entregas de código) por día | 0 a 29 |
| `meetings_per_day` | Reuniones por día | 0 a 9 |
| `screen_time` | Horas frente a la pantalla | 5.2 a 18.9 |
| `exercise_hours` | Horas de ejercicio por día | 0 a 2 |
| **`stress_level`** | **Nivel de estrés — lo que predecimos** | **0 a 100** |
| `burnout_level` | Categoría de burnout: Low / Medium / High | texto |

Las 10 primeras son las **predictoras**. `burnout_level` **no se usa para predecir**, porque está calculada a partir del estrés (sería data leakage, ver sección 13).

---

## 4. Librerías utilizadas y para qué sirve cada una

Una **librería** es un paquete de código ya hecho que alguien programó y que podemos reutilizar, para no escribir todo desde cero.

| Librería | ¿Qué es? (en simple) | ¿Para qué la usamos en este trabajo? |
|---|---|---|
| **Pandas** (`pandas`) | Es como un **Excel dentro de Python**: maneja tablas (llamadas *DataFrames*). | Es la protagonista de la limpieza: leer el CSV, contar vacíos (`isnull`), buscar duplicados (`duplicated`), normalizar texto (`str.strip`), **eliminar filas incompletas (`dropna`)**, calcular cuantiles y correlaciones, y armar todas las tablas de resultados. En la comparación usamos también `fillna` para probar la alternativa de imputar. |
| **NumPy** (`numpy`) | La base matemática de Python: cálculos con muchos números a la vez, muy rápido. | Representar valores vacíos (`np.nan`), calcular la raíz cuadrada del RMSE (`np.sqrt`) y valores absolutos (`np.abs`). |
| **Matplotlib** (`matplotlib`) | La librería clásica para **hacer gráficos**. | Histograma de edad de inicio laboral, gráficos de barras de importancia, gráficos de dispersión (real vs. predicho) e histograma de residuos. |
| **Seaborn** (`seaborn`) | Se apoya en Matplotlib y hace **gráficos estadísticos más lindos** con menos código. | **Boxplots** para detectar outliers, histogramas con curva de densidad, gráfico de barras de burnout y el **mapa de calor** de correlaciones. |
| **Scikit-learn** (`sklearn`) | La librería más usada de **Machine Learning** en Python: trae modelos y herramientas listas. | Todo el modelado (detalle abajo). |
| **os** | Viene incluida con Python. Sirve para trabajar con archivos y carpetas. | Verificar que el archivo CSV exista antes de leerlo. |
| **Jupyter Notebook** (`notebook`) | El programa donde escribimos y ejecutamos el análisis. Mezcla código, texto explicativo y gráficos en un solo documento. | Es el "cuaderno" donde vive todo el trabajo. |

### 4.1 Herramientas de scikit-learn que usamos

| Herramienta | Para qué sirve |
|---|---|
| `train_test_split` | Divide los datos en **80% para entrenar** y **20% para evaluar**. Con `random_state=42` la división es siempre la misma, así los resultados se pueden repetir. |
| `StandardScaler` | Pone todas las variables en la **misma escala** (promedio 0, desvío 1) para comparar de forma justa cuánto pesa cada una. |
| `LinearRegression` | **Regresión lineal**: encuentra la fórmula (sumas y multiplicaciones) que mejor relaciona los hábitos con el estrés. También la usamos para calcular el VIF. |
| `DecisionTreeRegressor` | **Árbol de decisión**: predice haciendo preguntas en cadena, tipo "¿trabaja más de 9 horas? → sí → ¿tiene más de 10 bugs?…". Lo limitamos a 4 niveles. |
| `plot_tree` | **Dibuja** el árbol de decisión para poder leer sus preguntas. |
| `RandomForestRegressor` | **Bosque aleatorio**: entrena **300 árboles** distintos y promedia sus respuestas. Suele ser más preciso que un árbol solo y nos dice qué variables usó más. |
| `r2_score`, `mean_absolute_error`, `mean_squared_error` | Las **métricas** para medir qué tan bien predice cada modelo (R², MAE y, a partir de la última, el RMSE). |

### 4.2 ¿Por qué "sin StatsModels"?

**StatsModels** es una librería que da tests estadísticos formales (como los *p-valores*). La consigna pedía usar solo herramientas vistas en clase, así que la reemplazamos así:

- **VIF** (multicolinealidad): lo calculamos "a mano" con `LinearRegression`.
- **Significancia de las variables**: en vez de p-valores, comparamos **tres métodos distintos** para rankear la importancia. Si los tres coinciden, la conclusión es confiable.

---

## 5. Diagnóstico: ¿qué tan sucio estaba el dataset?

Antes de limpiar hay que **medir** los problemas. Un médico no receta sin diagnosticar, y con los datos pasa lo mismo. Revisamos ocho cosas:

| # | Qué revisamos | Cómo | Resultado |
|---|---|---|---|
| 1 | **Valores faltantes por columna** | `isnull().sum()` | ❌ **140 vacíos en cada una de las 12 columnas** (2%). |
| 2 | **Valores faltantes por fila** | Contar vacíos en cada fila | ❌ Los vacíos están **dispersos**: **1.510 filas (21.6%)** tienen al menos uno. |
| 3 | **Filas duplicadas** | `duplicated()` | ✅ Ninguna. |
| 4 | **Valores fuera de rango** | Definimos límites reales (ej.: sueño entre 0 y 24 h, estrés entre 0 y 100) | ✅ Ninguno. |
| 5 | **Decimales donde no corresponde** | Edad, bugs o reuniones deberían ser enteros | ✅ Todos enteros. |
| 6 | **Categorías mal escritas** | Variantes de `burnout_level` (espacios, mayúsculas) | ✅ Solo Low / Medium / High. |
| 7 | **Outliers** | Regla del rango intercuartílico (IQR) y boxplots | ✅ Ninguno. |
| 8 | **Coherencia lógica** | ¿Trabajo + sueño > 24 h? ¿Pantalla < trabajo? ¿Edad de inicio laboral? | ⚠️ **1.998 personas "empezaron a trabajar" antes de los 18**, algunas con 1 año de edad (edad − experiencia). |

### 5.1 El hallazgo más importante del diagnóstico

Que cada columna tenga **solo 2%** de vacíos parece poco. Pero como los vacíos caen **en filas distintas**, usar `dropna()` sobre el dataset tal como viene borraría **más del 20%** de las filas:

```
Filas sin ningún vacío:   5.490
Filas con 1 vacío:        1.352
Filas con 2 vacíos:         146
Filas con 3 vacíos:          12
```

Elegimos igual `dropna()`, pero **antes recuperamos lo que se podía recuperar sin inventar nada** (ver sección 6, paso 2). Así terminamos con **5.606 filas** en lugar de 5.490.

### 5.2 ¿Cómo funciona la regla IQR para detectar outliers?

1. Se ordenan los valores y se busca el **25%** (Q1) y el **75%** (Q3).
2. La distancia entre ellos es el **IQR** (rango intercuartílico): ahí vive la "mitad central" de los datos.
3. Un valor es **atípico** si está a más de **1.5 × IQR** por debajo de Q1 o por encima de Q3.

Es lo mismo que muestran los **boxplots**: los puntos sueltos fuera de los "bigotes" son los outliers. En este dataset no apareció ninguno.

---

## 6. Limpieza: qué hicimos y por qué

Trabajamos sobre una **copia** de los datos (`df = df_raw.copy()`) para no modificar nunca los originales.

### Paso 1 — Normalizar el texto de `burnout_level`
Sacamos espacios y unificamos mayúsculas (ej.: `" high"` → `"High"`). En este dataset no cambió nada, pero es una **medida preventiva** por si llegan datos mal cargados.

### Paso 2 — Reconstruir `burnout_level` faltante (138 valores recuperados)
`burnout_level` es una categoría **calculada a partir del estrés**. Primero averiguamos la regla exacta mirando los datos:

| Categoría | Estrés mínimo | Estrés máximo |
|---|---|---|
| Low | 0.00 | 35.00 |
| Medium | 35.02 | 69.97 |
| High | 70.01 | 100.00 |

**Regla:** Low si estrés ≤ 35 · Medium si está entre 35 y 70 · High si es > 70.

Verificamos que la regla **coincide en el 100%** de las filas conocidas. Recién entonces la usamos para completar las **138 filas** que tenían estrés pero no categoría. No es un invento: es aplicar la misma regla con la que se crearon los datos.

> **¿Por qué importa esto si usamos `dropna()`?** Porque `dropna()` borra cualquier fila con un vacío, **en cualquier columna**. Sin este paso perderíamos 116 filas solo por faltarles `burnout_level`, una columna que ni siquiera se usa en el modelo.

### Paso 3 — Eliminar las filas sin `stress_level` (140 filas)
`stress_level` es **lo que queremos predecir**. **No se puede inventar la respuesta**. Estas 140 filas se eliminan (7.000 → 6.860). Lo hacemos como paso explícito, aunque `dropna()` también las borraría, para que quede claro el motivo.

> 💡 **Mejora respecto del notebook de limpieza del equipo (`proCoders-limpiezaDataSet.ipynb`):** ese notebook rellenaba con la mediana **todas** las columnas numéricas, **incluido `stress_level`**. Eso le daba al modelo 140 respuestas inventadas, todas iguales (53.8). En esta versión esas filas se eliminan.

### Paso 4 — Marcar (no borrar) la inconsistencia edad/experiencia
Hay ~1.960 personas cuya **edad − experiencia < 18**, o sea que "empezaron a trabajar" siendo menores, algunas con 1 año. Es un error típico de datos generados al azar. **Decidimos no borrarlas**, porque:
- Son **demasiadas** (casi un tercio del dataset).
- **No sabemos cuál** de las dos columnas está mal: ¿la edad o la experiencia?
- **Ninguna de las dos influye** en el estrés (correlación ≈ 0).

En vez de borrarlas, las **marcamos** con una columna `inconsistente_edad_exp` y después **comprobamos** si afectan el resultado (sección 8.6). Spoiler: no lo afectan.

### Paso 5 — Eliminar las filas con vacíos en las predictoras: `dropna()`
Por último, aplicamos `df.dropna()`: se eliminan las **1.254 filas** a las que les falta al menos un dato en las predictoras.

```
Filas después de la limpieza:   6.860
Filas eliminadas por dropna():  1.254
Filas para el modelo:           5.606  (80.1% del dataset original)
```

**¿Por qué `dropna()` y no rellenar (imputar)?**
- El modelo aprende **solo de datos reales**, sin valores estimados.
- Con 5.606 filas **sobran datos** para entrenar con confianza.
- Es más **simple y transparente** de explicar.
- Lo comprobamos con datos (ver 6.2): `dropna()` dio mejores métricas.

### 6.1 Resumen de la limpieza

| Paso | Filas |
|---|---|
| Dataset original | 7.000 |
| − filas sin `stress_level` | 6.860 |
| − filas con vacíos en predictoras (`dropna()`) | **5.606** |

| | Antes | Después |
|---|---|---|
| Celdas vacías | 1.680 | **0** |
| `burnout_level` vacíos | 140 | **0** (138 reconstruidos + 2 eliminados con su fila) |

### 6.2 ¿Borrar o imputar? Lo comprobamos

| Estrategia | Filas usadas | R² | MAE |
|---|---|---|---|
| **Borrar filas incompletas (`dropna`) — la elegida** | **5.606** | **0.823** | **7.97** |
| Imputar con la mediana de train | 6.860 | 0.802 | 8.12 |

Con `dropna()` el modelo predice mejor, porque aprende y se evalúa **solo con datos reales**. Imputar conservaría ~1.250 filas más, pero agrega valores estimados que no son el dato verdadero.

> **Aclaración honesta (por si la profesora pregunta):** las dos estrategias se evalúan sobre tests distintos, porque el de `dropna()` solo tiene filas completas. Por eso la diferencia de R² es **orientativa**, no una prueba definitiva. La razón principal para elegir `dropna()` es que **no inventa datos** y que nos quedan suficientes filas.

---

## 7. El modelado paso a paso

### Paso 1 — Exploración (EDA) con los datos limpios
**EDA** = *Exploratory Data Analysis*, "conocer los datos antes de modelar". Hicimos estadísticas descriptivas, histogramas de cada variable, la distribución de burnout (Medium 51%, High 26%, Low 23%) y el **mapa de calor de correlaciones**.

### Paso 2 — Preparación
Aplicamos `dropna()` (5.606 filas), elegimos las 10 predictoras y el objetivo, y dividimos 80/20: **4.484 para entrenar y 1.122 para el test**.

### Paso 3 — Regresión lineal
Entrenamos el modelo y medimos R², MAE y RMSE. Como los datos están en **unidades reales** (horas, cafés, reuniones), los coeficientes se leen directo: *"cada hora extra de trabajo suma 4.9 puntos de estrés"*.

### Paso 4 — Coeficientes estandarizados
Los coeficientes crudos **no se pueden comparar entre sí**: "1 hora de ejercicio" y "1 commit" son cosas muy distintas. Con `StandardScaler` todas las variables quedan en la misma escala, y el tamaño del coeficiente indica la importancia real.

### Paso 5 — Multicolinealidad (VIF)
Para cada variable, intentamos predecirla con las demás. Si se predice muy bien, es "casi una copia" de otras.
- **VIF ≈ 1:** aporta información propia ✅
- **VIF > 5:** sospechoso ⚠️
- **VIF > 10:** grave ❌

### Paso 6 — Árbol de decisión y Random Forest
Dos modelos más para comparar, con el **mismo** train/test. Además **dibujamos el árbol** para ver qué preguntas hace.

### Paso 7 — Consistencia (en lugar de p-valores)
Rankeamos las variables con **tres métodos independientes** (correlación, coeficiente estandarizado e importancia del Random Forest). Si coinciden, la conclusión es robusta.

### Paso 8 — Prueba de robustez
Reentrenamos el modelo **sin** las filas con inconsistencia edad/experiencia y lo evaluamos con el mismo test. Así comprobamos si esas filas "ensucian" el resultado.

### Paso 9 — Diagnóstico de residuos
Tres gráficos para ver si el modelo se equivoca "parejo":
- **Real vs. Predicho:** los puntos deberían estar cerca de la diagonal roja.
- **Residuos vs. Predicho:** repartidos al azar alrededor de 0, sin formar curvas ni embudos.
- **Histograma de residuos:** forma de campana centrada en 0.

---

## 8. Resultados

### 8.1 Comparación de modelos

| Modelo | R² | MAE | RMSE | En palabras |
|---|---|---|---|---|
| **Regresión lineal** 🏆 | **0.823** | **7.97** | **9.90** | Explica el 82% del estrés; se equivoca en promedio ~8 puntos sobre 100. |
| Random Forest | 0.791 | 8.77 | 10.77 | Muy cerca, ~9 puntos de error. |
| Árbol de decisión | 0.586 | 12.41 | 15.16 | El más débil: un solo árbol de 4 niveles es demasiado simple. |

**¿Por qué ganó el modelo más simple?** Porque en estos datos las relaciones son mayormente **lineales**: más horas de trabajo equivale a más estrés, de forma bastante proporcional. Cuando el patrón es una "recta", la regresión lineal es la herramienta justa.

### 8.2 ¿Cuánto suma o resta cada hábito? (coeficientes en unidades reales)

Manteniendo todo lo demás igual:

| Hábito | Efecto en el estrés (sobre 100) |
|---|---|
| 🏃 +1 hora de ejercicio | **−5.1 puntos** |
| 🕐 +1 hora de trabajo | **+4.9 puntos** |
| 😴 +1 hora de sueño | **−3.8 puntos** |
| 📅 +1 reunión por día | **+2.8 puntos** |
| ☕ +1 unidad de cafeína | **+2.0 puntos** |
| 🐛 +1 bug por día | **+1.9 puntos** |
| Experiencia, pantalla, commits, edad | ≈ 0 (entre −0.05 y −0.01) |

### 8.3 Ranking de importancia (los tres métodos)

| Puesto | Variable | Correlación | Coef. estandarizado | Random Forest | **Promedio** |
|---|---|---|---|---|---|
| 1 | Horas de trabajo | 1 | 1 | 1 | **1.0** |
| 2 | Bugs por día | 3 | 2 | 2 | **2.3** |
| 3 | Reuniones por día | 4 | 3 | 3 | **3.3** |
| 4 | Horas de sueño | 5 | 4 | 4 | **4.3** |
| 5 | Cafeína | 6 | 5 | 5 | **5.3** |
| 6 | Tiempo de pantalla | 2 | 10 | 7 | **6.3** |
| 6 | Ejercicio | 7 | 6 | 6 | **6.3** |
| 8 | Experiencia | 8 | 7 | 10 | **8.3** |
| 9 | Commits por día | 10 | 8 | 8 | **8.7** |
| 10 | Edad | 9 | 9 | 9 | **9.0** |

**Los tres métodos coinciden en el top 5.** El único que "desentona" es el tiempo de pantalla: 2º en correlación pero último en el modelo lineal, por multicolinealidad (ver 8.5).

### 8.4 🤔 ¿Por qué el ejercicio tiene el efecto más grande por hora pero queda 6º en el ranking?

Es una pregunta clave para la defensa:
- **Por hora**, el ejercicio es el que más baja el estrés (−5.1 puntos).
- Pero en el dataset la gente hace **entre 0 y 2 horas** de ejercicio: la diferencia entre personas es chica.
- En cambio, las horas de trabajo van de **4 a 14**: hay mucha más diferencia entre personas.

Por eso, en la práctica, **las horas de trabajo explican mucho más las diferencias de estrés** entre personas, aunque "hora por hora" el ejercicio tenga más efecto. Los coeficientes estandarizados corrigen justamente esto.

### 8.5 Multicolinealidad (VIF)

| Variable | VIF |
|---|---|
| Horas de trabajo | **7.05** ⚠️ |
| Tiempo de pantalla | **7.05** ⚠️ |
| Todas las demás | 1.00 ✅ |

Horas de trabajo y pantalla están muy relacionadas: quien trabaja más horas, pasa más tiempo frente a la pantalla. Por eso el modelo le da todo el peso a las horas de trabajo y casi nada a la pantalla.

### 8.6 Prueba de robustez

| | Filas de train | R² | MAE |
|---|---|---|---|
| Con todas las filas | 4.484 | 0.8234 | 7.97 |
| Sin filas inconsistentes (edad/experiencia) | 3.155 | 0.8228 | 7.98 |

**Prácticamente idénticos.** Los coeficientes también casi no cambian. Conclusión: la inconsistencia **no afecta** el análisis, así que fue correcto no borrar esas filas.

---

## 9. Diferencias con la versión con StatsModels

Este notebook parte de `burnout_analysis.ipynb` (la versión con StatsModels): **mismo dataset, mismo objetivo, mismas 10 predictoras, mismos tres modelos con la misma configuración y la misma estrategia de `dropna()`**. Las diferencias son:

| | Versión con StatsModels | Esta versión |
|---|---|---|
| **Limpieza** | Solo `df.dropna()` directo | Diagnóstico completo (8 revisiones) + `burnout_level` reconstruido **antes** de `dropna()` |
| **Filas para el modelo** | 5.490 | **5.606** (+116 filas recuperadas) |
| **Train / Test** | 4.392 / 1.098 | 4.484 / 1.122 |
| **R² regresión lineal** | 0.826 | 0.823 |
| **R² Random Forest** | 0.782 | 0.791 |
| **R² Árbol** | 0.596 | 0.586 |
| **Significancia** | p-valores (StatsModels OLS) | Coeficientes estandarizados + VIF + ranking con 3 métodos |
| **Extras** | — | Comparación `dropna` vs. imputar, prueba de robustez edad/experiencia, revisión de coherencia lógica |

### 9.1 ¿Por qué cambian un poco los números?
Porque las **116 filas recuperadas** cambian qué personas caen en train y en test. Las diferencias son mínimas (R² 0.826 vs. 0.823) y **las conclusiones son las mismas**: mismo modelo ganador, mismas variables importantes y coeficientes muy parecidos (por ejemplo, horas de trabajo +4.88 en la versión original y +4.89 en esta).

### 9.2 ¿Cómo se reemplazan los p-valores?
StatsModels decía qué variables eran "estadísticamente significativas" (p < 0.05). Acá lo resolvemos de otra forma:

| Versión con StatsModels (p-valores) | Esta versión (3 métodos) |
|---|---|
| Significativas: trabajo, sueño, cafeína, bugs, reuniones, ejercicio | Top 6 del ranking: trabajo, bugs, reuniones, sueño, cafeína y ejercicio (empatado con pantalla) |
| No significativas: edad, commits, pantalla | Últimas del ranking / coeficiente ≈ 0: edad, commits, pantalla (en el modelo lineal) |
| `experience_years`: significativa (p = 0.029) | Puesto 8: efecto minúsculo (−0.05 puntos por año) |

**Coinciden.** El único matiz es `experience_years`, y ver la pregunta correspondiente en la sección 13.

---

## 10. Conclusiones

**Sobre la limpieza:**
1. El dataset parecía tener "solo 2%" de vacíos, pero estaban dispersos en el **21.6% de las filas**. Medirlo antes fue clave para decidir.
2. Antes de usar `dropna()` **recuperamos 138 valores de `burnout_level`** con su regla exacta, lo que salvó 116 filas.
3. El modelo usa **5.606 filas (80% del original), todas con datos 100% reales**: ningún valor fue inventado ni estimado.
4. Se detectó una inconsistencia lógica (edad/experiencia) y se **comprobó con datos** que no afecta los resultados.

**Sobre el estrés:**
1. **Sí, se puede predecir** el estrés a partir de los hábitos: el modelo explica el **82%** con un error promedio de **~8 puntos sobre 100**.
2. **Lo que más aumenta el estrés:** 🕐 horas de trabajo, 🐛 bugs por día, 📅 reuniones y ☕ cafeína.
3. **Lo que más lo reduce:** 😴 dormir más y 🏃 hacer ejercicio.
4. **Edad, experiencia, commits y tiempo de pantalla** prácticamente no aportan al modelo.
5. **El modelo más simple (regresión lineal) fue el mejor**, porque las relaciones son mayormente proporcionales.

---

## 11. Limitaciones (lo que el análisis NO dice)

- **`dropna()` descarta ~20% de las filas.** Quedan suficientes, pero el modelo no está preparado para predecir a alguien con datos incompletos.
- **Correlación no es causalidad.** Encontramos que quienes trabajan más horas tienen más estrés, pero eso **no prueba** que trabajar más *cause* el estrés. Para afirmarlo haría falta un experimento.
- **Horas de trabajo y pantalla están entrelazadas** (VIF ≈ 7). Por eso la pantalla tiene alta correlación con el estrés (+0.55), pero el modelo no le da peso: su efecto lo "absorben" las horas de trabajo.
- **La regresión lineal puede predecir fuera de rango.** En 20 de 1.122 casos predijo estrés menor a 0 o mayor a 100.
- **La inconsistencia edad/experiencia sugiere que los datos son simulados.** Las conclusiones valen para este dataset y habría que validarlas con datos reales.
- **No usamos p-valores** (no estaban entre las herramientas vistas). Lo compensamos validando con tres métodos.

---

## 12. Cómo ejecutar el notebook

### Requisitos (una sola vez)
```
pip install notebook pandas numpy matplotlib seaborn scikit-learn
```

### Pasos
1. Abrir la carpeta `burnout_analysis_limpieza` en el Explorador de Windows, escribir `cmd` en la barra de direcciones y apretar Enter.
2. En la terminal que se abre, ejecutar: `jupyter notebook` (si no lo reconoce: `python -m notebook`).
3. Se abre el navegador. Hacer clic en `burnout_analysis_limpieza_sin_statsmodels.ipynb`.
4. Menú **Run → Run All Cells**.
5. Para cerrar: cerrar la pestaña y apretar **Ctrl + C** en la terminal.

### Cómo usar Jupyter
- El notebook está dividido en **celdas** de texto y de código.
- **Shift + Enter** ejecuta la celda seleccionada; el resultado aparece debajo.
- `[*]` = ejecutándose; `[5]` = ya terminó.
- **El orden importa:** si algo falla, usar **Kernel → Restart Kernel and Run All Cells**.
- `developer_burnout_dataset_7000.csv` tiene que estar en la **misma carpeta** que el notebook.
- El notebook ya viene **con los resultados guardados**, así que se pueden ver sin ejecutarlo.

---

## 13. Guía para la defensa: preguntas probables y cómo responderlas

### Sobre la limpieza

**❓ ¿Por qué usaron `dropna()` si perdían el 20% de los datos?**
> Porque así el modelo aprende **solo de datos reales**, sin valores inventados. Aun perdiendo ese 20%, quedan **5.606 personas**, más que suficientes para entrenar. Además lo comprobamos: con `dropna()` el modelo tuvo mejores métricas que rellenando con la mediana.

**❓ ¿No hubiera sido mejor rellenar los vacíos (imputar)?**
> Lo probamos. Imputar con la mediana conserva más filas (6.860), pero el R² bajó de 0.823 a 0.802. Además, rellenar mete valores que no son reales: si a alguien le falta su horario de trabajo y le ponemos la mediana (9 h), el modelo aprende de un dato falso. Aclaramos que la comparación es orientativa, porque cada estrategia se evalúa con un test distinto.

**❓ Si usaron `dropna()`, ¿para qué reconstruyeron `burnout_level`?**
> Porque `dropna()` borra filas con vacíos en **cualquier** columna. Sin reconstruirla, habríamos perdido 116 filas útiles solo por faltarles una columna que el modelo ni usa. Reconstruirla con su regla exacta nos dejó 5.606 filas en vez de 5.490.

**❓ ¿No es "inventar datos" reconstruir `burnout_level`?**
> No, porque no estimamos: aplicamos **la misma regla con la que se generó** la columna (Low ≤ 35 < Medium ≤ 70 < High). Antes de usarla verificamos que coincide en el 100% de los 6.722 casos conocidos.

**❓ ¿Por qué borraron explícitamente las filas sin `stress_level` si `dropna()` ya lo hacía?**
> Para dejar claro el **motivo**: es la variable que queremos predecir y la respuesta no se puede inventar. Es una decisión distinta a la de las predictoras, y queríamos que se vea.

**❓ ¿Por qué no borraron las personas con edad/experiencia inconsistente?**
> Porque eran casi un tercio del dataset, no sabemos cuál de las dos columnas está mal y ninguna influye en el estrés. En lugar de suponer, **lo comprobamos**: sin esas filas, el R² pasa de 0.8234 a 0.8228. Es decir, no cambia.

**❓ ¿Encontraron outliers?**
> Los buscamos con la regla IQR y con boxplots en las 11 variables numéricas, y no apareció ninguno. Por eso no recortamos ni borramos valores.

### Sobre el modelo

**❓ ¿Por qué no usaron `burnout_level` para predecir?**
> Porque está **calculada a partir del estrés**. Usarla sería data leakage: como darle al alumno las respuestas del examen.

**❓ ¿Qué significa R² = 0.82?**
> Que el modelo explica el **82% de las diferencias de estrés** entre las personas. El 18% restante se debe a factores que no están en los datos o a azar.

**❓ ¿Por qué ganó la regresión lineal y no el Random Forest?**
> Porque las relaciones en estos datos son mayormente lineales (proporcionales). Si el patrón es una recta, un modelo complejo no tiene nada extra que descubrir, y además el lineal es más fácil de interpretar.

**❓ El ejercicio tiene el coeficiente más grande. ¿Por qué no es la variable más importante?**
> Porque el coeficiente mide el efecto **por hora**, y la gente hace entre 0 y 2 horas de ejercicio, o sea que varía poco. Las horas de trabajo van de 4 a 14, entonces explican mucho más las diferencias reales entre personas. Por eso usamos coeficientes estandarizados para comparar.

**❓ ¿Por qué el tiempo de pantalla correlaciona tanto con el estrés pero no pesa en el modelo?**
> Por multicolinealidad: pantalla y horas de trabajo casi miden lo mismo (VIF ≈ 7). Cuando el modelo ya conoce las horas de trabajo, la pantalla no agrega información nueva.

### Sobre las diferencias con la versión con StatsModels

**❓ ¿Por qué no usaron p-valores?**
> Porque requieren StatsModels, que no vimos en clase. Validamos con **tres métodos independientes** y coinciden en el top 5. Además, las variables que StatsModels marcaba como significativas son las mismas que quedan arriba en nuestro ranking.

**❓ ¿Por qué el R² es 0.823 y no 0.826 como en la versión con StatsModels?**
> Porque recuperamos 116 filas reconstruyendo `burnout_level` antes del `dropna()`, y eso cambia qué personas quedan en train y en test. La diferencia es mínima y las conclusiones son las mismas.

**❓ En la versión con StatsModels, `experience_years` daba significativa (p = 0.029). ¿Por qué acá queda casi última?**
> No se contradicen. "Significativa" quiere decir que el efecto probablemente no es cero, **no que sea grande**. Su efecto es de apenas **−0.05 puntos de estrés por año** de experiencia: 10 años más de experiencia restan medio punto sobre 100. Con miles de datos, hasta efectos minúsculos pueden dar significativos. En la práctica, no influye.

**❓ ¿Entonces trabajar muchas horas causa estrés?**
> El análisis muestra una **asociación fuerte**, pero no prueba causalidad. Para eso haría falta un experimento controlado.

**❓ ¿Qué es `random_state=42`?**
> Una "semilla" que hace que la división de datos y el Random Forest den **siempre el mismo resultado**. Así cualquiera puede reproducir el análisis. El 42 es una convención.
