# LOTO — Rigor estadístico y teoría de juegos aplicados a la lotería

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)](https://scikit-learn.org/)

Proyecto de análisis del Loto chileno construido sobre una tesis incómoda:
**no se puede predecir**. En vez de esquivar esa conclusión, el proyecto la demuestra
con rigor y después construye lo único que sí admite optimización.

Son dos mitades que se responden entre sí:

| | Pregunta | Respuesta | Módulo |
|---|---|---|---|
| **Parte 1** | ¿Se puede predecir qué números salen? | No, y se demuestra | `modelo_clasificacion.py` |
| **Parte 2** | Entonces, ¿qué *sí* se puede optimizar? | El premio, no la probabilidad | `generador_apuestas.py` |

El valor del proyecto no está en un modelo con buenas métricas, sino en la
**metodología para no engañarse**: un backtest mal construido sobre estos datos
produce con facilidad resultados que parecen señal y no lo son.

---

## Arquitectura

```
                    ┌──────────────────┐
                    │     app.py       │  Streamlit: GUI + orquestador
                    └────────┬─────────┘
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌────────────────────┐        ┌──────────────────────┐
   │  scraper_polla.py  │        │ generador_apuestas.py│
   │  run_scraper()     │        │ generate_bets()      │
   └─────────┬──────────┘        └──────────────────────┘
             │ resultados.xlsx
             ▼
   ┌────────────────────┐        ┌──────────────────────┐
   │ consolidar_datos.py│───────▶│ modelo_clasificacion │
   │ update_database()  │        │ backtest temporal    │
   └─────────┬──────────┘        └──────────────────────┘
             ▼
    data/loto_historico.csv   ← fuente única de verdad
```

Cada módulo expone su lógica como función importable (`run_scraper()`,
`update_database()`, `generate_bets()`), de modo que la GUI los **importa
directamente** en vez de invocar subprocesos: los errores suben como excepciones
y llegan a la interfaz con su traza real.

| Archivo | Rol |
|---|---|
| `app.py` | Interfaz Streamlit y orquestador del pipeline |
| `scraper_polla.py` | Scraping de polla.cl (Selenium + BeautifulSoup), con modo web en vivo vía SSE |
| `consolidar_datos.py` | Validación, merge incremental y reconstrucción de fechas |
| `modelo_clasificacion.py` | Backtest temporal y prueba de ausencia de señal |
| `generador_apuestas.py` | Optimización combinatoria por teoría de juegos |
| `data/loto_historico.csv` | 1442 sorteos (3803–5244), fuente única de verdad |

---

## Instalación y uso

```bash
pip install -r requirements.txt
streamlit run app.py
```

La interfaz tiene una barra lateral donde se configura **el número de columnas de la
cartilla física** (las reglas visuales dependen de esa grilla) y dos paneles:

1. **Actualización de datos** — ejecuta scraper y consolidador en secuencia, con avance
   en vivo, y muestra las estadísticas del maestro resultante.
2. **Generador inteligente** — aplica los filtros con la cartilla configurada y entrega
   las combinaciones en tabla, descargables como CSV.

Cada módulo funciona también por línea de comandos:

```bash
python modelo_clasificacion.py --test 400 --refit 25
python generador_apuestas.py -n 5 --columnas 6 --altos 3
python consolidar_datos.py resultados.xlsx
```

---

## Ingeniería de datos

El dataset maestro se consolidó a partir de tres fuentes solapadas. La fusión se
validó antes de descartar los originales: **cero discrepancias** en los 1256 sorteos
comunes.

### Reconstrucción de fechas

186 sorteos llegaron sin fecha. El sorteo se realiza **martes, jueves y domingo**, y
sobre los 1256 registros con fecha la correspondencia `sorteo N ↔ N-ésimo día de sorteo`
resultó ser una **biyección exacta**: los 1256 días del rango están todos ocupados, sin
huecos ni duplicados, sin una excepción en 8 años. Los 6 sorteos que parecían anómalos
son los especiales de Nochebuena y Año Nuevo, que solo cambian de hora.

La reconstrucción se validó **fuera de muestra**: anclando el calendario en el sorteo
5214 (2025-01-07), predice correctamente las fechas de 8 sorteos scrapeados en vivo
**250 sorteos más allá del ancla** (5457–5464, julio-agosto 2026), 8 de 8 exactos.

La columna `fecha_origen` conserva la trazabilidad:

| `fecha_origen` | Sorteos | Confianza |
|---|---|---|
| `registrada` | 1256 | Fecha original de la fuente |
| `reconstruida` | 186 | Derivada del calendario |

De las reconstruidas, 30 están confirmadas por el ancla externa. Las otras 156
(año 2016) extrapolan hacia atrás y **no tienen validación independiente**: si Polla
usó otro calendario en 2016, esas fechas estarían corridas. Los números de esos
sorteos no están en duda, solo sus fechas.

---

## Parte 1 — Demostrar la ausencia de señal

El enfoque ingenuo —y el que traía la versión original de este proyecto— es una
regresión que predice `Numero ganador 3` a partir de la fecha. Está roto de raíz: esa
columna no es un fenómeno físico, es el resultado de **ordenar** los 6 números. El
modelo aprende la estadística de orden (la posición 3 tiende a ~18), baja el RMSE y
aparenta funcionar sin haber aprendido nada del sorteo.

La reformulación correcta:

- **Multi-etiqueta.** Una fila por (sorteo, número), etiqueta binaria "salió / no salió".
  Un clasificador estima `P(k)` para los 41 números y los rankea.
- **Split temporal estricto.** Entrena con sorteos ≤ N y predice el N+1, con reajuste
  periódico y ventana expansiva.
- **Features causales.** Frecuencias en ventanas de 10/25/50/100/250 sorteos, frecuencia
  histórica, sorteos desde la última aparición, día de semana y mes.
- **Métrica real.** Aciertos sobre 6, contra la hipergeométrica que describe el azar.

### Prueba de perturbación contra data leakage

Afirmar "no hay leakage" no basta. La construcción de features se verifica
programáticamente: se altera por completo el sorteo *t* y se comprueba que `X[:t+1]`
quede **bit a bit idéntico**, y que `X[t+1:]` sí cambie (confirmando que las features
usan el pasado y solo el pasado). Sin esa prueba, cualquier resultado positivo sería
indistinguible de una fuga temporal.

### Resultados — backtest sobre los últimos 400 sorteos

| Estrategia | Aciertos /6 | z | p |
|---|---|---|---|
| Modelo (HistGradientBoosting) | 0.8300 | −1.19 | 0.235 |
| Azar puro | 0.9125 | +0.85 | 0.395 |
| Los más frecuentes ("calientes") | 0.8450 | −0.82 | 0.414 |
| Los más atrasados ("fríos") | 0.8975 | +0.48 | 0.631 |
| **Esperanza teórica del azar** | **0.8780** | | |

Ninguna estrategia se distingue del azar. Lo más concluyente no es esa tabla sino la
calibración de las probabilidades:

- **ROC AUC = 0.4926** — sin información (0.5 es el valor de una moneda).
- **Log loss = 0.4166**, frente a **0.4163** de predecir la constante 6/41. El modelo
  es medible y literalmente **peor que no modelar**.
- Las `P(k)` se dispersan entre 0.1135 y 0.2400 alrededor del 0.1463 teórico. Esa
  dispersión es ruido puro: es el aspecto exacto de un modelo sobreajustando a un
  proceso aleatorio.

Un chi-cuadrado sobre los 1442 sorteos da **p = 0.92**: los 41 números son
estadísticamente indistinguibles de uniformes. El sorteo es limpio, y el modelo lo
confirma desde el lado predictivo.

---

## Parte 2 — Optimización por teoría de juegos

Si toda combinación es igual de probable, la pregunta deja de ser *cuál sale* y pasa a
ser **con cuánta gente hay que repartir el pozo si sale**. El premio mayor se divide
entre los ganadores, así que jugar combinaciones impopulares no cambia la probabilidad
de ganar pero sí el monto esperado condicional a ganar. Eso es teoría de juegos, no
predicción: se optimiza contra el comportamiento de los otros apostadores, no contra
el sorteo.

| Regla | Sesgo humano que explota |
|---|---|
| ≥3 números sobre 31 | Fechas de nacimiento concentran el juego en 1–31, sobre todo en 1–12 |
| Sin 3+ enteros consecutivos | 1-2-3-4-5-6 y variantes |
| Sin progresión aritmética de 4+ términos | Patrones regulares de cualquier paso |
| Sin 3+ términos en paso redondo (5, 10) | 10-20-30-40 |
| Ni todos pares ni todos impares | Combinaciones "estéticas" |
| Repartido en 3+ decenas | Bloques concentrados |
| Máx. N por fila y por columna | Líneas y columnas rellenadas a mano |
| Ocupa 3+ filas y 3+ columnas | Cruces, diagonales, bordes, esquinas |
| No repite un resultado histórico | Mucha gente juega combinaciones ya sorteadas |

**Enumeración exhaustiva.** El espacio C(41,6) = 4.496.388 se recorre completo, así que
los porcentajes de descarte son exactos y no estimados por muestreo. Con la cartilla por
defecto sobreviven **388.084 combinaciones (8.63%)**. La selección final es un **muestreo
uniforme** sobre ese conjunto: privilegiar algo dentro del filtro sería reintroducir la
superstición por la puerta de atrás.

**Cartilla parametrizable.** Las reglas visuales dependen de la grilla del cartón físico,
configurable desde la interfaz. El espacio admisible cambia en consecuencia:

| Columnas | Admisibles | % del total |
|---|---|---|
| 5 | 379.322 | 8.44% |
| **6** (defecto) | **388.084** | **8.63%** |
| 7 | 383.388 | 8.53% |
| 10 | 356.648 | 7.93% |

**Rendimiento.** Las combinaciones se representan como máscaras de 41 bits, lo que
convierte la búsqueda de progresiones aritméticas en desplazamientos vectorizados: el
filtrado completo bajó de ~45 s a ~3 s por configuración, con el precálculo cacheado
entre llamadas.

---

## Advertencia

> La probabilidad de acertar los 6 es **1 en 4.496.388** para las combinaciones que
> genera esta herramienta y para cualquier otra. Los filtros **no mejoran esa
> probabilidad ni un ápice**: solo reducen la chance de compartir el pozo si se gana.
> El valor esperado del juego sigue siendo negativo. Este proyecto es un ejercicio de
> estadística y teoría de juegos, no una estrategia para ganar dinero.

## Stack

`Python 3.12` · `pandas` · `numpy` · `scipy` · `scikit-learn` · `Streamlit` ·
`Selenium` · `BeautifulSoup` · `Flask`
