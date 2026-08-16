# Borrador — Post de LinkedIn

> Texto listo para copiar y pegar. Debajo hay variantes de apertura y notas de uso.

---

## Versión principal

Construí un proyecto de análisis de datos sobre la lotería chilena y el resultado principal es un modelo que **no funciona**. Esa era exactamente la idea.

La mayoría de los proyectos de "predicción de lotería" fracasan en silencio: entrenan un modelo, reportan un RMSE bajo y nunca notan que la métrica medía otra cosa. En este caso, el enfoque ingenuo predecía "el número de la posición 3" como variable continua. Ese target no es un fenómeno físico — es el resultado de **ordenar** los 6 números. El modelo aprendía la estadística de orden, bajaba el error y aparentaba funcionar sin haber aprendido nada del sorteo.

**Lo reformulé como debía estar:**

→ Clasificación multi-etiqueta: una fila por (sorteo, número), estimando P(k) para los 41 números.
→ Split temporal estricto con ventana expansiva: entrenar hasta N, predecir N+1.
→ Métrica real: aciertos sobre 6, contra la distribución hipergeométrica del azar.

El punto clave fue **verificar la ausencia de data leakage en vez de afirmarla**. Implementé una prueba de perturbación: se altera por completo el sorteo *t* y se comprueba que la matriz de features quede bit a bit idéntica hasta ese punto, y que solo cambie hacia adelante. Sin esa prueba, cualquier resultado positivo es indistinguible de una fuga temporal.

**El veredicto, con números:**

• ROC AUC = 0.5067 — sin información (0.5 es una moneda al aire)
• Log loss = 0.4164, frente a 0.4163 de predecir la constante 6/41

El modelo es medible y literalmente **peor que no modelar**. Un chi-cuadrado sobre 1.662 sorteos da p = 0.72: el sorteo es limpio. Demostrar rigurosamente que no hay señal es un resultado, no un fracaso.

---

**¿Y entonces qué sí se puede optimizar? El premio, no la probabilidad.**

Si toda combinación es igual de probable, la pregunta deja de ser *cuál sale* y pasa a ser **con cuánta gente hay que repartir el pozo si sale**. Eso es teoría de juegos: se optimiza contra el comportamiento de los otros apostadores, no contra el sorteo.

El generador descarta las combinaciones que responden al comportamiento de rebaño:

• Fechas de nacimiento, que concentran el juego en el 1–31 (y sobre todo en el 1–12)
• Secuencias y pasos redondos: 1-2-3-4-5-6, 10-20-30-40
• Patrones visuales en la cartilla física: líneas, columnas, diagonales, bordes
• Combinaciones que ya salieron alguna vez

Enumero el espacio completo — C(41,6) = 4.496.388 — así que los porcentajes de descarte son exactos y no estimados por muestreo. Sobreviven 388.056 combinaciones (8,63%), y la selección final es un **muestreo uniforme** sobre ese conjunto: privilegiar algo dentro del filtro sería reintroducir la superstición por la puerta de atrás.

---

**El detalle de ingeniería del que quedé más contento:**

Filtrar 4,5 millones de combinaciones tardaba 45 segundos, inviable para una interfaz interactiva. Representé cada combinación como una **máscara de 41 bits**, lo que convierte "¿está presente el número v?" en un desplazamiento vectorizado sobre un array de enteros. Buscar progresiones aritméticas pasó de ~36 s a menos de 1 s, y el filtrado completo de **45 s a 3 s**.

Lo importante: el rewrite se validó como regresión — reproduce el conteo exacto de cada una de las once reglas.

---

**Stack:** Python · pandas · NumPy · scikit-learn · SciPy · Streamlit · Selenium

El pipeline completo (scraping → validación → consolidación → modelo → generador) está orquestado desde una interfaz en Streamlit, con cada módulo expuesto como función importable.

🔗 https://github.com/HangLoose84/loto-data-pipeline

*Aclaración necesaria: la probabilidad de acertar los 6 sigue siendo 1 en 4.496.388 para estas combinaciones y para cualquier otra. Los filtros no la mejoran ni un ápice — solo reducen la chance de compartir el pozo. El valor esperado del juego sigue siendo negativo. Esto es un ejercicio de estadística e ingeniería, no una estrategia para ganar dinero.*

#DataScience #DataEngineering #Python #GameTheory #MachineLearning #Statistics #ETL #Streamlit

---

## Variantes de apertura

**A — Más directa (recomendada si buscas alcance):**

> Pasé semanas construyendo un modelo de machine learning para predecir la lotería. Funciona perfecto: demuestra que no se puede predecir.

**B — Orientada a reclutadores técnicos:**

> Un backtest mal construido sobre series temporales produce con facilidad resultados que parecen señal y no lo son. Construí un proyecto entero alrededor de esa trampa, para no caer en ella.

**C — Enfocada en la parte de teoría de juegos:**

> No puedes mejorar tu probabilidad de ganar la lotería. Sí puedes mejorar cuánto cobras si ganas. La diferencia es teoría de juegos, y es medible.

---

## Notas de uso

- **Longitud:** la versión principal supera los 3.000 caracteres. LinkedIn corta con "ver más" alrededor de los 200, así que las primeras dos líneas cargan todo el peso — no las edites a la baja.
- **Formato:** LinkedIn no renderiza Markdown. Al pegar, los `**negrita**` aparecerán como asteriscos literales. Quítalos, o usa un conversor a caracteres Unicode en negrita.
- **Separadores:** las líneas `---` no existen en LinkedIn; reemplázalas por una línea en blanco o un separador visual como `— — —`.
- **Requisito previo:** el repositorio debe ser **público** antes de publicar, o el enlace dará 404 a todo el que no seas tú.
- **Hashtags:** LinkedIn favorece entre 3 y 5. Si prefieres recortar, quédate con #DataScience #DataEngineering #Python #GameTheory.
- **Imagen:** el post rinde bastante más con una captura. En `docs/app-generador.png` está la interfaz con la tabla de resultados.
