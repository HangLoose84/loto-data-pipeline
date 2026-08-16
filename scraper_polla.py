"""
Scraper de resultados historicos de polla.cl.

Expone tres capas, de la mas reutilizable a la mas concreta:

    parsear_fila(celdas)  -> dict | None   logica pura, testeable sin red
    iter_resultados(...)  -> Iterator      recorre las paginas y va emitiendo
    run_scraper(...)      -> DataFrame     consume el iterador y escribe el .xlsx

La app Flask con streaming en vivo (SSE) quedo como una cuarta capa encima de
iter_resultados, para no duplicar la logica de scraping.
"""
import re
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
URL = "https://www.polla.cl/es/view/resultados/5271"
DESTINO = BASE / "resultados.xlsx"
COLUMNAS = ["Sorteo", "Fecha", "Numero 1", "Numero 2", "Numero 3",
            "Numero 4", "Numero 5", "Numero 6", "Comodin"]

# El sitio concatena la etiqueta con el valor al extraer el texto de la celda
# ("Numero de sorteo5245"), asi que todo se saca por expresion regular.
RE_SORTEO = re.compile(r"(\d+)")
RE_FECHA = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2})?)")


def parsear_fila(celdas):
    """Convierte las celdas de un <tr> en un registro, o None si no es valido.

    Descarta encabezados, filas incompletas y sorteos sin resultado publicado
    ("Resultados no disponible"), que en la version anterior entraban como
    filas de vacios.
    """
    if len(celdas) < 4:
        return None

    m = RE_SORTEO.search(celdas[1])
    if not m:
        return None                      # encabezado u otra fila sin sorteo
    sorteo = int(m.group(1))

    # 6 numeros ganadores + comodin. Menos de 7 significa sorteo sin publicar.
    numeros = [int(n) for n in RE_SORTEO.findall(celdas[3])]
    if len(numeros) < 7:
        return None
    numeros = numeros[:7]
    if not all(1 <= n <= 41 for n in numeros):
        return None

    # La fecha es un extra: si el sitio cambia el formato, consolidar_datos.py
    # la reconstruye desde el calendario de sorteos.
    fecha = pd.NaT
    if len(celdas) > 2:
        f = RE_FECHA.search(celdas[2])
        if f:
            fecha = pd.to_datetime(f.group(1), dayfirst=True, errors="coerce")

    return dict(zip(COLUMNAS, [sorteo, fecha] + numeros))


def _crear_driver(headless=True):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opciones = webdriver.ChromeOptions()
    if headless:
        opciones.add_argument("--headless=new")
    opciones.add_argument("--disable-gpu")
    opciones.add_argument("--log-level=3")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=opciones)


def _filas_de(driver):
    """Registros validos de la tabla actualmente cargada."""
    from bs4 import BeautifulSoup

    tabla = BeautifulSoup(driver.page_source, "html.parser").find(
        "table", class_="table-results")
    if not tabla:
        return []
    filas = []
    for tr in tabla.find_all("tr"):
        celdas = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        fila = parsear_fila(celdas)
        if fila:
            filas.append(fila)
    return filas


def iter_resultados(url=URL, headless=True, max_paginas=None, pausa=0.4,
                    espera=15, reintentos=3, on_warn=None):
    """Recorre las paginas de resultados y va emitiendo un dict por sorteo.

    El avance de pagina se confirma esperando a que la tabla cambie de
    contenido, no con una pausa fija. Una pausa fija demasiado corta hacia
    releer la misma tabla; detectar eso y cortar el recorrido truncaba la
    descarga en silencio, dejando huecos en el historico.

    Solo se termina cuando "Siguiente pagina" queda inactiva, desaparece, o
    la tabla no cambia tras varios reintentos (y en ese caso se avisa).
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import NoSuchElementException, TimeoutException

    avisar = on_warn or (lambda m: None)
    driver = _crear_driver(headless)
    vistos = set()
    try:
        driver.get(url)
        WebDriverWait(driver, espera).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-results")))

        pagina = 0
        while max_paginas is None or pagina < max_paginas:
            pagina += 1
            filas = _filas_de(driver)
            if not filas:
                avisar(f"Pagina {pagina} sin filas validas; se corta aqui.")
                break

            for fila in filas:
                if fila["Sorteo"] not in vistos:
                    vistos.add(fila["Sorteo"])
                    yield fila

            marca = filas[0]["Sorteo"]
            try:
                boton = driver.find_element(By.LINK_TEXT, "Siguiente página")
                if "inactive" in (boton.find_element(By.XPATH, "..")
                                  .get_attribute("class") or ""):
                    break                        # fin real del historico
            except NoSuchElementException:
                break

            # Confirmar que la tabla efectivamente cambio antes de re-leerla.
            for intento in range(1, reintentos + 1):
                try:
                    driver.find_element(By.LINK_TEXT, "Siguiente página").click()
                except NoSuchElementException:
                    break
                try:
                    WebDriverWait(driver, espera).until(
                        lambda d: (_filas_de(d) or [{"Sorteo": marca}])[0]["Sorteo"] != marca)
                    break
                except TimeoutException:
                    if intento == reintentos:
                        avisar(
                            f"La pagina {pagina + 1} no cargo tras {reintentos} "
                            f"intentos; la descarga queda incompleta en el "
                            f"sorteo {marca}.")
                    time.sleep(pausa * intento)
            else:
                break
            if pausa:
                time.sleep(pausa)
    finally:
        driver.quit()


def run_scraper(destino=DESTINO, on_row=None, **kwargs):
    """Scrapea el historico completo y lo guarda en un .xlsx.

    on_row: callback opcional que recibe (n_filas, fila) para reportar avance.
    Devuelve el DataFrame escrito.
    """
    filas = []
    for fila in iter_resultados(**kwargs):
        filas.append(fila)
        if on_row:
            on_row(len(filas), fila)

    df = pd.DataFrame(filas, columns=COLUMNAS)
    if destino:
        df.to_excel(destino, index=False)
    return df


# --------------------------------------------------------------------------
# Interfaz web opcional: muestra el scraping en vivo mientras corre.
#   python scraper_polla.py
# --------------------------------------------------------------------------
def crear_app():
    from flask import Flask, Response

    app = Flask(__name__)

    PAGINA = """<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
    <title>Resultados de Loto en Tiempo Real</title><style>
    body{font-family:system-ui,Arial,sans-serif;margin:40px}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #ddd;padding:8px;text-align:center}
    tr:nth-child(even){background:#f2f2f2}
    th{background:#4CAF50;color:#fff}
    #estado{margin:16px 0;font-weight:bold}</style></head><body>
    <h1>Resultados de Loto en Tiempo Real</h1>
    <button id="startBtn">START</button><div id="estado"></div>
    <table><thead><tr>__TH__</tr></thead><tbody id="filas"></tbody></table>
    <script>
    document.getElementById("startBtn").addEventListener("click", function(){
      this.disabled = true;
      const estado = document.getElementById("estado");
      estado.innerText = "Descargando...";
      const src = new EventSource('/stream');
      let n = 0;
      src.onmessage = function(e){
        if(e.data.trim() === "FIN"){ src.close();
          estado.innerText = `Listo: ${n} sorteos guardados en resultados.xlsx`;
        } else {
          n++; estado.innerText = `Descargando... ${n} sorteos`;
          document.getElementById("filas").innerHTML += e.data;
        }
      };
      src.onerror = function(){ src.close(); estado.innerText = "Conexion interrumpida."; };
    });
    </script></body></html>""".replace(
        "__TH__", "".join(f"<th>{c}</th>" for c in COLUMNAS))

    @app.route("/")
    def index():
        return PAGINA

    @app.route("/stream")
    def stream():
        def generar():
            filas = []
            for fila in iter_resultados():
                filas.append(fila)
                celdas = "".join(
                    f"<td>{'' if pd.isna(v) else v}</td>" for v in fila.values())
                yield f"data: <tr>{celdas}</tr>\n\n"
            pd.DataFrame(filas, columns=COLUMNAS).to_excel(DESTINO, index=False)
            yield "data: FIN\n\n"
        return Response(generar(), mimetype="text/event-stream")

    return app


if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    crear_app().run(debug=False, threaded=True)
