"""
Interfaz grafica y orquestador del pipeline.

    streamlit run app.py

Importa las funciones de los otros modulos directamente (nada de os.system),
asi que cualquier error del scraper o del consolidador llega a la interfaz
como una excepcion y se muestra tal cual.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import generador_apuestas as gen
from consolidar_datos import MAESTRO, update_database

BASE = Path(__file__).parent
DESTINO_SCRAPE = BASE / "resultados.xlsx"

st.set_page_config(page_title="LOTO — Análisis y Generador",
                   page_icon="🎲", layout="wide")


# --------------------------------------------------------------------------
# Datos y precalculo
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Enumerando las 4.496.388 combinaciones…")
def precalentar():
    """Enumera el espacio una sola vez por proceso; el resto son segundos."""
    gen._base()
    return True


def leer_maestro():
    if not MAESTRO.exists():
        return None
    return pd.read_csv(MAESTRO, parse_dates=["fecha"])


def resumen_maestro(df):
    origen = df["fecha_origen"].value_counts().to_dict()
    return {
        "total_sorteos": len(df),
        "sorteo_min": int(df["sorteo"].min()),
        "sorteo_max": int(df["sorteo"].max()),
        "fecha_max": df["fecha"].max(),
        "fechas_registradas": int(origen.get("registrada", 0)),
        "fechas_reconstruidas": int(origen.get("reconstruida", 0)),
    }


def mostrar_stats(s, nuevos=None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sorteos en el maestro", f"{s['total_sorteos']:,}".replace(",", "."),
              delta=f"+{nuevos}" if nuevos else None)
    c2.metric("Rango de sorteos", f"{s['sorteo_min']}–{s['sorteo_max']}")
    fecha = s["fecha_max"]
    c3.metric("Último sorteo", fecha.strftime("%d-%m-%Y") if pd.notna(fecha) else "—")
    c4.metric("Fechas registradas",
              f"{s['fechas_registradas']:,}".replace(",", "."),
              delta=f"{s['fechas_reconstruidas']} reconstruidas",
              delta_color="off")


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Cartilla física")
    columnas = st.number_input(
        "Número de columnas de la cartilla física",
        min_value=3, max_value=10, value=gen.COLUMNAS_POR_DEFECTO, step=1,
        help="Ancho de la grilla del cartón. Define las reglas visuales "
             "(líneas, columnas y patrones). Si el cartón real tiene otro "
             "ancho, cámbialo aquí y los filtros se recalculan.")

    n_filas = -(-gen.UNIVERSO // columnas)
    st.caption(f"Grilla resultante: {n_filas} filas × {columnas} columnas "
               f"para los {gen.UNIVERSO} números.")

    st.subheader("Reglas de la apuesta")
    cantidad = st.slider("Cantidad de apuestas", 1, 20, 5)
    min_altos = st.slider(f"Mínimo de números sobre {gen.UMBRAL_FECHA}", 0, 6, 3,
                          help="Sesgo contra las fechas de nacimiento, que "
                               "concentran el juego en el 1–31.")
    max_por_linea = st.slider("Máximo por fila o columna", 1, 6, 3,
                              help="Evita líneas y bloques marcados a mano.")
    excluir_hist = st.checkbox("Excluir resultados ya sorteados", value=True)

    usar_seed = st.checkbox("Fijar semilla (reproducible)", value=False)
    seed = st.number_input("Semilla", value=42, step=1) if usar_seed else None

    st.divider()
    st.caption("El sorteo es justo: ninguna combinación es más probable que "
               "otra. Estos filtros no mejoran la probabilidad de ganar, solo "
               "reducen la chance de compartir el pozo.")


# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("🎲 LOTO — Análisis estadístico y generador de apuestas")
st.markdown(
    "Pipeline completo: **scraping → consolidación → generación**. "
    "El modelo predictivo (`modelo_clasificacion.py`) demuestra que el sorteo "
    "no tiene señal explotable; este generador parte de esa base y optimiza el "
    "**premio esperado** en vez de la probabilidad.")

maestro = leer_maestro()
if maestro is None:
    st.error(f"No se encontró el dataset maestro en `{MAESTRO}`. "
             "Ejecuta primero la actualización de datos.")
else:
    mostrar_stats(resumen_maestro(maestro))

st.divider()

tab_datos, tab_gen = st.tabs(["📥 Actualización de datos",
                              "🎯 Generador inteligente"])


# --------------------------------------------------------------------------
# 1. Actualizacion de datos
# --------------------------------------------------------------------------
with tab_datos:
    st.subheader("Actualizar el histórico desde polla.cl")
    st.markdown(
        "Ejecuta el **scraper** y el **consolidador** en secuencia. El scraper "
        "abre Chrome en modo headless y recorre las páginas de resultados; "
        "puede tardar varios minutos. El consolidador valida los datos, agrega "
        "solo los sorteos nuevos y completa las fechas ausentes con el "
        "calendario martes/jueves/domingo.")

    col_btn, col_op = st.columns([1, 2])
    with col_op:
        max_paginas = st.number_input(
            "Límite de páginas (0 = sin límite)", min_value=0, value=0, step=1,
            help="Útil para una prueba rápida sin recorrer todo el histórico.")
    lanzar = col_btn.button("🔄 Actualizar base de datos", type="primary",
                            use_container_width=True)

    if lanzar:
        registro = st.empty()
        try:
            with st.status("Actualizando…", expanded=True) as estado:
                st.write("**Paso 1/2 — Scraping de polla.cl**")
                avance = st.empty()

                from scraper_polla import run_scraper

                def on_row(n, fila):
                    if n % 10 == 0 or n == 1:
                        avance.write(f"↓ {n} sorteos descargados "
                                     f"(último: {fila['Sorteo']})")

                df_scrape = run_scraper(
                    destino=DESTINO_SCRAPE, on_row=on_row,
                    max_paginas=int(max_paginas) or None)
                avance.write(f"✅ {len(df_scrape)} sorteos descargados.")

                if df_scrape.empty:
                    estado.update(label="El scraper no devolvió datos",
                                  state="error")
                    st.warning("No se obtuvo ninguna fila. Revisa la conexión "
                               "o si el sitio cambió de estructura.")
                else:
                    st.write("**Paso 2/2 — Consolidación y validación**")
                    lineas = []
                    _, stats = update_database(DESTINO_SCRAPE,
                                               log=lambda m: lineas.append(m))
                    for m in lineas:
                        st.write(f"· {m}")
                    estado.update(label="Base de datos actualizada",
                                  state="complete", expanded=False)
                    st.session_state["stats_update"] = stats

        except ModuleNotFoundError as e:
            st.error(f"Falta una dependencia del scraper: `{e.name}`. "
                     "Instálala con `pip install -r requirements.txt`.")
        except Exception as e:
            st.error(f"**{type(e).__name__}:** {e}")
            st.caption("Causas habituales: Chrome no instalado, sin conexión, "
                       "o cambio en la estructura del sitio.")

    if "stats_update" in st.session_state:
        s = st.session_state["stats_update"]
        st.success("Dataset maestro actualizado.")
        mostrar_stats(s, nuevos=s.get("sorteos_nuevos"))
        st.caption(f"Guardado en `{s['ruta']}`")

    if maestro is not None:
        with st.expander("Ver últimos sorteos del maestro"):
            st.dataframe(maestro.head(15), use_container_width=True,
                         hide_index=True)


# --------------------------------------------------------------------------
# 2. Generador
# --------------------------------------------------------------------------
with tab_gen:
    st.subheader("Generar apuestas de baja popularidad")
    st.markdown(
        f"Enumera las **{4496388:,}**".replace(",", ".") +
        " combinaciones posibles, descarta las que "
        "responden a sesgos humanos conocidos y **muestrea uniformemente** "
        "sobre lo que queda — dentro del filtro no se privilegia ninguna "
        "combinación, que sería volver a la superstición.")

    if st.button("🎯 Generar apuestas", type="primary"):
        precalentar()
        try:
            with st.spinner("Aplicando filtros sobre el espacio completo…"):
                st.session_state["resultado"] = gen.generate_bets(
                    cantidad=cantidad, columnas=int(columnas),
                    min_altos=min_altos, max_por_linea=max_por_linea,
                    seed=int(seed) if seed is not None else None,
                    excluir_historicas=excluir_hist)
        except ValueError as e:
            st.error(str(e))

    r = st.session_state.get("resultado")
    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Espacio total", f"{r['total']:,}".replace(",", "."))
        c2.metric("Combinaciones admisibles",
                  f"{r['admisibles']:,}".replace(",", "."),
                  delta=f"{r['pct_admisible']}% del total", delta_color="off")
        c3.metric("Cartilla usada", f"{r['columnas']} columnas")

        st.markdown("#### Combinaciones recomendadas")
        apuestas = r["apuestas"]
        st.dataframe(
            apuestas, use_container_width=True, hide_index=True,
            column_config={
                **{f"N{i}": st.column_config.NumberColumn(f"N{i}", width="small")
                   for i in range(1, 7)},
                "Apuesta": st.column_config.NumberColumn("#", width="small"),
                "Suma": st.column_config.NumberColumn("Suma", width="small"),
            })

        st.download_button(
            "⬇️ Descargar como CSV",
            apuestas.to_csv(index=False).encode("utf-8"),
            file_name="apuestas_loto.csv", mime="text/csv")

        with st.expander("Embudo de descarte — cuánto elimina cada regla"):
            st.dataframe(r["reglas"], use_container_width=True, hide_index=True)
            st.caption("Los porcentajes son exactos: el espacio se enumera "
                       "completo, no se estima por muestreo.")

        st.warning(
            f"**La probabilidad de acertar los 6 sigue siendo 1 en "
            f"{r['total']:,}".replace(",", ".") +
            "** para estas combinaciones y para cualquier otra. El filtro no "
            "mejora esa probabilidad ni un ápice: solo reduce la chance de "
            "compartir el pozo si se gana. El valor esperado del juego sigue "
            "siendo negativo.")
    else:
        st.info("Configura la cartilla y las reglas en la barra lateral, "
                "y pulsa **Generar apuestas**.")
