"""
Consolida la salida cruda del scraper contra el dataset maestro
data/loto_historico.csv.

    cargar_scrape(ruta)     -> DataFrame normalizado
    reconstruir_fechas(df)  -> DataFrame con las fechas ausentes completadas
    update_database(...)    -> (DataFrame, dict de estadisticas)

Uso por linea de comandos:
    python consolidar_datos.py [resultados.xlsx]
    python consolidar_datos.py --solo-fechas    # solo recalcula fechas faltantes
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
MAESTRO = BASE / "data" / "loto_historico.csv"
NUMS = [f"n{i}" for i in range(1, 7)]
COLS = ["sorteo", "fecha", "fecha_origen"] + NUMS + ["comodin"]

# El sorteo se realiza martes, jueves y domingo. Verificado contra los 1256
# sorteos con fecha registrada (3959-5214, 2017-01-01 a 2025-01-07): la
# correspondencia sorteo <-> N-esimo dia de sorteo no tiene ni una excepcion.
DIAS_SORTEO = [1, 3, 6]  # lunes = 0
HORA_SORTEO = pd.Timedelta(hours=21)


def calendario(desde="2010-01-01", hasta="2035-12-31"):
    dias = pd.date_range(desde, hasta, freq="D")
    return dias[dias.dayofweek.isin(DIAS_SORTEO)]


def mapa_sorteo_fecha(conocidos):
    """Ancla el calendario en un sorteo de fecha conocida y lo extiende.

    Falla si el ancla no reproduce TODAS las fechas conocidas, para que un
    cambio de calendario no pase inadvertido.
    """
    cal = calendario()
    dias = list(cal.normalize())

    ancla = conocidos.sort_values("sorteo").iloc[-1]
    idx = dias.index(pd.Timestamp(ancla["fecha"]).normalize())
    base = int(ancla["sorteo"])
    mapa = {base + (j - idx): cal[j] for j in range(len(dias))}

    discrepancias = [
        int(s) for s, f in zip(conocidos["sorteo"], conocidos["fecha"])
        if mapa.get(int(s)) is None
        or mapa[int(s)].normalize() != pd.Timestamp(f).normalize()
    ]
    if discrepancias:
        raise SystemExit(
            f"El calendario martes/jueves/domingo no reproduce {len(discrepancias)} "
            f"fechas ya registradas (ej. sorteos {discrepancias[:5]}). "
            "Revisar si el calendario de sorteos cambio antes de reconstruir.")
    return mapa


def reconstruir_fechas(df, log=print):
    """Rellena 'fecha' donde falte y anota la procedencia de cada fecha."""
    df = df.copy()
    if "fecha_origen" not in df.columns:
        df["fecha_origen"] = None
    df.loc[df["fecha"].notna() & df["fecha_origen"].isna(),
           "fecha_origen"] = "registrada"

    faltan = df["fecha"].isna()
    if not faltan.any():
        return df

    mapa = mapa_sorteo_fecha(df[df["fecha"].notna()][["sorteo", "fecha"]])
    df.loc[faltan, "fecha"] = df.loc[faltan, "sorteo"].map(
        lambda s: mapa.get(int(s)) + HORA_SORTEO
        if mapa.get(int(s)) is not None else pd.NaT)
    df.loc[faltan & df["fecha"].notna(), "fecha_origen"] = "reconstruida"
    log(f"{int(faltan.sum())} fechas reconstruidas desde el calendario")
    return df


def cargar_scrape(ruta):
    """Lee y normaliza el .xlsx que produce scraper_polla.run_scraper()."""
    df = pd.read_excel(ruta)
    orig = {str(c).strip().lower(): c for c in df.columns}

    def buscar(*claves):
        for k in claves:
            if k in orig:
                return orig[k]
        return None

    col_sorteo = buscar("sorteo", "numero de sorteo", "número de sorteo")
    if col_sorteo is None:
        raise SystemExit(f"No se encontro la columna de sorteo en {ruta}")

    cols_num = [buscar(f"numero {i}", f"número {i}", f"n{i}")
                for i in range(1, 7)] + [buscar("comodin", "comodín", "numero 7")]
    if any(c is None for c in cols_num):
        raise SystemExit(f"Faltan columnas de numeros en {ruta}")

    out = pd.DataFrame({"sorteo": df[col_sorteo]})
    for destino, origen in zip(NUMS + ["comodin"], cols_num):
        out[destino] = df[origen]

    col_fecha = buscar("fecha", "fecha/hora")
    out["fecha"] = (pd.to_datetime(df[col_fecha], errors="coerce")
                    if col_fecha else pd.NaT)
    out["fecha_origen"] = None

    out = out.dropna(subset=["sorteo", "n1"]).drop_duplicates(subset=["sorteo"])
    return out.astype({"sorteo": int,
                       **{c: int for c in NUMS + ["comodin"]}})[COLS]


def validar(df, log=print):
    """Reporta filas sospechosas sin descartarlas en silencio."""
    fuera = df[(df[NUMS + ["comodin"]] < 1).any(axis=1)
               | (df[NUMS + ["comodin"]] > 41).any(axis=1)]
    repes = df[df[NUMS].nunique(axis=1) != 6]
    for nombre, malas in [("fuera de rango 1-41", fuera),
                          ("con numeros repetidos", repes)]:
        if len(malas):
            log(f"AVISO: {len(malas)} sorteos {nombre}: "
                f"{malas['sorteo'].tolist()[:10]}")
    return df


def update_database(ruta_scrape=None, log=print):
    """Integra un scrape al maestro y completa las fechas.

    ruta_scrape None -> solo recalcula las fechas faltantes.
    Devuelve (DataFrame maestro, dict de estadisticas).
    """
    maestro = pd.read_csv(MAESTRO, parse_dates=["fecha"])
    previos = len(maestro)

    if ruta_scrape is not None:
        nuevos = validar(cargar_scrape(ruta_scrape), log=log)
        # El maestro manda: solo se agregan sorteos que aun no existen.
        faltantes = nuevos[~nuevos["sorteo"].isin(maestro["sorteo"])]
        maestro = pd.concat([maestro, faltantes], ignore_index=True)
        log(f"{len(faltantes)} sorteos nuevos agregados "
            f"(el scrape traia {len(nuevos)})")

    salida = reconstruir_fechas(maestro, log=log)
    salida = (salida[COLS].sort_values("sorteo", ascending=False)
                          .reset_index(drop=True))
    salida.to_csv(MAESTRO, index=False, date_format="%Y-%m-%d %H:%M")

    origen = salida["fecha_origen"].value_counts().to_dict()
    stats = {
        "total_sorteos": len(salida),
        "sorteos_nuevos": len(salida) - previos,
        "sorteo_min": int(salida["sorteo"].min()),
        "sorteo_max": int(salida["sorteo"].max()),
        "fecha_min": salida["fecha"].min(),
        "fecha_max": salida["fecha"].max(),
        "fechas_registradas": int(origen.get("registrada", 0)),
        "fechas_reconstruidas": int(origen.get("reconstruida", 0)),
        "ruta": str(MAESTRO),
    }
    log(f"Total: {stats['total_sorteos']} sorteos "
        f"({stats['sorteo_min']} a {stats['sorteo_max']}) | "
        f"registradas: {stats['fechas_registradas']} | "
        f"reconstruidas: {stats['fechas_reconstruidas']}")
    return salida, stats


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    solo_fechas = "--solo-fechas" in sys.argv
    update_database(None if solo_fechas
                    else (args[0] if args else BASE / "resultados.xlsx"))
