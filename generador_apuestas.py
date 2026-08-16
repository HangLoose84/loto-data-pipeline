"""
Generador de apuestas por teoria de juegos, sin Machine Learning.

Premisa: el sorteo es justo (ver modelo_clasificacion.py), asi que ninguna
combinacion es mas probable que otra. Lo que SI se puede optimizar es el
premio esperado: el pozo se reparte entre todos los ganadores, de modo que
jugar combinaciones que poca gente elige no aumenta la probabilidad de
ganar, pero aumenta cuanto se cobra si se gana.

Las reglas apuntan a los sesgos documentados del apostador humano:

  - Fechas de nacimiento: concentran las apuestas en el 1-31, y sobre todo
    en el 1-12. Los numeros 32-41 estan sistematicamente sub-jugados.
  - Secuencias y pasos redondos: 1-2-3-4-5-6, 10-20-30-40.
  - Patrones visuales en la cartilla: lineas, columnas, diagonales, bordes.
  - Combinaciones historicas: mucha gente juega resultados ya sorteados.

El espacio C(41,6) = 4.496.388 se enumera completo, asi que los porcentajes
de descarte son exactos y el muestreo final es uniforme sobre el conjunto
admisible (no se privilegia ninguna combinacion dentro de el).

API:
    generate_bets(cantidad=5, columnas=6, ...) -> dict

Uso por linea de comandos:
    python generador_apuestas.py [-n 5] [--columnas 6] [--altos 3] [--seed 42]
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
MAESTRO = BASE / "data" / "loto_historico.csv"
NUMS = [f"n{i}" for i in range(1, 7)]
UNIVERSO = 41
POR_SORTEO = 6
UMBRAL_FECHA = 31        # 1-31 son elegibles como dia del mes
PASOS_REDONDOS = (5, 10)
COLUMNAS_POR_DEFECTO = 6

_cache = {}


# --------------------------------------------------------------------------
# Nucleo vectorizado
#
# Cada combinacion se representa tambien como una mascara de 41 bits, lo que
# convierte "esta presente el numero v?" en un desplazamiento sobre un vector
# de enteros. Buscar progresiones aritmeticas pasa de ~36 s a menos de 1 s.
# --------------------------------------------------------------------------
def todas_las_combinaciones():
    it = combinations(range(1, UNIVERSO + 1), POR_SORTEO)
    arr = np.fromiter((n for c in it for n in c), dtype=np.int8)
    return arr.reshape(-1, POR_SORTEO)


def _mascaras_bits(C):
    bits = np.zeros(len(C), dtype=np.int64)
    for i in range(POR_SORTEO):
        bits |= np.left_shift(np.int64(1), C[:, i].astype(np.int64) - 1)
    return bits


def _presente(bits, valores):
    """Vector booleano: el numero 'valores[i]' pertenece a la combinacion i."""
    v = valores.astype(np.int64)
    dentro = (v >= 1) & (v <= UNIVERSO)
    desp = np.where(dentro, v - 1, 0)
    return (((bits >> desp) & 1) == 1) & dentro


def _hay_progresion(C, bits, terminos, pasos=None):
    """True donde existe una progresion aritmetica de 'terminos' elementos.

    Para 'terminos'=k basta fijar los dos primeros elementos y comprobar que
    los k-2 siguientes esten presentes; con pasos fijos, basta fijar el primero.
    """
    hallada = np.zeros(len(C), dtype=bool)
    if pasos is None:
        pares = [(i, j) for i in range(POR_SORTEO)
                 for j in range(i + 1, POR_SORTEO)]
        for i, j in pares:
            primero = C[:, i].astype(np.int16)
            paso = C[:, j].astype(np.int16) - primero
            ok = np.ones(len(C), dtype=bool)
            for t in range(2, terminos):
                ok &= _presente(bits, primero + t * paso)
            hallada |= ok
    else:
        for paso in pasos:
            for i in range(POR_SORTEO):
                primero = C[:, i].astype(np.int16)
                ok = np.ones(len(C), dtype=bool)
                for t in range(1, terminos):
                    ok &= _presente(bits, primero + t * paso)
                hallada |= ok
    return hallada


def _popcount(x):
    tabla = np.array([bin(i).count("1") for i in range(256)], dtype=np.int8)
    total = np.zeros(len(x), dtype=np.int8)
    y = x.copy()
    for _ in range(8):
        total += tabla[(y & 0xFF).astype(np.uint8)]
        y >>= 8
    return total


def _conteos_por_grupo(idx, n_grupos):
    """Cuantos numeros caen en cada grupo (fila o columna de la cartilla)."""
    n = len(idx)
    cnt = np.zeros((n, n_grupos), dtype=np.int8)
    filas = np.arange(n)
    for i in range(POR_SORTEO):
        cnt[filas, idx[:, i]] += 1
    return cnt


def combinaciones_historicas():
    if "historicas" not in _cache:
        df = pd.read_csv(MAESTRO)
        _cache["historicas"] = {tuple(sorted(f))
                                for f in df[NUMS].to_numpy().tolist()}
    return _cache["historicas"]


def _base():
    """Precalculo compartido: no depende de ningun parametro del usuario.

    Se cachea porque enumerar el espacio y buscar progresiones es lo caro;
    cambiar 'columnas' en la interfaz solo recalcula las reglas de grilla.
    """
    if "base" in _cache:
        return _cache["base"]

    C = todas_las_combinaciones()
    bits = _mascaras_bits(C)
    pares = (C % 2 == 0).sum(axis=1)

    decadas = np.zeros(len(C), dtype=np.int64)
    for i in range(POR_SORTEO):
        decadas |= np.left_shift(np.int64(1), (C[:, i].astype(np.int64) - 1) // 10)

    estructurales = [
        ("sin 3 o mas enteros consecutivos",
         ~_hay_progresion(C, bits, 3, pasos=(1,))),
        ("sin progresion aritmetica de 4 o mas terminos",
         ~_hay_progresion(C, bits, 4)),
        (f"sin 3 o mas terminos en paso redondo {PASOS_REDONDOS}",
         ~_hay_progresion(C, bits, 3, pasos=PASOS_REDONDOS)),
        ("ni todos pares ni todos impares", (pares >= 2) & (pares <= 4)),
        ("repartido en 3 o mas decenas", _popcount(decadas) >= 3),
    ]

    historicas = combinaciones_historicas()
    es_historica = np.zeros(len(C), dtype=bool)
    if historicas:
        bits_hist = np.array(sorted(
            sum(1 << (n - 1) for n in c) for c in historicas), dtype=np.int64)
        pos = np.searchsorted(bits_hist, bits)
        pos = np.clip(pos, 0, len(bits_hist) - 1)
        es_historica = bits_hist[pos] == bits

    _cache["base"] = (C, bits, estructurales, ~es_historica)
    return _cache["base"]


def reglas_grilla(C, columnas, max_por_linea):
    """Reglas visuales. Son las unicas que dependen de la cartilla fisica."""
    n_filas = int(np.ceil(UNIVERSO / columnas))
    fila = ((C.astype(np.int16) - 1) // columnas)
    col = ((C.astype(np.int16) - 1) % columnas)

    cnt_fila = _conteos_por_grupo(fila, n_filas)
    cnt_col = _conteos_por_grupo(col, columnas)

    minimo = min(3, n_filas, columnas)
    return [
        (f"maximo {max_por_linea} numeros por fila (cartilla de {columnas} col.)",
         cnt_fila.max(axis=1) <= max_por_linea),
        (f"maximo {max_por_linea} numeros por columna (cartilla de {columnas} col.)",
         cnt_col.max(axis=1) <= max_por_linea),
        (f"ocupa {minimo} o mas filas distintas",
         (cnt_fila > 0).sum(axis=1) >= minimo),
        (f"ocupa {minimo} o mas columnas distintas",
         (cnt_col > 0).sum(axis=1) >= minimo),
    ]


def generate_bets(cantidad=5, columnas=COLUMNAS_POR_DEFECTO, min_altos=3,
                  max_por_linea=3, seed=None, excluir_historicas=True,
                  umbral_fecha=UMBRAL_FECHA):
    """Genera 'cantidad' apuestas filtradas segun las reglas de impopularidad.

    columnas: ancho de la cartilla fisica; define las reglas visuales.

    Devuelve un dict con:
        apuestas    DataFrame de las combinaciones elegidas
        reglas      DataFrame con el embudo de descarte (exacto)
        admisibles  tamano del espacio que pasa todos los filtros
        total       C(41,6)
    """
    if not 1 <= columnas <= UNIVERSO:
        raise ValueError(f"columnas debe estar entre 1 y {UNIVERSO}")
    if not 0 <= min_altos <= POR_SORTEO:
        raise ValueError(f"min_altos debe estar entre 0 y {POR_SORTEO}")

    C, _, estructurales, no_historica = _base()
    total = len(C)

    reglas = [(f"al menos {min_altos} numeros sobre {umbral_fecha}",
               (C > umbral_fecha).sum(axis=1) >= min_altos)]
    reglas += estructurales
    reglas += reglas_grilla(C, columnas, max_por_linea)
    if excluir_historicas:
        reglas.append(("no repite un resultado historico", no_historica))

    viva = np.ones(total, dtype=bool)
    embudo = []
    for nombre, ok in reglas:
        viva &= ok
        embudo.append({"regla": nombre, "sobreviven": int(viva.sum()),
                       "% del total": round(100 * viva.sum() / total, 2)})

    admisibles = np.flatnonzero(viva)
    if len(admisibles) == 0:
        raise ValueError(
            "Ninguna combinacion pasa los filtros. Baja min_altos o sube "
            "max_por_linea.")

    rng = np.random.default_rng(seed)
    elegidas = C[rng.choice(admisibles, size=min(cantidad, len(admisibles)),
                            replace=False)]
    elegidas = np.sort(elegidas, axis=1)

    apuestas = pd.DataFrame(elegidas, columns=[f"N{i}" for i in range(1, 7)])
    apuestas.insert(0, "Apuesta", range(1, len(apuestas) + 1))
    apuestas["Suma"] = elegidas.sum(axis=1)
    apuestas[f"Sobre {umbral_fecha}"] = (elegidas > umbral_fecha).sum(axis=1)

    return {
        "apuestas": apuestas,
        "reglas": pd.DataFrame(embudo),
        "admisibles": int(len(admisibles)),
        "total": int(total),
        "pct_admisible": round(100 * len(admisibles) / total, 2),
        "columnas": columnas,
    }


def main(**kw):
    cantidad = kw.pop("cantidad")
    print("Enumerando el espacio completo de combinaciones...")
    r = generate_bets(cantidad=cantidad, **kw)
    print(f"C(41,6) = {r['total']:,} combinaciones\n")

    print(f"{'Regla':<58}{'sobreviven':>12}{'% total':>10}")
    print("-" * 80)
    for _, f in r["reglas"].iterrows():
        print(f"{f['regla']:<58}{f['sobreviven']:>12,}{f['% del total']:>9}%")

    print(f"\nEspacio admisible: {r['admisibles']:,} combinaciones "
          f"({r['pct_admisible']}% del total)")
    print(f"\n{cantidad} apuestas (muestreo uniforme sobre el espacio admisible):\n")
    for _, f in r["apuestas"].iterrows():
        nums = "  ".join(f"{f[f'N{i}']:2d}" for i in range(1, 7))
        print(f"  {f['Apuesta']}.  {nums}    suma {f['Suma']:3d}")

    print(f"\nProbabilidad de acertar los 6: 1 en {r['total']:,} para CUALQUIER "
          "combinacion,\nincluidas estas. El filtro no mejora esa probabilidad: "
          "solo reduce la\nchance de compartir el pozo con otro ganador.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--cantidad", type=int, default=5)
    ap.add_argument("--columnas", type=int, default=COLUMNAS_POR_DEFECTO,
                    help="ancho de la cartilla fisica (reglas visuales)")
    ap.add_argument("--altos", dest="min_altos", type=int, default=3,
                    help=f"minimo de numeros sobre {UMBRAL_FECHA}")
    ap.add_argument("--max-linea", dest="max_por_linea", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    main(**vars(ap.parse_args()))
