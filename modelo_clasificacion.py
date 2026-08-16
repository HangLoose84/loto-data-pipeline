"""
Modelo honesto: en vez de predecir "el numero de la posicion 3" como una
variable continua (lo que solo aprende la estadistica de orden), estima
P(k) = probabilidad de que el numero k salga en el proximo sorteo, para
k = 1..41.

Tres decisiones que separan esto de un backtest enganoso:

1. Split temporal estricto. Se entrena con sorteos <= N y se predice el N+1.
   Ningun sorteo futuro entra al entrenamiento, ni siquiera a traves de las
   features (todas se calculan con ventanas que terminan en N).
2. Formulacion multi-etiqueta. Una fila por (sorteo, numero), etiqueta binaria
   "salio / no salio". Un unico clasificador aprende P(k) y los 41 numeros
   se rankean por probabilidad.
3. Metrica real: aciertos sobre 6. El RMSE sobre posiciones ordenadas es
   irrelevante; lo que importa es cuantos numeros del sorteo se aciertan.

El resultado esperado -- y el punto del ejercicio -- es que el modelo NO
supere al azar de forma significativa.

Uso:
    python modelo_clasificacion.py [--test N] [--refit N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

BASE = Path(__file__).parent
MAESTRO = BASE / "data" / "loto_historico.csv"
NUMS = [f"n{i}" for i in range(1, 7)]
UNIVERSO = 41       # numeros del 1 al 41
POR_SORTEO = 6      # numeros que salen en cada sorteo
VENTANAS = (10, 25, 50, 100, 250)


def matriz_apariciones(df):
    """Matriz binaria (n_sorteos x 41): 1 si el numero k salio en ese sorteo.

    df debe venir ordenado cronologicamente (sorteo ascendente).
    """
    M = np.zeros((len(df), UNIVERSO), dtype=np.int8)
    filas = np.repeat(np.arange(len(df)), POR_SORTEO)
    cols = df[NUMS].to_numpy().ravel() - 1
    M[filas, cols] = 1
    return M


def construir_features(M, fechas):
    """Features causales por (sorteo t, numero k), calculadas solo con t' < t.

    Devuelve X con forma (n_sorteos, 41, n_features). La fila t nunca mira
    el sorteo t ni ninguno posterior.
    """
    n = len(M)
    acum = np.cumsum(M, axis=0)                      # apariciones hasta t inclusive
    previo = np.vstack([np.zeros((1, UNIVERSO)), acum[:-1]])  # hasta t-1

    feats, nombres = [], []

    t = np.arange(n).reshape(-1, 1)

    # Frecuencia relativa en las ultimas W extracciones (numeros "calientes").
    # ventana(t) = previo[t] - previo[t-w] = apariciones en los sorteos
    # [t-w, t-1], es decir sin tocar nunca el sorteo t.
    for w in VENTANAS:
        atrasado = np.vstack([np.zeros((min(w, n), UNIVERSO)),
                              previo[:max(n - w, 0)]])
        feats.append((previo - atrasado) / np.maximum(np.minimum(t, w), 1))
        nombres.append(f"freq_{w}")

    # Frecuencia historica acumulada (desde el inicio de la serie)
    feats.append(previo / np.maximum(t, 1))
    nombres.append("freq_total")

    # Sorteos transcurridos desde la ultima aparicion (numeros "frios")
    gap = np.zeros((n, UNIVERSO))
    ultima = np.full(UNIVERSO, -1)
    for i in range(n):
        gap[i] = np.where(ultima >= 0, i - ultima, i + 1)
        ultima = np.where(M[i] == 1, i, ultima)
    feats.append(gap)
    nombres.append("gap")

    # Identidad del numero y estacionalidad del sorteo
    feats.append(np.tile(np.arange(1, UNIVERSO + 1), (n, 1)))
    nombres.append("numero")
    feats.append(np.tile(fechas.dt.dayofweek.to_numpy().reshape(-1, 1),
                         (1, UNIVERSO)))
    nombres.append("dia_semana")
    feats.append(np.tile(fechas.dt.month.to_numpy().reshape(-1, 1),
                         (1, UNIVERSO)))
    nombres.append("mes")

    return np.stack(feats, axis=-1), nombres


def aciertos_top6(probas, reales):
    """Cuenta cuantos de los 6 numeros mas probables salieron de verdad."""
    top = np.argsort(-probas)[:POR_SORTEO] + 1
    return len(set(top) & set(reales))


def evaluar(nombre, aciertos):
    """Compara la media de aciertos contra el azar puro.

    Bajo la hipotesis nula (sorteo justo, eleccion sin informacion) los
    aciertos siguen una hipergeometrica: 6 extraidos de 41, 6 marcados.
    """
    a = np.asarray(aciertos, dtype=float)
    dist = stats.hypergeom(UNIVERSO, POR_SORTEO, POR_SORTEO)
    mu, sigma = dist.mean(), dist.std()
    err = sigma / np.sqrt(len(a))
    z = (a.mean() - mu) / err
    p = stats.norm.sf(abs(z)) * 2
    print(f"  {nombre:<28} {a.mean():.4f}   z = {z:+5.2f}   p = {p:.3f}"
          f"   {'SUPERA AL AZAR' if p < 0.05 and z > 0 else 'indistinguible del azar'}")
    return a.mean(), z, p


def main(n_test, cada):
    df = (pd.read_csv(MAESTRO, parse_dates=["fecha"])
            .sort_values("sorteo").reset_index(drop=True))
    M = matriz_apariciones(df)
    X, nombres = construir_features(M, df["fecha"])
    n = len(df)
    inicio = n - n_test

    print(f"Dataset: {n} sorteos ({df['sorteo'].iloc[0]}-{df['sorteo'].iloc[-1]})")
    print(f"Entrenamiento inicial: sorteos hasta el indice {inicio} | "
          f"Test: {n_test} sorteos | reajuste cada {cada}\n")
    print(f"Features ({len(nombres)}): {', '.join(nombres)}\n")

    modelo = None
    hits_modelo, hits_azar, hits_frecuentes, hits_frios = [], [], [], []
    y_true_all, y_prob_all = [], []
    rng = np.random.default_rng(42)

    for t in range(inicio, n):
        # Reajuste periodico con ventana expansiva: solo sorteos < t
        if modelo is None or (t - inicio) % cada == 0:
            X_tr = X[:t].reshape(-1, X.shape[-1])
            y_tr = M[:t].ravel()
            modelo = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=42)
            modelo.fit(X_tr, y_tr)

        probas = modelo.predict_proba(X[t])[:, 1]
        reales = df.loc[t, NUMS].to_numpy()

        hits_modelo.append(aciertos_top6(probas, reales))
        y_true_all.append(M[t])
        y_prob_all.append(probas)

        # Baselines calculados con la misma informacion causal
        hits_azar.append(len(set(rng.choice(UNIVERSO, POR_SORTEO,
                                            replace=False) + 1) & set(reales)))
        conteo = M[:t].sum(axis=0)
        hits_frecuentes.append(aciertos_top6(conteo.astype(float), reales))
        hits_frios.append(aciertos_top6(X[t][:, nombres.index("gap")], reales))

    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)

    print("Aciertos promedio sobre 6 numeros:\n")
    evaluar("Modelo (HistGradientBoost)", hits_modelo)
    evaluar("Baseline azar puro", hits_azar)
    evaluar("Baseline mas frecuentes", hits_frecuentes)
    evaluar("Baseline mas atrasados", hits_frios)
    print(f"\n  {'Esperanza teorica del azar':<28} "
          f"{POR_SORTEO * POR_SORTEO / UNIVERSO:.4f}")

    print("\nCalidad de las probabilidades estimadas:")
    print(f"  ROC AUC : {roc_auc_score(y_true_all, y_prob_all):.4f}  (0.5 = sin informacion)")
    print(f"  Log loss: {log_loss(y_true_all, y_prob_all):.4f}")
    base = np.full_like(y_prob_all, POR_SORTEO / UNIVERSO)
    print(f"  Log loss prediciendo siempre 6/41: {log_loss(y_true_all, base):.4f}")
    print(f"  Rango de P(k) estimadas: [{y_prob_all.min():.4f}, {y_prob_all.max():.4f}]"
          f"  (constante 6/41 = {POR_SORTEO / UNIVERSO:.4f})")

    dist = pd.Series(hits_modelo).value_counts().sort_index()
    print(f"\nDistribucion de aciertos del modelo: {dist.to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=400,
                    help="sorteos finales reservados para el backtest")
    ap.add_argument("--refit", type=int, default=25,
                    help="cada cuantos sorteos se reentrena el modelo")
    a = ap.parse_args()
    main(a.test, a.refit)
