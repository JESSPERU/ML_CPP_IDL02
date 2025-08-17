"""
=============================================
IDL02 - Proyecto de No Supervisado
Tema: Clustering (K-Means y DBSCAN) + PCA + Métricas
=============================================
"""

# 0) IMPORTACIÓN DE LIBRERÍAS
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# 1) FUNCIONES AUXILIARES
def load_data(path):
    """
    Carga del CSV y verificación de columnas requeridas:
    - ComprasTotales
    - FrecuenciaCompras
    - MontoPromedioCompra
    Eliminación de nulos para evitar errores al escalar/clusterear.
    """
    df = pd.read_csv(path)
    cols = ["ComprasTotales", "FrecuenciaCompras", "MontoPromedioCompra"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # Si falta alguna columna, se alerta
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")
    df = df.dropna(subset=cols).copy()
    return df, cols


def scale_features(df, cols):
    """
    Escalo con StandardScaler para que todas las variables queden
    en la misma escala (media 0, varianza 1). Esto es clave porque
    K-Means/DBSCAN usan distancias.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    return X, scaler


def kmeans_sweep(X, kmin=2, kmax=8, random_state=42):
    """
    Pruebo varios k en K-Means y guardo:
    - Inercia (para el elbow)
    - Silhouette (para medir separación/compacidad)
    - El modelo y las etiquetas por cada k, para luego elegir el mejor.
    """
    inertias, silhouettes, labels_by_k = [], [], {}
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        y = km.fit_predict(X)
        labels_by_k[k] = (km, y)
        inertias.append(km.inertia_)
        # Silhouette solo tiene sentido si hay más de un cluster
        s = silhouette_score(X, y) if len(set(y)) > 1 else np.nan
        silhouettes.append(s)
    return inertias, silhouettes, labels_by_k


def plot_elbow(k_values, inertias, outpath):
    """Gráfico del método del codo (inercia vs k)."""
    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("k (número de clusters)")
    plt.ylabel("Inercia (SSE)")
    plt.title("Método del codo - K-Means")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_silhouette(k_values, silhouettes, outpath):
    """Gráfico de silhouette por cada k probado."""
    plt.figure()
    plt.plot(k_values, silhouettes, marker="o")
    plt.xlabel("k (número de clusters)")
    plt.ylabel("Coeficiente de Silueta")
    plt.title("Silueta por k - K-Means")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def k_distance_plot(X, k, outpath):
    """
    k-distance plot para orientar el valor de eps en DBSCAN.
    La “rodilla” de esta curva suele ser una buena pista para eps.
    """
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    kdist = np.sort(distances[:, -1])  # distancia al k-ésimo vecino
    plt.figure()
    plt.plot(kdist)
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {k}° vecino")
    plt.title("k-distance plot (sugerencia para eps en DBSCAN)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Corro DBSCAN con los hiperparámetros que pase por consola.
    eps = radio de vecindad, min_samples = puntos mínimos en la vecindad.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y = db.fit_predict(X)
    return db, y


def run_pca(X, n_components=2):
    """
    Aplico PCA para reducir a 2D (ideal para graficar).
    También serviré la varianza explicada.
    """
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    return pca, Z


def plot_pca_scatter(Z, labels, title, outpath):
    """Gráfico 2D de PCA coloreado por etiquetas de cluster."""
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=labels)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pca_variance(pca, outpath):
    """Gráfico de varianza explicada acumulada por PCA."""
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    plt.figure()
    plt.plot(range(1, len(var) + 1), cum, marker="o")
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza explicada acumulada")
    plt.title("PCA - Varianza explicada")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def profile_clusters(df, labels, cols):
    """
    Pequeño “perfil” por cluster: media, mediana y cantidad.
    Esto me ayuda a interpretar segmentos para las recomendaciones.
    """
    prof = df.groupby(labels)[cols].agg(["mean", "median", "count"]).round(2)
    return prof


# 2) MAIN: DONDE JUNTO TODO 

def main():
    # --- 2.1) Argumentos por consola para que el profe vea que parametrizo ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/historial_compras.csv",
                        help="Ruta al CSV de clientes (por defecto: data/historial_compras.csv)")
    parser.add_argument("--kmin", type=int, default=2, help="k mínimo para K-Means")
    parser.add_argument("--kmax", type=int, default=8, help="k máximo para K-Means")
    parser.add_argument("--eps", type=float, default=1.8, help="eps para DBSCAN")
    parser.add_argument("--min_samples", type=int, default=3, help="min_samples para DBSCAN")
    args = parser.parse_args()

    # --- 2.2) Creo carpetas de salida si no existen ---
    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --- 2.3) Cargo datos y los dejo limpios/listos ---
    df, cols = load_data(args.data)

    # --- 2.4) Escalado (clave para distancias) ---
    X, scaler = scale_features(df, cols)

    # --- 2.5) K-Means: pruebo varios k y guardo inercia/silueta ---
    k_values = list(range(args.kmin, args.kmax + 1))
    inertias, silhouettes, labels_by_k = kmeans_sweep(X, args.kmin, args.kmax)

    # Gráficos para justificar elección de k
    plot_elbow(k_values, inertias, "figures/elbow_kmeans.png")
    plot_silhouette(k_values, silhouettes, "figures/silhouette_by_k.png")

    # --- 2.6) Elijo k: priorizo la mejor silueta; si no hay, uso inercia mínima como fallback ---
    valid_pairs = [(k, s) for k, s in zip(k_values, silhouettes) if not np.isnan(s)]
    if valid_pairs:
        best_k = max(valid_pairs, key=lambda t: t[1])[0]
    else:
        # Si por alguna razón no hay silhouette válida, me quedo con el "codo" (aprox por inercia)
        best_k = k_values[int(np.argmin(inertias))]

    km, y_km = labels_by_k[best_k]
    df_km = df.copy()
    df_km["KMeans_Cluster"] = y_km
    df_km.to_csv("outputs/segmented_kmeans.csv", index=False)

    # --- 2.7) DBSCAN: hago k-distance plot y luego corro con los hiperparámetros que puse ---
    k_distance_plot(X, args.min_samples, "figures/kdist_dbscan.png")
    db, y_db = run_dbscan(X, eps=args.eps, min_samples=args.min_samples)
    df_db = df.copy()
    df_db["DBSCAN_Cluster"] = y_db
    df_db.to_csv("outputs/segmented_dbscan.csv", index=False)

    # --- 2.8) PCA a 2 componentes: varianza y gráficos por algoritmo ---
    pca, Z = run_pca(X, n_components=2)
    plot_pca_variance(pca, "figures/pca_variance.png")
    plot_pca_scatter(Z, y_km, f"PCA (2D) - K-Means (k={best_k})", "figures/pca_kmeans.png")
    plot_pca_scatter(Z, y_db, f"PCA (2D) - DBSCAN (eps={args.eps}, min_samples={args.min_samples})", "figures/pca_dbscan.png")

    # --- 2.9) Métricas de evaluación (lo que pide el trabajo) ---
    sil_km = silhouette_score(X, y_km) if len(set(y_km)) > 1 else np.nan

    # Para DBSCAN, la silueta tiene sentido si realmente armó >=2 clusters (y no todo ruido)
    labels_db = set(y_db)
    if len(labels_db) > 1 and not (labels_db == {-1}):
        sil_db = silhouette_score(X, y_db)
    else:
        sil_db = np.nan

    # --- 2.10) Perfil de segmentos para poder dar recomendaciones ---
    prof_km = profile_clusters(df, y_km, cols)
    prof_db = profile_clusters(df, y_db, cols) if len(set(y_db)) > 1 else None

    # --- 2.11) Resumen en TXT para llevar al informe ---
    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Resumen IDL02 - Versión Estudiante ===\n")
        f.write(f"Variables utilizadas: {cols}\n")
        f.write(f"Mejor k (K-Means): {best_k}\n")
        f.write(f"Silueta K-Means: {sil_km:.3f}\n")
        if not np.isnan(sil_db):
            f.write(f"Silueta DBSCAN: {sil_db:.3f}\n")
        else:
            f.write("Silueta DBSCAN: N/A (clusters insuficientes o solo ruido)\n")

        f.write("\n-- Perfil de clusters K-Means (media/mediana/cantidad) --\n")
        f.write(str(prof_km))

        if prof_db is not None:
            f.write("\n\n-- Perfil de clusters DBSCAN --\n")
            f.write(str(prof_db))

    # --- 2.12) Mensajito final en consola para saber que todo ok ---
    print("[OK] Pipeline terminado.")
    print(f"Mejor k: {best_k} | Silueta K-Means: {sil_km:.3f}")
    if not np.isnan(sil_db):
        print(f"Silueta DBSCAN: {sil_db:.3f}")
    else:
        print("Silueta DBSCAN: N/A (clusters insuficientes o solo ruido)")


# 3) ARRANQUE DEL SCRIPT 
if __name__ == "__main__":
    main()