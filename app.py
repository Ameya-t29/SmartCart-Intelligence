from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "smartcart_customers.csv"


def make_encoder():
    # Compatible with different scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data(df):
    """Reproduce the main preprocessing/feature-engineering flow from the notebook."""
    df = df.copy()

    # Required columns from the original notebook.
    required = [
        "ID", "Year_Birth", "Education", "Marital_Status", "Income",
        "Kidhome", "Teenhome", "Dt_Customer", "Recency",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    # Same missing-value treatment.
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Same feature engineering.
    df["Age"] = 2026 - pd.to_numeric(df["Year_Birth"], errors="coerce")

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], dayfirst=True, errors="coerce"
    )
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (
        reference_date - df["Dt_Customer"]
    ).dt.days

    spending_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df["Total_Spending"] = df[spending_cols].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Same education grouping.
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "PhD": "Postgraduate",
        "Master": "Postgraduate"
    })

    # Same living arrangement grouping.
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Absurd": "Alone",
        "YOLO": "Alone"
    })

    # Same columns removed before encoding.
    cols_to_drop = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome",
        "Teenhome", "Dt_Customer"
    ] + spending_cols

    cleaned = df.drop(columns=cols_to_drop)

    # Same outlier rules from the notebook.
    cleaned = cleaned[(cleaned["Age"] < 90) & (cleaned["Income"] < 600_000)]
    cleaned = cleaned.dropna()

    # Encode categorical columns.
    cat_cols = ["Education", "Living_With"]
    encoder = make_encoder()
    encoded = encoder.fit_transform(cleaned[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=cleaned.index
    )

    numeric_df = cleaned.drop(columns=cat_cols)
    model_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Standard scaling.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df)

    # The notebook uses 3D PCA before the clustering experiments.
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Display data keeps customer identity + useful human-readable fields.
    display_df = df.loc[cleaned.index].copy()
    display_df["Age"] = cleaned["Age"]
    display_df["Total_Spending"] = cleaned["Total_Spending"]
    display_df["Total_Children"] = cleaned["Total_Children"]
    display_df["Customer_Tenure_Days"] = cleaned["Customer_Tenure_Days"]

    return model_df, X_pca, display_df, pca


def calculate_model_data(model_name="kmeans", n_clusters=4):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Put smartcart_customers.csv inside the data folder."
        )

    raw = pd.read_csv(DATA_PATH)
    model_df, X_pca, display_df, pca = prepare_data(raw)

    n_clusters = int(np.clip(n_clusters, 2, min(10, len(X_pca) - 1)))

    # KMeans and Agglomerative are both part of the notebook.
    if model_name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_pca)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else 0

    result = display_df.copy()
    result["cluster"] = labels
    result["PCA1"] = X_pca[:, 0]
    result["PCA2"] = X_pca[:, 1]
    result["PCA3"] = X_pca[:, 2]

    # Cluster summary, using the same spirit as X.groupby("clusters").mean()
    numeric_summary = model_df.copy()
    numeric_summary["cluster"] = labels
    summary = numeric_summary.groupby("cluster").mean(numeric_only=True)

    # Elbow data from the notebook.
    max_k = min(10, len(X_pca) - 1)
    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        wcss.append(float(km.inertia_))

    # Silhouette data from the notebook.
    sil_k = list(range(2, max_k + 1))
    sil_scores = []
    for k in sil_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_pca)
        sil_scores.append(float(silhouette_score(X_pca, labels_k)))

    # A simple automatic elbow estimate, used only as a UI hint.
    elbow_k = 4
    if len(wcss) >= 3:
        points = np.array([[k + 1, wcss[k]] for k in range(len(wcss))])
        a, b = points[0], points[-1]
        line = b - a
        line_norm = np.linalg.norm(line)
        if line_norm > 0:
            distances = []
            for p in points:
                distances.append(
                    abs(np.cross(line, p - a)) / line_norm
                )
            elbow_k = int(np.argmax(distances) + 1)

    cluster_counts = (
        result["cluster"].value_counts().sort_index().to_dict()
    )

    return {
        "rows": result,
        "summary": summary,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "wcss": wcss,
        "wcss_k": list(range(1, max_k + 1)),
        "silhouette": sil_scores,
        "silhouette_k": sil_k,
        "silhouette_score": float(score),
        "elbow_k": elbow_k,
        "cluster_counts": cluster_counts,
        "model": model_name,
        "k": n_clusters,
        "total_rows": len(result),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def dashboard():
    try:
        model_name = request.args.get("model", "kmeans").lower()
        if model_name not in {"kmeans", "agglomerative"}:
            model_name = "kmeans"

        k = int(request.args.get("k", 4))
        data = calculate_model_data(model_name, k)
        rows = data["rows"]

        # Compact JSON-safe response for the browser.
        preview = rows[
            [
                "ID", "Age", "Income", "Education", "Living_With",
                "Recency", "Total_Spending", "Total_Children",
                "cluster", "PCA1", "PCA2", "PCA3"
            ]
        ].copy()

        preview = preview.replace({np.nan: None})
        records = preview.to_dict(orient="records")

        summary = []
        for cluster_id, row in data["summary"].iterrows():
            summary.append({
                "cluster": int(cluster_id),
                "customers": int((rows["cluster"] == cluster_id).sum()),
                "avg_income": round(float(rows.loc[rows["cluster"] == cluster_id, "Income"].mean()), 2),
                "avg_spending": round(float(rows.loc[rows["cluster"] == cluster_id, "Total_Spending"].mean()), 2),
                "avg_recency": round(float(rows.loc[rows["cluster"] == cluster_id, "Recency"].mean()), 2),
                "avg_age": round(float(rows.loc[rows["cluster"] == cluster_id, "Age"].mean()), 2),
            })

        return jsonify({
            "success": True,
            "model": data["model"],
            "k": data["k"],
            "total_rows": data["total_rows"],
            "silhouette_score": data["silhouette_score"],
            "elbow_k": data["elbow_k"],
            "pca_variance": data["pca_variance"],
            "wcss_k": data["wcss_k"],
            "wcss": data["wcss"],
            "silhouette_k": data["silhouette_k"],
            "silhouette": data["silhouette"],
            "cluster_counts": data["cluster_counts"],
            "summary": summary,
            "records": records,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/customer/<int:customer_id>")
def customer(customer_id):
    try:
        data = calculate_model_data(
            request.args.get("model", "kmeans"),
            int(request.args.get("k", 4))
        )
        rows = data["rows"]
        found = rows[rows["ID"] == customer_id]
        if found.empty:
            return jsonify({"success": False, "error": "Customer not found"}), 404

        row = found.iloc[0]
        cluster_rows = rows[rows["cluster"] == row["cluster"]]

        return jsonify({
            "success": True,
            "customer": {
                "ID": int(row["ID"]),
                "Age": int(row["Age"]),
                "Income": float(row["Income"]),
                "Education": str(row["Education"]),
                "Living_With": str(row["Living_With"]),
                "Recency": int(row["Recency"]),
                "Total_Spending": float(row["Total_Spending"]),
                "Total_Children": int(row["Total_Children"]),
                "cluster": int(row["cluster"]),
                "PCA1": float(row["PCA1"]),
                "PCA2": float(row["PCA2"]),
                "PCA3": float(row["PCA3"]),
                "NumWebPurchases": int(row.get("NumWebPurchases", 0)),
                "NumCatalogPurchases": int(row.get("NumCatalogPurchases", 0)),
                "NumStorePurchases": int(row.get("NumStorePurchases", 0)),
            },
            "cluster_stats": {
                "customers": int(len(cluster_rows)),
                "avg_income": float(cluster_rows["Income"].mean()),
                "avg_spending": float(cluster_rows["Total_Spending"].mean()),
                "avg_recency": float(cluster_rows["Recency"].mean()),
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
