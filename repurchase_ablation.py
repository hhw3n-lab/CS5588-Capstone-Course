"""
Retail360 — 60-day repurchase ablation (Challenge 2 research extension)

RQ1: Do customer–product community features improve 60-day repeat-purchase
     prediction beyond RFM and temporal features?

Protocol
--------
1. Freeze cutoff date t.
2. Features from transactions strictly before t.
3. Label y=1 if customer purchases in [t, t+60d].
4. Chronological split; same split for every ablation.
5. Report PR-AUC, ROC-AUC, Lift@10%.

Usage
-----
  python experiments/repurchase_ablation.py \\
      --data data/online_retail_II.csv \\
      --cutoff 2011-06-01 \\
      --min-invoices 3

If --data is omitted, a synthetic demo cohort is used so the script runs offline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def load_transactions(path: str | None) -> pd.DataFrame:
    if path is None:
        # Synthetic cohort for offline demo
        rng = np.random.default_rng(7)
        rows = []
        base = datetime(2010, 12, 1)
        for cid in range(1000, 1200):
            n_inv = int(rng.integers(1, 15))
            for k in range(n_inv):
                day = int(rng.integers(0, 400))
                qty = int(rng.integers(1, 20))
                price = float(rng.uniform(0.5, 15))
                stock = f"S{int(rng.integers(1, 80)):03d}"
                rows.append(
                    {
                        "Invoice": f"I{cid}-{k}",
                        "StockCode": stock,
                        "Description": f"Item {stock}",
                        "Quantity": qty,
                        "InvoiceDate": base + timedelta(days=day),
                        "UnitPrice": price,
                        "CustomerID": cid,
                        "Country": "United Kingdom",
                    }
                )
        df = pd.DataFrame(rows)
        return df

    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    # Standardise column names if needed
    rename = {
        "Customer ID": "CustomerID",
        "InvoiceDate": "InvoiceDate",
        "UnitPrice": "UnitPrice",
        "Quantity": "Quantity",
        "StockCode": "StockCode",
        "Invoice": "Invoice",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df


def build_labels_and_features(
    df: pd.DataFrame,
    cutoff: datetime,
    horizon_days: int = 60,
    min_invoices: int = 3,
) -> pd.DataFrame:
    pre = df[df["InvoiceDate"] < cutoff].copy()
    post = df[
        (df["InvoiceDate"] >= cutoff)
        & (df["InvoiceDate"] < cutoff + timedelta(days=horizon_days))
    ].copy()

    # Invoice counts pre-cutoff
    pre["Invoice"] = pre["Invoice"].astype(str)
    inv_counts = pre.groupby("CustomerID")["Invoice"].nunique()
    eligible = inv_counts[inv_counts >= min_invoices].index

    pre = pre[pre["CustomerID"].isin(eligible)]
    post_buyers = set(post["CustomerID"].unique())

    # RFM-style features
    last_date = pre.groupby("CustomerID")["InvoiceDate"].max()
    first_date = pre.groupby("CustomerID")["InvoiceDate"].min()
    freq = pre.groupby("CustomerID")["Invoice"].nunique()
    monetary = (pre["Quantity"] * pre["UnitPrice"]).groupby(pre["CustomerID"]).sum()
    diversity = pre.groupby("CustomerID")["StockCode"].nunique()
    aov = monetary / freq
    recency = (cutoff - last_date).dt.days
    tenure = (last_date - first_date).dt.days
    avg_gap = tenure / freq.clip(lower=1)

    # Simple community proxy: hash of top stock cluster (placeholder for Leiden ID)
    # Real pipeline should build bipartite graph + Leiden; here we use dominant product bucket.
    top_stock = (
        pre.groupby(["CustomerID", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values("Quantity", ascending=False)
        .groupby("CustomerID")
        .first()["StockCode"]
    )
    community = top_stock.astype(str).str[:3]  # coarse product family proxy

    out = pd.DataFrame(
        {
            "recency": recency,
            "frequency": freq,
            "monetary": monetary,
            "diversity": diversity,
            "aov": aov,
            "tenure": tenure,
            "avg_gap": avg_gap,
            "community": community,
        }
    ).reindex(eligible)
    out = out.fillna(0)
    out["y"] = out.index.map(lambda c: 1 if c in post_buyers else 0)
    return out


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.1) -> float:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    order = np.argsort(-y_score)[:k]
    base = y_true.mean()
    if base == 0:
        return 0.0
    return float(y_true[order].mean() / base)


def eval_model(X_train, y_train, X_test, y_test, model_name: str = "lgb"):
    if model_name == "dummy":
        # Predict prevalence
        p = np.full(len(y_test), y_train.mean())
        return p
    if model_name == "lr":
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    # LightGBM or sklearn fallback
    if HAS_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=4, verbose=-1
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=80)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


FEATURE_SETS = {
    "A_dummy": [],
    "B_rfm": ["recency", "frequency", "monetary"],
    "C_temporal": ["recency", "frequency", "monetary", "tenure", "avg_gap"],
    "D_diversity": [
        "recency",
        "frequency",
        "monetary",
        "tenure",
        "avg_gap",
        "diversity",
        "aov",
    ],
    "E_community": [
        "recency",
        "frequency",
        "monetary",
        "tenure",
        "avg_gap",
        "diversity",
        "aov",
        "community_enc",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to Online Retail II CSV")
    ap.add_argument("--cutoff", default="2011-06-01")
    ap.add_argument("--min-invoices", type=int, default=3)
    ap.add_argument("--out", default="results/ablation_metrics.csv")
    args = ap.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d")
    df = load_transactions(args.data)
    feat = build_labels_and_features(df, cutoff, min_invoices=args.min_invoices)

    # Encode community
    le = LabelEncoder()
    feat["community_enc"] = le.fit_transform(feat["community"].astype(str))

    # Chronological-ish split by customer id order as a simple holdout
    # (replace with true time-window split when full multi-cutoff pipeline is ready)
    ids = np.array(sorted(feat.index))
    split = int(0.7 * len(ids))
    train_ids, test_ids = ids[:split], ids[split:]
    train = feat.loc[train_ids]
    test = feat.loc[test_ids]

    rows = []
    for name, cols in FEATURE_SETS.items():
        if name == "A_dummy":
            scores = eval_model(
                train[["recency"]], train["y"], test[["recency"]], test["y"], "dummy"
            )
        else:
            scores = eval_model(
                train[cols], train["y"], test[cols], test["y"], "lgb"
            )
        y = test["y"].values
        pr = average_precision_score(y, scores) if y.sum() > 0 else 0.0
        roc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.5
        lift = lift_at_k(y, scores, 0.1)
        rows.append(
            {
                "condition": name,
                "n_train": len(train),
                "n_test": len(test),
                "pos_rate_test": float(y.mean()),
                "PR_AUC": round(pr, 4),
                "ROC_AUC": round(roc, 4),
                "Lift@10%": round(lift, 3),
            }
        )
        print(f"{name:16s}  PR-AUC={pr:.3f}  ROC-AUC={roc:.3f}  Lift@10%={lift:.2f}x")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
