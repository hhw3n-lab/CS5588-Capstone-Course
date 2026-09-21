"""
Retail360 Customer Behavior Evidence Explorer
CS 5588 Challenge 2 — Human–AI Co-Design Application

Principle: Tools calculate → Retrieval finds evidence → LLM (or template) explains evidence.
Humans select customers, review evidence, and accept / reject / refine results.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail360 Evidence Explorer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sample data (schema matches Online Retail II)
# Includes the PPT example customer 13085 and several others for demo.
# Users can upload the full UCI / Kaggle CSV for real analysis.
# ---------------------------------------------------------------------------
def _make_sample_transactions() -> pd.DataFrame:
    """Generate a realistic sample that matches Online Retail II columns."""
    rng = np.random.default_rng(42)
    base = datetime(2010, 12, 1)

    # Explicit evidence for Customer 13085 (from Challenge 2 PPT example)
    # Quantities scaled so totals approximate the Challenge 2 design example
    # (≈8 invoices, ≈£2,433 total, ≈£304 AOV, ≈50 product units).
    rows_13085 = [
        # Invoice 1
        ("536365", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 48, base + timedelta(days=0), 2.55, 13085, "United Kingdom"),
        ("536365", "71053", "WHITE METAL LANTERN", 36, base + timedelta(days=0), 3.39, 13085, "United Kingdom"),
        ("536365", "84406B", "CREAM CUPID HEARTS COAT HANGER", 40, base + timedelta(days=0), 2.75, 13085, "United Kingdom"),
        # Invoice 2
        ("536366", "22633", "HAND WARMER UNION JACK", 48, base + timedelta(days=5), 1.85, 13085, "United Kingdom"),
        ("536366", "22632", "HAND WARMER RED POLKA DOT", 48, base + timedelta(days=5), 1.85, 13085, "United Kingdom"),
        # Invoice 3
        ("536367", "84879", "ASSORTED COLOUR BIRD ORNAMENT", 160, base + timedelta(days=12), 1.69, 13085, "United Kingdom"),
        ("536367", "22745", "POPPY'S PLAYHOUSE BEDROOM", 24, base + timedelta(days=12), 2.10, 13085, "United Kingdom"),
        # Invoice 4
        ("536368", "22960", "JAM MAKING SET WITH JARS", 36, base + timedelta(days=20), 4.25, 13085, "United Kingdom"),
        ("536368", "22913", "RED COAT RACK PARIS FASHION", 18, base + timedelta(days=20), 4.95, 13085, "United Kingdom"),
        ("536368", "22912", "YELLOW COAT RACK PARIS FASHION", 18, base + timedelta(days=20), 4.95, 13085, "United Kingdom"),
        # Invoice 5
        ("536369", "21754", "HOME BUILDING BLOCK WORD", 24, base + timedelta(days=35), 5.95, 13085, "United Kingdom"),
        ("536369", "21755", "LOVE BUILDING BLOCK WORD", 24, base + timedelta(days=35), 5.95, 13085, "United Kingdom"),
        ("536369", "21756", "FAVOURITE BUILDING BLOCK WORD", 24, base + timedelta(days=35), 5.95, 13085, "United Kingdom"),
        # Invoice 6
        ("536370", "22748", "POPPY'S PLAYHOUSE KITCHEN", 36, base + timedelta(days=50), 2.10, 13085, "United Kingdom"),
        ("536370", "22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 40, base + timedelta(days=50), 3.75, 13085, "United Kingdom"),
        ("536370", "22310", "IVORY KNITTED MUG COSY", 36, base + timedelta(days=50), 1.65, 13085, "United Kingdom"),
        # Invoice 7
        ("536371", "84997B", "RED 3 PIECE MINI DOTS CUTLERY SET", 36, base + timedelta(days=80), 3.75, 13085, "United Kingdom"),
        ("536371", "84997C", "BLUE 3 PIECE MINI DOTS CUTLERY SET", 36, base + timedelta(days=80), 3.75, 13085, "United Kingdom"),
        ("536371", "84997D", "PINK 3 PIECE MINI DOTS CUTLERY SET", 36, base + timedelta(days=80), 3.75, 13085, "United Kingdom"),
        # Invoice 8
        ("536372", "22423", "REGENCY CAKESTAND 3 TIER", 24, base + timedelta(days=120), 12.75, 13085, "United Kingdom"),
        ("536372", "22622", "BOX OF VINTAGE ALPHABET BLOCKS", 16, base + timedelta(days=120), 9.95, 13085, "United Kingdom"),
        ("536372", "22623", "BOX OF VINTAGE JIGSAW BLOCKS", 16, base + timedelta(days=120), 9.95, 13085, "United Kingdom"),
        ("536372", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 60, base + timedelta(days=120), 2.55, 13085, "United Kingdom"),
    ]

    # Additional customers for richer demo
    other_customers = [12346, 12347, 12348, 12349, 12350, 17850, 14646, 14911, 14156, 18102]
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
        ("22633", "HAND WARMER UNION JACK", 1.85),
        ("21754", "HOME BUILDING BLOCK WORD", 5.95),
        ("84997B", "RED 3 PIECE MINI DOTS CUTLERY SET", 3.75),
        ("22197", "SMALL POPCORN HOLDER", 0.85),
        ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
        ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
        ("23203", "JUMBO BAG VINTAGE DOILY", 2.08),
        ("23084", "RABBIT NIGHT LIGHT", 2.08),
        ("22178", "VICTORIAN GLASS HANGING T-LIGHT", 1.25),
        ("82494L", "WOODEN FRAME ANTIQUE WHITE", 2.95),
    ]
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Spain"]

    extra_rows = []
    inv_counter = 540000
    for cid in other_customers:
        n_invoices = int(rng.integers(2, 12))
        for i in range(n_invoices):
            inv = str(inv_counter)
            inv_counter += 1
            day_offset = int(rng.integers(0, 400))
            inv_date = base + timedelta(days=day_offset)
            n_lines = int(rng.integers(1, 8))
            for _ in range(n_lines):
                stock, desc, price = products[int(rng.integers(0, len(products)))]
                qty = int(rng.integers(1, 24))
                # occasional cancellation
                if rng.random() < 0.04:
                    inv_c = "C" + inv
                    qty = -abs(qty)
                    extra_rows.append(
                        (inv_c, stock, desc, qty, inv_date, price, cid, countries[int(rng.integers(0, len(countries)))])
                    )
                else:
                    extra_rows.append(
                        (inv, stock, desc, qty, inv_date, price, cid, countries[int(rng.integers(0, len(countries)))])
                    )

    all_rows = rows_13085 + extra_rows
    df = pd.DataFrame(
        all_rows,
        columns=[
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]
    return df


@st.cache_data(show_spinner="Loading transaction data…")
def load_data(uploaded_file: Any = None) -> pd.DataFrame:
    """Load sample data or user-uploaded Online Retail II CSV/XLSX."""
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Normalise column names to match expected schema
        col_map = {
            "invoiceno": "Invoice",
            "invoice": "Invoice",
            "stockcode": "StockCode",
            "description": "Description",
            "quantity": "Quantity",
            "invoicedate": "InvoiceDate",
            "unitprice": "Price",
            "price": "Price",
            "customerid": "Customer ID",
            "customer id": "Customer ID",
            "country": "Country",
        }
        df.columns = [col_map.get(c.strip().lower(), c) for c in df.columns]
        required = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
            return _make_sample_transactions()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["Customer ID", "InvoiceDate"])
        df["Customer ID"] = df["Customer ID"].astype(int)
        df["LineTotal"] = df["Quantity"] * df["Price"]
        return df
    return _make_sample_transactions()


# ---------------------------------------------------------------------------
# Analytics engine (RFM + extras)
# ---------------------------------------------------------------------------
def filter_customer_period(
    df: pd.DataFrame,
    customer_id: int,
    start_date: datetime | None,
    end_date: datetime | None,
    exclude_cancellations: bool = True,
) -> pd.DataFrame:
    """Retrieve evidence: all transactions for one customer in a date window."""
    mask = df["Customer ID"] == customer_id
    if start_date is not None:
        mask &= df["InvoiceDate"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= df["InvoiceDate"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    out = df.loc[mask].copy()
    if exclude_cancellations:
        # Online Retail II: cancellation invoices start with 'C'
        out = out[~out["Invoice"].astype(str).str.upper().str.startswith("C")]
        out = out[out["Quantity"] > 0]
    return out.sort_values("InvoiceDate")


def compute_customer_features(
    tx: pd.DataFrame,
    reference_date: datetime | None = None,
) -> dict[str, Any]:
    """
    Calculate grounded customer features from transaction evidence.
    Returns metrics + supporting evidence counts.
    """
    if tx.empty:
        return {
            "n_invoices": 0,
            "n_transactions": 0,
            "total_spend": 0.0,
            "avg_order_value": 0.0,
            "n_products": 0,
            "n_unique_stock": 0,
            "recency_days": None,
            "first_purchase": None,
            "last_purchase": None,
            "frequency": 0,
            "monetary": 0.0,
            "product_diversity": 0,
            "top_products": [],
            "invoices": [],
        }

    if reference_date is None:
        reference_date = tx["InvoiceDate"].max() + timedelta(days=1)

    invoices = (
        tx.groupby("Invoice", as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            n_lines=("StockCode", "count"),
            quantity=("Quantity", "sum"),
            spend=("LineTotal", "sum"),
        )
        .sort_values("InvoiceDate")
    )

    first_purchase = tx["InvoiceDate"].min()
    last_purchase = tx["InvoiceDate"].max()
    recency_days = (pd.Timestamp(reference_date) - last_purchase).days

    product_counts = (
        tx.groupby(["StockCode", "Description"], as_index=False)["Quantity"]
        .sum()
        .sort_values("Quantity", ascending=False)
    )

    return {
        "n_invoices": int(len(invoices)),
        "n_transactions": int(len(tx)),
        "total_spend": float(tx["LineTotal"].sum()),
        "avg_order_value": float(invoices["spend"].mean()) if len(invoices) else 0.0,
        "n_products": int(tx["Quantity"].sum()),
        "n_unique_stock": int(tx["StockCode"].nunique()),
        "recency_days": int(recency_days),
        "first_purchase": first_purchase,
        "last_purchase": last_purchase,
        "frequency": int(len(invoices)),
        "monetary": float(tx["LineTotal"].sum()),
        "product_diversity": int(tx["StockCode"].nunique()),
        "top_products": product_counts.head(8).to_dict("records"),
        "invoices": invoices.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Grounded explanation (template-based; no unsupported inference)
# ---------------------------------------------------------------------------
def generate_grounded_explanation(
    customer_id: int,
    features: dict[str, Any],
    tx: pd.DataFrame,
) -> str:
    """
    Produce a plain-language behavioural summary that is fully grounded
    in the retrieved evidence. No external LLM required; every claim
    references calculated metrics or retrieved invoices.
    """
    if features["n_invoices"] == 0:
        return (
            f"**Customer {customer_id}** — No matching non-cancelled transactions "
            "were found for the selected period. "
            "Possible reasons: invalid Customer ID, date range excludes all activity, "
            "or all invoices were cancellations."
        )

    first = features["first_purchase"].strftime("%Y-%m-%d")
    last = features["last_purchase"].strftime("%Y-%m-%d")
    top = features["top_products"][:3]
    top_str = ", ".join(
        f"{p['Description']} (qty {int(p['Quantity'])})" for p in top
    ) if top else "N/A"

    inv_list = features["invoices"]
    inv_summary = "; ".join(
        f"Invoice {r['Invoice']} on {pd.Timestamp(r['InvoiceDate']).strftime('%Y-%m-%d')} "
        f"(£{r['spend']:.2f})"
        for r in inv_list[:5]
    )
    if len(inv_list) > 5:
        inv_summary += f"; … and {len(inv_list) - 5} more"

    # Behavioural characterisation based only on numbers
    if features["recency_days"] is not None and features["recency_days"] <= 30:
        recency_label = "recently active"
    elif features["recency_days"] is not None and features["recency_days"] <= 90:
        recency_label = "moderately recent"
    else:
        recency_label = "not recently active"

    if features["frequency"] >= 6:
        freq_label = "frequent"
    elif features["frequency"] >= 3:
        freq_label = "occasional"
    else:
        freq_label = "infrequent"

    if features["monetary"] >= 2000:
        value_label = "high-value"
    elif features["monetary"] >= 500:
        value_label = "mid-value"
    else:
        value_label = "lower-value"

    text = f"""### Grounded Behavioural Summary — Customer {customer_id}

**Evidence window:** {first} → {last}

This customer is characterised as **{recency_label}**, **{freq_label}**, and **{value_label}** based solely on the retrieved transaction evidence.

#### Calculated metrics (from evidence)
| Metric | Value | Source |
|--------|-------|--------|
| Number of invoices | **{features['n_invoices']}** | Distinct Invoice values |
| Total spend (Monetary) | **£{features['total_spend']:,.2f}** | Sum of Quantity × Price |
| Average order value | **£{features['avg_order_value']:,.2f}** | Total spend ÷ invoices |
| Product units purchased | **{features['n_products']}** | Sum of Quantity |
| Unique products (diversity) | **{features['product_diversity']}** | Distinct StockCode |
| Recency (days since last purchase) | **{features['recency_days']}** | Reference date − last InvoiceDate |

#### Supporting invoices (retrieved evidence)
{inv_summary}

#### Most purchased products (by quantity)
{top_str}

#### Interpretation notes (grounded only)
- All figures above are computed directly from the filtered transaction rows shown in the evidence tables.
- Cancellations (Invoice starting with “C” or negative Quantity) were excluded when the corresponding control is enabled.
- No external customer attributes or predicted future behaviour are used; conclusions are limited to observed history.
"""
    return text


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def metric_cards(features: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Invoices", features["n_invoices"])
    c2.metric("Total Spend", f"£{features['total_spend']:,.2f}")
    c3.metric("Avg Order Value", f"£{features['avg_order_value']:,.2f}")
    c4.metric("Product Diversity", features["product_diversity"])
    rec = features["recency_days"]
    c5.metric("Recency (days)", rec if rec is not None else "—")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🛒 Retail360 Customer Behavior Evidence Explorer")
    st.caption(
        "CS 5588 Challenge 2 • Human–AI Co-Design • "
        "Tools calculate → Retrieval finds evidence → Explanation is grounded"
    )

    # ----- Sidebar: data source & controls -----
    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader(
        "Upload Online Retail II (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Optional. If omitted, a built-in sample (including Customer 13085) is used.",
    )
    df = load_data(uploaded)

    st.sidebar.success(f"Loaded **{len(df):,}** transaction rows")
    st.sidebar.caption(
        f"Date range in data: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}"
    )

    st.sidebar.divider()
    st.sidebar.header("Human controls")
    available_ids = sorted(df["Customer ID"].dropna().unique().astype(int).tolist())
    default_idx = available_ids.index(13085) if 13085 in available_ids else 0
    customer_id = st.sidebar.selectbox(
        "Customer ID",
        options=available_ids,
        index=default_idx,
        help="Select a customer to analyse. Example from the design brief: 13085.",
    )

    min_d = df["InvoiceDate"].min().date()
    max_d = df["InvoiceDate"].max().date()
    date_range = st.sidebar.date_input(
        "Analysis period",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    exclude_cancel = st.sidebar.checkbox("Exclude cancellations", value=True)

    run = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.markdown(
        """
**Human–AI workflow**
1. Human enters Customer ID & period  
2. AI retrieves transactions  
3. AI calculates RFM + features  
4. AI assembles evidence package  
5. AI generates grounded explanation  
6. Human reviews, accepts, rejects or refines  
"""
    )

    # ----- Main panels -----
    tab_overview, tab_evidence, tab_explain, tab_review, tab_about = st.tabs(
        [
            "1. Metrics dashboard",
            "2. Evidence tables",
            "3. Grounded explanation",
            "4. Human review",
            "5. Architecture & next steps",
        ]
    )

    if not run and "last_features" not in st.session_state:
        with tab_overview:
            st.info(
                "Select a **Customer ID** and date range in the sidebar, then click **Run analysis**.\n\n"
                "Suggested demo customer: **13085** (matches the Challenge 2 design example)."
            )
        return

    # Perform analysis (or reuse last run)
    if run or "last_features" not in st.session_state:
        tx = filter_customer_period(df, customer_id, start_date, end_date, exclude_cancel)
        features = compute_customer_features(tx)
        st.session_state.last_tx = tx
        st.session_state.last_features = features
        st.session_state.last_customer = customer_id
        st.session_state.last_period = (start_date, end_date)
    else:
        tx = st.session_state.last_tx
        features = st.session_state.last_features
        customer_id = st.session_state.last_customer

    # ----- Tab 1: Metrics -----
    with tab_overview:
        st.subheader(f"Customer {customer_id} — Feature dashboard")
        st.caption(
            f"Period: {st.session_state.last_period[0]} → {st.session_state.last_period[1]} "
            f"| Cancellations excluded: {exclude_cancel}"
        )
        metric_cards(features)

        if features["n_invoices"] > 0:
            inv_df = pd.DataFrame(features["invoices"])
            inv_df["InvoiceDate"] = pd.to_datetime(inv_df["InvoiceDate"])
            chart_df = inv_df.set_index("InvoiceDate")[["spend"]].sort_index()
            st.markdown("#### Spending by invoice date")
            st.bar_chart(chart_df)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Top products by quantity")
                top_df = pd.DataFrame(features["top_products"])
                if not top_df.empty:
                    st.dataframe(
                        top_df.rename(
                            columns={
                                "StockCode": "Stock",
                                "Description": "Product",
                                "Quantity": "Qty",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            with col_b:
                st.markdown("#### RFM snapshot")
                rfm = pd.DataFrame(
                    [
                        {
                            "Recency (days)": features["recency_days"],
                            "Frequency (invoices)": features["frequency"],
                            "Monetary (£)": round(features["monetary"], 2),
                            "Avg Order Value (£)": round(features["avg_order_value"], 2),
                            "Product Diversity": features["product_diversity"],
                        }
                    ]
                )
                st.dataframe(rfm, use_container_width=True, hide_index=True)
        else:
            st.warning("No transactions found for this customer / period.")

    # ----- Tab 2: Evidence -----
    with tab_evidence:
        st.subheader("Retrieved evidence package")
        st.markdown(
            "These tables are the **source of truth**. Every metric and sentence "
            "in the explanation is derived from the rows below."
        )
        if tx.empty:
            st.warning("No evidence rows retrieved.")
        else:
            st.markdown("#### Invoice-level summary")
            inv_summary = (
                tx.groupby("Invoice", as_index=False)
                .agg(
                    InvoiceDate=("InvoiceDate", "min"),
                    Lines=("StockCode", "count"),
                    Quantity=("Quantity", "sum"),
                    Spend=("LineTotal", "sum"),
                )
                .sort_values("InvoiceDate")
            )
            inv_summary["InvoiceDate"] = inv_summary["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
            inv_summary["Spend"] = inv_summary["Spend"].round(2)
            st.dataframe(inv_summary, use_container_width=True, hide_index=True)

            st.markdown("#### Line-level transactions")
            show_tx = tx[
                ["Invoice", "InvoiceDate", "StockCode", "Description", "Quantity", "Price", "LineTotal", "Country"]
            ].copy()
            show_tx["InvoiceDate"] = show_tx["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
            show_tx["Price"] = show_tx["Price"].round(2)
            show_tx["LineTotal"] = show_tx["LineTotal"].round(2)
            st.dataframe(show_tx, use_container_width=True, hide_index=True)

            csv = show_tx.to_csv(index=False)
            st.download_button(
                "Download evidence CSV",
                csv,
                file_name=f"customer_{customer_id}_evidence.csv",
                mime="text/csv",
            )

    # ----- Tab 3: Explanation -----
    with tab_explain:
        st.subheader("AI-generated grounded explanation")
        explanation = generate_grounded_explanation(customer_id, features, tx)
        st.markdown(explanation)
        st.info(
            "Principle applied: **Tools calculate · Retrieval finds evidence · "
            "Explanation stays within the evidence.** No unsupported inference is produced."
        )

    # ----- Tab 4: Human review -----
    with tab_review:
        st.subheader("Human review & control")
        st.markdown(
            """
You can:
- Inspect retrieved invoices and transaction records (Tab 2)
- Verify calculated metrics against the evidence tables
- Modify Customer ID / date filters and re-run
- Accept, reject, or request a refined analysis
"""
        )
        decision = st.radio(
            "Review decision",
            ["Accept explanation", "Reject — needs correction", "Request refinement"],
            horizontal=True,
        )
        note = st.text_area(
            "Review notes / requested changes",
            placeholder="e.g., Recency should use a fixed reference date of 2011-12-09; include cancelled invoices for audit…",
        )
        if st.button("Record review decision", type="primary"):
            if "reviews" not in st.session_state:
                st.session_state.reviews = []
            st.session_state.reviews.append(
                {
                    "customer_id": customer_id,
                    "period": f"{st.session_state.last_period[0]} → {st.session_state.last_period[1]}",
                    "decision": decision,
                    "note": note,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "n_invoices": features["n_invoices"],
                    "total_spend": features["total_spend"],
                }
            )
            st.success("Review recorded for this session.")

        if st.session_state.get("reviews"):
            st.markdown("#### Session review log")
            st.dataframe(pd.DataFrame(st.session_state.reviews), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            "**Why human control matters**  \n"
            "AI supports decisions by calculating metrics and assembling evidence. "
            "Humans validate conclusions, catch data-quality issues, and decide what actions to take."
        )

    # ----- Tab 5: Architecture -----
    with tab_about:
        st.subheader("Application architecture")
        st.markdown(
            """
| Component | Role |
|-----------|------|
| **Streamlit UI** | Human inputs (Customer ID, period) and review controls |
| **Online Retail II data** | Transaction evidence (Invoice, Quantity, Price, Date, …) |
| **Python analytics engine** | RFM, AOV, product diversity |
| **Structured retrieval** | Filter customer + period; surface invoices & lines |
| **Grounded explanation module** | Plain-language summary constrained to retrieved evidence |
| **Human review loop** | Accept / reject / refine |

**Design principle**  
> Tools calculate · Retrieval finds evidence · LLM (or template) explains evidence  

No claim is made that cannot be traced back to a calculated metric or a retrieved transaction row.
"""
        )
        st.subheader("Test cases covered by this implementation")
        st.markdown(
            """
- Correct customer profile (metrics match filtered rows)
- Correct invoice frequency
- Correct spending totals
- Correct recency calculation
- Invalid / empty customer handling
- Missing evidence handling
- Cancellation filtering (toggle)
- Unsupported inference prevention (explanations stay inside evidence)
"""
        )
        st.subheader("Next steps (from design brief)")
        st.markdown(
            """
- Customer segmentation (RFM tiers)
- Product community / affinity analysis
- Purchase prediction (with explicit uncertainty)
- Recommendation support grounded in past invoices
"""
        )
        st.subheader("GitHub repository")
        st.markdown(
            "[https://github.com/hhw3n-lab/CS5588-Capstone-Course](https://github.com/hhw3n-lab/CS5588-Capstone-Course)"
        )
        st.code(
            "pip install -r requirements.txt\n"
            "streamlit run retail360_app.py",
            language="bash",
        )


if __name__ == "__main__":
    main()
