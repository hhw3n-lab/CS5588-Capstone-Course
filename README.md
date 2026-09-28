# Data

## Sample (in-app)

`retail360_app.py` embeds a small synthetic sample matching the Online Retail II schema, including demo customer **13085**. No extra download is required for the basic Streamlit demo.

## Full benchmark (recommended for experiments)

**UCI Online Retail II** — Daqing Chen  
DOI: [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)

1. Download the CSV from UCI or Kaggle.
2. Place it here as `online_retail_II.csv` (or pass `--data path/to/file.csv` to the experiment script).
3. **Do not commit** the full raw file if it is large; keep download instructions only.

### Column expectations

`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

### Protocol reminder

Features must be computed only on transactions **before** the prediction cutoff date to avoid leakage.
