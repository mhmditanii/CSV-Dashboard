# CSV Insights

Upload any CSV. Get instant charts and plain-English insights — no technical knowledge required.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/mhmditanii/CSV-Dashboard.git
cd 
pip install -r requirements.txt
```

### 2. Set up Ollama (optional but recommended)

Ollama runs the LLM locally — no API key, no cost, no data leaving your machine.

```bash
# Install Ollama: https://ollama.com/download
# Then pull a model:
ollama pull llama3
# Start the server (runs in background):
ollama serve
```

If Ollama is not running, the app still works — it falls back to rule-based insights automatically.

### 3. Run

```bash
python app.py
```

Open [http://localhost:8050](http://localhost:8050)

---

## Project Structure

```
csv-insights/
├── data/                       # Data Test
│   ├── nyc_airbnb.csv          
│   └── titanic.csv
│
├── src/
│   ├── app/
│   │   ├── app.py              # Dash app — layout, callbacks, UI assembly
│   │   └── css/
│   │       └── style.css       # Custom Dash CSS (auto-loaded by Dash)
│   │
│   ├── core/
│   │   ├── chart_gen.py        # Auto-selects and builds Plotly figures
│   │   ├── data_loader.py      # CSV ingestion via DuckDB, Parquet caching
│   │   ├── insight_engine.py   # Hybrid: deterministic bullets + Ollama narrative
│   │   ├── schema_detector.py  # Column semantic type detection (the core module) 
│   │   └── stats_engine.py     # Distributions, correlations, outlier detection
│   │
│   └── requirements.txt
│
└── README.md
```

---

## How It Works

```
CSV Upload
    │
    ▼
data_loader.py       DuckDB reads CSV → caches as Parquet
    │
    ▼
schema_detector.py   Classifies every column: numeric / categorical /
                     datetime / boolean / identifier / geographic / text
    │
    ▼
stats_engine.py      Distributions, correlations, outliers, null rates
    │
    ▼
chart_generator.py   Picks chart types based on semantic types,
                     builds Plotly figures automatically
    │
    ▼
insight_engine.py    Rule-based bullets (always) +
                     Ollama narrative (if available)
    │
    ▼
app.py               Renders dashboard in Dash
```

---

## Tested Datasets

| Dataset     | Rows   | Columns | Notes                          |
|-------------|--------|---------|--------------------------------|
| Titanic     | 891    | 12      | Mixed types, missing values    |
| NYC Airbnb  | 48,895 | 16      | Dates, geo coordinates, prices |

Download links:
- Titanic: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- Airbnb: https://raw.githubusercontent.com/erkansirin78/datasets/master/AB_NYC_2019.csv

---

## Design Decisions & Trade-offs

### Why DuckDB for CSV loading?
DuckDB's `read_csv_auto` is 3–5× faster than `pd.read_csv` on files > 10k rows, auto-detects delimiters/encoding, and can query Parquet directly. The Parquet cache means subsequent interactions never re-parse the CSV.

### Why not HDF5?
HDF5 requires PyTables and handles mixed string/numeric DataFrames poorly. Parquet is DuckDB-native, smaller, and column-oriented — a better fit for this workload.

### Why Ollama instead of OpenAI/Claude API?
Local LLM = no API cost, no latency on network, no data leaving the user's machine. The quality trade-off is documented and the app degrades gracefully if Ollama is unavailable.

### Why Dash over Streamlit?
Dash gives full layout control (CSS grid, custom components), proper callback architecture, and production-grade routing. Streamlit's full-script re-run model is awkward for large DataFrame state.

### What's cut (MVP scope)
- No authentication
- No persistent storage / database (Parquet cache is session-scoped)
- No async job queue (Celery/Redis) — large files block the callback
- No multi-file comparison
- No CSV validation / sanitisation (security concern for production)
- No cost controls / rate limiting on LLM

### What's next
- Background processing with Celery for files > 50MB
- User-driven chart customisation (filter by column, change chart type)
- Export dashboard as PDF / PNG
- Replace Ollama with pluggable LLM provider (Anthropic, OpenAI)
- Add authentication for multi-user deployments

---

## Running in Production

```bash
pip install gunicorn
gunicorn app:server -w 2 -b 0.0.0.0:8050
```

Note: set `debug=False` in `app.run(...)` for production.
