# Insights — Dash + Ollama CSV Dashboard MVP

A Dash MVP for the case study: upload any CSV and generate a non-technical dashboard with useful charts, tables, and plain-English insights.

## What changed in this version

This version intentionally uses Ollama much more heavily, but keeps the full CSV local.

Flow:

1. User uploads CSV.
2. The app loads the file locally.
3. Python profiles columns and computes candidate insight views using NumPy/Pandas.
4. Ollama receives only:
   - column profiles
   - valid candidate chart/table ideas
   - 2 sample rows
5. Ollama chooses the best 8–12 cards, explains why those features belong together, and rewrites:
   - dashboard summary
   - card titles
   - x-axis/y-axis labels
   - table labels
   - takeaways
6. Python renders and calculates everything locally.

The full CSV is never sent to Ollama.

## Run

```bash
pip install -r requirements.txt
ollama pull llama3
python app.py
```

Optional:

```bash
export OLLAMA_MODEL="llama3.1"
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Why Dash

Dash is used because it is straightforward for a product-style upload workflow, supports interactive Plotly charts, and keeps the MVP easy to demo locally.

## Design choices

- **Ollama is the planner/polisher**, not the calculator.
- **Python owns correctness**: data loading, validation, filtering, calculations, and rendering.
- **Candidate-based planning** prevents Ollama from inventing bad column combinations.
- **Tables and charts are both first-class** because business users often understand ranked summaries faster than dense charts.
- **No raw scatter plots** for large datasets. Numeric relationships are shown as binned trends or tables.

## Failure mode

If Ollama is unavailable, the app falls back to deterministic selection and wording.

## Known limitations

- CSV parsing uses Pandas for reliability, while profiling and scoring rely heavily on NumPy arrays.
- Ollama quality depends on the local model. `llama3.1` or a larger model generally gives better wording than small local models.
- This is still an MVP, not a BI replacement.

## Suggested demo datasets

- Titanic CSV
- NYC Airbnb 2019 CSV
- Any business CSV with dates, categories, and numeric measures
