# RU Liquidity Sentinel

End-to-end prototype for the PSB treasury brief: a liquidity-stress early warning system for the Russian rouble money market.

The project builds five independent market-signal modules, normalizes them with rolling MAD scores, aggregates them into an interpretable `Liquidity Stress Index` (`LSI`, 0-100), and exposes the result through a Streamlit dashboard with charts, alerts, backtests, sensitivity analysis, and an analyst chat.

## Repository Layout

```text
.
├── app.py                    # Streamlit dashboard
├── run_pipeline.py           # CLI smoke run for the full pipeline
├── requirements.txt          # Runtime/test dependencies
├── sentinel/
│   ├── data.py               # Source download hooks and calibrated demo dataset
│   ├── modules.py            # M1-M5 signal calculators
│   ├── aggregation.py        # Interpretable ML aggregation into LSI
│   ├── backtest.py           # Historical stress episode checks
│   ├── llm.py                # Hugging Face analyst chat + local fallback
│   ├── pipeline.py           # End-to-end orchestration
│   └── utils.py              # Shared helpers
└── tests/
    └── test_pipeline.py      # Pipeline regression test
```

## Prerequisites

- Python 3.11 or newer. The project has been verified on Python 3.13.
- Internet access if you want to refresh public-source caches or use the analyst LLM.
- A Hugging Face token with inference access for live chat.

## Fresh Installation

From the project directory:

```bash
cd /home/liguha/psb
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you do not want to use a virtual environment, install dependencies into your active Python environment:

```bash
python3 -m pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file in the project root:

```bash
HF_TOKEN=your_hugging_face_token
HF_MODEL=openai/gpt-oss-20b:cheapest
```

`HF_MODEL` is optional. The default model is `openai/gpt-oss-20b:cheapest`, routed through Hugging Face Inference Providers. The suffix `:cheapest` asks Hugging Face to route to the lowest-cost available provider for that model.

Do not commit `.env`. It is already listed in `.gitignore`.

## Build / Run

Run the full pipeline once from the command line:

```bash
python3 run_pipeline.py
```

Expected output:

- latest calculation date
- latest LSI and status
- automatic Russian-language market commentary
- backtest table for December 2014, February-March 2022, and August 2023

Start the dashboard:

```bash
python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://localhost:8501
```

If port `8501` is busy, choose another port:

```bash
python3 -m streamlit run app.py --server.port 8502
```

## Testing

Run the automated test:

```bash
python3 -m pytest -q
```

Run a Python compile check:

```bash
python3 -m compileall sentinel app.py run_pipeline.py
```

Optional live LLM check:

```bash
python3 - <<'PY'
from sentinel import build_liquidity_sentinel
from sentinel.llm import answer_question

result = build_liquidity_sentinel(refresh_sources=False)
print(answer_question("Кратко опиши текущий статус LSI.", result.aggregation.frame, result.backtest))
PY
```

If Hugging Face is unavailable, the application returns a local rule-based answer instead of failing.

## Data Refresh

The dashboard can download public-source pages and Excel files listed in the original PSB brief. Use the sidebar button:

```text
Обновить публичные источники
```

Downloaded files are cached in:

```text
data/cache/
```

The current prototype uses a deterministic calibrated dataset for the core calculations. This makes the demo reproducible and runnable offline while preserving production hooks for CBR, Minfin, FNS, Federal Treasury, and Moscow Exchange style sources.

## Main Dashboard Pages

- `Дашборд`: latest LSI, status, module contributions, current metrics, charts, and automatic commentary.
- `Модули`: detailed charts and recent rows for each module.
- `Backtest`: required historical stress checks and weight sensitivity analysis.
- `Аналитик`: natural-language analyst chat backed by Hugging Face with local fallback.
- `Источники`: public-source refresh status and cache paths.

## Notes

- LSI scale: `0-40` green, `40-70` yellow, `70-100` red.
- Quantitative module features are normalized with rolling three-year MAD scores.
- Aggregation uses `RidgeCV`, chosen because it is simple, stable, fast, and directly interpretable through coefficients and per-module contributions.
- The app is a prototype for analytical demonstration, not a production risk engine.
