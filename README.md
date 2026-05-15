# 📈 MCP Trading Research Server

A **Model Context Protocol (MCP)** server that lets Claude run quantitative backtests against real market data — entirely through natural language. Write a strategy in plain English, and Claude translates it into DuckDB SQL, executes it against minute-level parquet data, and returns a full performance report with an equity curve chart.

Available in two configurations: a **local version** (reads parquet files from disk) and a **cloud version** (reads from Cloudflare R2, deployed on Google Cloud Run).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL SETUP                              │
│                                                                 │
│   Claude Desktop ──stdio──► trading_server_local.py            │
│                                     │                           │
│                              DuckDB reads                       │
│                              ./TestData/*.parquet               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD SETUP                              │
│                                                                 │
│   Claude Desktop ──stdio──► bridge.py ──HTTP/SSE──►            │
│                                     │                           │
│                          Google Cloud Run                       │
│                          trading_server.py (FastAPI + MCP)     │
│                                     │                           │
│                          DuckDB reads s3://quant-data-poc/      │
│                          (Cloudflare R2)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Overview

| File | Purpose |
|---|---|
| `trading_server_local.py` | MCP server for local use — reads parquet files from `DATA_DIR` |
| `trading_server.py` | MCP server for cloud deployment — reads parquet files from Cloudflare R2 |
| `bridge.py` | Stdio↔HTTP bridge connecting Claude Desktop to the cloud server via SSE |
| `Dockerfile` | Container definition for deploying the cloud server to Google Cloud Run |
| `requirements.txt` | All Python dependencies |
| `.env.example` | Template for environment variables — copy to `.env` and fill in secrets |

---

## Quickstart — Local

**1. Clone and install dependencies**
```bash
git clone https://github.com/yourname/mcp-trading-server.git
cd mcp-trading-server
pip install -r requirements.txt
```

**2. Set up environment**
```bash
cp .env.example .env
# Edit .env and set DATA_DIR to point to your parquet folder
```

**3. Place parquet files**

Drop your `.parquet` files into the `DATA_DIR` folder. Files can be named `TSLA.parquet`, `AAPL_2024.parquet`, etc. The server resolves symbol names automatically using regex.

**4. Register with Claude Desktop**

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "trading-local": {
      "command": "python",
      "args": ["/absolute/path/to/trading_server_local.py"],
      "env": {
        "DATA_DIR": "/absolute/path/to/TestData"
      }
    }
  }
}
```

**5. Run**

Restart Claude Desktop — the server starts automatically. Ask Claude:
> *"List the available symbols and run a simple moving average crossover backtest on TSLA."*

---

## Quickstart — Cloud

**1. Configure secrets**
```bash
cp .env.example .env
# Fill in R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, SERVER_API_KEY
```

**2. Build and deploy**
```bash
gcloud run deploy mcp-trading-server \
  --source . \
  --region us-east1 \
  --set-env-vars R2_ACCESS_KEY_ID=...,R2_SECRET_ACCESS_KEY=...,R2_ENDPOINT=...,SERVER_API_KEY=...
```

**3. Configure bridge**

Edit `.env`:
```
MCP_API_KEY=your_server_api_key
MCP_SERVER_URL=https://your-service-name.run.app
```

**4. Register bridge with Claude Desktop**
```json
{
  "mcpServers": {
    "trading-cloud": {
      "command": "python",
      "args": ["/absolute/path/to/bridge.py"]
    }
  }
}
```

---

## MCP Tools

### `list_data_files`
Lists all available symbols in the data source.

### `inspect_columns(symbol)`
Returns the schema and a 5-row sample for a given symbol. Useful for exploring data before writing a strategy.

### `execute_research_query(sql_query, ...)`
The core backtest engine. Accepts a DuckDB SQL query and optional portfolio parameters.

| Parameter | Default | Description |
|---|---|---|
| `sql_query` | — | DuckDB SQL implementing the strategy |
| `starting_balance` | 25000 | Initial capital in USD |
| `fee_roundtrip` | 0.0015 | Round-trip transaction cost (e.g. 0.0015 = 15 bps) |
| `fee_fixed` | 0.0 | Fixed USD cost per round-trip |
| `position_size_type` | `"fixed"` | `"fixed"` (USD amount) or `"percent"` (% of balance) |
| `position_size_value` | 25000 | Size amount matching `position_size_type` |

**Output (when `trade_ret` column is present):**
- Full JSON performance report (Sharpe, Sortino, Max DD, Win Rate, etc.)
- Equity curve chart as a base64-encoded PNG

### `get_storage_stats`
Returns a count of symbols and files in the data source.

---

## SQL Guide for Claude

The raw data is **minutely OHLCV**. Always resample to your target timeframe first.

**Columns:** `datetime`, `open`, `high`, `low`, `close`, `turnover`, `tradeable`

> ⚠️ `datetime`, `open`, `high`, `low`, `close` are reserved words in DuckDB. Always wrap them in double quotes.

### Minimal daily-bar CTE
```sql
WITH daily_bars AS (
    SELECT
        DATE_TRUNC('day', "datetime") AS trade_date,
        arg_min("open", "datetime")   AS open_p,
        MAX("high")                   AS high_p,
        MIN("low")                    AS low_p,
        arg_max("close", "datetime")  AS close_p,
        SUM(turnover)                 AS turnover
    FROM 'TSLA.parquet'
    WHERE tradeable = 1
      AND (EXTRACT(HOUR FROM "datetime") < 16
        OR (EXTRACT(HOUR FROM "datetime") = 16
           AND EXTRACT(MINUTE FROM "datetime") = 0))
    GROUP BY DATE_TRUNC('day', "datetime")
)
```

### Required output columns
The performance engine requires exactly these two column names in the final `SELECT`:
- `trade_date AS datetime`
- `trade_ret` — per-bar return, **NULL when flat** (never `0`)

---

## Performance Engine

`calculate_quant_metrics()` handles:

- **Fee modeling** — charges `fee_roundtrip` only on entry and exit bars (not every bar held), preventing fee over-counting on trend strategies
- **Position sizing** — fixed USD or percent-of-equity compounding
- **QuantStats metrics** — Sharpe, Sortino, Max Drawdown via the `quantstats` library
- **Drawdown recovery** — calculates max time to recover from peak
- **Streak analysis** — max consecutive wins and losses
- **Equity curve chart** — dark-background PNG, downsampled to 1000 points for large datasets

---

## Security Notes

- All secrets are loaded from environment variables — never hardcoded
- The cloud server validates the `x-api-key` header on all non-SSE endpoints
- The SSE and `/messages` endpoints are intentionally public (required for MCP protocol handshake)
- Generate a strong `SERVER_API_KEY` with: `python -c "import secrets; print(secrets.token_hex(32))"`
- Add `.env` to `.gitignore` — never commit it

---

## Project Structure

```
mcp-trading-server/
├── trading_server_local.py   # Local MCP server
├── trading_server.py         # Cloud MCP server (FastAPI + R2)
├── bridge.py                 # Stdio↔HTTP bridge for cloud mode
├── Dockerfile                # Cloud Run container definition
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Must include .env and TestData/
└── README.md
```

---

## Tech Stack

- **[MCP (Model Context Protocol)](https://modelcontextprotocol.io)** — Claude tool interface
- **[DuckDB](https://duckdb.org)** — in-process SQL engine for parquet analytics
- **[FastAPI](https://fastapi.tiangolo.com)** — async web framework (cloud version)
- **[Cloudflare R2](https://developers.cloudflare.com/r2/)** — S3-compatible object storage
- **[Google Cloud Run](https://cloud.google.com/run)** — serverless container hosting
- **[QuantStats](https://github.com/ranaroussi/quantstats)** — portfolio analytics
- **[Matplotlib](https://matplotlib.org)** — equity curve charting

---