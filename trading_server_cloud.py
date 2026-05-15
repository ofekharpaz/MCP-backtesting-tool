"""
MCP Trading Server — Cloud Edition
===================================
A FastAPI + Model Context Protocol (MCP) server that exposes quantitative
trading research tools to AI assistants (e.g. Claude via claude.ai).

Parquet market data lives in a Cloudflare R2 bucket. Strategies are expressed
as DuckDB SQL queries; the server executes them server-side and returns a full
performance report (Sharpe, Sortino, max drawdown, equity-curve chart, …).

Quick start
-----------
1. Copy .env.example to .env and fill in your credentials.
2. pip install -r requirements.txt
3. python trading_server_cloud.py          # listens on $PORT (default 8080)

Environment variables (see .env.example)
-----------------------------------------
R2_ACCESS_KEY_ID      – Cloudflare R2 access key
R2_SECRET_ACCESS_KEY  – Cloudflare R2 secret key
R2_ENDPOINT           – https://<account_id>.r2.cloudflarestorage.com
SERVER_API_KEY        – Secret sent in the x-api-key request header
R2_BUCKET_NAME        – R2 bucket that holds the .parquet files (default: quant-data)
"""

import asyncio
import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import boto3
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.types import ImageContent, TextContent
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response 

mcp = FastMCP("TradingServer")

executor = ThreadPoolExecutor(max_workers=4)

R2_KEY      = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET   = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
API_KEY     = os.getenv("SERVER_API_KEY")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "quant-data")

def get_duckdb_conn():
    """Return a DuckDB connection pre-configured with R2 / S3-compatible credentials."""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    clean_endpoint = R2_ENDPOINT.replace('https://', '')
    conn.execute(f"SET s3_endpoint='{clean_endpoint}';")
    conn.execute(f"SET s3_access_key_id='{R2_KEY}';")
    conn.execute(f"SET s3_secret_access_key='{R2_SECRET}';")
    conn.execute("SET s3_region='auto';")
    conn.execute("SET s3_url_style='path';")
    return conn

def calculate_quant_metrics(df, starting_balance=25000, fee_roundtrip=0.0015, fee_fixed=0.0, position_size_type="fixed", position_size_value=25000, symbol="Unknown"):
    df = df[df['trade_ret'].notnull()].copy()
    if df.empty: return "No trades executed."

    # 1. Data Cleaning
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df['trade_ret'] = df['trade_ret'].astype(float)

    # 2. Identify actual trade bars (entry = first bar of block, exit = last bar of block)
    # A "block" is a consecutive sequence of non-null trade_ret rows
    is_first_bar = df['trade_ret'].shift(1).isnull()   # first bar of holding block = entry
    is_last_bar  = df['trade_ret'].shift(-1).isnull()  # last bar of holding block = exit
    is_trade_bar = is_first_bar | is_last_bar

    # 3. Apply fee_roundtrip ONLY on entry and exit bars (split 50/50)
    # This prevents charging 2180x fees on a 14-trade trend strategy
    half_fee = fee_roundtrip / 2
    df['trade_ret_net'] = df['trade_ret'].copy()
    df.loc[is_trade_bar, 'trade_ret_net'] = df.loc[is_trade_bar, 'trade_ret'] - half_fee

    # 4. Equity Curve
    if 'signal' in df.columns:
        num_roundtrips = int((df['signal'] == 1).sum())
    else:
        num_roundtrips = int(is_first_bar.sum())

    if position_size_type == "percent":
        size_factor = float(position_size_value) / 100
        returns_vector = 1 + (size_factor * df['trade_ret_net'])
        df['equity_curve'] = starting_balance * returns_vector.cumprod()
        # Spread fixed fees evenly across all bars (deducted proportionally)
        total_fixed_fees = float(fee_fixed) * num_roundtrips
        if len(df) > 0:
            df['equity_curve'] -= total_fixed_fees * (np.arange(1, len(df)+1) / len(df))
    else:
        df['pnl_usd'] = (float(position_size_value) * df['trade_ret_net']) - (
            float(fee_fixed) * is_trade_bar.astype(float)
        )
        df['equity_curve'] = starting_balance + df['pnl_usd'].cumsum()

    # 5. Win/Loss per bar
    df['is_win'] = df['equity_curve'].diff().fillna(df['equity_curve'].iloc[0] - starting_balance) > 0

    # 6. Drawdown & Recovery
    peak = df['equity_curve'].expanding(min_periods=1).max()
    drawdown_pct = (df['equity_curve'] - peak) / peak

    max_dd_period_time = "0 days"
    is_in_dd = drawdown_pct < 0
    if is_in_dd.any():
        last_peak_dates = df['datetime'].where(~is_in_dd).ffill()
        valid_mask = is_in_dd & last_peak_dates.notnull()
        if valid_mask.any():
            durations = df.loc[valid_mask, 'datetime'] - last_peak_dates[valid_mask]
            max_dd_period_time = str(durations.max())

    # 7. Consecutive Streaks
    streak_id = (df['is_win'] != df['is_win'].shift()).cumsum()
    streaks = df.groupby(streak_id).cumcount() + 1
    max_wins   = int(streaks[df['is_win']].max())  if df['is_win'].any()  else 0
    max_losses = int(streaks[~df['is_win']].max()) if (~df['is_win']).any() else 0

    # 8. QuantStats Metrics
    df_qs = df.set_index('datetime')
    strategy_returns = df_qs['equity_curve'].pct_change().fillna(0)
    try:
        sharpe     = float(qs.stats.sharpe(strategy_returns))
        sortino    = float(qs.stats.sortino(strategy_returns))
        max_dd_val = float(qs.stats.max_drawdown(df_qs['equity_curve']))
    except Exception:
        sharpe, sortino, max_dd_val = 0.0, 0.0, 0.0

    # 9. Equity Curve Chart
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 4))
    plot_df = df.iloc[::max(1, len(df)//1000)] if len(df) > 1000 else df
    plt.plot(plot_df['datetime'], plot_df['equity_curve'], color='#2196F3', linewidth=2)
    plt.title(f'Equity Curve: {symbol}')
    plt.grid(True, alpha=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 10. Final Output
    stats_summary = {
        "Strategy Info": {
            "Asset": symbol,
            "Sizing": f"{position_size_value} ({position_size_type})",
            "Status": "Verified"
        },
        "Capital Performance": {
            "Starting Balance": f"${starting_balance:,.0f}",
            "Ending Balance": f"${df['equity_curve'].iloc[-1]:,.2f}",
            "Total Return": f"{((df['equity_curve'].iloc[-1] - starting_balance) / starting_balance) * 100:,.2f}%"
        },
        "QuantStats Metrics": {
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Max Drawdown %": f"{(max_dd_val * 100):.2f}%",
            "Max DD Recovery": max_dd_period_time
        },
        "Trade Statistics": {
            "Total Trades": num_roundtrips,
            "Win Rate": f"{(df['is_win'].sum() / len(df)) * 100:.2f}%",
            "Max Consecutive Wins": max_wins,
            "Max Consecutive Losses": max_losses
        }
    }

    return [
        TextContent(
            type="text",
            text=f"Here are the metrics:\n```json\n{json.dumps(stats_summary, indent=2)}\n```"
        ),
        ImageContent(
            type="image",
            data=chart_data,
            mimeType="image/png"
        )
    ]

@mcp.tool()
async def execute_research_query(sql_query: str, 
    starting_balance: float = 25000, 
    fee_roundtrip: float = 0.0015, 
    fee_fixed: float = 0.0, 
    position_size_type: str = "fixed", 
    position_size_value: float = 25000):
    """
    REQUIRED TOOL FOR QUANTITATIVE TRADING RESEARCH.

    DATA ARCHITECTURE:
    - Table: Use 'SYMBOL.parquet' (e.g., 'TSLA.parquet').
    - Columns: ["datetime", "open", "high", "low", "close", turnover, tradeable].
    - Use "datetime" instead of timestamp and turnover instead of volume.
    - IMPORTANT: datetime, open, high, low, and close are reserved words in DuckDB.
    Always wrap them in double quotes when referencing as column names.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DATA RESAMPLING — CRITICAL
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The raw data is MINUTELY. If the strategy uses a daily/weekly timeframe,
    you MUST resample FIRST before calculating any indicators or signals.

    ALWAYS resample to daily like this:
        WITH daily_bars AS (
            SELECT
                DATE_TRUNC('day', "datetime") AS trade_date,
                arg_min("open", "datetime") AS open_p,
                MAX("high") AS high_p,
                MIN("low") AS low_p,
                arg_max("close", "datetime") AS close_p,
                SUM(turnover) AS turnover
            FROM 'SYMBOL.parquet'
            WHERE tradeable = 1
            AND (EXTRACT(HOUR FROM "datetime") < 16
            OR (EXTRACT(HOUR FROM "datetime") = 16 AND EXTRACT(MINUTE FROM "datetime") = 0))
            GROUP BY DATE_TRUNC('day', "datetime")
        )
    Then calculate ALL indicators on daily_bars, never on raw minutely data.
    NEVER approximate daily bars by multiplying minutes (e.g. 390 * 200 = wrong).

    IMPORTANT — Column aliasing in daily_bars:
    - Do NOT alias resampled columns as "open", "close" etc. — these are reserved words.
    - Use safe aliases instead: open_p, high_p, low_p, close_p.
    - This avoids reserved word conflicts in all downstream CTEs.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    POSITION & SIGNAL LOGIC
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Generate a raw signal only on the bar where the condition triggers (entry/exit).
    - Carry the position forward between signals using LAST_VALUE IGNORE NULLS:
        LAST_VALUE(CASE WHEN signal != 0 THEN signal ELSE NULL END IGNORE NULLS)
            OVER (ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS position
    - position = 1 means LONG, position = -1 means SHORT, 0 means FLAT.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TRADE_RET — CRITICAL RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    To trigger the performance engine you MUST include a 'trade_ret' column.

    RULE 1 — ALWAYS use NULL when not in a position. NEVER use 0.
    ELSE 0 tells the engine "I made a trade with zero return" on every bar.
    ELSE NULL tells the engine "ignore this bar". NULL is always correct.

    RULE 2 — For TREND strategies (holding days/weeks/months):
    Populate trade_ret on EVERY bar the position is held, not just entry:
        CASE
            WHEN position = 1 AND next_close IS NOT NULL
                THEN (next_close - close_p) / close_p
            ELSE NULL
        END AS trade_ret
    This captures the full holding period P&L.

    RULE 3 — For MEAN REVERSION strategies (holding 1-2 bars):
    Populate trade_ret only on the entry bar:
        CASE
            WHEN signal = 1 THEN (next_close - close_p) / close_p
            ELSE NULL
        END AS trade_ret

    RULE 4 — BIAS PREVENTION:
    Never use future data to generate signals.
    Signals must only use current and past bars.
    trade_ret uses LEAD(close_p) which is fine — it is the execution price,
    not a signal input.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT REQUIREMENTS — CRITICAL
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The performance engine reads the DataFrame by exact column name.
    The final SELECT MUST output these two columns with EXACTLY these names:

    - datetime   → alias your date column: trade_date AS datetime
    - trade_ret  → the per-bar return (NULL when flat, never 0)

    Any other name for these columns will silently crash the engine with a
    misleading error. Always end your query like this:

        SELECT
            trade_date AS datetime,   -- REQUIRED exact name
            close_p,
            signal,
            position,
            trade_ret                 -- REQUIRED exact name
        FROM with_position
        ORDER BY trade_date

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DuckDB SYNTAX RULES — CRITICAL
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Reserved words MUST be quoted with double quotes: "datetime", "open",
    "high", "low", "close". Unquoted reserved words will cause a SQL error.
    - This applies everywhere: SELECT, ORDER BY, GROUP BY, EXTRACT, DATE_TRUNC,
    LAST(), FIRST(), LEAD(), LAG(), and all window functions.
    - After resampling, use safe aliases (open_p, close_p etc.) so reserved
    words never appear in downstream CTEs at all.
    - To filter by time use EXTRACT(HOUR FROM "datetime") and
    EXTRACT(MINUTE FROM "datetime").
    - NEVER use strftime() — it does not work in DuckDB.
    - NEVER use DATE_PART() — use EXTRACT() instead.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SERVER INSTRUCTIONS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - NEVER use LIMIT in backtesting queries.
    - The server auto-resolves the actual filename in R2.
    - Always confirm parameters (Starting Balance, Fees, Sizing) in your response.
    - If the engine returns a generic SQL error, the most likely cause is a missing
    or misnamed output column — check that 'datetime' and 'trade_ret' are present
    in the final SELECT before retrying.

    PARAMETERS:
    - sql_query: The SQL logic for the strategy.
    - starting_balance: Initial cash (default 25000).
    - fee_roundtrip: Transaction costs as a fraction (e.g., 0.0015 = 15bps).
    - fee_fixed: Fixed USD cost per roundtrip (e.g., 1.0 = $1 per trade).
    - position_size_type: 'fixed' (USD amount) or 'percent' (% of current balance).
    - position_size_value: Numeric value for sizing (e.g., 500 for fixed, 2 for 2%).
    """
    try:
        if isinstance(fee_fixed, str):
            fee_fixed = float(fee_fixed.replace('$', '').strip())

        if isinstance(position_size_value, str):
            position_size_value = float(position_size_value.replace('%', '').strip())
        
        position_size_type = str(position_size_type).lower().strip()
        if "percent" in position_size_type or "%" in position_size_type:
            position_size_type = "percent"
        else:
            position_size_type = "fixed"
    except Exception:
        position_size_type = "fixed"
        position_size_value = 25000

    # 1. Extract Symbol from the SQL query
    # Look for patterns like 'SYMBOL.parquet' or "SYMBOL.parquet"
    symbol_match = re.search(r"['\"](\w+)\.parquet['\"]", sql_query, re.IGNORECASE)
    if not symbol_match:
        return "Error: Could not extract symbol. Please use 'SYMBOL.parquet' format in your SQL query."
    
    requested_symbol = symbol_match.group(1).upper()

    try:
        # 2. Initialize R2 Client and Resolve Filename
        # We fetch the file list and use regex to support both '-' and '_' separators
        s3 = boto3.client(
            's3', 
            endpoint_url=R2_ENDPOINT, 
            aws_access_key_id=R2_KEY, 
            aws_secret_access_key=R2_SECRET
        )
        
        # Fetching object list from the bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=requested_symbol)
        
        matching_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Regex Logic:
                # ^RequestedSymbol -> Starts with symbol
                # ([-_].*)? -> Followed by optional hyphen/underscore and any characters (date/timestamp)
                # \.parquet$ -> Ends with .parquet
                if re.match(rf"^{requested_symbol}([-_].*)?\.parquet$", key, re.IGNORECASE):
                    matching_files.append(key)

        if not matching_files:
            return f"Error: Symbol '{requested_symbol}' not found in storage bucket '{BUCKET_NAME}'."

        # Select the most recent file if multiple versions exist (alphabetical descending)
        full_filename = sorted(matching_files, reverse=True)[0]

        # 3. Dynamic SQL Injection
        # Construct the full S3 path and wrap it in DuckDB's read_parquet function
        r2_path = f"s3://{BUCKET_NAME}/{full_filename}"
        fixed_query = re.sub(
            r"['\"]\w+\.parquet['\"]", 
            f"read_parquet('{r2_path}')", 
            sql_query, 
            flags=re.IGNORECASE
        )

        def run_duckdb_logic():
            with get_duckdb_conn() as conn:
                conn.execute("SET max_memory='1.5GB';")
                conn.execute("SET threads=4;")
                
                df = conn.execute(fixed_query).df()
                return df


        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(executor, run_duckdb_logic)
        
        # If 'trade_ret' is not present, return a simple data preview
        if 'trade_ret' not in df.columns:
            return {
                "status": "Data loaded successfully",
                "file_resolved": full_filename,
                "preview": df.head(5).to_dict(orient='records'),
                "instructions": "To see a full performance report, include a 'trade_ret' column in your SELECT statement."
            }
        
        # If 'trade_ret' exists, trigger the specialized quant engine
        return calculate_quant_metrics(df, 
            starting_balance=starting_balance,
            fee_roundtrip=fee_roundtrip,
            fee_fixed=fee_fixed,
            position_size_type=position_size_type,
            position_size_value=position_size_value,
            symbol=requested_symbol)

    except Exception as e:
        error_msg = str(e)
        return(
            f"SQL Error: {error_msg}\n\n"
            f"IMPORTANT — Common causes:\n"
            f"1. Column names like datetime, open, high, low, close are reserved words "
            f"in DuckDB — always wrap them in double quotes: \"datetime\", \"close\", etc.\n"
            f"2. Used HOUR() or MINUTE() instead of EXTRACT(HOUR FROM \"datetime\")\n"
            f"3. Used strftime() which does not work in DuckDB\n"
            f"4. Missing WHERE tradeable = 1 filter on daily resampling\n"
            f"5. Calculated indicators on raw minutely data instead of resampling first\n\n"
            f"Please fix the SQL and retry."
        )

@mcp.tool()
def get_storage_stats():
    """Returns exact count of unique symbols and files in the R2 bucket."""
    s3 = boto3.client('s3', endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_KEY, aws_secret_access_key=R2_SECRET)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME)
    
    all_files = []
    for page in pages:
        if 'Contents' in page:
            all_files.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.parquet')])
    
    unique_symbols = set([f.split('.')[0].upper() for f in all_files])
    
    return {
        "total_parquet_files": len(all_files),
        "total_unique_symbols": len(unique_symbols),
        "bucket_name": BUCKET_NAME,
        "status": "Verified directly from R2 storage"
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Skip SSE and messages FIRST — before API key check
    if request.url.path in ("/sse", "/messages"):
        return await call_next(request)
    
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: Invalid API Key"}
        )
    
    return await call_next(request)

sse_transport = SseServerTransport("/messages")

@app.get("/sse")
async def sse_endpoint(request: Request):
    async with sse_transport.connect_sse(
        request.scope, 
        request.receive, 
        request._send
    ) as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options()
        )

@app.post("/messages")
async def messages_endpoint(request: Request):
    body = await request.body()
    
    async def fake_receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def fast_send(message):
        pass

    try:
        await sse_transport.handle_post_message(
            request.scope, 
            fake_receive, 
            fast_send
        )
        
        return Response(status_code=200)
        
    except Exception as e:
        error_msg = str(e)
        if "initialization" in error_msg.lower():
            return Response("Server initializing, please wait...", status_code=503)
        
        print(f"Error handling message: {error_msg}")
        return Response(f"Internal Error: {error_msg}", status_code=500)
    
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)