import os, duckdb, json, io, base64, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

mcp = FastMCP("TradingServer-Local")

DATA_DIR = os.getenv("DATA_DIR", "./TestData")


# ──────────────────────────────────────────────
#  Internal Helpers
# ──────────────────────────────────────────────

def get_parquet_path(symbol: str) -> str | None:
    """Resolve a symbol name to its parquet file path, supporting wildcard suffixes."""
    if not os.path.exists(DATA_DIR):
        return None
    for f in os.listdir(DATA_DIR):
        if re.match(rf"^{re.escape(symbol)}([-_].*)?\\.parquet$", f, re.IGNORECASE):
            return os.path.join(DATA_DIR, f).replace("\\", "/")
    return None


def calculate_quant_metrics(
    df: pd.DataFrame,
    starting_balance: float = 25000,
    fee_roundtrip: float = 0.0015,
    fee_fixed: float = 0.0,
    position_size_type: str = "fixed",
    position_size_value: float = 25000,
    symbol: str = "Unknown",
):
    """
    Full quant performance engine.
    Expects a DataFrame with columns: datetime, trade_ret.
    trade_ret must be NULL (NaN) when flat — never 0.
    """
    df = df[df["trade_ret"].notnull()].copy()
    if df.empty:
        return "No trades executed."

    # 1. Data Cleaning
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df["trade_ret"] = df["trade_ret"].astype(float)

    # 2. Identify entry/exit bars of each holding block
    is_first_bar = df["trade_ret"].shift(1).isnull()
    is_last_bar  = df["trade_ret"].shift(-1).isnull()
    is_trade_bar = is_first_bar | is_last_bar

    # 3. Apply fee_roundtrip only on entry and exit bars (split 50/50)
    half_fee = fee_roundtrip / 2
    df["trade_ret_net"] = df["trade_ret"].copy()
    df.loc[is_trade_bar, "trade_ret_net"] = df.loc[is_trade_bar, "trade_ret"] - half_fee

    # 4. Trade count
    if "signal" in df.columns:
        num_roundtrips = int((df["signal"] == 1).sum())
    else:
        num_roundtrips = int(is_first_bar.sum())

    # 5. Equity Curve
    if position_size_type == "percent":
        size_factor = float(position_size_value) / 100
        returns_vector = 1 + (size_factor * df["trade_ret_net"])
        df["equity_curve"] = starting_balance * returns_vector.cumprod()
        total_fixed_fees = float(fee_fixed) * num_roundtrips
        if len(df) > 0:
            df["equity_curve"] -= total_fixed_fees * (np.arange(1, len(df) + 1) / len(df))
    else:
        df["pnl_usd"] = (float(position_size_value) * df["trade_ret_net"]) - (
            float(fee_fixed) * is_trade_bar.astype(float)
        )
        df["equity_curve"] = starting_balance + df["pnl_usd"].cumsum()

    # 6. Win/Loss per bar
    df["is_win"] = (
        df["equity_curve"].diff().fillna(df["equity_curve"].iloc[0] - starting_balance) > 0
    )

    # 7. Drawdown & Recovery
    peak = df["equity_curve"].expanding(min_periods=1).max()
    drawdown_pct = (df["equity_curve"] - peak) / peak

    max_dd_period_time = "0 days"
    is_in_dd = drawdown_pct < 0
    if is_in_dd.any():
        last_peak_dates = df["datetime"].where(~is_in_dd).ffill()
        valid_mask = is_in_dd & last_peak_dates.notnull()
        if valid_mask.any():
            durations = df.loc[valid_mask, "datetime"] - last_peak_dates[valid_mask]
            max_dd_period_time = str(durations.max())

    # 8. Consecutive Streaks
    streak_id = (df["is_win"] != df["is_win"].shift()).cumsum()
    streaks = df.groupby(streak_id).cumcount() + 1
    max_wins   = int(streaks[df["is_win"]].max())  if df["is_win"].any()  else 0
    max_losses = int(streaks[~df["is_win"]].max()) if (~df["is_win"]).any() else 0

    # 9. QuantStats Metrics
    df_qs = df.set_index("datetime")
    strategy_returns = df_qs["equity_curve"].pct_change().fillna(0)
    try:
        sharpe     = float(qs.stats.sharpe(strategy_returns))
        sortino    = float(qs.stats.sortino(strategy_returns))
        max_dd_val = float(qs.stats.max_drawdown(df_qs["equity_curve"]))
    except Exception:
        sharpe, sortino, max_dd_val = 0.0, 0.0, 0.0

    # 10. Equity Curve Chart
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 4))
    plot_df = df.iloc[:: max(1, len(df) // 1000)] if len(df) > 1000 else df
    plt.plot(plot_df["datetime"], plot_df["equity_curve"], color="#2196F3", linewidth=2)
    plt.title(f"Equity Curve: {symbol}")
    plt.grid(True, alpha=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    chart_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # 11. Final Output
    stats_summary = {
        "Strategy Info": {
            "Asset": symbol,
            "Sizing": f"{position_size_value} ({position_size_type})",
            "Status": "Verified",
        },
        "Capital Performance": {
            "Starting Balance": f"${starting_balance:,.0f}",
            "Ending Balance": f"${df['equity_curve'].iloc[-1]:,.2f}",
            "Total Return": f"{((df['equity_curve'].iloc[-1] - starting_balance) / starting_balance) * 100:,.2f}%",
        },
        "QuantStats Metrics": {
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Max Drawdown %": f"{(max_dd_val * 100):.2f}%",
            "Max DD Recovery": max_dd_period_time,
        },
        "Trade Statistics": {
            "Total Trades": num_roundtrips,
            "Win Rate": f"{(df['is_win'].sum() / len(df)) * 100:.2f}%",
            "Max Consecutive Wins": max_wins,
            "Max Consecutive Losses": max_losses,
        },
    }

    return [
        TextContent(
            type="text",
            text=f"Here are the metrics:\n```json\n{json.dumps(stats_summary, indent=2)}\n```",
        ),
        ImageContent(
            type="image",
            data=chart_data,
            mimeType="image/png",
        ),
    ]


# ──────────────────────────────────────────────
#  MCP Tools
# ──────────────────────────────────────────────

@mcp.tool()
def list_data_files():
    """Lists all available symbols found in the local data directory."""
    if not os.path.exists(DATA_DIR):
        return f"Error: DATA_DIR '{DATA_DIR}' not found. Set the DATA_DIR environment variable."
    symbols = [
        re.sub(r"([-_].+)?\.parquet$", "", f, flags=re.IGNORECASE)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet")
    ]
    return sorted(set(symbols))


@mcp.tool()
def inspect_columns(symbol: str):
    """Returns the schema and first 5 rows for a given symbol."""
    path = get_parquet_path(symbol)
    if not path:
        return f"Error: Symbol '{symbol}' not found in '{DATA_DIR}'."

    with duckdb.connect() as conn:
        try:
            schema = conn.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 1"
            ).df()
            sample = conn.execute(
                f"SELECT * FROM read_parquet('{path}') LIMIT 5"
            ).df()
            return {
                "columns": schema[["column_name", "column_type"]].to_dict(orient="records"),
                "sample_data": sample.to_dict(orient="records"),
            }
        except Exception as e:
            return f"Error: {str(e)}"


@mcp.tool()
def execute_research_query(
    sql_query: str,
    starting_balance: float = 25000,
    fee_roundtrip: float = 0.0015,
    fee_fixed: float = 0.0,
    position_size_type: str = "fixed",
    position_size_value: float = 25000,
):
    """
    REQUIRED TOOL FOR QUANTITATIVE TRADING RESEARCH (LOCAL VERSION).

    DATA ARCHITECTURE:
    - Table reference: Use 'SYMBOL.parquet' (e.g., 'TSLA.parquet').
    - Columns: ["datetime", "open", "high", "low", "close", "turnover", "tradeable"].
    - datetime, open, high, low, close are reserved words in DuckDB.
      Always wrap them in double quotes: "datetime", "close", etc.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DATA RESAMPLING — CRITICAL
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Raw data is MINUTELY. Resample to daily before any indicator calculation:

        WITH daily_bars AS (
            SELECT
                DATE_TRUNC('day', "datetime") AS trade_date,
                arg_min("open", "datetime")   AS open_p,
                MAX("high")                   AS high_p,
                MIN("low")                    AS low_p,
                arg_max("close", "datetime")  AS close_p,
                SUM(turnover)                 AS turnover
            FROM 'SYMBOL.parquet'
            WHERE tradeable = 1
            AND (EXTRACT(HOUR FROM "datetime") < 16
            OR  (EXTRACT(HOUR FROM "datetime") = 16
                 AND EXTRACT(MINUTE FROM "datetime") = 0))
            GROUP BY DATE_TRUNC('day', "datetime")
        )

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    POSITION & SIGNAL LOGIC
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Carry position forward with LAST_VALUE IGNORE NULLS.
    - position = 1 LONG | -1 SHORT | 0 FLAT.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TRADE_RET — CRITICAL RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Use NULL when flat. NEVER use 0.
    - Trend strategies: populate on every bar held.
    - Mean reversion: populate on entry bar only.
    - Never use future data for signals (LEAD is fine for trade_ret only).

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT REQUIREMENTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Final SELECT MUST include:
        trade_date AS datetime   -- exact name required
        trade_ret                -- exact name required (NULL when flat)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DUCKDB SYNTAX RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Use EXTRACT() — never strftime() or DATE_PART().
    - Never use LIMIT in backtesting queries.
    - Reserved words ("datetime", "open", "high", "low", "close") must be double-quoted.

    PARAMETERS:
    - sql_query:           DuckDB SQL implementing the strategy.
    - starting_balance:    Initial capital (default 25000).
    - fee_roundtrip:       Round-trip cost as fraction (e.g. 0.0015 = 15 bps).
    - fee_fixed:           Fixed USD cost per round-trip (e.g. 1.0).
    - position_size_type:  'fixed' (USD) or 'percent' (% of balance).
    - position_size_value: Size value (e.g. 500 for fixed, 2 for 2%).
    """
    # -- Parameter normalisation
    try:
        if isinstance(fee_fixed, str):
            fee_fixed = float(fee_fixed.replace("$", "").strip())
        if isinstance(position_size_value, str):
            position_size_value = float(position_size_value.replace("%", "").strip())
        position_size_type = str(position_size_type).lower().strip()
        if "percent" in position_size_type or "%" in position_size_type:
            position_size_type = "percent"
        else:
            position_size_type = "fixed"
    except Exception:
        position_size_type = "fixed"
        position_size_value = 25000

    # -- Extract symbol from SQL
    symbol_match = re.search(r"['\"](\w+)\.parquet['\"]", sql_query, re.IGNORECASE)
    if not symbol_match:
        return "Error: Could not extract symbol. Use 'SYMBOL.parquet' format in your SQL query."
    requested_symbol = symbol_match.group(1).upper()

    # -- Resolve local path
    resolved_path = get_parquet_path(requested_symbol)
    if not resolved_path:
        return f"Error: Symbol '{requested_symbol}' not found in '{DATA_DIR}'."

    # -- Rewrite SQL to use the resolved local path
    fixed_query = re.sub(
        r"['\"]\w+\.parquet['\"]",
        f"read_parquet('{resolved_path}')",
        sql_query,
        flags=re.IGNORECASE,
    )

    try:
        with duckdb.connect() as conn:
            df = conn.execute(fixed_query).df()

        if "trade_ret" not in df.columns:
            return {
                "status": "Data loaded successfully",
                "file_resolved": resolved_path,
                "preview": df.head(5).to_dict(orient="records"),
                "instructions": (
                    "To see a full performance report, include a 'trade_ret' column "
                    "in your SELECT statement."
                ),
            }

        return calculate_quant_metrics(
            df,
            starting_balance=starting_balance,
            fee_roundtrip=fee_roundtrip,
            fee_fixed=fee_fixed,
            position_size_type=position_size_type,
            position_size_value=position_size_value,
            symbol=requested_symbol,
        )

    except Exception as e:
        return (
            f"SQL Error: {str(e)}\n\n"
            "IMPORTANT — Common causes:\n"
            "1. Reserved words (datetime, open, high, low, close) must be double-quoted.\n"
            "2. Used HOUR() or MINUTE() instead of EXTRACT(HOUR FROM \"datetime\").\n"
            "3. Used strftime() — not supported in DuckDB.\n"
            "4. Missing WHERE tradeable = 1 filter in daily resampling.\n"
            "5. Indicators calculated on raw minutely data instead of resampled daily bars.\n\n"
            "Please fix the SQL and retry."
        )


@mcp.tool()
def get_storage_stats():
    """Returns the count of symbols and parquet files in the local data directory."""
    if not os.path.exists(DATA_DIR):
        return f"Error: DATA_DIR '{DATA_DIR}' not found."

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    unique_symbols = set(
        re.sub(r"([-_].+)?\.parquet$", "", f, flags=re.IGNORECASE).upper()
        for f in all_files
    )
    return {
        "total_parquet_files": len(all_files),
        "total_unique_symbols": len(unique_symbols),
        "data_directory": DATA_DIR,
        "status": "Verified from local filesystem",
    }


if __name__ == "__main__":
    mcp.run()