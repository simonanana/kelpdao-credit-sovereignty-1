"""upgrade/upgrade_analysis.py — Submission-ready methodological upgrades.

Consolidates all v10-F, v11-F, and v12-F cells from the working notebooks
into one importable module. This layer sits *on top of* the original
`src/defi_event_study/` package: it reuses `config`, `data`, and the core
`did_core` / `spec_D` / `spec_B` helpers from `models`, and adds the
methodological upgrades required by the submission-ready manuscript.

Contents (grouped by paper section):

    §4.4 Price-flow decomposition        laspeyres_tvl / stable_tvl / decompose_flow_vs_price
    §4.5 Design-robust inference         randomization_inference / synthetic_control / synthetic_did
    §4.6 SUTVA bounds                    sutva_attribution_bounds / external_control_estimates
    §4.7 Persistence, whale, truncation  persistence_across_windows / ssr_truncation_robustness
    §3.2 Event-study (correct versions)  event_study_detrended / event_study_first_diff
    §5.2 On-chain utilization            fetch_onchain_reserve_history / plot_utilization
                                         reconcile_supply_borrow_apy
    §5   Variance tests (upgraded)       variance_tests_upgraded  (Brown-Forsythe + bootstrap CI)
    All  Table regeneration              regenerate_final_tables

Rule of thumb: `run_analysis.py` produces the *original* paper's tables and
figures; `upgrade/run_upgrade.py` produces the *submission-ready* additions.
Both write to `results/` with distinct filename prefixes (`FINAL_*`, `u*_*`).

Data snapshot for the manuscript: 2026-07-08. Re-running with `--no-cache`
fetches fresh data from DefiLlama; see `data/README.md`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.stats.stattools as stools
from scipy import stats
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Reuse the base package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defi_event_study import config as C  # noqa: E402
from defi_event_study.data import build_panel, fetch_tvl, fetch_token_flows, load_all_sources  # noqa: E402
from defi_event_study.models import did_core, run_four_specs  # noqa: E402


# ===========================================================================
# Extended donor pool for randomization inference and synthetic control
# ===========================================================================
DONOR_SLUGS_EXTENDED: dict[str, str] = {
    "Compound_V3": "compound-v3", "Fluid": "fluid-lending", "Euler_V2": "euler-v2",
    "Kamino_Solana": "kamino-lend", "Venus_BSC": "venus-core-pool",
    "JustLend": "justlend", "Benqi": "benqi-lending", "Radiant": "radiant-v2",
    "Suilend": "suilend", "Aave_V2": "aave-v2", "Maple": "maple",
    "Save_Solana": "save",
    # v11-F3 additions to reach ~20 units (target: exact one-sided p ≤ 0.05):
    "Moonwell": "moonwell", "ZeroLend": "zerolend", "Dolomite": "dolomite",
    "Seamless": "seamless-protocol", "Silo_V2": "silo-v2", "Silo": "silo-finance",
    "Gearbox": "gearbox", "Lista": "lista-lending", "Avalon": "avalon-finance",
    "Kinza": "kinza-finance",
}

EXTERNAL_CTRL: dict[str, str] = {
    "Kamino_Solana": "kamino-lend",
    "Venus_BSC": "venus-core-pool",
}

# WHALE adjustment scenarios (see paper §4.7 and MANUAL_FACTS M4):
# On-chain entity tracing did not confirm the media-reported $174M within
# the event window; the reported $83.6M USDT transfer predates the event.
# Only the media-ceiling scenario is enabled — as a conservative check.
WHALE_ADJ_SCENARIOS_USD = {"media_reported_ceiling": 174e6}

STABLES = {"USDC", "USDT", "DAI", "USDS", "GHO", "USDE", "PYUSD", "TUSD",
           "FRAX", "LUSD", "SUSDS", "SUSDE"}

NW_LAGS = 5
_HDR = {"User-Agent": "Academic-Research/1.0"}


def _get(url: str, retries: int = 3):
    """Minimal JSON GET with retries; returns None on failure."""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=40, headers=_HDR)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                return None
            time.sleep(2 ** i)
    return None


# ===========================================================================
# §4.4  Price-flow decomposition (v10-F4a / F4b)
# ===========================================================================
def laspeyres_tvl(token_panel: pd.DataFrame, event=C.EVENT_TS) -> pd.DataFrame:
    """Quantity-index TVL: token quantities valued at fixed pre-event prices.

    For each token, take the last observed unit price before the event
    (implied by valueUsd / qty) and value the whole series at that price.
    The resulting index is immune to post-event price movements — Δ can only
    come from flows.
    """
    d = token_panel.dropna(subset=["valueUsd", "qty"]).copy()
    d = d[d["qty"] > 0]
    pre = d[d["date"] < event]
    p0 = (pre.sort_values("date")
              .groupby("token").tail(1)
              .assign(p0=lambda x: x["valueUsd"] / x["qty"])
              [["token", "p0"]])
    d = d.merge(p0, on="token", how="inner")
    d["fixed_val"] = d["qty"] * d["p0"]
    return (d.groupby("date", as_index=False)["fixed_val"]
             .sum().rename(columns={"fixed_val": "tvlUsd"}))


def stable_tvl(token_panel: pd.DataFrame) -> pd.DataFrame:
    """Stablecoin-only TVL: prices ≈ 1 by construction, so Δ ≈ pure flow."""
    d = token_panel[token_panel["token"].str.upper().isin(STABLES)]
    d = d.dropna(subset=["valueUsd"])
    return (d.groupby("date", as_index=False)["valueUsd"]
             .sum().rename(columns={"valueUsd": "tvlUsd"}))


def decompose_flow_vs_price(token_panel: pd.DataFrame,
                            event=C.EVENT_TS, horizon: int = 7) -> pd.Series:
    """Additive decomposition of ΔV across tokens into flow and price parts.

    Uses ΔV_k = q̄_k·Δp_k + p̄_k·Δq_k (Bennet decomposition).
    Returns totals (dV, flow_comp, price_comp) in USD.
    """
    d = token_panel.dropna(subset=["valueUsd", "qty"]).copy()
    d = d[d["qty"] > 0]
    t0 = d[d["date"] < event].sort_values("date").groupby("token").tail(1)
    t1 = (d[d["date"] <= event + pd.Timedelta(days=horizon)]
            .sort_values("date").groupby("token").tail(1))
    m = t0.merge(t1, on="token", suffixes=("_0", "_1"))
    m["p_0"], m["p_1"] = m["valueUsd_0"] / m["qty_0"], m["valueUsd_1"] / m["qty_1"]
    m["dV"] = m["valueUsd_1"] - m["valueUsd_0"]
    m["flow_comp"] = (m["qty_1"] - m["qty_0"]) * (m["p_0"] + m["p_1"]) / 2
    m["price_comp"] = (m["p_1"] - m["p_0"]) * (m["qty_0"] + m["qty_1"]) / 2
    return m[["dV", "flow_comp", "price_comp"]].sum()


def spec_D_generic(panel: pd.DataFrame, treat: str, ctrl: str, ycol: str = "log_tvl") -> dict | None:
    """Spec D (lagged DV) on an arbitrary outcome column. Used by §4.4 tables."""
    df = panel[panel["protocol"].isin([treat, ctrl])].copy()
    df["treated"] = (df["protocol"] == treat).astype(int)
    df["did"] = df["treated"] * df["post"]
    df["lag_y"] = df.groupby("protocol")[ycol].shift(1)
    return did_core(df.rename(columns={ycol: "y"}), "y",
                    ["lag_y", "post", "treated", "did"],
                    label=f"SpecD {treat} vs {ctrl}")


# ===========================================================================
# §4.5  Design-robust inference (v10-F6, F7)
# ===========================================================================
def randomization_inference(donor_tvl: dict[str, pd.DataFrame],
                            true_did: float,
                            spec_D_fn=spec_D_generic,
                            control: str = "Spark") -> tuple[pd.DataFrame, float, float]:
    """Assign placebo treatment to each donor protocol; compare Aave to distribution.

    Returns (placebo_df, one_sided_p, two_sided_p) where p values follow
    Fisher-exact convention with the +1 correction of MacKinnon-Webb (2020).
    """
    src_all = {**donor_tvl}
    panel = build_panel(src_all)
    rows = []
    for name in donor_tvl:
        if name == control:
            continue
        r = spec_D_fn(panel, name, control)
        if r and np.isfinite(r["did_coef"]):
            rows.append({"protocol": name, "did": r["did_coef"], "p_naive": r["did_p"]})
    plc = pd.DataFrame(rows)
    n = len(plc)
    p_one = (1 + (plc["did"] <= true_did).sum()) / (1 + n)
    p_two = (1 + (plc["did"].abs() >= abs(true_did)).sum()) / (1 + n)
    return plc, float(p_one), float(p_two)


def _synth_weights(y1: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 (demeaned SSR fit)."""
    y1c = y1 - y1.mean()
    Y0c = Y0 - Y0.mean(axis=0)
    k = Y0c.shape[1]
    obj = lambda w: np.sum((y1c - Y0c @ w) ** 2)  # noqa: E731
    res = minimize(obj, np.full(k, 1 / k),
                   bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500})
    return res.x


def synthetic_control(panel: pd.DataFrame,
                      treated: str, donors: list[str],
                      ycol: str = "log_tvl",
                      event=C.EVENT_TS) -> tuple[pd.Series, pd.Series, float]:
    """Classic Abadie et al. (2010) synthetic control on log TVL.

    Returns (synthetic_series, gap_series, post_mean_ATT).
    """
    wide = (panel[panel["protocol"].isin([treated] + donors)]
              .pivot_table(index="date", columns="protocol", values=ycol)
              .dropna())
    donors = [c for c in wide.columns if c != treated]
    pre = wide.index < event
    w = _synth_weights(wide.loc[pre, treated].values, wide.loc[pre, donors].values)
    synth = wide[donors] @ w
    synth = synth + (wide.loc[pre, treated].mean() - synth[pre].mean())
    gap = wide[treated] - synth
    return synth, gap, float(gap[~pre].mean())


def rmspe_ratio_inference(panel: pd.DataFrame,
                          treated: str, donors: list[str],
                          ycol: str = "log_tvl",
                          event=C.EVENT_TS) -> tuple[pd.Series, float]:
    """Placebo-in-space inference via RMSPE post/pre ratio (Abadie et al. 2010)."""
    def one(target: str, others: list[str]) -> float:
        wide = (panel[panel["protocol"].isin([target] + others)]
                  .pivot_table(index="date", columns="protocol", values=ycol)
                  .dropna())
        others = [c for c in wide.columns if c != target]
        if len(others) < 3:
            return np.nan
        pre = wide.index < event
        w = _synth_weights(wide.loc[pre, target].values, wide.loc[pre, others].values)
        s = wide[others] @ w
        s = s + (wide.loc[pre, target].mean() - s[pre].mean())
        g = wide[target] - s
        pre_r = np.sqrt((g[pre] ** 2).mean())
        post_r = np.sqrt((g[~pre] ** 2).mean())
        return post_r / max(pre_r, 1e-9)

    ratios = {treated: one(treated, donors)}
    for d in donors:
        others = [c for c in donors if c != d]
        ratios[d] = one(d, others)
    rr = pd.Series(ratios).dropna().sort_values(ascending=False)
    rank = int(rr.index.get_loc(treated)) + 1
    return rr, rank / len(rr)


def synthetic_did(panel: pd.DataFrame,
                  treated: str, donors: list[str],
                  ycol: str = "log_tvl",
                  event=C.EVENT_TS) -> float:
    """Simplified Arkhangelsky et al. (2021) synthetic DiD estimator.

    Educational implementation: unit weights (§_synth_weights) + time weights
    fitted on pre-period donors to match donor post-period mean; combined
    with a double-difference. For camera-ready quality: use pysyncon or R
    synthdid, which add regularization and jackknife SEs.
    """
    wide = (panel[panel["protocol"].isin([treated] + donors)]
              .pivot_table(index="date", columns="protocol", values=ycol)
              .dropna())
    donors = [c for c in wide.columns if c != treated]
    pre = wide.index < event
    w = _synth_weights(wide.loc[pre, treated].values, wide.loc[pre, donors].values)

    Y0_pre = wide.loc[pre, donors].values          # T0 × J
    Y0_post_mean = wide.loc[~pre, donors].values.mean(axis=0)  # J
    T0 = Y0_pre.shape[0]
    obj = lambda lam: np.sum((Y0_post_mean - Y0_pre.T @ lam) ** 2)  # noqa: E731
    res = minimize(obj, np.full(T0, 1 / T0),
                   bounds=[(0, 1)] * T0,
                   constraints=[{"type": "eq", "fun": lambda l: l.sum() - 1}],
                   method="SLSQP", options={"maxiter": 800})
    lam = res.x

    y_tr_post = wide.loc[~pre, treated].mean()
    y_tr_pre = wide.loc[pre, treated].values @ lam
    y_co_post = Y0_post_mean @ w
    y_co_pre = (Y0_pre @ w) @ lam
    return float((y_tr_post - y_tr_pre) - (y_co_post - y_co_pre))


# ===========================================================================
# §4.6  SUTVA attribution bounds and external controls
# ===========================================================================
def sutva_attribution_bounds(tvl: dict[str, pd.DataFrame],
                             control: str = "Spark",
                             treated: str = "Aave_V3",
                             fractions: tuple[float, ...] = (0.0, 0.5, 1.0),
                             post_days: int = C.POST_DAYS) -> pd.DataFrame:
    """Bound the DiD by attributing s∈{fractions} of Spark's abnormal inflow to Aave.

    Constructs a counterfactual Spark path by extrapolating the pre-event
    linear drift; the observed excess is "abnormal inflow." Removing s×excess
    yields a de-contaminated Spark series; re-estimate Spec D on that panel.
    """
    sp = tvl[control].copy()
    pre_w = sp[(sp["date"] >= C.EVENT_TS - pd.Timedelta(days=30)) &
               (sp["date"] < C.EVENT_TS)]
    drift = (pre_w["tvlUsd"].iloc[-1] - pre_w["tvlUsd"].iloc[0]) / max(len(pre_w) - 1, 1)
    t0_lvl = pre_w["tvlUsd"].iloc[-1]

    rows = []
    for s in fractions:
        adj = sp.copy()
        post_m = adj["date"] >= C.EVENT_TS
        days = (adj.loc[post_m, "date"] - C.EVENT_TS).dt.days
        counterfactual = t0_lvl + drift * days
        abnormal = (adj.loc[post_m, "tvlUsd"] - counterfactual).clip(lower=0)
        adj.loc[post_m, "tvlUsd"] = adj.loc[post_m, "tvlUsd"] - s * abnormal
        panel = build_panel({treated: tvl[treated], control: adj}, post_days=post_days)
        r = spec_D_generic(panel, treated, control)
        if r:
            rows.append([f"s = {s:.0%}",
                         round(r["did_coef"], 4),
                         round(r["did_p"], 5),
                         round((np.exp(r["did_coef"]) - 1) * 100, 1)])
    return pd.DataFrame(rows, columns=["Attribution scenario", "SpecD_DiD", "p", "Pct_effect"])


def external_control_estimates(all_tvl: dict[str, pd.DataFrame],
                               treated: str = "Aave_V3",
                               externals: tuple[str, ...] = ("Kamino_Solana", "Venus_BSC"),
                               post_days: int = C.POST_DAYS) -> pd.DataFrame:
    """DiD against ecosystem-external controls with no rsETH exposure."""
    panel = build_panel(all_tvl, post_days=post_days)
    rows = []
    for ctrl in externals:
        if ctrl not in all_tvl:
            continue
        r = spec_D_generic(panel, treated, ctrl)
        if r:
            rows.append([ctrl, r["N"], round(r["did_coef"], 4),
                         round(r["did_p"], 5),
                         round((np.exp(r["did_coef"]) - 1) * 100, 1)])
    return pd.DataFrame(rows, columns=["External control", "N", "SpecD_DiD", "p", "Pct_effect"])


# ===========================================================================
# §4.7  Persistence, whale adjustment, SSR truncation
# ===========================================================================
def persistence_across_windows(tvl: dict[str, pd.DataFrame],
                               windows: tuple[int, ...] = (9, 30, 60),
                               treated: str = "Aave_V3",
                               control: str = "Spark") -> pd.DataFrame:
    """Re-estimate Spec D at multiple post-event windows for a persistence check."""
    rows = []
    for post in windows:
        panel = build_panel(tvl, post_days=post)
        r = spec_D_generic(panel, treated, control)
        if r:
            rows.append([post, r["N"], round(r["did_coef"], 4),
                         round(r["did_p"], 5),
                         round((np.exp(r["did_coef"]) - 1) * 100, 1)])
    return pd.DataFrame(rows,
                        columns=["Post_days", "N", "SpecD_DiD", "p", "Pct_effect"])


def ssr_truncation_robustness(tvl: dict[str, pd.DataFrame],
                              cut_date: str = "2026-05-25",
                              treated: str = "Aave_V3",
                              control: str = "Spark") -> dict:
    """§6.3 seventh limitation: robustness to truncating before the SSR reduction."""
    cut = pd.Timestamp(cut_date, tz="UTC")
    post_trunc = int((cut - C.EVENT_TS).days)
    r_full = spec_D_generic(build_panel(tvl, post_days=60), treated, control)
    r_tr = spec_D_generic(build_panel(tvl, post_days=post_trunc), treated, control)
    return {
        "full_T60": {"did": r_full["did_coef"], "p": r_full["did_p"]},
        f"truncated_T{post_trunc}": {"did": r_tr["did_coef"], "p": r_tr["did_p"]},
    }


def whale_adjustment_check(tvl: dict[str, pd.DataFrame],
                           control: str = "Spark",
                           treated: str = "Aave_V3",
                           post_days: int = C.POST_DAYS,
                           whale_usd: float = WHALE_ADJ_SCENARIOS_USD["media_reported_ceiling"],
                           t_plus: int = 2) -> dict:
    """Conservative upper-bound: remove media-reported whale from post-Spark TVL.

    On-chain entity tracing did not confirm this deposit in the event window
    (see MANUAL_FACTS['M4']); we run the removal purely as an upper-bound
    robustness check.
    """
    sp = tvl[control].copy()
    t2 = C.EVENT_TS + pd.Timedelta(days=t_plus)
    sp.loc[sp["date"] >= t2, "tvlUsd"] = sp.loc[sp["date"] >= t2, "tvlUsd"] - whale_usd
    r = spec_D_generic(build_panel({treated: tvl[treated], control: sp}, post_days=post_days),
                       treated, control)
    return {"did": r["did_coef"], "p": r["did_p"],
            "pct": round((np.exp(r["did_coef"]) - 1) * 100, 1),
            "whale_removed_usd": whale_usd}


# ===========================================================================
# §3.2  Event-study — correct implementations (v12-F2'')
#
# ⚠  v11-F2 (full-sample linear trend absorption) is DEPRECATED. That version
#     absorbed post-event drops into the fitted trend, producing spurious
#     pre-event drift. The two correct versions below fit the trend on the
#     pre-event sample only OR difference the outcome; both are used in the
#     paper's Figure 3 (panels a and b).
# ===========================================================================
def event_study_detrended(panel: pd.DataFrame,
                          treated: str = "Aave_V3",
                          control: str = "Spark",
                          base_week: int = -1,
                          week_clip: tuple[int, int] = (-12, 8)) -> pd.DataFrame:
    """Panel (a): per-protocol linear trend fit on PRE-event data only,
    then extrapolated to post-event and subtracted.
    """
    df = panel[panel["protocol"].isin([treated, control])].copy()
    df["treated"] = (df["protocol"] == treated).astype(int)
    df["rel_week"] = np.clip(np.floor((df["date"] - C.EVENT_TS).dt.days / 7).astype(int),
                             week_clip[0], week_clip[1])

    detrended = []
    for _, g in df.groupby("protocol"):
        g = g.sort_values("date").copy()
        pre = g[g["date"] < C.EVENT_TS]
        b = np.polyfit(pre["t"], pre["log_tvl"], 1)
        g["resid"] = g["log_tvl"] - np.polyval(b, g["t"])
        detrended.append(g)
    dd = pd.concat(detrended)

    dums = pd.get_dummies(dd["rel_week"], prefix="w", dtype=float)
    inter = dums.mul(dd["treated"], axis=0).add_prefix("tx_")
    X = pd.concat([dd[["treated"]], dums, inter], axis=1) \
          .drop(columns=[f"w_{base_week}", f"tx_w_{base_week}"])
    m = sm.OLS(dd["resid"].astype(float),
               sm.add_constant(X.astype(float))
              ).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    ks = sorted(dd["rel_week"].unique())
    return pd.DataFrame({
        "week": [k for k in ks if k != base_week],
        "coef": [m.params[f"tx_w_{k}"] for k in ks if k != base_week],
        "se":   [m.bse[f"tx_w_{k}"] for k in ks if k != base_week],
    })


def event_study_first_diff(panel: pd.DataFrame,
                           treated: str = "Aave_V3",
                           control: str = "Spark",
                           base_week: int = -1,
                           week_clip: tuple[int, int] = (-12, 8)) -> pd.DataFrame:
    """Panel (b): first-differenced (weekly growth) event study.

    This is the cleanest test of parallel trends: if pre-event leads are
    small in growth rates, the two series shared a common growth process.
    """
    df = panel[panel["protocol"].isin([treated, control])].copy()
    df["treated"] = (df["protocol"] == treated).astype(int)
    df = df.sort_values(["protocol", "date"])
    df["dlog"] = df.groupby("protocol")["log_tvl"].diff()
    df = df.dropna(subset=["dlog"])
    df["rel_week"] = np.clip(np.floor((df["date"] - C.EVENT_TS).dt.days / 7).astype(int),
                             week_clip[0], week_clip[1])

    dums = pd.get_dummies(df["rel_week"], prefix="w", dtype=float)
    inter = dums.mul(df["treated"], axis=0).add_prefix("tx_")
    X = pd.concat([df[["treated"]], dums, inter], axis=1) \
          .drop(columns=[f"w_{base_week}", f"tx_w_{base_week}"])
    m = sm.OLS(df["dlog"].astype(float),
               sm.add_constant(X.astype(float))
              ).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    ks = sorted(df["rel_week"].unique())
    return pd.DataFrame({
        "week": [k for k in ks if k != base_week],
        "coef": [m.params[f"tx_w_{k}"] for k in ks if k != base_week],
        "se":   [m.bse[f"tx_w_{k}"] for k in ks if k != base_week],
    })


# ===========================================================================
# §5.2  On-chain utilization — DefiLlama chartLendBorrow is now paywalled;
# we fall back to archival eth_call on the Aave ProtocolDataProvider.
# ===========================================================================
# ⚠ Verify this address on docs.aave.com Deployed Contracts before running;
# it changes with each Aave 3.x deployment.
AAVE_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"
SEL_GET_RESERVE_DATA = "0x35ea6a75"      # getReserveData(address)

PUBLIC_RPCS = [
    "https://eth.llamarpc.com",
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.flashbots.net",
    "https://eth.drpc.org",
]

AAVE_STABLECOINS = {
    "Aave V3 USDC Ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "Aave V3 DAI Ethereum":  "0x6B175474E89094C44Da98b954EedeAC495271d0F",
}


def _block_at(ts: float) -> int | None:
    """DefiLlama free block-timestamp mapping."""
    j = _get(f"https://coins.llama.fi/block/ethereum/{int(ts)}")
    return j.get("height") if j else None


def _eth_call(to: str, data: str, block_hex: str) -> str | None:
    """Round-robin over public archive-capable RPCs."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": to, "data": data}, block_hex]}
    for rpc in PUBLIC_RPCS:
        try:
            r = requests.post(rpc, json=payload, timeout=25, headers=_HDR)
            j = r.json()
            if "result" in j and j["result"] and j["result"] != "0x":
                return j["result"]
        except Exception:  # noqa: BLE001
            continue
    return None


def fetch_onchain_reserve_history(label: str, underlying: str,
                                  use_cache: bool = True) -> pd.DataFrame:
    """Daily archival eth_call to getReserveData for one Aave reserve.

    Sampling strategy: weekly from PRE−120 to PRE−46, daily from PRE−45
    through POST+60. Requires an archive-capable RPC in PUBLIC_RPCS.
    """
    cache = C.DATA_RAW / f"onchain_{underlying[2:10]}.csv"
    if use_cache and cache.exists():
        d = pd.read_csv(cache, parse_dates=["date"])
        d["date"] = pd.to_datetime(d["date"], utc=True)
        return d

    days = list(pd.date_range(C.EVENT_TS - pd.Timedelta(days=C.PRE_DAYS),
                              C.EVENT_TS - pd.Timedelta(days=46), freq="7D")) + \
           list(pd.date_range(C.EVENT_TS - pd.Timedelta(days=45),
                              C.EVENT_TS + pd.Timedelta(days=60), freq="D"))
    calldata = SEL_GET_RESERVE_DATA + underlying[2:].lower().rjust(64, "0")
    rows = []
    for dt in days:
        blk = _block_at(dt.timestamp())
        if not blk:
            continue
        res = _eth_call(AAVE_DATA_PROVIDER, calldata, hex(blk))
        if not res:
            continue
        h = res[2:]
        w = [int(h[i * 64:(i + 1) * 64], 16) for i in range(len(h) // 64)]
        # ReserveData layout: [0]=unbacked, [1]=accruedToTreasury,
        # [2]=totalAToken, [3]=totalStableDebt, [4]=totalVariableDebt,
        # [5]=liquidityRate (ray), [6]=variableBorrowRate (ray), ...
        if len(w) < 7 or w[2] == 0:
            continue
        total_a, sd, vd, liq_ray, bor_ray = w[2], w[3], w[4], w[5], w[6]
        rows.append({
            "date": dt,
            "U": (sd + vd) / total_a,
            "supply_apr_pct": liq_ray / 1e27 * 100,
            "borrow_apr_pct": bor_ray / 1e27 * 100,
        })
        time.sleep(0.15)
    d = pd.DataFrame(rows)
    if len(d):
        d.to_csv(cache, index=False)
    return d


def plot_utilization(util: dict[str, pd.DataFrame],
                     save_to: Path | None = None) -> plt.Figure:
    """Two-panel figure: U and borrow APR around the event (paper Figure 7)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    palette = {"Aave V3 DAI Ethereum": "#E04B4B", "Aave V3 USDC Ethereum": "#F4A261"}
    for key, color in palette.items():
        d = util.get(key, pd.DataFrame())
        if d.empty:
            continue
        w = d[(d["date"] >= C.EVENT_TS - pd.Timedelta(days=45)) &
              (d["date"] <= C.EVENT_TS + pd.Timedelta(days=60))]
        axes[0].plot(w["date"], w["U"], lw=1.8, color=color, label=key)
        axes[1].plot(w["date"], w["borrow_apr_pct"], lw=1.8, color=color, label=key)
    axes[0].axhline(0.80, color="gray", ls=":", lw=1)
    axes[0].text(C.EVENT_TS - pd.Timedelta(days=44), 0.815, "U* = 0.80",
                 fontsize=8, color="gray")
    for ax, ttl, yl in [
        (axes[0], "Aave V3 Utilization (on-chain, eth_call getReserveData)", "U"),
        (axes[1], "Aave V3 Variable Borrow APR (%)", "APR %"),
    ]:
        ax.axvline(C.EVENT_TS, color="k", ls="--", lw=1.3)
        ax.set_title(ttl); ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches="tight")
    return fig


def reconcile_supply_borrow_apy(util_dai: pd.DataFrame,
                                apy_defillama: pd.DataFrame,
                                reserve_factor: float = 0.25) -> dict:
    """Cross-validate the on-chain peak borrow APR against DefiLlama's peak supply APY.

    Identity: supply APY ≈ borrow APR × U × (1 − reserve factor)
    Paper §5.2 reports: 39.8% × 1.00 × 0.75 = 29.8% ≈ 29.9% (DefiLlama).
    Confirm reserve_factor at app.aave.com before publishing.
    """
    pk = util_dai[(util_dai["date"] >= C.EVENT_TS) &
                  (util_dai["date"] <= C.EVENT_TS + pd.Timedelta(days=C.POST_DAYS))]
    i = pk["borrow_apr_pct"].idxmax()
    borrow, u_at = float(pk.loc[i, "borrow_apr_pct"]), float(pk.loc[i, "U"])
    implied_supply = borrow * u_at * (1 - reserve_factor)
    defillama_max = float(
        apy_defillama[(apy_defillama["date"] >= C.EVENT_TS) &
                      (apy_defillama["date"] <= C.EVENT_TS + pd.Timedelta(days=C.POST_DAYS))]
        ["apy"].max()
    )
    return {
        "onchain_borrow_apr_peak_pct": borrow,
        "utilization_at_peak": u_at,
        "implied_supply_apy_pct": implied_supply,
        "defillama_supply_apy_peak_pct": defillama_max,
        "reserve_factor_used": reserve_factor,
    }


# ===========================================================================
# §5  Variance tests — upgraded (Brown-Forsythe + bootstrap CI, one-sided)
# ===========================================================================
def variance_tests_upgraded(apy: dict[str, pd.DataFrame],
                            pair_map: list[tuple[str, str, str]],
                            n_boot: int = 2000,
                            seed: int = 42) -> pd.DataFrame:
    """Brown-Forsythe (median-centered Levene) with one-sided p and bootstrap
    CI on the variance ratio σ²_Aave / σ²_Spark. Replaces v9 Levene table.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a_key, s_key, pair in pair_map:
        da, ds = apy[a_key], apy[s_key]
        start = max(da[da["date"] < C.EVENT_TS]["date"].min(),
                    ds[ds["date"] < C.EVENT_TS]["date"].min())
        av = da[(da["date"] >= start) & (da["date"] < C.EVENT_TS)]["apy"].dropna().values
        sv = ds[(ds["date"] >= start) & (ds["date"] < C.EVENT_TS)]["apy"].dropna().values

        _, p_lev = stats.levene(av, sv, center="mean")
        _, p_bf = stats.levene(av, sv, center="median")
        F = float(av.var(ddof=1) / sv.var(ddof=1))
        # One-sided test: H1 σ²_Aave > σ²_Spark
        p_bf_1s = float(p_bf / 2 if F > 1 else 1 - p_bf / 2)
        boots = [rng.choice(av, len(av)).var(ddof=1) /
                 rng.choice(sv, len(sv)).var(ddof=1) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append([pair, len(av), len(sv), round(F, 2),
                     round(lo, 2), round(hi, 2),
                     round(p_lev, 5), round(p_bf, 5), round(p_bf_1s, 5)])
    return pd.DataFrame(rows, columns=[
        "Pair", "N_Aave", "N_Spark", "F_ratio", "F_boot_lo", "F_boot_hi",
        "Levene_p", "BrownForsythe_p", "BF_p_one_sided",
    ])


# ===========================================================================
# Table regeneration on the current data snapshot
# ===========================================================================
def regenerate_final_tables(tvl: dict[str, pd.DataFrame],
                            apy: dict[str, pd.DataFrame],
                            out_dir: Path | None = None,
                            snapshot_tag: str = "FINAL") -> dict[str, pd.DataFrame]:
    """Rebuild every paper table from the current cached snapshot.

    Written to results/tables/{snapshot_tag}_*.csv. The manuscript numbers
    must be pulled from this call — not from the original run_analysis.py,
    which used the April snapshot.
    """
    out = Path(out_dir or C.RESULTS_TABLES)
    out.mkdir(parents=True, exist_ok=True)
    tables = {}

    # ----- Table 3: TVL summary -----
    panel = build_panel(tvl)
    rows = []
    for p in ["Aave_V3", "Morpho", "Spark"]:
        d = panel[panel["protocol"] == p]
        if d.empty:
            continue
        pre = d.loc[d["post"] == 0, "tvlUsd"].mean() / 1e9
        i0 = (d["date"] - C.EVENT_TS).abs().idxmin()
        i7 = (d["date"] - (C.EVENT_TS + pd.Timedelta(days=7))).abs().idxmin()
        t0, t7 = d.loc[i0, "tvlUsd"] / 1e9, d.loc[i7, "tvlUsd"] / 1e9
        rows.append([p, round(pre, 2), round(t0, 2), round(t7, 2),
                     round((t7 / t0 - 1) * 100, 1)])
    tables["table3"] = pd.DataFrame(
        rows, columns=["Protocol", "PreMean_B", "T0_B", "T7_B", "Pct_T0_T7"])
    tables["table3"].to_csv(out / f"{snapshot_tag}_table3_tvl_summary.csv", index=False)

    # ----- Table 4: DiD, two pairs -----
    rows = []
    for pair, ctrl in [("Aave_vs_Spark", "Spark"), ("Aave_vs_Morpho", "Morpho")]:
        for r in run_four_specs(panel, "Aave_V3", ctrl):
            if r is None:
                continue
            pct = np.nan if "FirstDiff" in r["label"] else round(
                (np.exp(r["did_coef"]) - 1) * 100, 1)
            rows.append([pair, r["label"], r["N"], round(r["R2"], 4),
                         round(r["DW"], 3), round(r["did_coef"], 4),
                         round(r["did_se"], 4), round(r["did_p"], 5), pct])
    tables["table4"] = pd.DataFrame(
        rows, columns=["Pair", "Spec", "N", "R2", "DW",
                       "DiD_coef", "HAC_SE", "p", "Pct_effect"])
    tables["table4"].to_csv(out / f"{snapshot_tag}_table4_did.csv", index=False)

    # ----- Table 5: APY stats -----
    rows = []
    for label, df in apy.items():
        if df.empty:
            continue
        pre = df[df["date"] < C.EVENT_TS]
        post = df[(df["date"] >= C.EVENT_TS) &
                  (df["date"] <= C.EVENT_TS + pd.Timedelta(days=C.POST_DAYS))]
        rows.append([label, len(pre),
                     round(pre["apy"].mean(), 2) if len(pre) else np.nan,
                     round(pre["apy"].std(), 3) if len(pre) else np.nan,
                     len(post),
                     round(post["apy"].std(), 3) if len(post) > 1 else np.nan,
                     round(post["apy"].max(), 2) if len(post) else np.nan])
    tables["table5"] = pd.DataFrame(
        rows, columns=["Pool", "N_pre", "PreMean", "PreSD",
                       "N_post", "PostSD", "PostMax"])
    tables["table5"].to_csv(out / f"{snapshot_tag}_table5_apy_stats.csv", index=False)

    return tables
