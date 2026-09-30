"""
ALL-SIC REGION RIDGE RANK-OPTIMIZATION ENGINE, VECTORIZED BACKTEST & ANALYTICS SUITE
====================================================================================
- Model: Ridge Regression with High-Penalty Shrinkage (alpha=5000.0) per SIC Region.
- Portfolio: ONCE A YEAR (end of December) the model is refit, every stock is scored, and
  each quintile is bought equal-weight and held untouched for 12 months (Jan-Dec). Repeat.
- Delisting & Rebalancing Policy:
    1. No -35% penalty imputation; uses CRSP DelRet if present, else 0.0.
    2. If a stock delists during the year, its final value is spread across the
       surviving names in proportion to their value (no other trading during the year).
    3. Look-Ahead Bias Eliminated: the model only trains on rows whose 12M outcome was
       fully known at the rebalance date; scoring uses all stocks alive on that date.
    4. 15 bps one-way cost on the annual buy and the annual sell.
    5. Q1-Q5 spread is charged trading costs on BOTH legs.

- Analytics (all written to OUTPUT_DIR, ready to commit to GitHub):
    * Console table of returns + Sharpe (and ~20 more stats) for every bucket
    * Risk/return, drawdown, tail-risk, CAPM alpha/beta (Newey-West), capture ratios
    * Probabilistic Sharpe Ratio, Sharpe 95% CI (Lo 2002)
    * Optional Fama-French 5 + Momentum attribution (drop in Ken French CSVs)
    * Rank IC / ICIR (overall, per SIC region, per raw feature), rolling IC
    * Ridge coefficient stability across refits and regions
    * Calendar-year, decade, and up/down-market breakdowns
    * Transaction-cost sensitivity (analytic, no re-simulation)
    * Year-over-year turnover, basket breadth, sector composition of the long leg
    * ~14 publication-quality PNG charts + README.md report with embedded tables
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, FuncFormatter
from scipy import stats as sps
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & PATHS
# ----------------------------------------------------------------------------
MONTHLY_PATH = r'C:\Users\user\Desktop\WRDS\prices\monthly_stock.parquet'
FUND_PATH = r'C:\Users\user\Desktop\WRDS\prices\annual_fundamentals.csv'
LINK_PATH = r'C:\Users\user\Desktop\WRDS\gvkey_and_permco.csv'
DELIST_PATH = r'C:\Users\user\Desktop\WRDS\prices\delisting_information.csv'

# Optional: raw Ken French monthly CSVs (as downloaded & unzipped from his data library).
# If present, the risk-free rate is used in Sharpe/Sortino and a FF5+MOM regression is run.
FF5_PATH = r'C:\Users\user\Desktop\WRDS\F-F_Research_Data_5_Factors_2x3.csv'
MOM_PATH = r'C:\Users\user\Desktop\WRDS\F-F_Momentum_Factor.csv'
RF_ANNUAL_FALLBACK = 0.0          # used only if FF5_PATH is missing

OUTPUT_DIR = 'results'            # figures/, tables/, README.md, summary.json
SHOW_PLOTS = False                # True -> also pop up figures interactively
MODERN_START = '2005-01-01'

MIN_PRICE = 5.0
FILING_LAG_DAYS = 90
STALE_TOLERANCE_DAYS = 400

FORWARD_HORIZON = 12
REBALANCE_MONTH = 12           # form portfolios at December month-end -> hold Jan..Dec
EMBARGO_MONTHS = 13
MIN_TRAIN_MONTHS = 60
TRAIN_WINDOW_MONTHS = 120
RIDGE_ALPHA = 5000.0           # Strong shrinkage toward equal weighting
SLIPPAGE_ONE_WAY = 0.0015      # 15 bps one-way execution drag
COST_GRID_BPS = [0, 5, 10, 15, 25, 50, 75, 100]
ROLL_WIN = 36
NW_LAGS = 12

FEATURE_COLS = [
    'F_RULE_OF_40',
    'F_CASH_ROA',
    'F_DELTA_GROSS_MARGIN',
    'F_OPERATING_LEVERAGE',
    'F_CASH_CONVERSION',
    'F_INTERNAL_FINANCING',
    'F_NET_BUYBACK',
    'F_MOM_12_2',
    'F_LOW_VOLATILITY'
]

# Standard US SIC Division Classifications
SIC_DIVISIONS = {
    'A_Agri_Forest_Fish': (100, 999),
    'B_Mining': (1000, 1499),
    'C_Construction': (1500, 1799),
    'D_Manufacturing': (2000, 3999),
    'E_Transp_Comm_Utils': (4000, 4999),
    'F_Wholesale_Trade': (5000, 5199),
    'G_Retail_Trade': (5200, 5999),
    'H_Finance_Ins_RealEst': (6000, 6799),
    'I_Services': (7000, 8999),
    'J_Public_Admin': (9100, 9999)
}

BUCKETS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
BUCKET_LABELS = {
    'Q1': 'Q1 (Top 20% - Long)',
    'Q2': 'Q2 (60-80%)',
    'Q3': 'Q3 (40-60% Median)',
    'Q4': 'Q4 (20-40%)',
    'Q5': 'Q5 (Bottom 20% - Short)',
    'BENCH': 'EW Universe Benchmark',
    'Q1_minus_Q5': 'Q1 - Q5 Long/Short',
    'Q1_minus_BENCH': 'Q1 - Benchmark (Active)',
}

# ----------------------------------------------------------------------------
# 2. STATISTICAL HELPERS & LABEL CALCULATOR
# ----------------------------------------------------------------------------
def _lower_map(cols):
    return {c.lower(): c for c in cols}

def newey_west_tstat(series, lags=12):
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(x)
    if n < 12:
        return np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return np.nan
    return mu / np.sqrt(var / n)

def hac_ols(y, X, lags=12):
    """OLS with intercept and Newey-West (Bartlett) HAC standard errors.
    Returns (coefs, tstats, r2) where index 0 is the intercept."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 1e-30, None))
    r2 = 1.0 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return b, b / se, r2

def assign_sic_region(sic_code):
    try:
        val = int(sic_code)
    except (ValueError, TypeError):
        return None
    for name, (low, high) in SIC_DIVISIONS.items():
        if low <= val <= high:
            return name
    return None

def compute_forward_cumret(series, horizon=12):
    """
    Computes forward cumulative return over up to `horizon` months.
    If a stock delists prematurely, compounds available returns and terminates.
    """
    vals = series.values
    n = len(vals)
    out = np.full(n, np.nan)

    for i in range(n):
        window = vals[i + 1 : min(i + 1 + horizon, n)]
        if len(window) > 0 and not np.isnan(window[0]):
            valid_rets = []
            for r in window:
                if np.isnan(r):
                    break
                valid_rets.append(r)
            if valid_rets:
                out[i] = np.prod(1.0 + np.array(valid_rets)) - 1.0
    return pd.Series(out, index=series.index)

def parse_french_csv(path):
    """Parse a raw Ken French monthly CSV (text header, monthly block, annual block).
    Returns a DataFrame indexed by month-end with values in DECIMALS."""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='latin-1') as fh:
        lines = fh.read().splitlines()
    header, rows = None, []
    for ln in lines:
        parts = [p.strip() for p in ln.split(',')]
        if header is None:
            if len(parts) > 1 and parts[0] == '' and any(p for p in parts[1:]):
                header = parts[1:]
            continue
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[:len(header) + 1])
        elif rows:
            break  # end of the monthly block
    if header is None or not rows:
        return None
    df = pd.DataFrame(rows, columns=['YYYYMM'] + header)
    df.index = pd.to_datetime(df.pop('YYYYMM'), format='%Y%m') + pd.offsets.MonthEnd(0)
    df = df.apply(pd.to_numeric, errors='coerce') / 100.0
    df.columns = [c.strip() for c in df.columns]
    return df

def load_factors():
    ff5 = parse_french_csv(FF5_PATH)
    mom = parse_french_csv(MOM_PATH)
    if ff5 is None:
        return None
    if mom is not None:
        mom.columns = ['MOM' if 'mom' in c.lower() else c for c in mom.columns]
        ff5 = ff5.join(mom[['MOM']], how='left')
    return ff5

# ----------------------------------------------------------------------------
# 3. 1:1 SECURITY LINKAGE
# ----------------------------------------------------------------------------
def load_clean_link():
    print("[1/6] Linking 1:1 PERMCO <-> PERMNO universe mapping...")
    m_ids = pd.read_parquet(MONTHLY_PATH, columns=['PERMCO', 'PERMNO']).drop_duplicates()
    valid_permcos = m_ids.groupby('PERMCO')['PERMNO'].nunique()[lambda s: s == 1].index
    valid_permnos = m_ids.groupby('PERMNO')['PERMCO'].nunique()[lambda s: s == 1].index
    clean_map = m_ids[m_ids['PERMCO'].isin(valid_permcos) & m_ids['PERMNO'].isin(valid_permnos)].copy()

    link = pd.read_csv(LINK_PATH)
    lm = _lower_map(link.columns)
    link_df = pd.DataFrame({
        'GVKEY': pd.to_numeric(link[lm['gvkey']], errors='coerce'),
        'PERMCO': pd.to_numeric(link[lm['permco']], errors='coerce')
    }).dropna().astype('int64').drop_duplicates()

    return link_df.merge(clean_map, on='PERMCO', how='inner')

# ----------------------------------------------------------------------------
# 4. COMPUTE FACTOR MATRIX & REGIONAL RANK TARGETS
# ----------------------------------------------------------------------------
def build_multi_sic_dataset(link_clean):
    print("[2/6] Loading multi-sector fundamentals and pricing tables...")
    cols = {'gvkey', 'datadate', 'sic', 'at', 'dltt', 'dlc', 'capx', 'oancf', 'ib', 'ebit', 'revt', 'gp', 'csho', 'prcc_f'}
    fund = pd.read_csv(FUND_PATH, usecols=lambda c: c.lower() in cols).rename(columns=str.upper)
    fund['DATADATE'] = pd.to_datetime(fund['DATADATE'])
    fund['GVKEY'] = pd.to_numeric(fund['GVKEY'], errors='coerce')
    fund = fund.dropna(subset=['GVKEY', 'AT', 'DATADATE', 'PRCC_F', 'CSHO', 'SIC'])
    fund['GVKEY'] = fund['GVKEY'].astype('int64')

    fund['SIC_NUM'] = pd.to_numeric(fund['SIC'], errors='coerce')
    fund['SIC_REGION'] = fund['SIC_NUM'].apply(assign_sic_region)
    fund = fund.dropna(subset=['SIC_REGION'])
    fund = fund[fund['PRCC_F'].abs() >= MIN_PRICE]
    fund = fund.sort_values(['GVKEY', 'DATADATE']).drop_duplicates(['GVKEY', 'DATADATE'], keep='last')

    fund['MCAP'] = (fund['PRCC_F'].abs() * fund['CSHO']).replace(0, np.nan)
    fund = fund[fund['MCAP'] > 0]

    for c in ['DLC', 'DLTT', 'CAPX', 'REVT', 'GP', 'EBIT']:
        if c in fund.columns:
            fund[c] = fund[c].fillna(0.0)

    at = fund['AT'].replace(0, np.nan)
    revt = fund['REVT'].replace(0, np.nan)
    cfo = fund['OANCF'].fillna(fund['IB'])
    fcf = cfo - fund['CAPX']

    fund['GM'] = fund['GP'] / revt
    fcf_margin = (fcf / revt).clip(-0.60, 0.60)

    g = fund.groupby('GVKEY')
    rev_growth = g['REVT'].pct_change().clip(-0.50, 1.50)

    fund['F_RULE_OF_40'] = rev_growth + fcf_margin
    fund['F_CASH_ROA'] = cfo / at
    fund['F_DELTA_GROSS_MARGIN'] = g['GM'].diff().clip(-0.40, 0.40)
    fund['F_OPERATING_LEVERAGE'] = (g['EBIT'].pct_change() - rev_growth).clip(-2.0, 2.0)
    fund['F_CASH_CONVERSION'] = (cfo - fund['IB']) / at
    fund['F_INTERNAL_FINANCING'] = (cfo / (fund['CAPX'] + 0.01 * at)).clip(-2.0, 10.0)
    fund['F_NET_BUYBACK'] = -g['CSHO'].pct_change().clip(-0.5, 0.5)

    first_rows = ~g.cumcount().astype(bool)
    diff_cols = ['F_RULE_OF_40', 'F_DELTA_GROSS_MARGIN', 'F_OPERATING_LEVERAGE', 'F_NET_BUYBACK']
    fund.loc[first_rows, diff_cols] = np.nan
    fund['EFFECTIVE_DATE'] = fund['DATADATE'] + pd.Timedelta(days=FILING_LAG_DAYS)

    # Monthly Price Data
    print("      Processing monthly price panel & delisting adjustments...")
    use_cols = ['PERMNO', 'PERMCO', 'MTHCALDT', 'MTHRET']
    m_prices = pd.read_parquet(MONTHLY_PATH, columns=use_cols)
    m_prices['MTHCALDT'] = pd.to_datetime(m_prices['MTHCALDT'])
    m_prices['PERMCO'] = pd.to_numeric(m_prices['PERMCO'], errors='coerce').astype('int64')
    m_prices = m_prices[m_prices['PERMCO'].isin(set(link_clean['PERMCO']))].copy()

    # Apply delisting return without -35% artificial penalty
    if os.path.exists(DELIST_PATH):
        delist = pd.read_csv(DELIST_PATH)
        delist['DelistingDt'] = pd.to_datetime(delist['DelistingDt'])
        delist['MTHCALDT'] = delist['DelistingDt'] + pd.offsets.MonthEnd(0)
        delist_m = delist.merge(link_clean[['PERMNO', 'PERMCO']].drop_duplicates(), on='PERMNO', how='inner')

        # Use recorded DelRet; fill missing values with 0.0 (no arbitrary -35% haircut)
        delist_m['DelRet_Clean'] = pd.to_numeric(delist_m['DelRet'], errors='coerce').fillna(0.0)

        delist_map = delist_m[['PERMCO', 'MTHCALDT', 'DelRet_Clean']].drop_duplicates(['PERMCO', 'MTHCALDT'])
        m_prices = m_prices.merge(delist_map, on=['PERMCO', 'MTHCALDT'], how='left')

        has_d = m_prices['DelRet_Clean'].notna()
        m_prices['MTHRET'] = m_prices['MTHRET'].fillna(0.0)
        m_prices.loc[has_d, 'MTHRET'] = (
            (1.0 + m_prices.loc[has_d, 'MTHRET']) * (1.0 + m_prices.loc[has_d, 'DelRet_Clean']) - 1.0
        ).clip(lower=-1.0)
        m_prices.drop(columns=['DelRet_Clean'], inplace=True)

    m_prices = m_prices.drop_duplicates(['PERMCO', 'MTHCALDT']).sort_values(['PERMCO', 'MTHCALDT']).reset_index(drop=True)
    m_prices['LOG_RET'] = np.log1p(m_prices['MTHRET'].clip(lower=-0.99))

    grp_p = m_prices.groupby('PERMCO')

    mom_12 = grp_p['LOG_RET'].transform(lambda s: s.rolling(12, min_periods=10).sum())
    m_prices['F_MOM_12_2'] = mom_12 - m_prices['LOG_RET']
    vol_12 = grp_p['MTHRET'].transform(lambda s: s.rolling(12, min_periods=8).std()).clip(lower=0.01)
    m_prices['F_LOW_VOLATILITY'] = -vol_12

    # Forward 12M Return Target
    print("      Calculating forward 12M returns...")
    m_prices['FWD_12M_RET'] = grp_p['MTHRET'].transform(
        lambda s: compute_forward_cumret(s, horizon=FORWARD_HORIZON)
    )

    f_perm = fund.merge(link_clean[['GVKEY', 'PERMCO']], on='GVKEY', how='inner')
    f_perm = f_perm.sort_values(['PERMCO', 'EFFECTIVE_DATE']).drop_duplicates(['PERMCO', 'EFFECTIVE_DATE'], keep='last')

    merged = pd.merge_asof(
        m_prices.sort_values('MTHCALDT'),
        f_perm.sort_values('EFFECTIVE_DATE'),
        left_on='MTHCALDT',
        right_on='EFFECTIVE_DATE',
        by='PERMCO',
        direction='backward',
        tolerance=pd.Timedelta(days=STALE_TOLERANCE_DAYS)
    ).dropna(subset=['MCAP', 'SIC_REGION'])

    universe = merged[merged['MCAP'] >= 150].copy()

    # Cross-sectional ranks per region per month
    print("      Normalizing factor ranks & targets within each SIC region...")
    rank_cols = []
    for f in FEATURE_COLS:
        r_name = f'{f}_RK'
        universe[r_name] = universe.groupby(['MTHCALDT', 'SIC_REGION'])[f].rank(pct=True).fillna(0.5) - 0.5
        rank_cols.append(r_name)

    universe['ALPHA_TARGET'] = universe.groupby(['MTHCALDT', 'SIC_REGION'])['FWD_12M_RET'].rank(pct=True) - 0.5

    # Return universe without dropping ALPHA_TARGET NaN so scoring sees all alive stocks
    return universe, m_prices, rank_cols

# ----------------------------------------------------------------------------
# 5. WALK-FORWARD RIDGE MODEL PER SIC REGION (ONCE A YEAR, NO LOOK-AHEAD)
# ----------------------------------------------------------------------------
def run_all_regions_ridge(dataset, rank_cols):
    """Once a year (at the end of REBALANCE_MONTH) each region's Ridge model is refit
    on data whose 12M outcome is fully known, then used to score every live stock."""
    print("[3/6] Annual Ridge refit + scoring per SIC region...")
    all_predictions, coef_log = [], []
    regions = sorted(dataset['SIC_REGION'].unique())

    for idx, reg in enumerate(regions, 1):
        sub_df = dataset[dataset['SIC_REGION'] == reg]
        dates = np.sort(sub_df['MTHCALDT'].unique())
        if len(dates) <= MIN_TRAIN_MONTHS:
            print(f"  [{idx}/{len(regions)}] {reg}: skipped (only {len(dates)} months of data)")
            continue

        # Formation dates: one per year, after the minimum history requirement
        form_dates = [d for d in dates[MIN_TRAIN_MONTHS:] if pd.Timestamp(d).month == REBALANCE_MONTH]
        reg_preds = []

        for form_date in form_dates:
            form_dt = pd.Timestamp(form_date)

            # Train only on rows whose forward 12M return had fully played out by form_dt
            train_cutoff = form_dt - pd.DateOffset(months=EMBARGO_MONTHS)
            train_start = train_cutoff - pd.DateOffset(months=TRAIN_WINDOW_MONTHS)
            train_pool = sub_df[(sub_df['MTHCALDT'] >= train_start) &
                                (sub_df['MTHCALDT'] <= train_cutoff)].dropna(subset=rank_cols + ['ALPHA_TARGET'])
            if len(train_pool) < 150:
                continue

            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, random_state=42)
            model.fit(train_pool[rank_cols].values, train_pool['ALPHA_TARGET'].values)
            coef_log.append({'SIC_REGION': reg, 'REFIT_DATE': form_dt, 'N_TRAIN': len(train_pool),
                             **dict(zip(FEATURE_COLS, model.coef_))})

            # Score every stock alive on the formation date (no target required)
            live = sub_df[sub_df['MTHCALDT'] == form_date].dropna(subset=rank_cols).copy()
            if len(live) >= 10:
                live['ALPHA_SCORE'] = model.predict(live[rank_cols].values)
                reg_preds.append(live[['MTHCALDT', 'PERMCO', 'SIC_REGION', 'ALPHA_SCORE']])

        if reg_preds:
            reg_out = pd.concat(reg_preds, ignore_index=True)
            all_predictions.append(reg_out)
            print(f"  [{idx}/{len(regions)}] {reg}: {len(reg_preds)} annual portfolios, {len(reg_out):,} stock-years scored")
        else:
            print(f"  [{idx}/{len(regions)}] {reg}: not enough data to train")

    if not all_predictions:
        raise RuntimeError("No predictions could be generated across the SIC regions.")

    master_scored = pd.concat(all_predictions, ignore_index=True)

    print("\n[4/6] Sorting all scored stocks into quintiles each year (Q1 = best)...")
    master_scored['QUINTILE'] = master_scored.groupby('MTHCALDT')['ALPHA_SCORE'].transform(
        lambda s: pd.qcut(s.rank(method='first'), q=5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']).astype(object)
        if len(s) >= 25 else pd.Series(np.nan, index=s.index)
    )
    master_scored = master_scored.dropna(subset=['QUINTILE'])
    master_scored['QUINTILE'] = master_scored['QUINTILE'].astype(str)
    return master_scored, pd.DataFrame(coef_log)

# ----------------------------------------------------------------------------
# 6. ANNUAL BUY-AND-HOLD BACKTEST
# ----------------------------------------------------------------------------
def run_quintile_backtest(scored_df, m_prices, slippage=SLIPPAGE_ONE_WAY):
    """Once a year: buy each quintile equal-weight, hold untouched for 12 months, sell, repeat.
    - Weights drift with prices during the year (true buy-and-hold, no monthly rebalancing).
    - A delisted stock's final value is spread across the survivors in proportion to their value.
    - Costs: the whole book is bought at `slippage` in month 1 and sold at `slippage` in month 12.
    - Benchmark: the same annual buy-and-hold applied to ALL scored stocks, before costs.
    Returns (quintile_series, benchmark, cost_engine, turnover) where cost_engine(slip)
    re-prices the quintiles at any other one-way cost without re-running the simulation."""
    print("      Simulating annual buy-and-hold portfolios...")
    all_dates = np.sort(m_prices['MTHCALDT'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    max_idx = len(all_dates) - 1

    h = scored_df[['MTHCALDT', 'PERMCO', 'QUINTILE']].copy()
    h['START_IDX'] = h['MTHCALDT'].map(date_to_idx)

    # One row per (stock, holding month 1..12)
    offsets = np.arange(1, FORWARD_HORIZON + 1)
    idx = h['START_IDX'].values[:, None] + offsets
    ok = (idx <= max_idx).ravel()
    exp = pd.DataFrame({
        'COHORT': np.repeat(h['MTHCALDT'].values, FORWARD_HORIZON)[ok],
        'PERMCO': np.repeat(h['PERMCO'].values, FORWARD_HORIZON)[ok],
        'QUINTILE': np.repeat(h['QUINTILE'].values, FORWARD_HORIZON)[ok],
        'OFFSET': np.tile(offsets, len(h))[ok],
        'HOLD_DT': all_dates[np.clip(idx.ravel(), 0, max_idx)][ok],
    })
    px = m_prices[['PERMCO', 'MTHCALDT', 'MTHRET']].rename(columns={'MTHCALDT': 'HOLD_DT'})
    exp = exp.merge(px, on=['PERMCO', 'HOLD_DT'], how='inner')   # delisted stocks drop out

    # Buy-and-hold weight = value of $1 invested at formation, as of the start of each month
    exp = exp.sort_values(['COHORT', 'QUINTILE', 'PERMCO', 'OFFSET'])
    growth = (1 + exp['MTHRET']).groupby([exp['COHORT'], exp['QUINTILE'], exp['PERMCO']]).cumprod()
    exp['W'] = (growth / (1 + exp['MTHRET'])).values
    exp['WR'] = exp['W'] * exp['MTHRET']

    def portfolio_returns(df):
        g = df.groupby(['HOLD_DT'])[['WR', 'W']].sum()
        return (g['WR'] / g['W']).sort_index()

    gross = {q: portfolio_returns(exp[exp['QUINTILE'] == q]) for q in BUCKETS}
    benchmark = portfolio_returns(exp).rename('BENCH')

    common = benchmark.index
    for q in BUCKETS:
        common = common.intersection(gross[q].index)
    common = common.sort_values()
    benchmark = benchmark.loc[common]

    # Month-in-holding-year for each calendar month (1 = buy month, 12 = sell month)
    hold_month = exp.groupby('HOLD_DT')['OFFSET'].max().reindex(common)
    trades = ((hold_month == 1).astype(int) + (hold_month == FORWARD_HORIZON).astype(int))

    def cost_engine(slip):
        out = {q: (gross[q].loc[common] - slip * trades).rename(q) for q in BUCKETS}
        # Long/short pays costs on BOTH legs
        out['Q1_minus_Q5'] = (gross['Q1'].loc[common] - gross['Q5'].loc[common] - 2 * slip * trades).rename('Q1_minus_Q5')
        return out

    # Turnover: share of each quintile that is new vs. last year's picks
    turnover = {}
    for q in ['Q1', 'Q5']:
        sets = scored_df[scored_df['QUINTILE'] == q].groupby('MTHCALDT')['PERMCO'].apply(set).sort_index()
        turnover[q] = pd.Series([np.nan] + [1 - len(sets.iloc[i] & sets.iloc[i - 1]) / max(len(sets.iloc[i]), 1)
                                            for i in range(1, len(sets))], index=sets.index)
    return cost_engine(slippage), benchmark, cost_engine, pd.DataFrame(turnover)

# ----------------------------------------------------------------------------
# 7. PERFORMANCE ANALYTICS
# ----------------------------------------------------------------------------
def drawdown_series(r):
    cum = (1 + r.fillna(0)).cumprod()
    return cum / cum.cummax() - 1

def drawdown_table(r, top=5):
    """Top-N drawdown episodes: peak, trough, recovery, depth, lengths (months)."""
    cum = (1 + r.fillna(0)).cumprod()
    dd = cum / cum.cummax() - 1
    episodes, in_dd, start = [], False, None
    for dt, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, dt
        elif v == 0 and in_dd:
            seg = dd.loc[start:dt]
            episodes.append((start, seg.idxmin(), dt, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), pd.NaT, seg.min()))
    rows = []
    idx = list(dd.index)
    for s, t, e, depth in sorted(episodes, key=lambda x: x[3])[:top]:
        peak = idx[max(idx.index(s) - 1, 0)]
        rows.append({
            'Peak': peak.strftime('%Y-%m'), 'Trough': t.strftime('%Y-%m'),
            'Recovery': e.strftime('%Y-%m') if pd.notna(e) else 'Not recovered',
            'Depth': depth,
            'Peak->Trough (m)': idx.index(t) - idx.index(peak),
            'Total Length (m)': (idx.index(e) if pd.notna(e) else len(idx) - 1) - idx.index(peak),
        })
    return pd.DataFrame(rows)

def perf_stats(r, bench=None, rf=None, long_short=False):
    """Comprehensive monthly-return statistics. `rf` = monthly risk-free Series (or None).
    Long/short (self-financing) portfolios are not charged the risk-free rate."""
    r = r.dropna()
    n = len(r)
    if n < 12:
        return {}
    rf_m = pd.Series(0.0, index=r.index) if (rf is None or long_short) else rf.reindex(r.index).fillna(0.0)
    ex = r - rf_m
    sq12 = np.sqrt(12)

    cum = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (12.0 / n) - 1
    ann_ret = r.mean() * 12
    vol = r.std(ddof=1) * sq12
    sr_m = ex.mean() / ex.std(ddof=1) if ex.std(ddof=1) > 0 else np.nan
    sharpe = sr_m * sq12
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * sq12
    sortino = ex.mean() * 12 / downside if downside > 0 else np.nan
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    # longest underwater stretch
    uw = (dd < 0).astype(int)
    longest_uw = int(uw.groupby((uw == 0).cumsum()).sum().max()) if uw.any() else 0
    skew = sps.skew(r, bias=False)
    kurt = sps.kurtosis(r, bias=False)             # excess kurtosis
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    # Lo (2002) iid Sharpe SE (annualised) and Bailey/Lopez de Prado Probabilistic Sharpe Ratio vs 0
    sr_se = np.sqrt((1 + 0.5 * sr_m ** 2) / n) * sq12
    psr_den = np.sqrt(max(1e-12, 1 - skew * sr_m + (kurt + 2) / 4.0 * sr_m ** 2))
    psr = sps.norm.cdf(sr_m * np.sqrt(n - 1) / psr_den)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()

    out = {
        'Months': n,
        'Ann. Return': ann_ret,
        'CAGR': cagr,
        'Ann. Vol': vol,
        'Sharpe': sharpe,
        'Sharpe 95% CI Lo': sharpe - 1.96 * sr_se,
        'Sharpe 95% CI Hi': sharpe + 1.96 * sr_se,
        'Prob. Sharpe > 0': psr,
        'Sortino': sortino,
        'Calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        'Max DD': mdd,
        'Longest DD (m)': longest_uw,
        'Hit Rate': (r > 0).mean(),
        'Profit Factor': gains / losses if losses > 0 else np.nan,
        'Best Month': r.max(),
        'Worst Month': r.min(),
        'Skew': skew,
        'Excess Kurt': kurt,
        'VaR 95% (m)': var95,
        'CVaR 95% (m)': cvar95,
        'NW t(mean)': newey_west_tstat(r, lags=NW_LAGS),
        'Growth of $1': cum.iloc[-1],
    }

    if bench is not None:
        b = bench.reindex(r.index)
        m = b.notna()
        rr, bb = r[m], b[m]
        rf_b = rf_m[m] if not long_short else pd.Series(0.0, index=rr.index)
        coefs, tstats, r2 = hac_ols((rr - rf_b).values, (bb - (rf.reindex(bb.index).fillna(0.0) if rf is not None else 0.0)).values.reshape(-1, 1), lags=NW_LAGS)
        active = rr - bb
        te = active.std(ddof=1) * sq12
        up, dn = bb > 0, bb < 0
        out.update({
            'Beta': coefs[1],
            'Alpha (ann.)': coefs[0] * 12,
            'Alpha NW t': tstats[0],
            'Correlation': rr.corr(bb),
            'Tracking Error': te,
            'Info Ratio': (active.mean() * 12 / te if te > 0 else np.nan) if not long_short else np.nan,
            'Up Capture': rr[up].mean() / bb[up].mean() if up.any() else np.nan,
            'Down Capture': rr[dn].mean() / bb[dn].mean() if dn.any() else np.nan,
        })
    return out

def build_stats_table(q_series, benchmark, rf=None):
    series = {q: q_series[q] for q in BUCKETS}
    series['BENCH'] = benchmark
    series['Q1_minus_Q5'] = q_series['Q1_minus_Q5']
    series['Q1_minus_BENCH'] = q_series['Q1'] - benchmark
    rows = {}
    for k, s in series.items():
        ls = k in ('Q1_minus_Q5', 'Q1_minus_BENCH')
        rows[BUCKET_LABELS[k]] = perf_stats(s, bench=None if k == 'BENCH' else benchmark, rf=rf, long_short=ls)
    return pd.DataFrame(rows).T

def monotonicity_stats(q_series):
    means = np.array([q_series[q].mean() for q in BUCKETS])
    rho, p = sps.spearmanr([5, 4, 3, 2, 1], means)
    diffs = np.diff(means[::-1])          # Q5->Q4->...->Q1 should be increasing
    return {'Spearman(rank, mean ret)': rho, 'p-value': p,
            'Monotonic steps (of 4)': int((diffs > 0).sum())}

def calendar_year_table(q_series, benchmark):
    df = pd.DataFrame({**{q: q_series[q] for q in BUCKETS}, 'Bench': benchmark,
                       'Q1-Q5': q_series['Q1_minus_Q5']})
    return df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)

def subperiod_table(q_series, benchmark, rf=None):
    df = pd.DataFrame({**{q: q_series[q] for q in BUCKETS}, 'Bench': benchmark,
                       'Q1-Q5': q_series['Q1_minus_Q5']})
    decade = (df.index.year // 10) * 10
    rows = []
    for d, g in df.groupby(decade):
        if len(g) < 12:
            continue
        row = {'Period': f'{d}s', 'Months': len(g)}
        for c in g.columns:
            s = perf_stats(g[c], rf=rf, long_short=(c == 'Q1-Q5'))
            row[f'{c} Ret'] = s['Ann. Return']
            row[f'{c} SR'] = s['Sharpe']
        rows.append(row)
    return pd.DataFrame(rows).set_index('Period')

def market_regime_table(q_series, benchmark):
    df = pd.DataFrame({**{q: q_series[q] for q in BUCKETS}, 'Q1-Q5': q_series['Q1_minus_Q5'],
                       'Bench': benchmark})
    terc = pd.qcut(benchmark, 3, labels=['Bear tercile', 'Middle tercile', 'Bull tercile'])
    out = df.groupby(terc, observed=False).mean() * 12
    out.index = out.index.astype(str)
    out.index.name = 'Regime'
    out.loc['Up months (Bench>0)'] = df[benchmark > 0].mean() * 12
    out.loc['Down months (Bench<0)'] = df[benchmark < 0].mean() * 12
    out['Months'] = list(terc.value_counts().reindex(out.index[:3]).values) + \
                    [(benchmark > 0).sum(), (benchmark < 0).sum()]
    return out

def cost_sensitivity_table(cost_engine, benchmark, rf=None):
    rows = []
    for bps in COST_GRID_BPS:
        qs = cost_engine(bps / 1e4)
        s1 = perf_stats(qs['Q1'], rf=rf)
        sls = perf_stats(qs['Q1_minus_Q5'], long_short=True)
        rows.append({'One-way cost (bps)': bps, 'Q1 Ann. Ret': s1['Ann. Return'], 'Q1 Sharpe': s1['Sharpe'],
                     'Q1 - Bench (ann.)': (qs['Q1'] - benchmark).mean() * 12,
                     'L/S Ann. Ret': sls['Ann. Return'], 'L/S Sharpe': sls['Sharpe'],
                     'L/S NW t': sls['NW t(mean)']})
    tbl = pd.DataFrame(rows).set_index('One-way cost (bps)')
    # break-even: one-way cost at which Q1 excess over benchmark hits zero (linear in cost)
    x = np.array(COST_GRID_BPS, dtype=float)
    y = tbl['Q1 - Bench (ann.)'].values
    slope = np.polyfit(x, y, 1)
    be = -slope[1] / slope[0] if slope[0] < 0 else np.nan
    return tbl, be

# ----------------------------------------------------------------------------
# 8. SIGNAL ANALYTICS (IC, COEFFICIENTS, CHURN, COMPOSITION)
# ----------------------------------------------------------------------------
def _xs_spearman(df, a, b, by, min_n=20):
    d = df.dropna(subset=[a, b])
    d = d.assign(_ra=d.groupby(by)[a].rank(), _rb=d.groupby(by)[b].rank())
    cnt = d.groupby(by)['_ra'].transform('size')
    d = d[cnt >= min_n]
    return d.groupby(by)[['_ra', '_rb']].apply(lambda g: g['_ra'].corr(g['_rb']))

def ic_summary(ic):
    """One IC per year; the 12M windows don't overlap, so a plain t-stat is valid."""
    ic = ic.dropna()
    sd = ic.std()
    return {'Mean IC': ic.mean(), 'IC Std': sd, 'ICIR': ic.mean() / sd if sd > 0 else np.nan,
            'IC > 0 (%)': (ic > 0).mean(), 't-stat': ic.mean() / sd * np.sqrt(len(ic)) if sd > 0 else np.nan,
            'Years': int(len(ic))}

def signal_analytics(master_scored, universe, rank_cols):
    print("      Computing rank ICs (model, per-region, per-feature)...")
    fwd = universe[['MTHCALDT', 'PERMCO', 'FWD_12M_RET'] + rank_cols]
    sc = master_scored.merge(fwd, on=['MTHCALDT', 'PERMCO'], how='left')

    ic = _xs_spearman(sc, 'ALPHA_SCORE', 'FWD_12M_RET', 'MTHCALDT').sort_index()
    ic_tbl = pd.DataFrame({'Ridge composite (pooled)': ic_summary(ic)}).T

    reg_ic = _xs_spearman(sc, 'ALPHA_SCORE', 'FWD_12M_RET', ['SIC_REGION', 'MTHCALDT'], min_n=10)
    reg_tbl = pd.DataFrame({reg: ic_summary(s.droplevel(0)) for reg, s in reg_ic.groupby(level=0)}).T

    feat_rows, feat_ic_series = {}, {}
    for f, rc in zip(FEATURE_COLS, rank_cols):
        s = _xs_spearman(sc, rc, 'FWD_12M_RET', 'MTHCALDT').sort_index()
        feat_ic_series[f] = s
        feat_rows[f] = ic_summary(s)
    feat_tbl = pd.DataFrame(feat_rows).T.sort_values('Mean IC', ascending=False)

    # IC decay: correlation of score with k-month-ahead 12M return? -> use score autocorrelation instead
    # Feature correlation (pooled, OOS period) of the within-region rank features
    feat_corr = sc[rank_cols].corr()
    feat_corr.index = feat_corr.columns = FEATURE_COLS

    breadth = master_scored.groupby(['MTHCALDT', 'QUINTILE']).size().unstack()[BUCKETS]

    comp = master_scored.groupby(['MTHCALDT', 'QUINTILE', 'SIC_REGION']).size()
    q1_comp = (comp.xs('Q1', level='QUINTILE').unstack(fill_value=0))
    q1_comp = q1_comp.div(q1_comp.sum(axis=1), axis=0)
    all_comp = master_scored.groupby(['MTHCALDT', 'SIC_REGION']).size().unstack(fill_value=0)
    all_comp = all_comp.div(all_comp.sum(axis=1), axis=0)
    tilt = pd.DataFrame({'Q1 weight': q1_comp.mean(), 'Universe weight': all_comp.mean()})
    tilt['Active tilt'] = tilt['Q1 weight'] - tilt['Universe weight']
    tilt = tilt.sort_values('Active tilt', ascending=False)

    return dict(ic=ic, ic_tbl=ic_tbl, reg_tbl=reg_tbl, feat_tbl=feat_tbl, feat_ic=feat_ic_series,
                feat_corr=feat_corr, breadth=breadth, q1_comp=q1_comp, tilt=tilt)

def factor_regression_table(q_series, benchmark, factors):
    if factors is None:
        return None
    fcols = [c for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'] if c in factors.columns]
    rows = {}
    series = {**{q: q_series[q] for q in BUCKETS}, 'Q1_minus_Q5': q_series['Q1_minus_Q5'], 'BENCH': benchmark}
    for k, s in series.items():
        d = pd.concat([s.rename('R'), factors], axis=1, join='inner').dropna()
        if len(d) < 36:
            continue
        y = d['R'] - (0 if k == 'Q1_minus_Q5' else d['RF'])
        b, t, r2 = hac_ols(y.values, d[fcols].values, lags=NW_LAGS)
        row = {'Alpha (ann.)': b[0] * 12, 'Alpha t': t[0]}
        for i, c in enumerate(fcols, 1):
            row[f'b_{c}'] = b[i]
            row[f't_{c}'] = t[i]
        row['R2'] = r2
        rows[BUCKET_LABELS[k]] = row
    return pd.DataFrame(rows).T

# ----------------------------------------------------------------------------
# 9. CONSOLE TABLES
# ----------------------------------------------------------------------------
def print_quintile_table(stats_tbl, title, mono=None):
    cols = [('Ann. Return', '{:>8.2%}'), ('CAGR', '{:>8.2%}'), ('Ann. Vol', '{:>8.2%}'),
            ('Sharpe', '{:>7.2f}'), ('Sortino', '{:>7.2f}'), ('Max DD', '{:>8.2%}'),
            ('Calmar', '{:>6.2f}'), ('Hit Rate', '{:>6.1%}'), ('NW t(mean)', '{:>6.2f}'),
            ('Alpha (ann.)', '{:>7.2%}'), ('Beta', '{:>5.2f}'), ('Growth of $1', '{:>8.2f}x')]
    heads = ['Ann.Ret', 'CAGR', 'Vol', 'Sharpe', 'Sortino', 'MaxDD', 'Calmar', 'Hit%', 'NW t', 'Alpha', 'Beta', 'Growth']
    widths = [8, 8, 8, 7, 7, 8, 6, 6, 6, 7, 5, 9]
    header = f"{'Bucket':<27}|" + '|'.join(f" {h:>{w}} " for h, w in zip(heads, widths))
    W = len(header)
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    print(header)
    print("-" * W)
    for i, (name, row) in enumerate(stats_tbl.iterrows()):
        if name == BUCKET_LABELS['BENCH']:
            print("-" * W)
        cells = []
        for (c, fmt), w in zip(cols, widths):
            v = row.get(c, np.nan)
            cells.append(f" {'--':>{w}} " if pd.isna(v) else f" {fmt.format(v)} ")
        print(f"{name:<27}|" + '|'.join(cells))
    print("=" * W)
    if mono:
        print(f"Monotonicity: Spearman(rank, mean return) = {mono['Spearman(rank, mean ret)']:.2f}  "
              f"| monotonic steps Q5->Q1: {mono['Monotonic steps (of 4)']}/4")
    ls = stats_tbl.loc[BUCKET_LABELS['Q1_minus_Q5']]
    print(f"Q1 - Q5 spread Newey-West t-stat ({NW_LAGS} lags): {ls['NW t(mean)']:.2f}  | "
          f"Sharpe 95% CI [{ls['Sharpe 95% CI Lo']:.2f}, {ls['Sharpe 95% CI Hi']:.2f}]  | "
          f"Prob. Sharpe>0: {ls['Prob. Sharpe > 0']:.1%}\n")

def print_df(df, title, fmt=None):
    print("\n" + title)
    print("-" * len(title))
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        d = df.copy()
        for c in d.columns:
            if str(c) in ('Months', 'Years', 'N', 'Peak->Trough (m)', 'Total Length (m)'):
                d[c] = d[c].astype('Int64')
        print(d.to_string(float_format=fmt or (lambda v: f'{v:,.4f}')))

# ----------------------------------------------------------------------------
# 10. MARKDOWN WRITER (no tabulate dependency)
# ----------------------------------------------------------------------------
PCT_COLS = {'Q1 turnover', 'Ann. Return', 'CAGR', 'Ann. Vol', 'Max DD', 'Hit Rate', 'Best Month', 'Worst Month',
            'VaR 95% (m)', 'CVaR 95% (m)', 'Alpha (ann.)', 'Tracking Error', 'Prob. Sharpe > 0',
            'Q1 weight', 'Universe weight', 'Active tilt', 'IC > 0 (%)', 'Depth',
            'Q1 Ann. Ret', 'Q1 - Bench (ann.)', 'L/S Ann. Ret'}

def fmt_cell(col, v):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return '--'
    c = str(col)
    if c in PCT_COLS or c.endswith(' Ret') or c in BUCKETS + ['Bench', 'Q1-Q5']:
        return f'{v:.2%}'
    if c in ('Months', 'Years', 'Longest DD (m)', 'N', 'Peak->Trough (m)', 'Total Length (m)') or (float(v).is_integer() and abs(v) > 100):
        return f'{int(v):,}'
    return f'{v:.2f}' if abs(v) >= 0.01 or v == 0 else f'{v:.4f}'

def df_to_md(df, index_name=''):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join([index_name or (df.index.name or '')] + cols) + ' |',
             '|' + '---|' + '|'.join(['---:'] * len(cols)) + '|']
    for idx, row in df.iterrows():
        lines.append('| ' + ' | '.join([str(idx)] + [fmt_cell(c, v) for c, v in zip(df.columns, row.values)]) + ' |')
    return '\n'.join(lines)

# ----------------------------------------------------------------------------
# 11. CHARTS
# ----------------------------------------------------------------------------
# Ordered diverging encoding: Q1 (strong blue) -> Q3 (neutral gray) -> Q5 (strong red)
QCOL = {'Q1': '#184f95', 'Q2': '#6da7ec', 'Q3': '#9a9994', 'Q4': '#e88a8a', 'Q5': '#b3261e'}
C_BENCH = '#0b0b0b'
C_LS = '#4a3aa7'
C_ACCENT = '#eb6834'
INK2 = '#52514e'

def _style():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
        'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.8,
        'legend.frameon': False, 'lines.linewidth': 1.6,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

def _save(fig, name, figdir):
    path = os.path.join(figdir, name)
    fig.savefig(path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path

def make_charts(q_series, benchmark, stats_tbl, sig, coefs, cal, cost_tbl, turnover, figdir):
    _style()
    figs = {}
    pct = PercentFormatter(1.0, decimals=0)

    # 1) Growth of $1, log scale
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for q in BUCKETS:
        cum = (1 + q_series[q].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, color=QCOL[q], lw=2.2 if q in ('Q1', 'Q5') else 1.3, label=BUCKET_LABELS[q])
    bc = (1 + benchmark.fillna(0)).cumprod()
    # direct-label only the long, short and benchmark ends; nudge apart so they never collide
    ends = sorted([(float((1 + q_series[q].fillna(0)).cumprod().iloc[-1]), q) for q in ('Q1', 'Q5')] +
                  [(float(bc.iloc[-1]), 'Bench')])
    placed = []
    for v, k in ends:
        y = np.log10(v)
        if placed and y - placed[-1] < 0.045:
            y = placed[-1] + 0.045
        placed.append(y)
        ax.annotate(f'{k} {v:,.1f}x', (bc.index[-1], 10 ** y), xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=INK2, annotation_clip=False)
    ax.plot(bc.index, bc.values, color=C_BENCH, ls='--', lw=1.4, label=BUCKET_LABELS['BENCH'])
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:,.0f}' if v >= 1 else f'${v:.2f}'))
    ax.set_title('Growth of $1 by model quintile (annual rebalance, log scale, net of 15 bps one-way costs)')
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    figs['growth'] = _save(fig, '01_growth_of_1_quintiles.png', figdir)

    # 2) Long/short cumulative + drawdown
    ls = q_series['Q1_minus_Q5']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    cum = (1 + ls.fillna(0)).cumprod()
    a1.plot(cum.index, cum.values, color=C_LS, lw=2)
    a1.set_title('Q1 - Q5 long/short spread: cumulative growth of $1')
    dd = drawdown_series(ls)
    a2.fill_between(dd.index, dd.values, 0, color=C_LS, alpha=0.25, lw=0)
    a2.plot(dd.index, dd.values, color=C_LS, lw=1)
    a2.yaxis.set_major_formatter(pct)
    a2.set_title('Drawdown', fontsize=10)
    figs['ls'] = _save(fig, '02_long_short_cumulative_drawdown.png', figdir)

    # 3) Bars: annualised return and Sharpe by quintile (two panels, one axis each)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rets = [stats_tbl.loc[BUCKET_LABELS[q], 'Ann. Return'] for q in BUCKETS]
    srs = [stats_tbl.loc[BUCKET_LABELS[q], 'Sharpe'] for q in BUCKETS]
    b_ret = stats_tbl.loc[BUCKET_LABELS['BENCH'], 'Ann. Return']
    b_sr = stats_tbl.loc[BUCKET_LABELS['BENCH'], 'Sharpe']
    for ax, vals, bval, ttl, f in [(a1, rets, b_ret, 'Annualised return', '{:.1%}'), (a2, srs, b_sr, 'Sharpe ratio', '{:.2f}')]:
        bars = ax.bar(BUCKETS, vals, color=[QCOL[q] for q in BUCKETS], width=0.62, edgecolor='white', linewidth=2)
        ax.axhline(bval, color=C_BENCH, ls='--', lw=1)
        ax.annotate('Benchmark', (4.35, bval), va='bottom', ha='right', fontsize=8.5, color=INK2)
        for b, v in zip(bars, vals):
            ax.annotate(f.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords='offset points', ha='center', fontsize=9, color='#0b0b0b')
        ax.set_title(ttl)
        ax.grid(axis='x', visible=False)
    a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    figs['bars'] = _save(fig, '03_return_and_sharpe_by_quintile.png', figdir)

    # 4) Rolling 36M Sharpe
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, c, lab in [('Q1', QCOL['Q1'], 'Q1'), ('Q5', QCOL['Q5'], 'Q5'), ('Q1_minus_Q5', C_LS, 'Q1 - Q5')]:
        s = q_series[k]
        rs = s.rolling(ROLL_WIN).mean() / s.rolling(ROLL_WIN).std() * np.sqrt(12)
        ax.plot(rs.index, rs.values, color=c, lw=1.6, label=lab)
    rb = benchmark.rolling(ROLL_WIN).mean() / benchmark.rolling(ROLL_WIN).std() * np.sqrt(12)
    ax.plot(rb.index, rb.values, color=C_BENCH, ls='--', lw=1.1, label='Benchmark')
    ax.axhline(0, color='#b5b4ae', lw=1)
    ax.set_title(f'Rolling {ROLL_WIN}-month Sharpe ratio')
    ax.legend(ncol=4, loc='upper left')
    figs['roll_sharpe'] = _save(fig, '04_rolling_sharpe.png', figdir)

    # 5) Underwater curves
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, c, lab in [('Q1', QCOL['Q1'], 'Q1 long'), ('Q1_minus_Q5', C_LS, 'Q1 - Q5')]:
        d = drawdown_series(q_series[k])
        ax.plot(d.index, d.values, color=c, lw=1.4, label=lab)
    d = drawdown_series(benchmark)
    ax.plot(d.index, d.values, color=C_BENCH, ls='--', lw=1.1, label='Benchmark')
    ax.yaxis.set_major_formatter(pct)
    ax.set_title('Underwater curves (drawdown from running peak)')
    ax.legend(loc='lower left', ncol=3)
    figs['underwater'] = _save(fig, '05_drawdowns.png', figdir)

    # 6) Calendar-year heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, 0.26 * len(cal) + 1)))
    lim = np.nanpercentile(np.abs(cal.values), 95)
    im = ax.imshow(cal.values, aspect='auto', cmap='RdBu', vmin=-lim, vmax=lim)
    ax.set_xticks(range(cal.shape[1]))
    ax.set_xticklabels(cal.columns)
    ax.set_yticks(range(len(cal)))
    ax.set_yticklabels(cal.index, fontsize=8)
    ax.grid(False)
    for i in range(cal.shape[0]):
        for j in range(cal.shape[1]):
            v = cal.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > 0.6 * lim else '#0b0b0b')
    ax.set_title('Calendar-year returns')
    fig.colorbar(im, ax=ax, format=pct, shrink=0.6)
    figs['calendar'] = _save(fig, '06_calendar_year_heatmap.png', figdir)

    # 7) Monthly return heatmap of the L/S spread
    m = ls.to_frame('r')
    m['Y'], m['M'] = m.index.year, m.index.month
    piv = m.pivot_table(index='Y', columns='M', values='r')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.24 * len(piv) + 1)))
    lim = np.nanpercentile(np.abs(piv.values), 95)
    im = ax.imshow(piv.values, aspect='auto', cmap='RdBu', vmin=-lim, vmax=lim)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.grid(False)
    ax.set_title('Q1 - Q5 monthly returns')
    fig.colorbar(im, ax=ax, format=PercentFormatter(1.0, decimals=1), shrink=0.6)
    figs['monthly_heat'] = _save(fig, '07_long_short_monthly_heatmap.png', figdir)

    # 8) Return distribution of L/S with normal overlay
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = ls.dropna()
    ax.hist(x, bins=50, density=True, color=C_LS, alpha=0.55, edgecolor='white', linewidth=0.8)
    grid = np.linspace(x.min(), x.max(), 300)
    ax.plot(grid, sps.norm.pdf(grid, x.mean(), x.std()), color=C_BENCH, lw=1.3, ls='--', label='Normal fit')
    ax.axvline(x.quantile(0.05), color=C_ACCENT, lw=1.2, label=f'5% VaR {x.quantile(0.05):.1%}')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f'Q1 - Q5 monthly return distribution (skew {sps.skew(x):.2f}, excess kurt {sps.kurtosis(x):.2f})')
    ax.legend()
    figs['dist'] = _save(fig, '08_long_short_distribution.png', figdir)

    # 9) Rolling IC
    ic = sig['ic']
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(ic.index, ic.values, width=250, color='#6da7ec', label='Rank IC (each annual portfolio)')
    ax.plot(ic.index, ic.rolling(5, min_periods=3).mean(), color='#184f95', lw=2, label='5-year average')
    ax.axhline(ic.mean(), color=C_ACCENT, lw=1.2, ls='--', label=f'Mean IC {ic.mean():.3f}')
    ax.axhline(0, color='#b5b4ae', lw=1)
    ax.set_title('Information coefficient: Spearman(score, forward 12M return)')
    ax.legend(ncol=3, loc='upper left')
    figs['ic'] = _save(fig, '09_rank_ic_timeseries.png', figdir)

    # 10) Per-feature mean IC (horizontal bars)
    ft = sig['feat_tbl'].sort_values('Mean IC')
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = ['#2a78d6' if v >= 0 else '#e34948' for v in ft['Mean IC']]
    ax.barh(ft.index, ft['Mean IC'], color=cols, height=0.6, edgecolor='white', linewidth=2)
    for i, (v, t) in enumerate(zip(ft['Mean IC'], ft['t-stat'])):
        ax.annotate(f'{v:.3f} (t={t:.1f})', (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords='offset points',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8.5, color=INK2)
    ax.axvline(0, color='#b5b4ae', lw=1)
    ax.grid(axis='y', visible=False)
    ax.set_title('Univariate rank IC of each within-region feature (OOS period)')
    figs['feat_ic'] = _save(fig, '10_feature_ic.png', figdir)

    # 11) Ridge coefficient heatmap (region x feature, mean across refits)
    if coefs is not None and len(coefs):
        cm = coefs.groupby('SIC_REGION')[FEATURE_COLS].mean()
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(cm) + 2))
        lim = np.abs(cm.values).max()
        im = ax.imshow(cm.values, aspect='auto', cmap='RdBu', vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(FEATURE_COLS)))
        ax.set_xticklabels([f.replace('F_', '') for f in FEATURE_COLS], rotation=30, ha='right', fontsize=8.5)
        ax.set_yticks(range(len(cm)))
        ax.set_yticklabels(cm.index, fontsize=8.5)
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm.values[i, j]:.2g}', ha='center', va='center', fontsize=7,
                        color='white' if abs(cm.values[i, j]) > 0.6 * lim else '#0b0b0b')
        ax.set_title('Average Ridge coefficient by SIC region (blue = positive loading)')
        fig.colorbar(im, ax=ax, shrink=0.7)
        figs['coef_heat'] = _save(fig, '11_ridge_coefficients_by_region.png', figdir)

        # 12) Coefficient drift over time (averaged across regions, refit dates grouped by year)
        cy = coefs.assign(Y=coefs['REFIT_DATE'].dt.year).groupby('Y')[FEATURE_COLS].mean()
        fig, ax = plt.subplots(figsize=(11, 4.8))
        pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', '#52514e']
        for f, c in zip(FEATURE_COLS, pal):
            ax.plot(cy.index, cy[f], color=c, lw=1.6, label=f.replace('F_', ''))
        ax.axhline(0, color='#b5b4ae', lw=1)
        ax.set_title('Ridge coefficient drift across annual refits (mean over regions)')
        ax.legend(ncol=3, fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.08))
        figs['coef_drift'] = _save(fig, '12_ridge_coefficient_drift.png', figdir)

    # 13) Transaction-cost sensitivity
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(cost_tbl.index, cost_tbl['Q1 - Bench (ann.)'], color=QCOL['Q1'], marker='o', ms=5, label='Q1 minus benchmark')
    a1.plot(cost_tbl.index, cost_tbl['L/S Ann. Ret'], color=C_LS, marker='o', ms=5, label='Q1 - Q5')
    a1.axhline(0, color='#b5b4ae', lw=1)
    a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    a1.set_title('Annualised excess return vs cost')
    a1.set_xlabel('One-way cost (bps)')
    a1.legend()
    a2.plot(cost_tbl.index, cost_tbl['Q1 Sharpe'], color=QCOL['Q1'], marker='o', ms=5, label='Q1 Sharpe')
    a2.plot(cost_tbl.index, cost_tbl['L/S Sharpe'], color=C_LS, marker='o', ms=5, label='Q1 - Q5 Sharpe')
    a2.set_title('Sharpe ratio vs cost')
    a2.set_xlabel('One-way cost (bps)')
    a2.legend()
    figs['cost'] = _save(fig, '13_cost_sensitivity.png', figdir)

    # 14) Long-leg sector composition over time
    qc = sig['q1_comp']
    comp = qc.groupby(qc.index.year).mean()          # annual average (pandas-version agnostic)
    comp = comp[comp.mean().sort_values(ascending=False).index]
    top = comp.columns[:6]
    comp_plot = comp[top].copy()
    comp_plot['Other'] = comp.drop(columns=top).sum(axis=1)
    pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#b5b4ae']
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.stackplot(comp_plot.index, comp_plot.T.values, labels=comp_plot.columns, colors=pal[:comp_plot.shape[1]],
                 edgecolor='white', linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title('Q1 (long leg) composition by SIC division, annual average')
    ax.legend(ncol=4, fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.08))
    figs['composition'] = _save(fig, '14_long_leg_sector_composition.png', figdir)

    # 15) Breadth and turnover
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    br = sig['breadth']
    a1.plot(br.index, br['Q1'], color=QCOL['Q1'], lw=1.6, label='Names per quintile')
    a1.set_title('Breadth: stocks per quintile at each annual rebalance')
    a2.plot(turnover.index, turnover['Q1'], color=QCOL['Q1'], lw=1.6, marker='o', ms=4, label='Q1')
    a2.plot(turnover.index, turnover['Q5'], color=QCOL['Q5'], lw=1.6, marker='o', ms=4, label='Q5')
    a2.set_ylim(0, 1)
    a2.yaxis.set_major_formatter(pct)
    a2.set_title("Turnover: share of this year's picks that are new vs last year")
    a2.legend(ncol=2)
    figs['breadth'] = _save(fig, '15_breadth_and_turnover.png', figdir)

    # 16) Risk / return scatter
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for k in BUCKETS + ['BENCH']:
        row = stats_tbl.loc[BUCKET_LABELS[k]]
        c = QCOL.get(k, C_BENCH)
        ax.scatter(row['Ann. Vol'], row['Ann. Return'], s=90, color=c, edgecolor='white', linewidth=2, zorder=3,
                   marker='D' if k == 'BENCH' else 'o')
        ax.annotate('Bench' if k == 'BENCH' else k, (row['Ann. Vol'], row['Ann. Return']), xytext=(7, 0),
                    textcoords='offset points', va='center', fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('Annualised volatility')
    ax.set_ylabel('Annualised return')
    ax.set_title('Risk vs return by quintile')
    figs['scatter'] = _save(fig, '16_risk_return_scatter.png', figdir)

    return figs

# ----------------------------------------------------------------------------
# 12. REPORT
# ----------------------------------------------------------------------------
def write_readme(path, sections, figs, figdir_rel):
    def img(key, alt):
        return f'![{alt}]({figdir_rel}/{os.path.basename(figs[key])})\n' if key in figs else ''
    s = sections
    md = []
    md.append('# All-SIC Ridge Rank Engine: Backtest & Analytics Report\n')
    md.append(f"_Out-of-sample period: **{s['start']} to {s['end']}** ({s['months']} months). "
              f"Generated {pd.Timestamp.today():%Y-%m-%d}._\n")
    md.append('## Headline\n')
    md.append(s['headline'] + '\n')
    md.append(img('growth', 'Growth of $1'))
    md.append('## Methodology\n')
    md.append(
        '- **Universe:** CRSP/Compustat common stocks with a clean 1:1 PERMCO-PERMNO-GVKEY link, price >= $5, '
        'market cap >= $150M, fundamentals lagged 90 days (stale after 400 days).\n'
        f'- **Features ({len(FEATURE_COLS)}):** ' + ', '.join(f'`{f}`' for f in FEATURE_COLS) +
        '; each converted to a cross-sectional percentile rank **within SIC division and month**.\n'
        f'- **Model:** one Ridge regression per SIC division (alpha = {RIDGE_ALPHA:,.0f}, no intercept), '
        f'target = within-division rank of forward 12M return. Refit every December on a rolling {TRAIN_WINDOW_MONTHS}-month '
        f'window, using only rows whose 12M outcome was already known ({EMBARGO_MONTHS}-month embargo); first prediction after {MIN_TRAIN_MONTHS} months of history.\n'
        '- **Portfolio:** once a year (end of December) every stock is scored and sorted into quintiles (Q1 = highest). '
        'Each quintile is bought equal-weight and held untouched for 12 months, then the process repeats. '
        'Delisted stocks drop out and their value is spread across the survivors. CRSP delisting returns used where present.\n'
        f'- **Costs:** {SLIPPAGE_ONE_WAY * 1e4:.0f} bps one-way on the annual buy and sell. '
        'Benchmark = the same annual equal-weight buy-and-hold of all scored stocks, before costs.\n'
        f"- **Risk-free:** {s['rf_note']}\n")
    md.append('## Performance by quintile (full sample)\n')
    md.append(df_to_md(s['main_short'], 'Bucket') + '\n')
    md.append(f"\n{s['mono_line']}\n")
    md.append(img('bars', 'Return and Sharpe by quintile'))
    md.append(img('scatter', 'Risk vs return'))
    if s.get('modern_short') is not None:
        md.append(f'## Modern regime ({MODERN_START[:4]}-present)\n')
        md.append(df_to_md(s['modern_short'], 'Bucket') + '\n')
    md.append('## Long/short spread\n')
    md.append(img('ls', 'Long/short'))
    md.append('### Worst drawdowns (Q1 - Q5)\n')
    md.append(df_to_md(s['dd_ls'].set_index('Peak'), 'Peak') + '\n')
    md.append(img('underwater', 'Drawdowns'))
    md.append(img('roll_sharpe', 'Rolling Sharpe'))
    md.append(img('dist', 'Return distribution'))
    md.append(img('monthly_heat', 'Monthly heatmap'))
    md.append('## Full risk statistics\n')
    md.append('<details><summary>Expand all statistics</summary>\n\n' + df_to_md(s['main_full'].T, 'Statistic') + '\n\n</details>\n')
    if s.get('ff') is not None:
        md.append('## Fama-French 5-factor + Momentum attribution\n')
        md.append('Newey-West t-stats (12 lags). Long-only buckets regressed in excess of RF; the spread is self-financing.\n\n')
        md.append(df_to_md(s['ff'], 'Bucket') + '\n')
    md.append('## Signal quality\n')
    md.append('### Rank information coefficient (score vs. realised forward 12M return)\n')
    md.append('One IC per annual portfolio; the 12-month windows do not overlap, so these are independent observations.\n\n')
    md.append(df_to_md(s['ic_tbl'], 'Signal') + '\n')
    md.append(img('ic', 'Rank IC'))
    md.append('### IC by SIC division\n')
    md.append(df_to_md(s['reg_tbl'], 'SIC division') + '\n')
    md.append('### Univariate feature ICs\n')
    md.append(df_to_md(s['feat_tbl'], 'Feature') + '\n')
    md.append(img('feat_ic', 'Feature IC'))
    md.append('### Model coefficients\n')
    md.append(img('coef_heat', 'Ridge coefficients'))
    md.append(img('coef_drift', 'Coefficient drift'))
    md.append('## Robustness\n')
    md.append('### Transaction-cost sensitivity\n')
    md.append(df_to_md(s['cost_tbl'], 'One-way cost (bps)') + '\n')
    be = s['breakeven']
    md.append(f"\nBreak-even one-way cost for Q1 to match the benchmark: **{be:,.0f} bps**.\n" if pd.notna(be) else '')
    md.append(img('cost', 'Cost sensitivity'))
    md.append('### By decade\n')
    md.append(df_to_md(s['decade'], 'Period') + '\n')
    md.append('### Market regimes (annualised mean return)\n')
    md.append(df_to_md(s['regime'], 'Regime') + '\n')
    md.append('### Calendar-year returns\n')
    md.append('<details><summary>Expand table</summary>\n\n' + df_to_md(s['cal'], 'Year') + '\n\n</details>\n')
    md.append(img('calendar', 'Calendar-year heatmap'))
    md.append('## Portfolio construction diagnostics\n')
    md.append('### Long-leg sector tilt vs scored universe\n')
    md.append(df_to_md(s['tilt'], 'SIC division') + '\n')
    md.append(img('composition', 'Sector composition'))
    md.append(img('breadth', 'Breadth and turnover'))
    md.append('\n---\n*Backtest results are hypothetical, computed on historical data, and do not represent live trading.*\n')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))

def report(q_series, benchmark, cost_engine, turnover, master_scored, universe, rank_cols, coefs):
    print("\n[5/6] Generating statistical report...")
    tabdir = os.path.join(OUTPUT_DIR, 'tables')
    figdir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(tabdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    factors = load_factors()
    rf = factors['RF'] if factors is not None and 'RF' in factors.columns else None
    if rf is None and RF_ANNUAL_FALLBACK:
        rf = pd.Series((1 + RF_ANNUAL_FALLBACK) ** (1 / 12) - 1, index=benchmark.index)
    rf_note = ('1M T-bill from Ken French library (Sharpe/Sortino are excess of RF).' if factors is not None
               else f'constant {RF_ANNUAL_FALLBACK:.2%} p.a. (add Ken French CSVs for T-bill excess returns).')

    # ---- Main tables
    main = build_stats_table(q_series, benchmark, rf)
    mono = monotonicity_stats(q_series)
    print_quintile_table(main, "FULL SAMPLE: AGGREGATE ALL-SIC RIDGE RANK ENGINE  (returns & Sharpe by bucket)", mono)

    modern = None
    mod_mask = benchmark.index >= MODERN_START
    if mod_mask.sum() > 24:
        mq = {k: v[mod_mask] for k, v in q_series.items()}
        modern = build_stats_table(mq, benchmark[mod_mask], rf)
        print_quintile_table(modern, f"MODERN REGIME: AGGREGATE ALL-SIC RIDGE ENGINE ({MODERN_START[:4]} - PRESENT)",
                             monotonicity_stats(mq))

    # ---- Secondary analytics
    sig = signal_analytics(master_scored, universe, rank_cols)
    cal = calendar_year_table(q_series, benchmark)
    dec = subperiod_table(q_series, benchmark, rf)
    regime = market_regime_table(q_series, benchmark)
    cost_tbl, breakeven = cost_sensitivity_table(cost_engine, benchmark, rf)
    dd_ls = drawdown_table(q_series['Q1_minus_Q5'], top=5)
    ff = factor_regression_table(q_series, benchmark, factors)
    coef_summary = None
    if coefs is not None and len(coefs):
        coef_summary = coefs.groupby('SIC_REGION')[FEATURE_COLS].agg(['mean', 'std'])

    pf = lambda v: f'{v:,.4f}'
    print_df(sig['ic_tbl'], 'RANK IC - RIDGE COMPOSITE vs FORWARD 12M RETURN', pf)
    print_df(sig['reg_tbl'], 'RANK IC BY SIC DIVISION', pf)
    print_df(sig['feat_tbl'], 'UNIVARIATE FEATURE RANK ICs', pf)
    print_df(cost_tbl, 'TRANSACTION-COST SENSITIVITY', pf)
    if pd.notna(breakeven):
        print(f"Break-even one-way cost (Q1 vs benchmark): {breakeven:,.0f} bps")
    print_df(dec, 'ANNUALISED RETURN & SHARPE BY DECADE', pf)
    print_df(regime, 'MARKET REGIME ANALYSIS (annualised mean returns)', pf)
    print_df(dd_ls.set_index('Peak'), 'TOP 5 DRAWDOWNS - Q1 minus Q5', pf)
    print_df(sig['tilt'], 'LONG-LEG SECTOR TILT vs SCORED UNIVERSE', pf)
    if ff is not None:
        print_df(ff, 'FAMA-FRENCH 5 + MOMENTUM ATTRIBUTION (NW t-stats)', pf)
    print(f"\nAverage yearly turnover (share of picks that are new vs last year): "
          f"Q1 {turnover['Q1'].mean():.1%} | Q5 {turnover['Q5'].mean():.1%}")

    # ---- Save CSVs
    main.to_csv(os.path.join(tabdir, 'performance_full_sample.csv'))
    if modern is not None:
        modern.to_csv(os.path.join(tabdir, 'performance_modern.csv'))
    pd.DataFrame(q_series).assign(BENCH=benchmark).to_csv(os.path.join(tabdir, 'monthly_returns.csv'))
    cal.to_csv(os.path.join(tabdir, 'calendar_year_returns.csv'))
    dec.to_csv(os.path.join(tabdir, 'decade_breakdown.csv'))
    regime.to_csv(os.path.join(tabdir, 'market_regimes.csv'))
    cost_tbl.to_csv(os.path.join(tabdir, 'cost_sensitivity.csv'))
    dd_ls.to_csv(os.path.join(tabdir, 'long_short_drawdowns.csv'), index=False)
    sig['ic'].rename('IC').to_csv(os.path.join(tabdir, 'annual_rank_ic.csv'))
    sig['ic_tbl'].to_csv(os.path.join(tabdir, 'ic_summary.csv'))
    sig['reg_tbl'].to_csv(os.path.join(tabdir, 'ic_by_region.csv'))
    sig['feat_tbl'].to_csv(os.path.join(tabdir, 'ic_by_feature.csv'))
    sig['feat_corr'].to_csv(os.path.join(tabdir, 'feature_rank_correlation.csv'))
    sig['tilt'].to_csv(os.path.join(tabdir, 'long_leg_sector_tilt.csv'))
    sig['breadth'].to_csv(os.path.join(tabdir, 'breadth.csv'))
    if coefs is not None and len(coefs):
        coefs.to_csv(os.path.join(tabdir, 'ridge_coefficients_all_refits.csv'), index=False)
        coef_summary.to_csv(os.path.join(tabdir, 'ridge_coefficients_summary.csv'))
    if ff is not None:
        ff.to_csv(os.path.join(tabdir, 'factor_attribution.csv'))

    # ---- Charts
    print("\n[6/6] Rendering charts and writing README report...")
    figs = make_charts(q_series, benchmark, main, sig, coefs, cal, cost_tbl, turnover, figdir)
    turnover.to_csv(os.path.join(tabdir, 'yearly_turnover.csv'))

    # ---- README
    short_cols = ['Ann. Return', 'CAGR', 'Ann. Vol', 'Sharpe', 'Sortino', 'Max DD', 'Calmar', 'Hit Rate',
                  'NW t(mean)', 'Alpha (ann.)', 'Beta', 'Info Ratio', 'Growth of $1']
    ls_row = main.loc[BUCKET_LABELS['Q1_minus_Q5']]
    q1_row = main.loc[BUCKET_LABELS['Q1']]
    b_row = main.loc[BUCKET_LABELS['BENCH']]
    headline = (
        f"| Metric | Q1 (long) | Benchmark | Q1 - Q5 |\n|---|---:|---:|---:|\n"
        f"| Annualised return | {q1_row['Ann. Return']:.2%} | {b_row['Ann. Return']:.2%} | {ls_row['Ann. Return']:.2%} |\n"
        f"| Sharpe ratio | {q1_row['Sharpe']:.2f} | {b_row['Sharpe']:.2f} | {ls_row['Sharpe']:.2f} |\n"
        f"| Max drawdown | {q1_row['Max DD']:.1%} | {b_row['Max DD']:.1%} | {ls_row['Max DD']:.1%} |\n"
        f"| Newey-West t (mean) | {q1_row['NW t(mean)']:.2f} | {b_row['NW t(mean)']:.2f} | {ls_row['NW t(mean)']:.2f} |\n"
        f"| Mean rank IC | | | {sig['ic_tbl'].iloc[0]['Mean IC']:.3f} (t = {sig['ic_tbl'].iloc[0]['t-stat']:.2f}) |\n")
    sections = dict(
        start=f'{benchmark.index[0]:%Y-%m}', end=f'{benchmark.index[-1]:%Y-%m}', months=len(benchmark),
        headline=headline, rf_note=rf_note,
        main_short=main[[c for c in short_cols if c in main.columns]],
        main_full=main,
        modern_short=modern[[c for c in short_cols if c in modern.columns]] if modern is not None else None,
        mono_line=(f"Monotonicity: Spearman correlation between quintile rank and mean return = "
                   f"**{mono['Spearman(rank, mean ret)']:.2f}**; {mono['Monotonic steps (of 4)']}/4 steps monotonic."),
        dd_ls=dd_ls, ff=ff, ic_tbl=sig['ic_tbl'], reg_tbl=sig['reg_tbl'], feat_tbl=sig['feat_tbl'],
        cost_tbl=cost_tbl, breakeven=breakeven, decade=dec, regime=regime, cal=cal, tilt=sig['tilt'])
    write_readme(os.path.join(OUTPUT_DIR, 'README.md'), sections, figs, 'figures')

    summary = {
        'period': [sections['start'], sections['end']],
        'full_sample': {k: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()} for k, row in main.iterrows()},
        'monotonicity': {k: float(v) for k, v in mono.items()},
        'ic': {k: float(v) for k, v in sig['ic_tbl'].iloc[0].items()},
        'breakeven_cost_bps': None if pd.isna(breakeven) else float(breakeven),
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDone. Report: {os.path.abspath(os.path.join(OUTPUT_DIR, 'README.md'))}")
    print(f"      Figures: {len(figs)} PNGs in {os.path.abspath(figdir)}")
    print(f"      Tables:  CSVs in {os.path.abspath(tabdir)}")

# ----------------------------------------------------------------------------
# 13. ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    if not SHOW_PLOTS:
        matplotlib.use('Agg')
    link_clean = load_clean_link()
    universe, m_prices, rank_cols = build_multi_sic_dataset(link_clean)
    master_scored, coefs = run_all_regions_ridge(universe, rank_cols)
    quintile_series, benchmark_series, cost_engine, turnover = run_quintile_backtest(master_scored, m_prices)
    report(quintile_series, benchmark_series, cost_engine, turnover,
           master_scored, universe, rank_cols, coefs)
