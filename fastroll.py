"""
ALL-SIC REGION RIDGE RANK-OPTIMIZATION ENGINE & VECTORIZED BACKTEST
- Model: Ridge Regression with High-Penalty Shrinkage (alpha=5000.0) per SIC Region.
- Delisting & Rebalancing Policy:
    1. No -35% penalty imputation; uses CRSP DelRet if present, else 0.0.
    2. If a stock delists inside the 12-month holding window, it drops out after its
       terminal month and remaining capital is automatically equal-weighted into
       the surviving names of that sleeve.
    3. Look-Ahead Bias Eliminated: ALPHA_TARGET NaNs dropped only for model fitting,
       while scoring operates on all active stocks alive at formation date T.
    4. Vectorized 12-month overlapping sleeve backtest with 15 bps slippage.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & PATHS
# ----------------------------------------------------------------------------
MONTHLY_PATH = r'C:\Users\user\Desktop\WRDS\prices\monthly_stock.parquet'
FUND_PATH = r'C:\Users\user\Desktop\WRDS\prices\annual_fundamentals.csv'
LINK_PATH = r'C:\Users\user\Desktop\WRDS\gvkey_and_permco.csv'
DELIST_PATH = r'C:\Users\user\Desktop\WRDS\prices\delisting_information.csv'

MIN_PRICE = 5.0
FILING_LAG_DAYS = 90
STALE_TOLERANCE_DAYS = 400

FORWARD_HORIZON = 12
EMBARGO_MONTHS = 13
MIN_TRAIN_MONTHS = 60
TRAIN_WINDOW_MONTHS = 120
RIDGE_ALPHA = 5000.0           # Strong shrinkage toward equal weighting
SLIPPAGE_ONE_WAY = 0.0015      # 15 bps one-way execution drag

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

# ----------------------------------------------------------------------------
# 3. 1:1 SECURITY LINKAGE
# ----------------------------------------------------------------------------
def load_clean_link():
    print("[1/5] Linking 1:1 PERMCO <-> PERMNO universe mapping...")
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
    print("[2/5] Loading multi-sector fundamentals and pricing tables...")
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

    agg_benchmark = universe.groupby('MTHCALDT')['MTHRET'].mean().sort_index().rename('AGGREGATE_BENCHMARK_RET')

    # Return universe without dropping ALPHA_TARGET NaN so scoring sees all alive stocks
    return universe, m_prices, agg_benchmark, rank_cols

# ----------------------------------------------------------------------------
# 5. WALK-FORWARD RIDGE MODEL PER SIC REGION (NO LOOK-AHEAD)
# ----------------------------------------------------------------------------
def run_all_regions_ridge(dataset, rank_cols):
    print("[3/5] Starting multi-region Ridge training and prediction loop...")
    all_predictions = []
    regions = sorted(dataset['SIC_REGION'].unique())

    for idx, reg in enumerate(regions, 1):
        print(f"\n---> [{idx}/{len(regions)}] Training Ridge Engine for Region: {reg}")
        sub_df = dataset[dataset['SIC_REGION'] == reg].copy()
        dates = np.sort(sub_df['MTHCALDT'].unique())
        start_idx = MIN_TRAIN_MONTHS

        if len(dates) <= start_idx:
            print(f"      [SKIP] Insufficient monthly observations ({len(dates)}) for {reg}.")
            continue

        current_model = None
        last_refit_year = None
        trained_refits = 0
        reg_preds = []

        for t_idx in range(start_idx, len(dates)):
            pred_date = dates[t_idx]
            pred_dt = pd.Timestamp(pred_date)

            # Annual Walk-Forward Refit with 13M Embargo
            if (last_refit_year != pred_dt.year) or (current_model is None):
                train_cutoff = pred_dt - pd.DateOffset(months=EMBARGO_MONTHS)
                train_start = train_cutoff - pd.DateOffset(months=TRAIN_WINDOW_MONTHS)

                # Training pool: Drop missing forward targets ONLY here
                train_pool = sub_df[
                    (sub_df['MTHCALDT'] >= train_start) &
                    (sub_df['MTHCALDT'] <= train_cutoff)
                ].dropna(subset=rank_cols + ['ALPHA_TARGET'])

                if len(train_pool) >= 150:
                    X_train = train_pool[rank_cols].values
                    y_train = train_pool['ALPHA_TARGET'].values

                    current_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, random_state=42)
                    current_model.fit(X_train, y_train)
                    last_refit_year = pred_dt.year
                    trained_refits += 1

            # Scoring pool: All active candidates on this date (no target requirement)
            test_candidates = sub_df[sub_df['MTHCALDT'] == pred_date].dropna(subset=rank_cols).copy()
            if current_model is not None and len(test_candidates) >= 10:
                X_test = test_candidates[rank_cols].values
                test_candidates['ALPHA_SCORE'] = current_model.predict(X_test)
                reg_preds.append(test_candidates[['MTHCALDT', 'PERMCO', 'SIC_REGION', 'ALPHA_SCORE']])

        if reg_preds:
            reg_out = pd.concat(reg_preds, ignore_index=True)
            all_predictions.append(reg_out)
            print(f"      [SUCCESS] Generated {len(reg_out):,} out-of-sample predictions across {trained_refits} annual refits.")
        else:
            print(f"      [NOTICE] Insufficient cross-sectional density for {reg}.")

    if not all_predictions:
        raise RuntimeError("No predictions could be generated across the SIC regions.")

    master_scored = pd.concat(all_predictions, ignore_index=True)

    print("\n[4/5] Pooling regional predictions and creating aggregate monthly quintiles (Q1 to Q5)...")
    master_scored['QUINTILE'] = master_scored.groupby('MTHCALDT')['ALPHA_SCORE'].transform(
        lambda s: pd.qcut(s, q=5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']) if len(s) >= 25 else np.nan
    )
    master_scored = master_scored.dropna(subset=['QUINTILE'])
    return master_scored

# ----------------------------------------------------------------------------
# 6. FAST VECTORIZED 12-MONTH OVERLAPPING SLEEVE BACKTEST
# ----------------------------------------------------------------------------
def run_quintile_backtest(scored_df, m_prices, agg_benchmark):
    print("      Simulating 12-month overlapping sleeves with survivor rebalancing...")
    all_dates = np.sort(m_prices['MTHCALDT'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    holdings = scored_df[['MTHCALDT', 'PERMCO', 'QUINTILE']].copy()
    holdings['START_IDX'] = holdings['MTHCALDT'].map(date_to_idx)

    offsets = np.arange(1, FORWARD_HORIZON + 1)
    holding_indices = holdings['START_IDX'].values[:, None] + offsets  # (N, 12)
    max_idx = len(all_dates) - 1
    valid_mask = holding_indices <= max_idx

    n_rows = len(holdings)
    cohort_dates = np.repeat(holdings['MTHCALDT'].values, FORWARD_HORIZON)
    permcos = np.repeat(holdings['PERMCO'].values, FORWARD_HORIZON)
    quintiles = np.repeat(holdings['QUINTILE'].values, FORWARD_HORIZON)
    flat_indices = np.clip(holding_indices.ravel(), 0, max_idx)
    flat_offsets = np.tile(offsets, n_rows)
    flat_valid = valid_mask.ravel()

    expanded = pd.DataFrame({
        'COHORT': cohort_dates[flat_valid],
        'HOLD_DT': all_dates[flat_indices[flat_valid]],
        'PERMCO': permcos[flat_valid],
        'QUINTILE': quintiles[flat_valid],
        'OFFSET': flat_offsets[flat_valid]
    })

    px_simple = m_prices[['PERMCO', 'MTHCALDT', 'MTHRET']].rename(columns={'MTHCALDT': 'HOLD_DT'})

    # Inner join automatically keeps a stock only for months it actively traded.
    # When a stock delists, it contributes its final month's return and drops out.
    expanded = expanded.merge(px_simple, on=['PERMCO', 'HOLD_DT'], how='inner')

    # Apply one-way slippage upon entry (OFFSET == 1)
    expanded.loc[expanded['OFFSET'] == 1, 'MTHRET'] -= SLIPPAGE_ONE_WAY

    # Apply one-way slippage upon exit (either at month 12 or last traded month if delisted earlier)
    max_offset_per_pick = expanded.groupby(['COHORT', 'PERMCO'])['OFFSET'].transform('max')
    expanded.loc[expanded['OFFSET'] == max_offset_per_pick, 'MTHRET'] -= SLIPPAGE_ONE_WAY

    # Computing mean over active names dynamically rebalances proceeds to remaining survivors
    sleeve_ret = expanded.groupby(['QUINTILE', 'COHORT', 'HOLD_DT'])['MTHRET'].mean().reset_index()

    # Average across all concurrent active sleeves on each calendar month
    quintile_series = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = sleeve_ret[sleeve_ret['QUINTILE'] == q]
        q_mth = sub.groupby('HOLD_DT')['MTHRET'].mean().sort_index()
        common_idx = q_mth.index.intersection(agg_benchmark.index)
        quintile_series[q] = q_mth.loc[common_idx]

    common_idx = quintile_series['Q1'].index.intersection(agg_benchmark.index)
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        quintile_series[q] = quintile_series[q].loc[common_idx]

    benchmark_series = agg_benchmark.loc[common_idx]
    quintile_series['Q1_minus_Q5'] = quintile_series['Q1'] - quintile_series['Q5']

    return quintile_series, benchmark_series

def annualize_monthly(monthly_ret):
    r = monthly_ret.dropna()
    if len(r) < 12:
        return dict(CAGR=np.nan, VOL=np.nan, SHARPE=np.nan, MDD=np.nan, MULT=np.nan)
    cum = (1 + r).cumprod()
    yrs = len(r) / 12.0
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    dd = (cum / cum.cummax() - 1).min()
    return dict(CAGR=cagr, VOL=vol, SHARPE=cagr / vol if vol > 0 else np.nan, MDD=dd, MULT=cum.iloc[-1])

def print_quintile_table(q_series, benchmark, title):
    W = 88
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    print(f"{'Quintile / Basket':<26} | {'CAGR':>8} | {'Vol':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Growth':>8}")
    print("-" * W)

    order = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    labels = {
        'Q1': 'Q1 (Top 20% - Long)',
        'Q2': 'Q2 (60-80%)',
        'Q3': 'Q3 (40-60% Median)',
        'Q4': 'Q4 (20-40%)',
        'Q5': 'Q5 (Bottom 20% - Short)'
    }

    for q in order:
        s = annualize_monthly(q_series[q])
        print(f"{labels[q]:<26} | {s['CAGR']:>7.2%} | {s['VOL']:>7.2%} | {s['SHARPE']:>7.2f} | {s['MDD']:>7.2%} | {s['MULT']:>7.2f}x")

    print("-" * W)
    bench_stats = annualize_monthly(benchmark)
    print(f"{'Aggregate Universe Benchmark':<26} | {bench_stats['CAGR']:>7.2%} | {bench_stats['VOL']:>7.2%} | {bench_stats['SHARPE']:>7.2f} | {bench_stats['MDD']:>7.2%} | {bench_stats['MULT']:>7.2f}x")

    spread = q_series['Q1_minus_Q5']
    spr_mean = spread.mean() * 12
    spr_vol = spread.std() * np.sqrt(12)
    spr_sharpe = spr_mean / spr_vol if spr_vol > 0 else np.nan
    tstat = newey_west_tstat(spread, lags=12)
    print(f"{'Q1 - Q5 Long/Short':<26} | {spr_mean:>7.2%} | {spr_vol:>7.2%} | {spr_sharpe:>7.2f} | {'--':>8} | {'--':>8}")
    print("=" * W)
    print(f"Aggregate Q1 - Q5 Spread Newey-West t-stat (12 lags): {tstat:>6.2f}\n")

def report(q_series, benchmark):
    print("\n[5/5] Generating final aggregate statistical report and plots...")
    print_quintile_table(q_series, benchmark, "FULL SAMPLE: AGGREGATE ALL-SIC RIDGE RANK ENGINE")

    mod_mask = benchmark.index >= '2005-01-01'
    if mod_mask.sum() > 24:
        mod_q_series = {k: v[mod_mask] for k, v in q_series.items()}
        print_quintile_table(mod_q_series, benchmark[mod_mask], "MODERN REGIME: AGGREGATE ALL-SIC RIDGE ENGINE (2005 - PRESENT)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    colors = {'Q1': 'darkgreen', 'Q2': 'forestgreen', 'Q3': 'gray', 'Q4': 'coral', 'Q5': 'firebrick'}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        cum = (1 + q_series[q].fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, label=q, color=colors[q], linewidth=2.0 if q in ['Q1', 'Q5'] else 1.2)

    bench_cum = (1 + benchmark.fillna(0)).cumprod()
    ax1.plot(bench_cum.index, bench_cum.values, label="Equal-Weight All-SIC Universe", color='navy', linestyle=':', linewidth=1.8)
    ax1.set_yscale('log')
    ax1.set_title("Aggregate All-SIC Ridge Rank Quintiles (Q1 to Q5)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    spread_cum = (1 + q_series['Q1_minus_Q5'].fillna(0)).cumprod()
    ax2.plot(spread_cum.index, spread_cum.values, color='purple', linewidth=1.8)
    ax2.set_title("Cumulative Aggregate Q1 - Q5 Long/Short Spread", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Growth of $1")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    link_clean = load_clean_link()
    universe, m_prices, agg_benchmark, rank_cols = build_multi_sic_dataset(link_clean)
    master_scored = run_all_regions_ridge(universe, rank_cols)
    quintile_series, benchmark_series = run_quintile_backtest(master_scored, m_prices, agg_benchmark)
    report(quintile_series, benchmark_series)