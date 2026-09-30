**Dataset**
The dataset contained price (CRSP) and fundamental (Compustat) data for all active and delisted US Equities going back to the 1970s to 2026. Thus, it is survivorship bias free. 


**Model choice**
A ridge regression model is chosen to avoid overfitting. One model is built for each sector (which is defined as a SIC code range). 10 sectors are built. 
| Division | Sector | SIC Codes |
|:---:|:---|:---:|
| A | Agriculture, Forestry & Fishing | 0100–0999 |
| B | Mining | 1000–1499 |
| C | Construction | 1500–1799 |
| D | Manufacturing | 2000–3999 |
| E | Transportation, Communications & Utilities | 4000–4999 |
| F | Wholesale Trade | 5000–5199 |
| G | Retail Trade | 5200–5999 |
| H | Finance, Insurance & Real Estate | 6000–6799 |
| I | Services | 7000–8999 |
| J | Public Administration | 9100–9999 |


**Feature engineering**
To generate a training example, an (X, Y) pair, I used data from the last 2 years. Y was the cross sectional alpha (bounded between 0.5 and -0.5, 0 meaning that its return = the market (equal weighted) mean). X contains fundamental data, such as past cash return on assets, change in gross margin, operating leverage, and also price data like the 12-2 momentum. To prevent lookahead bias, when we are predicting the return from T to T+12 months, we use the report from a fiscal year that ended at before T-3, and the one before that, because companies take <90 days (mandated by the SEC) to release their reports after fiscal year end. 
| Feature | Formula | Description |
|:---|:---|:---|
| Rule of 40 | Revenue Growth + FCF Margin | Growth and profitability combined |
| Cash ROA | CFO / Total Assets | Cash return on assets |
| Δ Gross Margin | GM(t) − GM(t−1), where GM = Gross Profit / Revenue | Year-over-year change in gross margin |
| Operating Leverage | EBIT Growth − Revenue Growth | Whether profits grow faster than sales |
| Cash Conversion | (CFO − Net Income) / Total Assets | Earnings quality (accruals) |
| Internal Financing | CFO / (CapEx + 0.01 × Total Assets) | Ability to fund investment from operating cash flow |
| Net Buyback | −(Shares(t) / Shares(t−1) − 1) | Share count reduction (buybacks minus issuance) |
| Momentum (12-2) | Σ log returns from month t−11 to t−1 | Trailing 1-year return, skipping the most recent month |
| Low Volatility | −σ(monthly returns over trailing 12 months) | Lower volatility scores higher |

*CFO = operating cash flow (net income used if missing); FCF = CFO − CapEx; FCF Margin = FCF / Revenue. Fundamentals are annual and lagged 90 days after fiscal year-end. Each feature is converted to a percentile rank within its SIC division each month before entering the model.*

**Training example**
| Column | Value | Meaning |
|:---|---:|:---|
| MTHCALDT | 2015-06-30 | Formation month |
| PERMCO | 20436 | CRSP company ID |
| SIC_REGION | D_Manufacturing | Sector the stock is ranked within |
| F_RULE_OF_40_RK | 0.31 | 81st percentile vs. Manufacturing peers |
| F_CASH_ROA_RK | 0.27 | 77th percentile |
| F_DELTA_GROSS_MARGIN_RK | 0.08 | 58th percentile |
| F_OPERATING_LEVERAGE_RK | 0.12 | 62nd percentile |
| F_CASH_CONVERSION_RK | 0.19 | 69th percentile |
| F_INTERNAL_FINANCING_RK | 0.34 | 84th percentile |
| F_NET_BUYBACK_RK | 0.22 | 72nd percentile |
| F_MOM_12_2_RK | 0.15 | 65th percentile |
| F_LOW_VOLATILITY_RK | 0.29 | 79th percentile |
| FWD_12M_RET | 18.4% | Realized return over the next 12 months (not a model input) |
| **ALPHA_TARGET** | **0.24** | **Label: 74th percentile of forward return within sector** |

**Model Training**
A new model for built for each sector each January. The latest training example had its target return period end 1 month before January. 

**Model testing** 
I did a walk forward test of the equal weighted portfolios of the five quintiles of predicted returns. The stocks are held for a year. If they become delisted, the delisting return from CRSP is applied and the portfolio rebalances to equal weight the other survivors. 


**Quintile returns**
<img width="1902" height="682" alt="image" src="https://github.com/user-attachments/assets/014a7609-3254-47d4-b17f-c00cc619c801" />

| Bucket | CAGR | Ann. Vol | Sharpe |
|:---|---:|---:|---:|
| Q1 (Top 20% – Long) | 14.23% | 16.04% | 0.92 |
| Q2 (60–80%) | 13.53% | 16.32% | 0.86 |
| Q3 (40–60% Median) | 12.85% | 17.71% | 0.78 |
| Q4 (20–40%) | 12.29% | 19.26% | 0.70 |
| Q5 (Bottom 20% – Short) | 10.29% | 24.55% | 0.52 |
| **EW Universe Benchmark** | **12.80%** | **18.36%** | **0.75** |
| Q1 − Q5 Long/Short | 0.37% | 12.64% | 0.09 |
| Q1 − Benchmark (Active) | 0.77% | 4.49% | 0.19 |

**Feature Importances for the sectors**
<img width="1531" height="1007" alt="11_ridge_coefficients_by_region" src="https://github.com/user-attachments/assets/4061e1a1-acf4-4cdb-98f5-8d31495f82b7" />

**Rolling ICs**
<img width="1465" height="618" alt="09_rank_ic_timeseries" src="https://github.com/user-attachments/assets/e0018c97-8a9a-4068-9c63-4a0c92a37920" />


