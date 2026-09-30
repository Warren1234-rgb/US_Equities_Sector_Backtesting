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
| Feature | Formula | In Plain English |
|:---|:---|:---|
| Rule of 40 | Revenue Growth + FCF Margin | Is the company growing fast *and* making cash? High scores do both. |
| Cash ROA | Operating Cash Flow / Total Assets | How much real cash the business generates for every dollar of assets it owns. |
| Δ Gross Margin | Gross Margin this year − last year | Is the company getting pricing power or cutting costs? Rising margins are a good sign. |
| Operating Leverage | EBIT Growth − Revenue Growth | Are profits growing faster than sales? That means the business is scaling efficiently. |
| Cash Conversion | (Operating Cash Flow − Net Income) / Total Assets | Are reported earnings backed by actual cash? Low scores can signal accounting-driven profits. |
| Internal Financing | Operating Cash Flow / CapEx | Can the company pay for its own investment without borrowing or issuing stock? |
| Net Buyback | −% Change in Shares Outstanding | Is the company buying back shares (good for holders) or diluting them? |
| Momentum (12-2) | Past-year return, skipping last month | Stocks that have been going up tend to keep going up. |
| Low Volatility | −Volatility of monthly returns (past 12 months) | Calmer stocks have historically delivered better risk-adjusted returns. |

*All features are ranked against same-sector peers each month, so a stock is only compared to companies in its own industry.*

**Training example**
| Feature | Sector Percentile |
|:---|---:|
| Rule of 40 | 81st |
| Cash ROA | 77th |
| Δ Gross Margin | 58th |
| Operating Leverage | 62nd |
| Cash Conversion | 69th |
| Internal Financing | 84th |
| Net Buyback | 72nd |
| Momentum (12-2) | 65th |
| Low Volatility | 79th |
| **Target: Forward 12M Return** | **74th** |

*Example training row: a Manufacturing stock in June 2015. Each feature is ranked against same-sector peers that month; the model learns to predict the stock's forward-return percentile from its feature percentiles.*

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


