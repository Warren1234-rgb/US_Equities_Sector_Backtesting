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
Here are the features chosen. 


- Rule of 40
- Cash ROA
- Δ Gross Margin
- Operating Leverage
- Cash Conversion
- Internal Financing
- Net Buyback
- Momentum (12-2)
- Low Volatility


To prevent lookahead bias, when we are predicting the return from T to T+12 months, we use the report from a fiscal year that ended at before T-3, and the one before that, because companies take <90 days (mandated by the SEC) to release their reports after fiscal year end. 


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

| Bucket | CAGR | Ann. Vol | Sharpe |
|:---|---:|---:|---:|
| Q1 (Top 20% – Long) | 14.11% | 15.81% | 0.92 |
| Q2 (60–80%) | 13.26% | 15.93% | 0.87 |
| Q3 (40–60% Median) | 13.00% | 17.13% | 0.80 |
| Q4 (20–40%) | 12.94% | 18.74% | 0.75 |
| Q5 (Bottom 20% – Short) | 11.04% | 23.97% | 0.56 |
| **EW Universe Benchmark** | **13.40%** | **17.78%** | **0.80** |


**Feature Importances for the sectors**
These are the feature weights for each feature, using that feature only (univariate regression). 
<img width="1529" height="667" alt="10_feature_ic" src="https://github.com/user-attachments/assets/fb1ee1b5-b770-4a4c-874d-afb0b62077ad" />


**Rolling ICs**
These are the Pearson correlations between the predicted and actual return (percentiles). ![Uploading 10_feature_ic.png…]()

<img width="1465" height="618" alt="09_rank_ic_timeseries" src="https://github.com/user-attachments/assets/7cc1694a-6138-4700-a9e9-f0e11371d855" />


