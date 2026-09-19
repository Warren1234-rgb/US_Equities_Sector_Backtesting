**Dataset**
The dataset contained price (CRSP) and fundamental (Compustat) data for all active and delisted US Equities going back to the 1970s to 2026. Thus, it is survivorship bias free. 


**Model choice**
A ridge regression model is chosen to avoid overfitting. One model is built for each sector (which is defined as a SIC code range). 10 sectors are built. 



**Feature engineering**
To generate a training example, an (X, Y) pair, I used data from the last 2 years, and the model predicts 1 year ahead. X contains fundamental data, such as past cash return on assets, change in gross margin, operating leverage, and also price data like the 12-2 momentum. To prevent lookahead bias, when we are predicting the return from T to T+12 months, we use the report from a fiscal year that ended at before T-3, and the one before that, because companies take <90 days (mandated by the SEC) to release their reports after fiscal year end. 

**Model Training**
A new model for built for each sector each January. The latest training example had its target return period end 1 month before January. 

**Model testing** 
I did a walk forward test of the equal weighted portfolios of the five quintiles of predicted returns. The stocks are held for a year. If they become delisted, the delisting return from CRSP is applied and the portfolio rebalances to equal weight the other survivors. 


**Quintile returns**
<img width="1902" height="682" alt="image" src="https://github.com/user-attachments/assets/014a7609-3254-47d4-b17f-c00cc619c801" />
