# Multi-Region SIC Ridge Rank Engine

A quantitative research engine that trains regularized Ridge rank models independently across all 10 standard SIC divisions using CRSP and Compustat data.

## Features
- **Independent Division Models:** Walk-forward rolling estimation (120M window, 13M embargo) fit independently per SIC region.
- **Dynamic Survivor Rebalancing:** Delisting returns recorded without arbitrary penalty haircuts; remaining capital dynamically equal-weighted across surviving names.
- **Look-Ahead Bias Elimination:** Forward targets isolated strictly to the training pool; out-of-sample prediction scores all active candidates alive at date T.
- **Vectorized Backtester:** Overlapping 12-month sleeves executed via vectorized array broadcasting with 15 bps one-way slippage.

## Requirements
Install the required dependencies:
\\\ash
pip install -r requirements.txt
\\\

## Data Sources
Expects WRDS/CRSP monthly pricing, annual Compustat fundamentals, link tables, and delisting files.
