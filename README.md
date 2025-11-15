# Project 4: Pairs Trading

A Python-based implementation of a pairs-trading strategy, including cointegration testing, model construction, back‐testing, portfolio value tracking, and metrics.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Data Requirements](#data-requirements)
* [Setup & Installation](#setup-installation)
* [Running the Code](#running-the-code)
* [File/Module Structure](#filemodule-structure)
* [Dependencies](#dependencies)
* [How to Extend](#how-to-extend)
* [License](#license)

## Overview

This project implements a quantitative pairs-trading strategy, where two assets whose price series are cointegrated are selected; a model is built (e.g., via a Kalman filter structure) and a back-test is run to simulate entering and exiting positions when the spread signals trade opportunities. The performance is measured via custom metrics and visualizations.

## Features

* Cointegration testing of candidate asset pairs (via cointegration_functions.py, cointegration_test.py).
* Construction of a model (e.g., in models.py / Kalman_structure.py) to track spread/relationship.
* Portfolio value tracking (portfolio_value.py) and back‐testing (Backtesting.py).
* Utility functions for data handling (data_utils.py).
* Metric computations in metrics.py.
* Plotting/visualization in plots.py.
* A main script main.py to tie everything together.

## Data Requirements

To use this project you will need:

1. *Historical price data* for two (or more) assets (e.g., equities, ETFs). The data should be in a format with at least:

   * Date/time index (or a column for date)
   * Closing price (or adjusted close) for each asset
   * Preferably in CSV or other standard tabular format
2. *Data frequency*: daily (or intraday if you adapt) – the code assumes a consistent time‐step.
3. *Preprocessing*: The data should be cleaned (no large gaps, missing values handled). Your data_utils.py module handles some data‐wrangling, but you should verify your data is properly formatted.
4. *Folder/Path structure*: The code expects input data files to be placed in a folder (for example data/) or a path you specify in main.py. You will need to update main.py or config to point to your data location.
5. *Configuration*: You may need to specify which pair of assets to analyze, the start‐date/end‐date, and other parameters (spread threshold, look-back windows, etc). There may be a section in main.py or in a configuration section to adjust.

## Setup & Installation

1. Clone this repository:

   bash
   git clone https://github.com/arantzaghg/Project_4_Pairs_Trading.git
   cd Project_4_Pairs_Trading
   
2. (Recommended) Create a virtual environment:

   bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   
3. Install required Python packages:

   bash
   pip install -r requirements.txt
   

   The requirements.txt file lists the dependencies (e.g., pandas, numpy, matplotlib, statsmodels, etc).
4. Verify that your data folder and file paths match the expected structure in data_utils.py / main.py.

## Running the Code

Once setup is complete:

1. Place your input data files (e.g., CSVs) into the appropriate folder (e.g., data/).
2. Open main.py and configure:

   * Asset symbols / file names
   * Date range
   * Model parameters (if any)
3. Run the main script:

   bash
   python main.py
   
4. The code will execute the sequence: load data → test cointegration → build model → run back‐test → compute metrics → generate plots.
5. Review outputs: back-test results, portfolio value evolution, metrics, and saved plots (e.g., output folder plots/ if configured).

## File / Module Structure

Here’s a brief overview of the key files:

* data_utils.py — functions for loading, cleaning, and preparing data for analysis.
* cointegration_functions.py — implements cointegration tests (e.g., Engle-Granger) and candidate pair identification.
* cointegration_test.py — a script to run cointegration tests on chosen assets/pairs.
* models.py — defines the model structure (e.g., the spread model, parameter estimation).
* Kalman_structure.py — alternative modelling structure using a Kalman filter approach for estimating spread dynamics.
* Backtesting.py — runs the trading simulation: entering/exiting positions, tracking portfolio value.
* portfolio_value.py — functions to compute portfolio value over time given trade history.
* metrics.py — functions to compute performance metrics (e.g., Sharpe ratio, max drawdown, win-loss ratio).
* plots.py — visualization functions: plot portfolio value over time, spread over time, signals, etc.
* main.py — orchestrates the workflow: parameters, calls to modules, output generation.
* requirements.txt — lists Python dependencies.
* LICENSE — MIT license for the code.

## Dependencies

The main dependencies (as listed in requirements.txt) include but may not be limited to:

* pandas
* numpy
* matplotlib / seaborn
* statsmodels (for cointegration tests)
* scipy
  You may also need additional libraries depending on your modelling (e.g., filterpy or custom Kalman filter libraries) if used. Ensure your requirements.txt reflects everything required.

## How to Extend

Here are ideas for extending the project:

* Add intraday data support (minute-level or hourly) and adapt the back‐testing logic accordingly.
* Add parameter optimization (e.g., grid‐search over spread thresholds, look‐back windows).
* Add transaction cost modelling (commissions, slippage).
* Add more assets/pairs dynamically by scanning a universe of assets.
* Add a live‐trading module or paper-trading integration.
* Add reporting/export (e.g., CSV/Excel of results, interactive dashboards).
* Add risk management constraints (max position size, stop-loss, time in trade limits).

## License

This project is licensed under the MIT License.



