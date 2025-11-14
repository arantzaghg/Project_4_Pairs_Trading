from itertools import combinations
import pandas as pd
import yfinance as yf
from cointegration_functions import correlation, ols_adf, johansen
from data_utils import split_data

def cointegration_test():
    ticker = {
    "Clothing_and_Apparel": ["COLM", "CPRI", "DKS", "DECK", "BIRK", "ASO", "GES", "BOOT"],
    "Financials":["MS", "SCHW", "CMA", "NTRS", "AMP", "BEN", "LPLA"],   
    "Airlines": ["CPA", "ALGT", "VLRS", "AER", "CHH"],
    "Food_and_Beverage": ["KHC", "HSY", "MNST", "CELH", "POST", "TAP"],
    "Entertainment_and_Media": ["NWSA", "CHTR", "SPOT", "IMAX",  "BILI"],
    "Automotive": ["HMC", "TM", "STLA", "VWAGY", "VLVLY", "LI", "XPEV", "BYDDY"]
    }

    pairs_tickers = {sec: list(combinations(tks, 2)) for sec, tks in ticker.items()}

    results = []

    for sector, pairs in pairs_tickers.items():
        print(f"\n=== Descargando datos del sector: {sector} ===")
        for t1, t2 in pairs:
            print(f"Descargando {t1} y {t2}...")

            data = yf.download([t1, t2], period="15y", interval="1d", progress=False)

            close = data['Close'].dropna()
            close.columns = [col[0] if isinstance(col, tuple) else col for col in close.columns]
            close = close[[t1, t2]].dropna()

            train_data, _ = split_data(close)

            corr = correlation(train_data, window=252)
            _, adf_pvalue = ols_adf(train_data)
            eigenvector, critical_value95, trace_stat = johansen(train_data)
            johansen_pass = 1 if trace_stat > critical_value95 else 0

            results.append({
                'sector': sector,
                'pair': f'{t1}-{t2}',
                'corr': float(corr),
                'pvalue_adf': float(adf_pvalue),
                'johansen_pass': johansen_pass,
                'eigenvector': eigenvector,
                'Strength': trace_stat/critical_value95,
            })

    df_results = pd.DataFrame(results)
    df_filtrado = (
        df_results[
            (df_results['corr'] > 0.60) &
            (df_results['pvalue_adf'] < 0.05) &
            (df_results['johansen_pass'] == 1)
        ]
        .sort_values(by='Strength', ascending=False)
        .reset_index(drop=True)
    )

    print(df_results)
    print(df_filtrado)

if __name__ == "__main__":
    cointegration_test()