import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models import Operation
import statsmodels.api as st

def plot_tickers(data: pd.DataFrame):

    colors = ['cornflowerblue', 'rosybrown']

    plt.figure(figsize=(8, 4))
    for i, col in enumerate(data.columns):
        plt.plot(data.index, data[col], label=col, color=colors[i], linewidth=2)

    plt.title("Prices of Tickers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="Tickers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_value(test_data: pd.DataFrame, portfolio_value: pd.Series):

    plt.figure(figsize=(8, 4))
    plt.plot(test_data.index, portfolio_value, label="Portfolio Value", color='cornflowerblue')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.legend()
    plt.show()

def plot_spread(data: pd.DataFrame):

    data = data.copy()
    x_spread = st.add_constant(data.iloc[:, 0])
    y_spread = data.iloc[:, 1]

    model = st.OLS(y_spread, x_spread).fit()
    residuals_spread = model.resid
    mean_residuals_spread = residuals_spread.mean()

    plt.figure(figsize=(8, 4))
    plt.plot(data.index, residuals_spread, label="Spread", color='cornflowerblue')
    plt.axhline(y=mean_residuals_spread, color='red', linestyle='--', linewidth=1.5, label='Mean Spread')
    plt.title("Spread Over Time")
    plt.xlabel("Date")
    plt.ylabel("Spread Value")
    plt.grid()
    plt.legend()
    plt.show()
def plot_vecm_norm(test_data: pd.DataFrame, vecm: pd.Series, theta: float):

    plt.figure(figsize=(8, 4))
    plt.plot(test_data.index, vecm, label="VECM_Norm", color='cornflowerblue')

    
    plt.axhline(y=theta, color='green', linestyle='--', linewidth=1.5, label=f'+θ ({theta:.2f})')
    plt.axhline(y=-theta, color='red', linestyle='--', linewidth=1.5, label=f'-θ ({-theta:.2f})')
    plt.title("VECM_Norm Over Time")
    plt.xlabel("Date")
    plt.ylabel("VECM_Norm Value")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_hr(test_data: pd.DataFrame, hr_values: pd.Series):

    plt.figure(figsize=(8, 4))
    plt.plot(test_data.index, hr_values, label="Hedge Ratio", color='cornflowerblue')
    plt.title("Hedge Ratio Over Time")
    plt.xlabel("Date")
    plt.ylabel("Hedge Ratio")
    plt.grid()
    plt.legend()
    plt.show()


def plot_real_vs_hat(data: pd.DataFrame, y_real, y_hat):

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, y_real, label='Real', color='cornflowerblue', linewidth=2)
    plt.plot(data.index, y_hat, label='Predicted', color='rosybrown', linestyle='--', linewidth=2)
    
    plt.title("Real vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(all_trades: list[Operation]):

    returns = []
    for position in all_trades:
        returns_per_share = position.exit_price / position.entry_price - 1
        if position.type == 'SHORT':
            returns_per_share = -returns_per_share
        returns.append(returns_per_share)

    plt.figure(figsize=(8, 4))
    sns.histplot(returns, bins=15, kde=True, color='cornflowerblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.title("Returns Distribution per Trade")
    plt.xlabel("Returns per Trade")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

