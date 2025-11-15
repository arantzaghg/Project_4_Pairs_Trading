from data_utils import get_asset_data, split_data, add_overlay
from Backtesting import backtest
from cointegration_functions import johansen
from plots import plot_portfolio_value, plot_spread, plot_vecm_norm, plot_hr, plot_tickers, plot_returns_distribution, plot_real_vs_hat
from metrics import all_metrics


def main():

    # Data Preparation

    tickers = ["MS", "SCHW"]
    data = get_asset_data(tickers)
    train_data, test_data = split_data(data)
    test_data_lp = add_overlay(train_data, test_data, overlay_size=252)
    eigenvector, _, _= johansen(train_data)
    theta = 0.33


    # Backtesting

    portfolio_value, final_cash, stats, p2_values, p2_hat_values, vecm_values, vecm_hat_values, vecm_norm_values, hr_values,\
        borrow_costs, commission_costs, all_trades = backtest(test_data_lp,1000000, eigenvector, theta)

    # Portfolio results

    print(f"\n--- PORTFOLIO VALUE ---")
    print(f"Final portfolio value: {portfolio_value.iloc[-1]:.2f}")
    print(f"Final cash: {final_cash:.2f}")
    
    # Metrics

    print(f"\n--- METRICS ---")
    print(all_metrics(portfolio_value))

    # Cost analysis

    print(f"\n--- COST ANALYSIS ---")
    print(f"Borrow Costs: {borrow_costs:.2f}")
    print(f"Commissions: {commission_costs:.2f}")

    # Trade statistics

    print(f"\n--- TRADE STATISTICS ---")
    print(
    f"Total trades: {int(stats['total_trades'])}\n"
    f"Wins: {int(stats['Wins'])}\n"
    f"Losses: {int(stats['Losses'])}\n"
    f"Win rate: {stats['win_rate']:.2%}\n"
    f"Win/Loss ratio: {stats['avg_win_loss']:.4f}\n"
    f"Profit factor: {stats['profit_factor']:.4f}"
    )

    # Plots
    plot_tickers(data)
    plot_portfolio_value(test_data_lp, portfolio_value)
    plot_spread(train_data)
    plot_real_vs_hat(test_data, p2_values, p2_hat_values)
    plot_vecm_norm(test_data, vecm_norm_values, theta)
    plot_real_vs_hat(test_data, vecm_values, vecm_hat_values)
    plot_hr(test_data, hr_values)
    plot_returns_distribution(all_trades)



if __name__ == "__main__":
    main()