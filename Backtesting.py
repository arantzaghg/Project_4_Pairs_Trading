from models import Operation
from portfolio_value import get_portfolio_value
from cointegration_functions import johansen
from Kalman_structure import KalmanFilterReg
import numpy as np
import pandas as pd


def backtest(data: pd.DataFrame,cash: float, initial_eig, theta) -> tuple[pd.Series, float, float, int, int, int, int, float]: 
    
    n_shares = 100
    COM = 0.125 / 100
    BORROW_RATE = (0.25 / 100) / 252

    data = data.copy().dropna()

    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []
    all_trades = []

    port_hist = []

    hedge_ratio = KalmanFilterReg(Q_filter=0.01, R_filter=0.0001)
    k_eigenvector = KalmanFilterReg(Q_filter=0.01, R_filter=0.0001)
    k_eig = initial_eig

    hr_values = []
    p2_values = []
    p2_hat_values = []

    eig1_values = []
    eig2_values = []

    eig1_h_values = []
    eig2_h_values = []

    vecm_values = []
    vecm_hat_values =[]
    vecm_norm_values =[]

    y = data.columns[0]
    x = data.columns[1]

    borrow_costs = []
    commission_costs = []
    pnl_values = []

    for i, row in data.iterrows():

        post = data.index.get_loc(i)
        if post < 252:
            port_hist.append(get_portfolio_value(cash, active_long_positions, active_short_positions, n_shares, y, x, row[y], row[x]))
            continue

        # Get prices
        p1 = row[y]
        p2 = row[x]

        # Kalman Filter Hedge Ratio
        y_k = p1
        x_k = p2

        hedge_ratio.update(x_k, y_k)
        w0, w1 = hedge_ratio.params
        hr = w1
        hr_values.append(hr)

        p2_hat = w0 + w1 * p1
        p2_hat_values.append(p2_hat)
        p2_values.append(p2)


        # Kalman Filter Eigenvector
        x1 = p1
        x2 = p2

        if post >= 252:
            try:
                k_eig,, = johansen(data.iloc[post - 252:post, :])
            except:
                pass

        eig1, eig2 = k_eig
        eig1_values.append(eig1)
        eig2_values.append(eig2)

        vecm = eig1 * x1 + eig2 * x2
        vecm_values.append(vecm)
        k_eigenvector.update(x1, x2, vecm)

        eig1_h, eig2_h = k_eigenvector.params
        eig1_h_values.append(eig1_h)
        eig2_h_values.append(eig2_h)

        vecm_hat = eig1_h * x1 + eig2_h * x2
        vecm_hat_values.append(vecm_hat)
        vecm_sample = vecm_hat_values[-252:]

        # Normalize VECM
        if len(vecm_sample) >= 252:
            vecm_m = np.nanmean(vecm_sample)
            vecm_std = np.nanstd(vecm_sample)

            if vecm_std != 0:
                vecm_norm = (vecm_hat - vecm_m) / vecm_std
            else:
                vecm_norm = np.nan
        else:
            vecm_norm = np.nan

        vecm_norm_values.append(vecm_norm)


        # Check signals

        #Open positions

        if vecm_norm > theta and not active_long_positions and not active_short_positions:

            #  Long ticker y

            available_cash = cash * 0.4
            n_shares_long_position = available_cash // (p1 * (1 + COM))

            if available_cash > p1 * n_shares_long_position * (1 + COM) and n_shares_long_position > 0:
                cost_y = p1 * n_shares_long_position * (1 + COM)
                cash -= cost_y
                long_y = Operation(
                    ticker=y,
                    time=i,
                    entry_price=p1,
                    exit_price=0.0,
                        n_shares=n_shares_long_position,
                        type='LONG'
                    )
                
                active_long_positions.append(long_y)
                all_trades.append(long_y)
            
            # Short ticker x
            n_shares_short_position = int(n_shares_long_position * hr)
            cost_x = p2 * n_shares_short_position * COM
            if cash > cost_x and n_shares_short_position > 0:
                cash -= cost_x
                short_x = Operation(
                    ticker=x,
                    time=i,
                    entry_price=p2,
                    exit_price=0.0,
                        n_shares=n_shares_short_position,
                        type='SHORT'
                    )
                
                active_short_positions.append(short_x)
                all_trades.append(short_x)


        if vecm_norm < -theta and not active_long_positions and not active_short_positions:

            # Short ticker y

            available_cash = cash * 0.4
            n_shares_short_position = available_cash // (p1 * (1 + COM))

            if available_cash > p1 * n_shares_short_position * (1 + COM) and n_shares_short_position > 0:
                cost_y = p1 * n_shares_short_position * COM
                cash -= cost_y
                short_y = Operation(
                    ticker=y,
                    time=i,
                    entry_price=p1,
                    exit_price=0.0,
                        n_shares=n_shares_short_position,
                        type='SHORT'
                    )
                
                active_short_positions.append(short_y)
                all_trades.append(short_y)

            # Long ticker x
            n_shares_long_position = int(n_shares_short_position * hr)
            cost_x = p2 * n_shares_long_position * (1 + COM)
            if cash > cost_x and n_shares_long_position > 0:
                cash -= cost_x
                long_x = Operation(
                    ticker=x,
                    time=i,
                    entry_price=p2,
                    exit_price=0.0,
                        n_shares=n_shares_long_position,
                        type='LONG'
                    )
                
                active_long_positions.append(long_x)
                all_trades.append(long_x)

        # Close positions
        if abs(vecm_norm) < 0.05:

            # Close long positions
            for position in active_long_positions.copy():
                if position.ticker == y:
                    cash += row[y] * position.n_shares * (1 - COM)
                    commission_costs.append(row[y] * position.n_shares * COM)
                    pnl_values.append((row[y] * position.n_shares * (1 - COM)) - (position.entry_price * position.n_shares * (1+COM)))
                    position.exit_price = row[y]

                if position.ticker == x:
                    cash += row[x] * position.n_shares * (1 - COM)
                    commission_costs.append(row[x] * position.n_shares * COM)
                    pnl_values.append((row[x] * position.n_shares * (1 - COM)) - (position.entry_price * position.n_shares * (1+COM)))
                    position.exit_price = row[x]
                    
                active_long_positions.remove(position)

            # Close short positions

            # Borrow cost
            for position in active_short_positions.copy():
                if position.ticker == y:
                    cash -= row[y] * position.n_shares * BORROW_RATE
                    borrow_costs.append(row[y] * position.n_shares * BORROW_RATE)
                if position.ticker == x:
                    cash -= row[x] * position.n_shares * BORROW_RATE
                    borrow_costs.append(row[x] * position.n_shares * BORROW_RATE)

            # Close short positions
            for position in active_short_positions.copy():
                if position.ticker == y:
                    pnl = (position.entry_price - row[y]) * position.n_shares
                    com = row[y] * position.n_shares * COM
                    cash += pnl - com
                    commission_costs.append(com)
                    pnl_values.append(pnl)
                    position.exit_price = row[y]

                if position.ticker == x:
                    pnl = (position.entry_price - row[x]) * position.n_shares
                    com = row[x] * position.n_shares * COM
                    cash += pnl - com
                    commission_costs.append(com)
                    pnl_values.append(pnl)
                    position.exit_price = row[x]

                active_short_positions.remove(position)
        

        port_hist.append(get_portfolio_value(cash, active_long_positions, active_short_positions, n_shares, y, x, row[y], row[x]))

    # Close remaining positions at the end of the backtest

    for position in active_long_positions.copy():
        if position.ticker == y:
            cash += row[y] * position.n_shares * (1 - COM)
            commission_costs.append(row[y] * position.n_shares * COM)
            pnl_values.append((row[y] * position.n_shares * (1 - COM)) - (position.entry_price * position.n_shares * (1+COM)))
            position.exit_price = row[y]

        if position.ticker == x:
            cash += row[x] * position.n_shares * (1 - COM)
            commission_costs.append(row[x] * position.n_shares * COM)
            pnl_values.append((row[x] * position.n_shares * (1 - COM)) - (position.entry_price * position.n_shares * (1+COM)))
            position.exit_price = row[x]

    for position in active_short_positions.copy():
        if position.ticker == y:
            pnl = (position.entry_price - row[y]) * position.n_shares
            com = row[y] * position.n_shares * COM
            cash += pnl - com
            commission_costs.append(com)
            pnl_values.append(pnl)
            position.exit_price = row[y]

        if position.ticker == x:
            pnl = (position.entry_price - row[x]) * position.n_shares
            com = row[x] * position.n_shares * COM
            cash += pnl - com
            commission_costs.append(com)
            pnl_values.append(pnl)
            position.exit_price = row[x]

    active_long_positions = []
    active_short_positions = []

    # Cost summaries
    borrow_costs = sum(borrow_costs)
    commission_costs = sum(commission_costs)

    # Trade statistics
    pnl = np.array(pnl_values)
    stats = {
        "total_trades": pnl.size,
        "Wins" : np.sum(pnl > 0),
        "Losses" : np.sum(pnl <= 0),
        "win_rate": np.sum(pnl > 0) / pnl.size if pnl.size > 0 else 0.0,
        "avg_win": pnl[pnl > 0].mean() if np.any(pnl > 0) else 0.0,
        "avg_loss": pnl[pnl < 0].mean() if np.any(pnl < 0) else 0.0,
        "avg_win_loss": (
            (pnl[pnl > 0].mean() / abs(pnl[pnl < 0].mean()))
            if np.any(pnl < 0) else np.nan
        ),
        "profit_factor": (
            pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())
            if np.any(pnl < 0) else np.nan
        )
    }


    return pd.Series(port_hist), cash, stats,  p2_values, p2_hat_values, vecm_values, vecm_hat_values, \
    vecm_norm_values, hr_values, borrow_costs, commission_costs, all_trades