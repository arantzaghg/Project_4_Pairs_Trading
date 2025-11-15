from models import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation],n_shares: float, y_ticker: float, x_ticker: float, y_p: float, x_p: float) -> float:
    """
    Calculate the total portfolio value.
    
    Parameters:
    cash : float: The available cash in the portfolio.
    long_ops : list[Operation]: List of long position operations.
    short_ops : list[Operation]: List of short position operations.
    n_shares : float: Number of shares held.
    y_ticker : float: Ticker symbol for asset Y.
    x_ticker : float: Ticker symbol for asset X.
    y_p : float: Current price of asset Y.
    x_p : float: Current price of asset X.
    
    Returns:
    float: Total portfolio value.
    """
    
    port_val = cash

    # add long positions value
    for position in long_ops:
        if position.ticker == x_ticker:
            port_val += x_p * position.n_shares
        if position.ticker == y_ticker:
            port_val += y_p * position.n_shares

    # add short positions value
    for position in short_ops:
        if position.ticker == x_ticker:
            port_val += (position.entry_price - x_p) * position.n_shares
        if position.ticker == y_ticker:
            port_val += (position.entry_price - y_p) * position.n_shares

    return port_val
