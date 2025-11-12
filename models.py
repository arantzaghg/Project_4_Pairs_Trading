from dataclasses import dataclass

@dataclass
class Operation:
    ticker: str
    time: str
    entry_price: float
    exit_price: float
    n_shares: int
    type: str