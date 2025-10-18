import numpy as np
from datetime import datetime

def year_fraction(t1: datetime, t2: datetime, basis: int = 365) -> float:
    return max((t2 - t1).days, 0) / basis

def annualize_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(252.0)
