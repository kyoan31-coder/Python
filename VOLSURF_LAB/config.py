from pydantic import BaseModel

class Settings(BaseModel):
    default_ticker: str = "SPY"
    risk_free_rate: float = 0.02  # annualized
    data_cache_dir: str = "data_cache"

settings = Settings()
