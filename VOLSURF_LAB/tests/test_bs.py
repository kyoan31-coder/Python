from models.black_scholes import bs_price
def test_bs_non_negative():
    assert bs_price(100,100,0.02,0.2,0.5,True) >= 0
