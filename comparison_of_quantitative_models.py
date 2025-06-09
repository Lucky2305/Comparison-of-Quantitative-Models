
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from european_options import european_options_gbm

ticker = "RELIANCE.NS"
display_data = yf.download(ticker, start="2020-01-01", end="2024-12-31")

data = yf.download("RELIANCE.NS", period="1y")
prices = data['Close']

returns = np.log(prices / prices.shift(1)).dropna()

S_0 = 1411
K = 1370
dt = 1/252
total_time = 19/252
r = 0.06221
N =19
sigma = 0.231943
mu = -0.06868
s_paths = np.zeros((10000, N + 1))
s_paths[:, 0] = S_0

from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


black_scholes = black_scholes_call(S_0, K, total_time, r, sigma)


def binomial_tree_call(S, K, T, r, sigma, n_steps):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Asset price tree
    price_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        price_tree[i, -1] = S * (u ** i) * (d ** (n_steps - i))

    # Option value tree
    option_tree= np.zeros_like(price_tree)
    option_tree[:, -1] = np.maximum(0, price_tree[:, -1] - K)
    for t in range(n_steps - 1, -1, -1):
        for i in range(t + 1):
            option_tree[i, t] = np.exp(-r * dt) * (
                    p * option_tree[i + 1, t + 1] + (1 - p) * option_tree[i, t + 1]
            )
    return option_tree[0, 0]

n_steps = 500
binomial_price = binomial_tree_call(S_0, K, total_time, r, sigma, n_steps)
european_options_gbm(1)
print(f"Black Scholes price: {black_scholes}")
print(f"Binomial Tree price: {binomial_price}")