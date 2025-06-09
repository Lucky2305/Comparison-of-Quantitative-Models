def european_options_gbm(n):
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt


    # Download historical data
    ticker = "RELIANCE.NS"  # Example: Apple stock
    display_data = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    #print(display_data.head())

    data = yf.download("RELIANCE.NS", period="1y")
    prices = data['Close']
    price_array = prices.values
    #print(len(price_array))


    returns = np.log(prices / prices.shift(1)).dropna()
    #annual_volatility = returns.std() * np.sqrt(252)
    #historical_drift = returns.mean() * 252
    #print(f"Historical Drift{historical_drift}")
    #print(f"Annual Volatility{annual_volatility}")
    #formula for GBM s_t = s_0*exp((mu-sigma^2/2+sigma*z*sqrt(dt)))

    S_0 = 1411
    K = 1370
    dt = 1/252
    total_time = 19/252
    r = 0.06221
    N =19
    sigma = 0.231943
    mu = -0.06868
    s_paths = np.zeros((100000, N + 1))
    s_paths[:, 0] = S_0

    for i in range(100000):
        for  t in range(1,N+1):
            z = np.random.normal()
            s_paths[i, t] = s_paths[i, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * z * np.sqrt(dt))

    s_final_day = []
    for i in range(100000):
        s_final_day.append(s_paths[i][19])

    #calculating payoff for calls and puts
    call = []
    put = []
    for i in range(100000):
        call.append(max(s_final_day[i]-K,0))
        put.append(max(K-s_final_day[i],0))
    call_mean = np.mean(call)
    put_mean = np.mean(put)
    discounted_call = call_mean*np.exp(-r*total_time)
    discounted_put = put_mean*np.exp(-r*total_time)

    #ploting the gbm eqn
    num_paths_to_plot = 10
    plt.figure(figsize=(10, 6))
    for i in range(num_paths_to_plot):
        plt.plot(s_paths[i], linewidth=1)

    plt.title("Sample Simulated Stock Price Paths (GBM)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
    return print(f"Discounted call:{discounted_call} \n Discounted put:{discounted_put}")
n = int(input("Enter the number of times to simulate: "))
for i in range(n):
    european_options_gbm(n)

#Call option: 64 > 63 → Market call option is more expensive than your model says → Possibly overpriced call → Selling the call might be profitable if you expect price to revert.
#Put option: 20 > 12 → Market put option is cheaper than your model says → Possibly underpriced put → Buying a put might be profitable.
