# =============================================
# Streamlit Portfolio Optimization Dashboard
# =============================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import date
import plotly.graph_objects as go

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")

# -----------------------------
# 1. Streamlit App Title
# -----------------------------
st.title("Portfolio Optimization Dashboard – No Short Selling")
st.markdown("""
This app helps you allocate your investment across selected stocks for **maximum risk-adjusted returns**, without short selling.
""")

# -----------------------------
# 2. Sidebar Inputs
# -----------------------------
st.sidebar.header("Inputs")

tickers_input = st.sidebar.text_input(
    "Enter Tickers (comma-separated)",
    "RELIANCE.NS, INFY.NS, HDFCBANK.NS, TCS.NS, ICICIBANK.NS"
)

# Minimum selectable date
min_start_date = date(2000, 1, 1)

# Start date input
start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2018, 1, 1),
    min_value=min_start_date,
    max_value=date.today()
)

# End date input
end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=start_date,
    max_value=date.today()
)

# Total investment input
total_investment = st.sidebar.number_input(
    "Total Investment (₹)", value=10000000, step=100000
)

# Risk-free rate input
rf_annual = st.sidebar.number_input(
    "Risk-Free Rate (annual, %)", value=6.0, step=0.1
)
risk_free_rate = (1 + rf_annual / 100) ** (1 / 252) - 1  # daily rate

# -----------------------------
# 3. Functions
# -----------------------------
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=risk_free_rate):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def min_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def stress_test_plot(port_returns, start, end, title):
    stress_period = port_returns[start:end]
    if stress_period.empty:
        st.warning(f"No data available for {title}")
        return
    cum_returns = (1 + stress_period).cumprod()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cum_returns, label="Portfolio")
    ax.set_title(f"Stress Test: {title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# 4. Main Analysis
# -----------------------------
if st.button("Analyze Portfolio"):
    
    tickers = [t.strip() for t in tickers_input.split(",")]
    
    if not tickers or tickers == ['']:
        st.error("Please enter at least one valid ticker symbol.")
    else:
        try:
            # Download data
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            if data.empty:
                st.error("No data fetched. Check ticker symbols or date range.")
                st.stop()

            returns = data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            st.success("Data downloaded successfully!")
            
            # Optimization – Max Sharpe, no short selling
            num_assets = len(mean_returns)
            bounds = tuple((0,1) for _ in range(num_assets))
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            initial_weights = [1/num_assets]*num_assets
            
            result = minimize(
                neg_sharpe,
                initial_weights,
                args=(mean_returns, cov_matrix),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500}
            )
            
            optimal_weights = result.x
            opt_return, opt_vol, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            
            # ---------------- Portfolio Allocation Table ----------------
            alloc_df = pd.DataFrame({
                "Ticker": tickers,
                "Weight": optimal_weights
            }).set_index("Ticker")

            alloc_df["Weight"] = (alloc_df["Weight"] * 100).round(2)   # in %
            alloc_df["Start Price"] = data.iloc[0].round(2)            # price at start_date
            alloc_df["End Price"] = data.iloc[-1].round(2)             # price at end_date
            alloc_df["No. of Shares"] = (total_investment * alloc_df["Weight"] / 100 / alloc_df["End Price"]).round(2)
            alloc_df["Value (₹)"] = (alloc_df["No. of Shares"] * alloc_df["End Price"]).round(2)

            st.subheader("📌 Optimal Portfolio Allocation")
            st.dataframe(
                alloc_df.style.format({
                    "Weight": "{:.2f}%",
                    "Start Price": "₹{:.2f}",
                    "End Price": "₹{:.2f}",
                    "No. of Shares": "{:.2f}",
                    "Value (₹)": "₹{:.2f}"
                }),
                use_container_width=True
            )
            
            # Interactive Donut Chart for Allocation
            fig_pie = go.Figure(data=[go.Pie(
                labels=alloc_df.index,
                values=alloc_df["Weight"],
                hole=0.4,  # donut style
                textinfo="label+percent",
                hovertemplate="Ticker: %{label}<br>Weight: %{percent}<br>Value: ₹%{customdata:,.0f}",
                customdata=alloc_df["Value (₹)"]
            )])

            fig_pie.update_layout(
                title_text="Optimal Allocation (%)",
                showlegend=True
            )

            st.plotly_chart(fig_pie, use_container_width=True)

            # Portfolio returns
            port_returns = (returns * optimal_weights).sum(axis=1)
            
            # Risk Metrics
            max_drawdown = (1 + port_returns).cumprod().div((1 + port_returns).cumprod().cummax()) - 1
            VaR_95 = np.percentile(port_returns, 5)
            CVaR_95 = port_returns[port_returns <= VaR_95].mean()
            
            st.subheader("Portfolio Metrics")
            st.write(f"**Expected Annual Return (%):** {opt_return*252:.2%}")
            st.write(f"**Annualized Volatility (%):** {opt_vol*np.sqrt(252):.2%}")
            st.write(f"**Sharpe Ratio:** {opt_sharpe*np.sqrt(252):.4f}")
            st.write(f"**Max Drawdown:** {max_drawdown.min():.2%}")
            st.write(f"**Value-at-Risk (95%):** {VaR_95:.2%} – Maximum daily loss expected 95% of the time")
            st.write(f"**Conditional VaR (95%):** {CVaR_95:.2%} – Average loss in the worst 5% of days")
            
            # Cumulative portfolio value
            cum_portfolio_value = total_investment * (1 + port_returns).cumprod()
            final_amount = cum_portfolio_value[-1]
            cumulative_return_percent = (final_amount / total_investment - 1) * 100
            
            # Portfolio value as percentage gain/loss
            cum_portfolio_percent = (cum_portfolio_value / total_investment - 1) * 100
            
            # -----------------------------
            # Plot interactive single chart
            # -----------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_portfolio_value.index,
                y=cum_portfolio_value,
                mode='lines',
                name='Portfolio Value (₹)',
                hovertemplate='Date: %{x}<br>Value: ₹%{y:,.0f}<br>Gain/Loss: %{customdata:.2f}%',
                customdata=cum_portfolio_percent.values
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Final Portfolio Value:** ₹{final_amount:,.0f}")
            st.write(f"**Cumulative Return (%):** {cumulative_return_percent:.2f}%")
            
            # Stress Testing
            st.subheader("Stress Testing")
            stress_test_plot(port_returns, "2020-01-01", "2020-06-30", "COVID-19 Crash")
            stress_test_plot(port_returns, "2008-01-01", "2009-03-31", "2008 Financial Crisis")
            
            # Efficient Frontier
            st.subheader("Efficient Frontier")
            returns_range = np.linspace(mean_returns.min(), mean_returns.max(), 50)
            volatility_list = []
            
            for target_return in returns_range:
                constraints_ef = [
                    {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "eq", "fun": lambda x, tr: np.dot(x, mean_returns) - tr, "args": (target_return,)}
                ]
                result_ef = minimize(min_volatility, initial_weights, args=(mean_returns, cov_matrix),
                                     method="SLSQP", bounds=bounds, constraints=constraints_ef)
                if result_ef.success:
                    volatility_list.append(result_ef.fun)
                else:
                    volatility_list.append(np.nan)
            
            fig2, ax = plt.subplots(figsize=(10,6))
            ax.plot(np.array(volatility_list)*np.sqrt(252), np.array(returns_range)*252, linestyle='--', color='black', label='Efficient Frontier')
            ax.scatter(opt_vol*np.sqrt(252), opt_return*252, color='red', s=200, label='Optimal Portfolio')
            ax.set_title("Efficient Frontier")
            ax.set_xlabel("Annualized Volatility")
            ax.set_ylabel("Annualized Return")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig2)
            
            # -----------------------------
            # Explanations Section
            # -----------------------------
            with st.expander("📖 What do these numbers and graphs mean?"):
                st.markdown("""
**Portfolio Allocation:**  
Shows the proportion of your investment in each stock (Weight), start & end prices, number of shares, and current value.  
Optimal allocation is calculated to **maximize risk-adjusted return** without short selling.

**Expected Annual Return (%):**  
Average return your portfolio is expected to generate in a year.

**Annualized Volatility (%):**  
Measures portfolio fluctuations. Higher volatility = higher risk.

**Sharpe Ratio:**  
Risk-adjusted return = (Return – Risk-Free Rate) / Volatility. Higher is better.

**Max Drawdown (%):**  
Largest peak-to-trough loss experienced by the portfolio.

**Value-at-Risk (95%):**  
Maximum daily loss expected 95% of the time. For example, VaR = -2% means on 95% of days, loss won't exceed 2%.

**Conditional VaR (CVaR 95%):**  
Average loss in the worst 5% of days. Gives a sense of potential extreme losses.

**Portfolio Value Over Time Chart:**  
Interactive chart shows your portfolio’s ₹ value. Hover to see **percentage gain/loss** compared to initial investment.

**Stress Testing:**  
Shows portfolio performance during extreme market crises (COVID-19, 2008). Helps assess portfolio resilience.

**Efficient Frontier:**  
Shows the set of optimal portfolios giving the **highest return for each risk level**. Red star = your portfolio.
                """)
            
        except Exception as e:
            st.error(f"Error fetching data or processing: {e}")
