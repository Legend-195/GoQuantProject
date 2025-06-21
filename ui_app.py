import streamlit as st
import numpy as np
import time
import pickle
import threading
import websocket
import json
import pandas as pd
import joblib
import gc
from sklearn.preprocessing import StandardScaler
from fee_model import calculate_fee
from datetime import datetime
# Memory management

gc.set_threshold(700, 10, 10) # garbage collection threshold in streamlit
# ---------- Load Models ----------

quantile_model = joblib.load("quantile_regression_model.pkl") #Regression model efficiency
scaler = joblib.load("slippage_scaler.pkl")
logistic_model = joblib.load("logistic_regression_model.pkl")
logistic_scaler = joblib.load("logistic_scaler.pkl")


# ---------- Globals ----------
orderbook_data = {}
last_slippage = {"value": None}
current_order_size = 100
lock = threading.Lock() #network communication
simulation_active = threading.Event()
last_update_time = {"start": None, "end": None}
# ---------- Slippage Estimation ----------
def compute_depth(orderbook, levels=5):
    return sum(float(qty) for _, qty in orderbook.get("asks", [])[:levels])
def compute_imbalance(orderbook):
    best_bid_qty = float(orderbook["bids"][0][1]) if orderbook.get("bids") else 0
    best_ask_qty = float(orderbook["asks"][0][1]) if orderbook.get("asks") else 0
    total = best_bid_qty + best_ask_qty
    return (best_bid_qty - best_ask_qty) / total if total else 0

def predict_maker_taker_prob(orderbook, order_size, volatility, order_type):
    if not orderbook.get("bids") or not orderbook.get("asks"):
        return None, None
    best_bid = float(orderbook["bids"][0][0])
    best_ask = float(orderbook["asks"][0][0])
    spread = best_ask - best_bid
    imbalance = compute_imbalance(orderbook)
    mid_price = (best_bid + best_ask) / 2
    order_size_usd = order_size 
    order_type_encoded = 1 if order_type.lower() == "market" else 0

    input_df = pd.DataFrame([[spread,  imbalance,volatility,order_size_usd, order_type_encoded]],
                            columns=["spread","imbalance","volatility", "order_size_usd",  "order_type_encoded"])
    input_scaled = logistic_scaler.transform(input_df)
    probs = logistic_model.predict_proba(input_scaled)[0]
    return probs[0], probs[1]
def estimate_live_slippage(orderbook, usd_amount):
    if not orderbook.get("bids") or not orderbook.get("asks"):
        return None
    best_bid = float(orderbook["bids"][0][0])
    best_ask = float(orderbook["asks"][0][0])
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    depth = compute_depth(orderbook)
    order_size = usd_amount

    input_df = pd.DataFrame([[spread, order_size, depth]], columns=["spread", "order_size", "depth"])
    input_scaled = scaler.transform(input_df)
    slippage_bps = quantile_model.predict(input_scaled)[0]

    with lock:
        last_slippage["value"] = round(slippage_bps, 6)
# ----------------- Market Impact (Almgren-Chriss) ------------------

def temporary_impact(volume, alpha, eta):
    impact = eta * volume ** alpha
    return impact

def permanent_impact(volume, beta, gamma):
    impact = gamma * volume ** beta
    return impact

def hamiltonian(inventory, sell_amount, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step=0.5):
    temp_impact = risk_aversion * sell_amount * permanent_impact(sell_amount / time_step, beta, gamma)
    perm_impact = risk_aversion * (inventory - sell_amount) * time_step * temporary_impact(sell_amount / time_step, alpha, eta)
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step * ((inventory - sell_amount) ** 2)
    H = temp_impact + perm_impact + exec_risk
    return np.clip(H, -500, 500)

# ---------- Optimized optimal_execution() ----------
def optimal_execution(time_steps, total_shares, risk_aversion, alpha, beta, gamma, eta):
    bin_size = max(1, total_shares // 1000)
    bins = np.arange(0, total_shares + 1, bin_size)
    num_bins = len(bins)
    log_value_function = np.full((time_steps, num_bins), np.inf)
    best_moves = np.zeros((time_steps, num_bins), dtype=int)
    time_step_size = 0.5

    for i, shares in enumerate(bins):
        log_value_function[-1, i] = temporary_impact(shares / time_step_size, alpha, eta)
        best_moves[-1, i] = shares

    for t in reversed(range(time_steps - 1)):
        for i, shares in enumerate(bins):
            best_val = np.inf
            best_n = 0
            for j in range(i + 1):
                sell_amt = bins[j]
                remaining = shares - sell_amt
                if remaining < 0:
                    continue
                k = np.searchsorted(bins, remaining)
                if k >= num_bins:
                    continue
                future_val = log_value_function[t + 1, k]
                H = hamiltonian(shares, sell_amt, risk_aversion, alpha, beta, gamma, eta)
                total = future_val + H
                if total < best_val:
                    best_val = total
                    best_n = sell_amt
            log_value_function[t, i] = best_val
            best_moves[t, i] = best_n

    inventory_path = [total_shares]
    trajectory = []
    shares = total_shares
    for t in range(1, time_steps):
        idx = np.searchsorted(bins, shares) #DATA STRUCTURE SELECTION NUMPY
        sell = best_moves[t, idx]
        trajectory.append(sell)
        shares -= sell
        inventory_path.append(shares)

    return np.array(inventory_path), np.array(trajectory) #DATA STRUCTURE SELECTION NUMPY

def compute_market_impact(order_size):
    time_steps = 20
    risk_aversion = 0.01
    alpha = 1
    beta = 1
    gamma = 0.05
    eta = 0.05

    _, optimal_trajectory = optimal_execution(time_steps, order_size, risk_aversion, alpha, beta, gamma, eta)

    temp_impact_total = 0
    perm_impact_total = 0
    time_step_size = 0.5

    for v in optimal_trajectory:
        temp_impact_total += temporary_impact(v / time_step_size, alpha, eta)
        perm_impact_total += permanent_impact(v / time_step_size, beta, gamma)

    return round(temp_impact_total + perm_impact_total, 10)

# ---------- WebSocket ----------
def on_message(ws, message):
    global orderbook_data
    start_time = time.time()  # Start timestamp for data processing latency
    try:
        data = json.loads(message) #network communication
        with lock:
            last_update_time["start"] = time.time()  # Capture start
            orderbook_data = {
                "bids": data.get("bids", []),          # dict data structure selection 
                "asks": data.get("asks", [])
            }
        if simulation_active.is_set():
            estimate_live_slippage(orderbook_data, current_order_size)
        end_time = time.time()
        processing_latency = (end_time - start_time) * 1000  # ms
        print(f"[Latency] Data Processing Latency: {processing_latency:.3f} ms") #data processing
    except Exception as e:
        print(f"Message parsing error: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")
    time.sleep(3)
    start_websocket()

def on_open(ws):
    print("WebSocket connected")

def start_websocket():
    def run():
        ws = websocket.WebSocketApp(
            "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever()
    threading.Thread(target=run, daemon=True).start() #thread management optimization

# ---------- Start WebSocket ----------
start_websocket()

# ---------- Page Config ----------
st.set_page_config(page_title="GoQuant Trade Simulator", layout="wide")

# ---------- CSS Styling ----------
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 2rem; }
        .metric-box {
            background-color: #ffffff;
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-label {
            color: #888888;
            font-size: 0.9rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("## 🚀 GoQuant Real-Time Trade Simulator")

# ---------- Columns ----------
left_col, right_col = st.columns(2)

# ---------- Input Section ----------
with left_col:
    st.markdown("### 🧮 Input Parameters")

    with st.form("input_form"):
        exchange = st.selectbox("Exchange", options=["OKX"], index=0, disabled=True)
        asset = st.text_input("Spot Asset (e.g., BTC-USDT)", value="BTC-USDT")
        order_type = st.radio("Order Type", options=["Market", "Limit"], horizontal=True)

        quantity = st.slider("Quantity (~USD)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
        volatility = st.number_input("Volatility", min_value=0.0, value=0.001, step=0.0001, format="%.5f")
        fee_tier = st.selectbox("Fee Tier", ["Tier 1", "Tier 2", "Tier 3", "Custom"])

        submit_button = st.form_submit_button("🔎 Simulate Trade")

# ---------- Simulation Trigger ----------
if submit_button:
    current_order_size = quantity
    simulation_active.set()
    st.session_state["simulate"] = True

# ---------- Output Section ----------
if "simulate" in st.session_state and st.session_state["simulate"]:
    with right_col:
        st.markdown("### 📊 Output Metrics")

        slippage_ph = st.empty()
        fees_ph = st.empty()
        impact_ph = st.empty()
        cost_ph = st.empty()
        ratio_ph = st.empty()
        latency_ph = st.empty()
        trade_role = "taker" if order_type.lower() == "market" else "maker"
        while True:
            loop_start_time = time.time()
            with lock : 
                slippage = last_slippage["value"]
            if slippage is not None:
                ui_update_start = time.time()
                fees = calculate_fee(quantity, order_type=trade_role, tier=fee_tier)
                impact = compute_market_impact(quantity)
                net_cost = slippage + fees + impact
                maker_prob, taker_prob = predict_maker_taker_prob(orderbook_data, quantity, volatility, order_type)
                if maker_prob is not None:
                  maker_percent = round(maker_prob*100,1)
                  taker_percent = round(taker_prob*100,1)
                  maker_taker_ratio = f"{maker_percent}% Maker / {taker_percent}% Taker"
                else:
                 maker_taker_ratio = "N/A"
                last_update_time["end"] = time.time()
                internal_latency_ms = round((last_update_time["end"] - last_update_time["start"]) * 1000, 2) if last_update_time["start"] else "N/A"
                latency = f"{internal_latency_ms} ms"

                slippage_ph.markdown(f'<div class="metric-box"><div class="metric-label">Expected Slippage</div><div class="metric-value">📉 {slippage} bps</div></div>', unsafe_allow_html=True)
                fees_ph.markdown(f'<div class="metric-box"><div class="metric-label">Expected Fees</div><div class="metric-value">💸 {fees}</div></div>', unsafe_allow_html=True)
                impact_ph.markdown(f'<div class="metric-box"><div class="metric-label">Market Impact</div><div class="metric-value">📈 {impact}</div></div>', unsafe_allow_html=True)
                cost_ph.markdown(f'<div class="metric-box"><div class="metric-label">Net Cost</div><div class="metric-value">🧾 {net_cost}</div></div>', unsafe_allow_html=True)
                ratio_ph.markdown(f'<div class="metric-box"><div class="metric-label">Maker/Taker Proportion</div><div class="metric-value">⚖️ {maker_taker_ratio}</div></div>', unsafe_allow_html=True)
                latency_ph.markdown(f'<div class="metric-box"><div class="metric-label">Internal Latency</div><div class="metric-value">⏱️ {latency}</div></div>', unsafe_allow_html=True)
                ui_update_end = time.time()
                ui_latency = (ui_update_end - ui_update_start) * 1000
                print(f"[Latency] UI Update Latency: {ui_latency:.3f} ms")

                loop_end_time = time.time()
                total_loop_latency = (loop_end_time - loop_start_time) * 1000
                print(f"[Latency] End-to-End Loop Latency: {total_loop_latency:.3f} ms")
            time.sleep(1.5) #thread management optimization
            gc.collect() # Memory management
else:
    with right_col:
        st.info("Fill in the input parameters on the left and click **Simulate Trade**.")
