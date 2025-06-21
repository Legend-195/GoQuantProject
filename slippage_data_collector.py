import websocket
import json
import csv
import time
from fee_model import calculate_fee  # Import fee calculator
import numpy as np
import threading
import pickle
import statsmodels.api as sm
import random
# Shared data & lock for thread-safe updates
latest_metrics = {}
metrics_lock = threading.Lock()

# WebSocket URL (GoQuant BTC-USDT-SWAP endpoint)
SOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"

# CSV setup
csv_filename = "slippage_data.csv"
header = ["spread", "order_size", "depth", "slippage"]

try:
    with open(csv_filename, "x", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
except FileExistsError:
    pass
latest_metrics = {}
metrics_lock = threading.Lock()
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
        idx = np.searchsorted(bins, shares)
        sell = best_moves[t, idx]
        trajectory.append(sell)
        shares -= sell
        inventory_path.append(shares)

    return np.array(inventory_path), np.array(trajectory)

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



def estimate_slippage(orderbook, usd_amount):
    
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])

    if not bids or not asks:
        return None

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2

    filled_qty = 0
    total_cost = 0

    for price, quantity in asks:
        price = float(price)
        quantity = float(quantity)
        trade_value = price * quantity

        if total_cost + trade_value >= usd_amount:
            needed_usd = usd_amount - total_cost
            needed_qty = needed_usd / price
            filled_qty += needed_qty
            total_cost += needed_qty * price
            break
        else:
            filled_qty += quantity
            total_cost += trade_value

    if filled_qty == 0:
        return None

    avg_execution_price = total_cost / filled_qty
    slippage = (avg_execution_price - mid_price) / mid_price
   
    return round(slippage*10000, 6) #in bps


# ----------------- WebSocket Handling ------------------

def on_message(ws, message):
    start_time = time.perf_counter()
    data = json.loads(message)
    bids = data.get("bids", [])
    asks = data.get("asks", [])

    if not bids or not asks:
        print("Insufficient data, skipping...")
        return

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = round(best_ask - best_bid, 6)
    depth = sum(float(qty) for _, qty in bids[:10]) + sum(float(qty) for _, qty in asks[:10])
    order_size =random.randint(50,200)
    slippage = estimate_slippage(data, usd_amount=order_size)
    fee = calculate_fee(order_amount_usd=order_size, order_type="taker", tier="Tier 1")
    impact=compute_market_impact(100)

    if slippage is not None:
        row= [ spread, order_size, depth, slippage]
        with open(csv_filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(" Fee , Market Impact , Net cost",fee, impact, slippage+fee+impact)
    else:
        print("Slippage calculation failed, skipping row.")

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")

def on_open(ws):
    print("WebSocket Connection Opened")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET_URL,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()
