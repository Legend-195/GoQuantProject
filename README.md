# ⚡ GoQuant Real-Time Trade Simulator

This project is a **real-time trade execution simulator** developed as part of the GoQuant internship selection challenge. It is built in Python and processes **live Level 2 (L2) order book data** from OKX via WebSocket, applying advanced models to simulate realistic trading costs and market behavior.

## 📈 Key Features

### ✅ Real-Time WebSocket Integration
- Live L2 order book stream from:  
  `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`
- Continuously updates with each market tick for accurate, low-latency decision-making.

### 🔮 Execution Parameter Prediction

#### 1. **Expected Slippage**
- Modeled using **Quantile Regression** (`statsmodels.QuantReg`) at the 50th percentile.
- Inputs: `spread`, `USD amount`, `depth`
- Reflects the median execution cost in basis points (bps).

#### 2. **Expected Fees**
- Calculated using a **rule-based fee model** depending on order type (maker/taker) and tier level.
- Example: `Tier 1` taker fee = 0.06%, maker fee = 0.02%

#### 3. **Expected Market Impact**
- Estimated using the **Almgren-Chriss Optimal Execution Model** with dynamic programming.
- Incorporates:
  - **Temporary Impact**: \( \eta \cdot v^\alpha \)
  - **Permanent Impact**: \( \gamma \cdot v^\beta \)
  - **Execution Risk**: depends on volatility, inventory, and time horizon

#### 4. **Net Execution Cost**
- Computed as:  
  \[
  \text{Net Cost} = \text{Slippage} + \text{Fees} + \text{Market Impact}
  \]

#### 5. **Maker/Taker Proportion**
- Classified using **Logistic Regression** trained on labeled WebSocket snapshots.
- Inputs: `order_price`, `order_type`, `side`, `spread`, `depth`, `bid/ask levels`
- Output: probability of a trade being a taker.

#### 6. **Internal Latency**
- Measures tick-wise processing time using `time.perf_counter()` for profiling system performance.

---

## 🧠 Machine Learning Models Used

### 📌 Quantile Regression (Slippage)
- Library: `statsmodels`
- Trained on features: `spread`, `usd_amount`, `depth`
- Model Results:
  - **MSE**: 5.39e-05  
  - **R²**: 0.798  
  - **50th percentile** (median): chosen to reflect realistic execution cost in volatile financial environments.

### 📌 Logistic Regression (Maker/Taker)
- Library: `scikit-learn`
- Model Accuracy: **100%** on labeled real-time data  
- Uses one-hot encoded `order_type` and `side` features  
- Highly interpretable and suitable for binary classification

---

## 📊 UI Integration with Streamlit

A clean, two-panel layout:
- **Left Panel** – User Inputs:
  - Exchange (OKX)
  - Asset (BTC-USDT-SWAP)
  - Order Type
  - Quantity (USD)
  - Volatility
  - Fee Tier

- **Right Panel** – Real-Time Outputs:
  - Predicted Slippage
  - Fees
  - Market Impact
  - Net Cost
  - Maker/Taker Probability
  - Internal Latency

---

## ⚙️ Performance Optimizations

- Efficient NumPy-based dynamic programming for the Almgren-Chriss model
- Minimal latency via direct in-memory processing of order book JSON messages
- Quantile regression fallback for missing data ensures model robustness
- Lazy evaluation of model inference to reduce computation during idle ticks

---

## 📦 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
