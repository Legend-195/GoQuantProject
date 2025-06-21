⚡ GoQuant Real-Time Trade Simulator
This is a high-performance real-time trading simulator built for the GoQuant internship program. The system processes live L2 order book data via WebSocket, models slippage, fees, market impact, and maker/taker behavior, and delivers instantaneous analytics via a Streamlit-based user interface.

🔍 Project Objectives
Build a realistic market simulation environment

Analyze microstructure-level trading costs

Provide accurate real-time trade analytics and visual feedback

Help traders understand execution risk and latency in fast-moving markets

💻 System Overview
The system integrates several components:

✅ WebSocket Integration
Connects to GoQuant's live L2 order book feed:
wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP

Parses real-time bid/ask levels and computes microstructure statistics like spread and depth

Feeds tick-wise data into predictive models with latency benchmarking

📈 Slippage Prediction (Quantile Regression)
Uses a quantile regression model trained on historical tick-level data

Input features: spread, order size (USD), depth

Output: slippage in basis points (bps)

50th percentile (median) is modeled to provide robust slippage expectations under asymmetric market conditions

💸 Fee Estimation (Rule-Based)
Rule-based module based on exchange fee tier (e.g., Tier 1, Tier 2, etc.)

Calculates expected transaction cost depending on order type (maker or taker)

📉 Market Impact Estimation (Almgren-Chriss Model)
Implements a dynamic programming approach based on Almgren-Chriss optimal execution theory

Models temporary and permanent price impacts

Parameters include risk aversion, volatility, and execution trajectory over time

Estimates expected price movement due to the order itself

🧠 Maker/Taker Prediction (Logistic Regression)
Uses a logistic regression classifier to estimate probability of a trade being executed as maker or taker

Input features: spread, depth, order price, order size, order type, order side

Output: binary classification → maker (0) or taker (1)

⚙️ Internal Latency Measurement
Measures the internal processing time per market tick from WebSocket input to output generation

Helps benchmark system performance under real-time constraints

🎨 Streamlit UI
The interface is built using Streamlit and includes:

🔧 Input Panel (Left)
Exchange (OKX)

Spot Asset (e.g., BTC-USDT-SWAP)

Order Type (market)

Quantity (~USD value)

Volatility (from external source)

Fee Tier (Tier 1, Tier 2, etc.)

📊 Output Panel (Right)
Expected Slippage

Expected Fees

Expected Market Impact

Net Cost (slippage + fee + market impact)

Maker/Taker Proportion

Internal Latency (seconds)

🚀 Performance Optimization Techniques
Vectorized matrix operations (NumPy) for fast computation

Pretrained ML models loaded via Pickle/Joblib

Efficient use of dynamic programming for market impact

Internal latency profiling using time.perf_counter()

Asynchronous WebSocket handling for maker/taker stream

📁 Folder Structure
bash
Copy
Edit
project-root/
│
├── models/                     # Pretrained regression/classification models
├── ui/                         # Streamlit interface code
├── utils/                      # Market impact and fee logic
├── data/                       # Historical slippage/maker-taker datasets
├── slippage_collector.py       # Main script to collect slippage + live latency
├── maker_taker_collector.py    # Script to label and classify maker/taker
├── main.py                     # Core backend integration
├── requirements.txt
└── README.md                   # You're reading this
📊 Example Metrics
Quantile Regression

MSE: 5.39e-05

R² Score: 0.79

50th percentile slippage predicted using spread, depth, and order size

Logistic Regression

Accuracy: 100%

Used for real-time classification of maker vs taker execution

🧠 Learnings
Deep understanding of order book dynamics

Execution risk modeling with financial mathematics

Real-time system design and optimization

UI/UX integration for financial analytics

📌 How to Run
bash
Copy
Edit
pip install -r requirements.txt
streamlit run ui/main.py

