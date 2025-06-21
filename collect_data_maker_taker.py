import asyncio
import websockets
import json
import csv
import os
import random

async def collect_data():
    uri = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    retries = 0

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                print("✅ Connected to WebSocket")

                while True:
                    message = await websocket.recv()
                    data = json.loads(message)

                    best_bid = float(data["bids"][0][0])
                    best_ask = float(data["asks"][0][0])
                    spread = best_ask - best_bid
                    depth_bid = sum(float(bid[1]) for bid in data["bids"][:5])
                    depth_ask = sum(float(ask[1]) for ask in data["asks"][:5])
                    depth = depth_bid + depth_ask
                    imbalance = (depth_bid - depth_ask) / depth if depth != 0 else 0

                    # 🔁 Randomized order features
                    order_type = random.choice(["market", "limit"])
                    side = random.choice(["buy", "sell"])
                    volatility = round(random.uniform(5, 50), 2)
                    order_size_usd = round(random.uniform(50, 1000), 2)

                    # ✅ Maker/taker classification logic
                    if order_type == "market":
                        maker_taker = "taker"
                        limit_price = None
                    else:
                        mid_price = (best_bid + best_ask) / 2
                        limit_price = round(mid_price + random.uniform(-0.5, 0.5), 2)
                        if side == "buy":
                            maker_taker = "maker" if limit_price < best_ask else "taker"
                        else:
                            maker_taker = "maker" if limit_price > best_bid else "taker"

                    row = {
                        "spread": spread,
                        "imbalance": imbalance,
                        "volatility": volatility,
                        "order_size_usd": order_size_usd,
                        "side": side,
                        "order_type": order_type,
                        "maker_taker": maker_taker
                    }

                    print(row)

                    file_exists = os.path.exists("maker_taker_data.csv")
                    with open("maker_taker_data.csv", mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=row.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)

                    await asyncio.sleep(1)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"⚠️ WebSocket closed: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            retries += 1
            if retries > 5:
                print("❌ Too many retries. Exiting.")
                break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(collect_data())
