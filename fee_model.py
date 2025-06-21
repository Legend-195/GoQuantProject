# Filename: fee_model.py

def calculate_fee(order_amount_usd, order_type="taker", tier="Tier 1"):
    """
    Calculates estimated trading fee for a given order type and fee tier.
    
    Parameters:
    - order_amount_usd: float, value of order in USD
    - order_type: str, either 'maker' or 'taker'
    - tier: str, one of 'Tier 1', 'Tier 2', 'Tier 3'

    Returns:
    - float, estimated fee in USD (rounded to 6 decimal places)
    """
    FEE_TIERS = {
        "Tier 1": {"maker": 0.0008, "taker": 0.001},
        "Tier 2": {"maker": 0.0006, "taker": 0.0008},
        "Tier 3": {"maker": 0.0005, "taker": 0.0007}
    }

    if tier not in FEE_TIERS:
        raise ValueError(f"Invalid fee tier: {tier}")

    if order_type not in FEE_TIERS[tier]:
        raise ValueError(f"Invalid order type: {order_type}")

    fee_rate = FEE_TIERS[tier][order_type]
    fee = order_amount_usd * fee_rate
    return round(fee, 6)
