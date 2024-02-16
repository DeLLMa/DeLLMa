from collections import deque
from typing import List, Tuple, Optional
import pandas as pd
from itertools import combinations

FRUITS = {
    "2021": [
        "apple",
        "avocado",
        "grape",
        "grapefruit",
        "lemon",
        "peach",
        "pear",
    ],
}

AGNOSTIC_STATES = [
    "climate condition",
    "supply chain disruptions",
    "economic health",
    "market sentiment and investor psychology",
    "political events and government policies",
    "natural disasters and other 'black swan' events",
    "geopolitical issues",
]

FRUIT_STATES = {
    "2021": {
        # product-agnostic state variables
        "agnostic": {
            "climate condition": "the climate condition of the next agricultural season in California",
            "supply chain disruptions": "the supply chain disruptions of the next agricultural season in California",
        },
        # product-specific state variables
        "specific": {
            # 'demand change': 'the demand change of the next agricultural season in California',
            "price change": lambda c: f"the change in price per unit of {c} for the next agricultural season in California",
            "yield change": lambda c: f"the change in yield of {c} for the next agricultural season in California",
        },
    },
}

STOCKS = ["AMD", "DIS", "GME", "GOOGL", "META", "NVDA", "SPY"]
STOCKS_SYMBOL_TO_NAME_MAP = {
    "AMD": "Advanced Micro Devices",
    "DIS": "The Walt Disney Company",
    "GME": "GameStop Corp",
    "GOOGL": "Alphabet, i.e. Google",
    "META": "Meta Platforms, i.e. Facebook",
    "NVDA": "NVIDIA",
    "SPY": "S&P 500",
}


def get_combinations(
    agent_name: str, source_year: Optional[str] = None
) -> List[Tuple[str, ...]]:
    combs = []
    if agent_name == "farmer":
        products = FRUITS[source_year]
    elif agent_name == "trader":
        products = STOCKS
    else:
        raise ValueError("agent_name must be either 'farmer' or 'trader'")

    for i in range(2, len(products) + 1):
        for c in combinations(products, i):
            combs.append(c)

    return combs


def merge_by_commodity(
    df_x: pd.DataFrame | str,
    df_y: pd.DataFrame | str,
    on: str = "Commodity",
) -> pd.DataFrame:
    if type(df_x) == str:
        df_x = pd.read_csv(df_x)
    if type(df_y) == str:
        df_y = pd.read_csv(df_y)
    df = pd.merge(df_x, df_y, on=on)
    return df
