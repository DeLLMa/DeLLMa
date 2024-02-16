import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Optional
import pandas as pd

from agent.agent import (
    PROJECT_ROOT,
    DeLLMaAgent,
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)

sys.path.append(PROJECT_ROOT)
from utils.data_utils import STOCKS, STOCKS_SYMBOL_TO_NAME_MAP


class TradeAgent(DeLLMaAgent):
    stocks: List[str] = STOCKS
    stocks_symbol_to_name_map: Dict[str, str] = STOCKS_SYMBOL_TO_NAME_MAP
    system_content = (
        "You are a expert economist helping traders decide what stock to buy today."
    )
    period = "month"

    states: Dict[str, Dict[str, str]] = {
        # product-agnostic state variables
        "agnostic": {
            "economic health": "the overall health of the economy significantly impacts stock prices. Indicators such as GDP growth, unemployment rates, inflation, and interest rates can affect investor sentiment and thus stock prices. For instance, high inflation or rising interest rates often lead to a decrease in stock prices",
            "market sentiment and investor psychology": "The collective mood and expectations of investors can greatly influence stock prices. This can include reactions to news, trends, and other market movements. Investor psychology often drives market trends and bubbles, and can lead to overvaluation or undervaluation of stocks",
            "political events and government policies": "Political stability, government policies, and regulatory changes can have a significant impact on stock prices. For example, new regulations in a sector can increase costs for companies, affecting their profits and stock prices. Conversely, deregulation or favorable policies can lead to stock market rallies",
            "natural disasters and other 'black swan' events": "Natural disasters, pandemics, and other unexpected events can have a significant impact on stock prices. For example, the COVID-19 pandemic caused a sharp drop in stock prices in early 2020, as investors reacted to the uncertainty and economic disruption caused by the virus",
            "geopolitical issues": "Events such as wars, international conflicts, or major political changes in one country can have ripple effects across the global economy. These events can affect investor confidence and risk tolerance, leading to fluctuations in stock markets around the world",
        },
        # product-specific state variables
        "specific": {
            f"merges and major acquisitions related to": f"Announcements of mergers and acquisitions can have a substantial effect on stock prices. Typically, the stock of a company being acquired rises, while the acquiring company's stock might experience mixed reactions depending on the perceived benefits and costs of the acquisition.",
            f"regulatory changes and legal issues happened to": f"Changes in regulations or legal challenges specific to a company or its industry can affect its stock price. For example, new environmental regulations might impact energy companies, or antitrust investigations can target tech giants, affecting their stocks.",
            f"financial health of": f"A company's financial health, as revealed in its earnings reports, balance sheets, and future earnings guidance, can heavily influence its stock price. Positive news, like higher than expected profits or successful new product launches, can boost stock prices, while negative news like losses or declining sales can cause them to drop.",
            f"company growth of": f"A company's growth prospects can have a significant impact on its stock price. For example, if a company is expected to grow rapidly in the future, its stock price might rise, while a company with stagnant growth might see its stock price decline.",
            f"company product launches of": f"New product launches can have a significant impact on a company's stock price. For example, if a company launches a new product that is expected to be successful, its stock price might rise, while a product that is seen as a failure can cause it to drop.",
        },
    }

    unit: str = "dollars"
    product: str = "stock"

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(PROJECT_ROOT, "data/stocks/"),
        raw_context_fname: str = "stock.csv",
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[dataclass] = None,
        preference_config: Optional[dataclass] = None,
        agent_name: str = "trader",
        history_length: int = 24,
    ):
        assert set(choices).issubset(set(self.stocks))
        self.choices = sorted(set(choices))
        self.path = path
        utility_prompt = f"I'm a trader planning my next move. I would like to maximize my profit with '{action_config.budget}' dollars."
        self.source_date = "2023-12-01"
        self.target_date = "2023-12-29"
        super().__init__(
            path,
            raw_context_fname,
            temperature,
            utility_prompt,
            state_config,
            action_config,
            preference_config,
            agent_name,
        )
        self.history_length = history_length

        if (
            self.state_config.state_enum_mode != "base"
            and len(self.state_config.states) == 0
        ):
            self.state_config.states = self._format_state_dict()

    def _format_state_dict(self):
        state2desc = self.states["agnostic"].copy()
        for choice, variable in product(
            self.choices, sorted(self.states["specific"].keys())
        ):
            state2desc[
                f"{variable} {self.stocks_symbol_to_name_map[choice]} ({choice.upper()})".lower()
            ] = self.states["specific"][variable]
        return state2desc

    def _format_stock_context(self, stock_symbol: str):
        df = pd.read_csv(os.path.join(self.path, f"{stock_symbol.upper()}.csv"))
        df_month = df.groupby(pd.PeriodIndex(df["Date"], freq="M"))["Close"].mean()
        query = f"""Below are the information about stock {stock_symbol} (i.e. {self.stocks_symbol_to_name_map[stock_symbol]}). Units are in dollars per share.
    Current Price: {df[df.Date == self.source_date]['Open'].values[0]:.2f}.
    Historical Prices:\n"""
        for month in df_month.index[-self.history_length - 1 : -1]:
            query += f"""\t{month}: {df_month[month]:.2f}\n"""
        query += "\n"
        return query

    def prepare_context(self) -> str:
        context = f"""Below are the stocks I am considering: {", ".join(self.choices)}. I would like to know which stock I should buy based on the information of their historical prices in the last {self.history_length} months.
I can only buy one stock and I have a budget of {self.action_config.budget} dollars. I would like to maximize my profit. Today is {self.source_date}. I'm buying stocks today and will sell them at the end of the month ({self.target_date}).\n\n"""
        for p in self.choices:
            context += self._format_stock_context(p)
        return context


if __name__ == "__main__":
    # Example to produce the belief distribution prompt
    agent = TradeAgent(
        choices=["AMD", "DIS", "GME", "GOOGL", "META", "NVDA", "SPY"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(),
    )
    belief_distribution_prompt = agent.prepare_belief_dist_generation_prompt()

    # Example to produce the full dellma prompt
    agent = TradeAgent(
        choices=["AMD", "DIS", "GME", "GOOGL", "META", "NVDA", "SPY"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(pref_enum_mode="order", sample_size=50),
    )
    dellma_prompt = agent.prepare_dellma_prompt()
    print(dellma_prompt)
