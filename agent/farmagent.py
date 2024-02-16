import os
import sys
import json
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
from utils.prompt_utils import inference, format_query
from utils.data_utils import merge_by_commodity, FRUITS, FRUIT_STATES


class FarmAgent(DeLLMaAgent):
    products: Dict[str, List[str]] = FRUITS
    states = FRUIT_STATES
    system_content = "You are a helpful agricultural expert helping farmers decide what produce to plant next year."
    period = "year"
    unit: str = "acres"
    product: str = "fruit"

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(PROJECT_ROOT, "data/agriculture/"),
        raw_context_fname: str = "fruit-sept-2021.txt",
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[dataclass] = None,
        preference_config: Optional[dataclass] = None,
        agent_name: str = "farmer",
    ):
        self.choices = sorted(set(choices))
        source_year = raw_context_fname.split("-")[-1].split(".")[0]
        target_year = str(int(source_year) + 1)
        assert set(choices).issubset(set(self.products[source_year]))
        self.stats = merge_by_commodity(
            os.path.join(path, "stats", f"CA-{source_year}.csv"),
            os.path.join(path, "stats", f"CA-{target_year}.csv"),
        )
        utility_prompt = f"I'm a farmer in California planning what fruit to plant next year. I would like to maximize my profit with '{action_config.budget}' acres of land."

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

        self.source_year = source_year

        if (
            self.state_config.state_enum_mode != "base"
            and len(self.state_config.states) == 0
        ):
            self.state_config.states = self._format_state_dict()

        self.context_cache = self.cache_context(
            self.raw_context_fname,
            self.cache_context_fname,
            self.stats,
        )

    def _format_state_dict(self):
        state2desc = self.states[self.source_year]["agnostic"].copy()
        for choice, variable in product(
            self.choices, sorted(self.states[self.source_year]["specific"].keys())
        ):
            state2desc[f"{choice} {variable}"] = self.states[self.source_year][
                "specific"
            ][variable](choice)
        return state2desc

    def _format_summary_prompt(self, report: str):
        query = f"Below is an agriculture report published by the USDA:\n\n{report}\n\n"
        format_instruction = f"""Please write a detailed summary of the report.
You should format your response as a JSON object. The JSON object should contain the following keys:
- 'summary': a string that summarize, in detail, the overview of the report. Your summary should include price, yield, production, and other information relevant to a farmer making decisions about what to plant. You should also include key factors, such as weather, supply chain, and demand, that affect the market."""
        for p in self.products[self.source_year]:
            format_instruction += f"""
- '{p}': a string that describes, in detail, information pertaining to {p} in the report. You should include information on {p} prices and production, as well as factors that affect them."""
        format_instruction += f"""
- 'factors': a list of strings that enumerates the factors that affect the market, based on the report. You should include at least 5 factors, ranked in decreasing order of importance."""
        query = format_query(query, format_instruction)
        return query

    def cache_context(
        self,
        raw_fname: str,
        cache_fname: str,
        price_yield_df: pd.DataFrame,
    ):
        if cache_fname is not None and os.path.exists(cache_fname):
            return json.load(open(cache_fname, "r"))

        query = self._format_summary_prompt(open(raw_fname).read())
        response = inference(query, temperature=self.temperature)

        if type(response) != dict:
            raise NotImplementedError

        for p in self.products[self.source_year]:
            if p not in price_yield_df["Commodity"].values:
                continue
            response[p] = {
                "summary": response[p],
                "yield": price_yield_df[price_yield_df["Commodity"] == p][
                    "Yield_x"
                ].item(),
                "price": price_yield_df[price_yield_df["Commodity"] == p][
                    "Price per Unit_x"
                ].item(),
            }
        json.dump(response, open(cache_fname, "w"), indent=4)
        return response

    def _format_product_context(
        self, product: str, context_cache: Dict[str, str | Dict[str, str]]
    ):
        return f"""
- {product}:
    - Product Summary: {context_cache['summary']}
    - California Price and Yield Statistics: the average {product} yield is {context_cache['yield']} and the average price per unit is {context_cache['price']}."""

    def prepare_context(
        self,
        context_cache: Optional[Dict[str, str]] = None,
    ) -> str:
        if context_cache is None:
            context_cache = getattr(self, "context_cache", None)

        if context_cache is None:
            raise ValueError("Context cache not found.")

        context = f"""Below is an agriculture report published by the USDA. It gives an overview of the fruit and nut market in the United States, with an additional focus on information pertaining to {", ".join(self.choices)}.

Market Overview: {context_cache['summary']}
"""
        for p in self.choices:
            context += self._format_product_context(p, context_cache[p])
        return context


if __name__ == "__main__":
    # Example to produce the belief distribution prompt
    agent = FarmAgent(
        choices=["apple", "avocado", "grape", "grapefruit", "lemon", "peach", "pear"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(),
    )
    belief_distribution_prompt = agent.prepare_belief_dist_generation_prompt()
    """
    # Generate belief distribution:
    belief_distribution = inference(
        belief_distribution_prompt,
        system_content="You are a helpful agricultural expert helping farmers decide what produce to plant next year.",
        temperature=0.0,
    )
    """
    # Example to produce the full dellma prompt
    agent = FarmAgent(
        choices=["apple", "avocado", "grape", "grapefruit", "lemon", "peach", "pear"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(
            pref_enum_mode="pairwise-minibatch", sample_size=50
        ),
    )
    dellma_prompt = agent.prepare_dellma_prompt()
    # print(dellma_prompt)
    """
    # Generate dellma response:
    dellma_response = inference(
        dellma_prompt,
        system_content="You are a helpful agricultural expert helping farmers decide what produce to plant next year.",
        temperature=0.0,
    )
    """
