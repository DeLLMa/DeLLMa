import os
import json
import sys
import warnings
from itertools import product
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable

import numpy as np
import yaml

PROJECT_ROOT = os.environ["HOME"] + "/dellma/"
sys.path.append(PROJECT_ROOT)

from utils.prompt_utils import format_query


@dataclass
class ActionConfig:
    action_enum_mode: str = "base"
    budget: int = 10


@dataclass
class StateConfig:
    state_enum_mode: str = "base"
    states: Dict[str, str] = field(default_factory=dict)
    topk: int = 3


@dataclass
class PreferenceConfig:
    pref_enum_mode: str = "base"
    sample_size: int = 50  # number of states to be sampled from the belief distribution
    minibatch_size: int = 50  # size of the minimbatch
    overlap_pct: float = 0.2  # percentage of overlap between minibatches


class DeLLMaAgent:
    temperature: float = 0.0
    belief2score: Dict[str, float] = {
        "very likely": 6,
        "likely": 5,
        "somewhat likely": 4,
        "somewhat unlikely": 3,
        "unlikely": 2,
        "very unlikely": 1,
    }

    unit: str = "units"
    product: str = "fruit"

    def __init__(
        self,
        path: str,
        raw_context_fname: str | None,
        temperature: float = 0.0,
        utility_prompt: Optional[str] = None,
        state_config: Optional[dataclass] = None,
        action_config: Optional[dataclass] = None,
        preference_config: Optional[dataclass] = None,
        agent_name: str = "farmer",
    ):
        if not os.path.exists(os.path.join(path, "reports", "summary")):
            os.makedirs(os.path.join(path, "reports", "summary"))

        self.cache_context_fname = os.path.join(
            path, "reports", "summary", raw_context_fname.split(".")[0] + ".json"
        )

        self.raw_context_fname = os.path.join(path, "reports", raw_context_fname)
        self.temperature = temperature
        self.utility_prompt = utility_prompt
        self.state_config = state_config
        self.action_config = action_config
        self.preference_config = preference_config
        if agent_name not in ["farmer", "trader"]:
            raise ValueError("Agent name must be either farmer or trader.")
        self.agent_name = agent_name  # farmer or trader

    def cache_context(
        self,
        raw_fname: str,
        cache_fname: str,
        **kwargs,
    ) -> str | Dict[str, str]:
        raise NotImplementedError

    def cache_state_beliefs(
        self,
        state_beliefs: Dict[str, Dict[str, str]],
    ):
        if hasattr(self, "source_year"):
            fname = f"cache/{self.agent_name}_{self.source_year}_states.json"
        else:
            fname = f"cache/{self.agent_name}_states.json"
        json.dump(
            state_beliefs,
            open(os.path.join(PROJECT_ROOT, fname), "w"),
            indent=4,
        )

    def load_state_beliefs(self) -> Dict[str, Tuple[List[str], List[float]]]:
        """Load and convert natural language beliefs to probabilities
        @return: belief_dist: Dict[str, Tuple[List[str], List[float]]]
                              key: names of state variables that are relevant to the agent
                              value: tuple of lists of state values and their corresponding probabilities
        """
        if hasattr(self, "source_year"):
            fname = f"cache/{self.agent_name}_{self.source_year}_states.json"
        else:
            fname = f"cache/{self.agent_name}_states.json"
        _full_state_beliefs = json.load(
            open(os.path.join(PROJECT_ROOT, fname), "r"),
        )

        self.belief_dist = {}
        for state, val2belief in _full_state_beliefs.items():
            if state in self.state_config.states:
                total_score = sum([self.belief2score[v] for v in val2belief.values()])
                self.belief_dist[state] = (
                    list(val2belief.keys()),
                    [self.belief2score[v] / total_score for v in val2belief.values()],
                )

        return self.belief_dist

    def sample_state(self) -> List[str]:
        """Sample a state from the belief distribution self.belief_dist
        @return: sampled_state: List[str]
                 a list of strings that describe the sampled state
        """
        if not hasattr(self, "belief_dist"):
            self.load_state_beliefs()
        sampled_state = []
        for state, (vals, probs) in self.belief_dist.items():
            sampled_state.append(f"{state}: {np.random.choice(vals, p=probs)}")
        return sampled_state

    def sample_state_action_pairs_batch(self) -> List[str]:
        # sample_size * len(action_strs)
        state_batch = [
            self.sample_state() for _ in range(self.preference_config.sample_size)
        ] * len(self.action_strs)
        state_batch = np.array(state_batch)
        action_batch = np.repeat(self.action_strs, self.preference_config.sample_size)
        action_batch = np.array(action_batch)
        # shuffle state and action using the same index
        idx = np.arange(len(state_batch))
        np.random.shuffle(idx)
        state_batch = state_batch[idx]
        action_batch = action_batch[idx]
        stride = int(
            self.preference_config.minibatch_size
            * (1 - self.preference_config.overlap_pct)
        )
        state_action_batch = []
        for i in range(0, len(state_batch), stride):
            minibatch = []
            j = min(
                len(state_batch),
                i + self.preference_config.minibatch_size,
            )
            for k in range(i, j):
                sampled_state = state_batch[k]
                sampled_action = action_batch[k]
                minibatch.append(
                    f"- State-Action Pair {k+1-i}. State: {', '.join(sampled_state)}; {sampled_action}"
                )
            state_action_batch.append(minibatch)
            if j == len(state_batch):
                break
        return state_action_batch

    def sample_state_action_pairs(self) -> str:
        """Sample a set of state-action pairs, wherein
            - states are sampled from the state belief distribution
            - actions are sampled uniformly from the action space
        @return: state_action_pairs: List[str]
                 a list of strings that describe the sampled state-action pairs
        """
        state_action_pairs = []
        for i in range(self.preference_config.sample_size):
            sampled_state = self.sample_state()
            sampled_action = np.random.choice(self.action_strs)
            state_action_pairs.append(
                f"- State-Action Pair {i+1}. State: {', '.join(sampled_state)}; {sampled_action}"
            )
        return state_action_pairs

    def prepare_context(self) -> str:
        raise NotImplementedError

    def prepare_actions(self) -> List[List[Tuple[str, str]]]:
        """Format actions into a list of lists of tuples, wherein each tuple is (action, budget)
        @return: actions: List[List[Tuple[str, str]]]
                 a list of lists of tuples, wherein each tuple is (action, budget)
        """
        if self.action_config is None:
            raise ValueError("Action config not found.")
        action_enum_mode = self.action_config.action_enum_mode
        budget = self.action_config.budget
        if action_enum_mode != "base":
            raise NotImplementedError

        # implement base actions
        choices = getattr(self, "choices", [])
        self.actions = [[(c, budget)] for c in choices]

        # Piecing together action strings
        self.action_strs = []
        for i, action in enumerate(self.actions):
            self.action_strs.append(
                f"Action {i+1}. "
                + ", ".join(f"{c}: {a} {self.unit}" for c, a in action)
            )
        merged_action_str = "\n".join(self.action_strs)
        return f"Below are the actions I can take:\n{merged_action_str}"

    def prepare_state_prompt(self) -> Dict[str, List[str]]:
        """Prompt the model with context and state variables, and return the model's belief distribution over the state variables
        N.B. This function is NOT used for the final DeLLMa decision prompt, but is used to generate the state belief distribution
        """
        if self.state_config is None:
            raise ValueError("State config not found.")
        state_enum_mode = self.state_config.state_enum_mode
        if state_enum_mode == "base":
            if len(self.state_config.states) > 0:
                warnings.warn("States are not used in base mode but provided.")
            state_prompt = ""
        elif state_enum_mode == "sequential":
            state_prompt = f"""I would like to adopt a decision making under uncertainty framework to make my decision. The goal of you, the decision maker, is to choose an optimal action, while accounting for uncertainty in the unknown state. Previously, you have already provided a forecast of future state variables relevant to planting decisions. The state is a vector of {len(self.state_config.states)} elements, each of which is a random variable. The state variables (and their most probable values) are enumerated below:"""
            if hasattr(self, "source_year"):
                fname = f"cache/{self.agent_name}_{self.source_year}_states.json"
            else:
                fname = f"cache/{self.agent_name}_states.json"
            _full_state_beliefs = json.load(
                open(
                    os.path.join(PROJECT_ROOT, fname),
                    "r",
                ),
            )
            for state in self.state_config.states.keys():
                state_prompt += f"\n- {state}: {_full_state_beliefs[state]}"
        else:
            raise NotImplementedError
        return state_prompt

    def prepare_preference_prompt(self) -> str | List[str]:
        """Prompt the model with context, actions, (and potentially states), and return the model's preference/decision over the actions

        Available preference enum modes:
        - base: the model is asked to make a decision based on the context and actions WITHOUT any state information
        - rank: the model is asked to rank the state-action pairs sampled from the state belief distribution and action space
        """
        if self.preference_config is None:
            raise ValueError("Preference config not found.")
        pref_enum_mode = self.preference_config.pref_enum_mode
        # implement base preference
        if pref_enum_mode == "base":
            preference_prompt = "I would like to know which action I should take based on the information provided above."
            format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the action you recommend the {self.agent_name} to take. The output format should be the same as the format of the actions listed above, e.g. {self.action_strs[0]}
- explanation: a string that describes, in detail, the reasoning behind your decision. """
            if self.product == "fruit":
                format_instruction += f"""You should include information on the expected yield and price of each fruit, as well as factors that affect them."""
            elif self.product == "stock":
                format_instruction += f"""You should include information on the expected price of each stock, as well as factors that affect them."""
            else:
                raise NotImplementedError
            return format_query(
                preference_prompt, format_instruction=format_instruction
            )
        else:
            preference_prompt = "Below, I have sampled a set of state-action pairs, wherein states are sampled from the state belief distribution you provided and actions are sampled uniformly from the action space. I would like to construct a utility function from your comparisons of state-action pairs\n\n"

            if "rank" in pref_enum_mode:
                format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the state-action pair you recommend the {self.agent_name} to take. The output format should be the same as the format of the state-action pairs listed above, e.g. State-Action Pair 5.
- rank: a list of integers that ranks the state-action pairs in decreasing rank of preference. For example, if you think the first state-action pair is the most preferred, the second state-action pair is the second most preferred, and so on. For example, [1, 2, 3, 4, 5].
- explanation: a string that describes, in detail, the reasoning behind your decision. """
                if self.product == "fruit":
                    format_instruction += f"""You should include information on the expected yield and price of each fruit, as well as factors that affect them."""
                elif self.product == "stock":
                    format_instruction += f"""You should include information on the expected price of each stock, as well as factors that affect them."""
                else:
                    raise NotImplementedError
            elif "pairwise" in pref_enum_mode:
                format_instruction = f"""You should format your response as a JSON object. The JSON object should contain the following keys:
- decision: a string that describes the state-action pair you recommend the {self.agent_name} to take. The output format should be the same as the format of the state-action pairs listed above, e.g. State-Action Pair 5.
- pair: a list of lists of integers that describes your pairwise preference between state-action pairs. For example, if you think the first state-action pair is more preferred than the second state-action pair, the second state-action pair is more preferred than the third state-action pair, and so on. For example, [[1, 2], [2, 3], [3, 4], [4, 5]].
- explanation: a string that describes, in detail, the reasoning behind your decision. """
                if self.product == "fruit":
                    format_instruction += f"""You should include information on the expected yield and price of each fruit, as well as factors that affect them."""
                elif self.product == "stock":
                    format_instruction += f"""You should include information on the expected price of each stock, as well as factors that affect them."""
                else:
                    raise NotImplementedError
        if "minibatch" in pref_enum_mode:
            state_action_batch = self.sample_state_action_pairs_batch()
        else:
            state_action_batch = [self.sample_state_action_pairs()]

        preference_prompts = []
        for state_action_pairs in state_action_batch:
            preference_prompts.append(
                format_query(
                    preference_prompt + "\n\n".join(state_action_pairs) + "\n\n",
                    format_instruction=format_instruction,
                )
            )
        return preference_prompts

    def prepare_belief_dist_generation_prompt(self) -> str:
        """Prompt the model with context and state variables to generate the model's belief distribution over the state variables
        N.B. This function is NOT used for the final DeLLMa decision prompt, but is used to generate the state belief distribution
        """
        context = self.prepare_context()

        state_prompt = ""
        if self.state_config is None:
            raise ValueError("State config not found.")
        state_enum_mode = self.state_config.state_enum_mode
        if state_enum_mode == "base":
            raise ValueError(
                "State enum mode must be cannot be base for belief dist generation."
            )
        elif state_enum_mode == "sequential":
            # enumerate through each state dimenion in tandem

            state_prompt = f"""I would like to adopt a decision making under uncertainty framework to make my decision. The goal of you, the decision maker, is to choose an optimal action, while accounting for uncertainty in the unknown state. The first step of this procedure is for you to produce a belief distribution over the future state. The state is a vector of {len(self.state_config.states)} elements, each of which is a random variable. The state variables are enumerated below:"""

            for state, desc in self.state_config.states.items():
                state_prompt += f"\n- {state}: {desc}"

            format_instruction = f"""You should format your response as a JSON object with {len(self.state_config.states.keys())} keys, wherein each key should be a state variable from the list above. 

Each key should map to a JSON object with {self.state_config.topk} keys, each of which is a string that describes the value of the state variable. Together, these keys should enumerate the top {self.state_config.topk} most likely values of the state variable. Each key should map to your belief verbalized in natural language. If the state variable is continuous (e.g. changes to a quantity), you should discretize it into {self.state_config.topk} bins.

You should strictly choose your belief from the following list: 'very likely', 'likely', 'somewhat likely', 'somewhat unlikely', 'unlikely', 'very unlikely'.
For example, if one of the state variable is 'climate condition', and the top 3 most likely values are 'drought', 'heavy precipitation', and 'snowstorm', then your response should be formatted as follows:
{{
    "climate condition": {{
        "drought": "somewhat likely",
        "heavy precipitation": "very likely",
        "snowstorm": "unlikely"
    }},
    ...
}}
"""
            state_prompt = format_query(
                state_prompt,
                format_instruction=format_instruction,
            )
        else:
            raise NotImplementedError

        return f"{context}\n\n{state_prompt}"

    def prepare_dellma_prompt(self) -> str | List[str]:
        """Implements full DeLLMa prompt for the agent"""
        context = self.prepare_context()  # how the context is prepared
        actions = self.prepare_actions()  # how the actions are enumerated
        state = self.prepare_state_prompt()  # how the state-action pair is enumerated
        preference = (
            self.prepare_preference_prompt()
        )  # how the preference/comparison is elicited
        if isinstance(preference, list):
            dellma_prompts = []
            for p in preference:
                dellma_prompts.append(
                    f"{context}\n\n{self.utility_prompt}\n\n{actions}\n\n{state}\n\n{p}"
                )
            return dellma_prompts
        return f"{context}\n\n{self.utility_prompt}\n\n{actions}\n\n{state}\n\n{preference}"

    def prepare_chain_of_thought_prompt(self, version: str = "utility") -> str:
        """Implements chain-of-thought prompt for the agent"""

        def dict_to_str(d):
            return yaml.dump(d, default_flow_style=False)

        context = self.prepare_context()
        actions = self.prepare_actions()
        # assumes base preference mode
        preference = (
            self.prepare_preference_prompt()
        )  # how the preference/comparison is elicited
        prefix = f"""{context}\n\n{self.utility_prompt}\n\n{actions}\n\n"""
        if version == "utility":
            state_prompt = f"""{prefix}First think about the unknown factors that would affect your final decisions. """
            utility_prompt_func = (
                lambda s: f"""{prefix}Now I have enumerated the unknown factors that would affect my final decisions:\n\n {dict_to_str(s)} \n\nGiven these unknow factors, think about the possiblity that each factor would occur within a {self.period}. 
    You should format your response as a JSON object, where in each key should be a factor variable listed above. 

    Each key should map to a JSON object with {self.state_config.topk} keys, each of which is a string that describes the value of the factor variable. Together, these keys should enumerate the top {self.state_config.topk} most likely values of the factor variable. Each key should map to your belief verbalized in natural language. If the factor variable is continuous (e.g. changes to a quantity), you should discretize it into {self.state_config.topk} bins.

    You should strictly choose your belief from the following list: 'very likely', 'likely', 'somewhat likely', 'somewhat unlikely', 'unlikely', 'very unlikely'.
    For example, if one of the factor variable is 'climate condition', and the top 3 most likely values are 'drought', 'heavy precipitation', and 'snowstorm', then your response should be formatted as follows:
    {{
        "climate condition": {{
            "drought": "somewhat likely",
            "heavy precipitation": "very likely",
            "snowstorm": "unlikely"
        }},
        ...
    }}"""
            )
            decision_prompt_func = (
                lambda s, u: f"""{prefix}Now I have enumerated the unknown factors that would affect my final decisions:\n\n {dict_to_str(s)} \n\nI also empirically estimated the possibility of occurrence of each possible factor:\n\n {dict_to_str(u)} \n\nGiven these unknow factors and the possibility estimates of these factors' occurrences, think about your final decision. \n\n {preference}"""
            )

            chain = [state_prompt, utility_prompt_func, decision_prompt_func]
        elif version == "reward":
            reward_prompt = f"""{prefix}For each action, think about how much money you would get if you take this action.
        You should format your response as a JSON object, where in each key should be an action listed above. Please include all actions.
        
        Each key should map to a JSON object with 3 keys, one is 'reward', which is a float that describes the reward of the action, and the other is 'unit', which is a string that describes the unit of the reward. Then you should put your reasoning of this action under key 'explanation'. For example, if the reward of action 'buy 100 shares of AMC' is 1000 dollars, then your response should be formatted as follows:
    {{
        ""action 1: buy AMC"": {{
            ""reward"": 1000,
            ""unit"": ""dollars""
            ""explanation"": ""I think the price of AMC will increase in the next month, so I will buy 100 shares of AMC.""
        }},
        ...
    }}"""
            decision_prompt_func = (
                lambda s: f"""{prefix}Now I have enumerated the my predictions of the reward of taking each possible action:\n\n {dict_to_str(s)} \n\nGiven these predictions, think about your final decision. \n\n {preference}"""
            )

            chain = [reward_prompt, decision_prompt_func]

        return chain
