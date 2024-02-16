import os
import json
from typing import List, Dict, Callable
from time import sleep
import pandas as pd

import openai
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY"]
CLIENT = OpenAI()

SUMMARY_PROMPT = (
    "You are a helpful agricultural expert studying a report published by the USDA"
)
ANALYST_PROMPT = "You are a helpful agricultural expert helping farmers decide what produce to plant next year."


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"


def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
):
    success = False
    while not success:
        try:
            response = CLIENT.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query + "<json>"},
                ],
                temperature=temperature,
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response = response.choices[0].message.content
    except Exception as e:
        print(e)
        response = ""

    try:
        response = json.loads(response.lower())
    except:
        response = response

    return response


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.7,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
):
    responses = []
    for _ in range(num_samples):
        if use_chain_of_thought:
            response = chain_of_thought_inference(
                chain=query, system_content=system_content, temperature=temperature
            )["response"]
        else:
            response = inference(query, system_content, temperature)
        responses.append(response)

    decisions = [r["decision"] for r in responses]
    majority_decision = max(set(decisions), key=decisions.count)
    response = {
        "decision": majority_decision,
        "explanation": responses,
    }
    return response


def chain_of_thought_inference(
    chain: List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.5,
):
    history = {}
    for query in chain:
        if isinstance(query, str):
            response = inference(query, system_content, temperature)
        else:
            previous_results = [history[k] for k in history.keys()]
            query = query(*previous_results)
            response = inference(query, system_content, temperature)
        history[query] = response

    return {
        "query": [{"prompt": q, "response": r} for q, r in history.items()],
        "response": response,
    }


def summarize(
    fname: str, products: List[str], temperature: float = 0.0
) -> Dict[str, str]:
    products = sorted(p.lower() for p in products)
    summary_fname = fname.split(".")[0] + "-" + "-".join(products) + ".json"
    if os.path.exists(summary_fname):
        # print(f"Summary file {summary_fname} already exists.")
        return json.load(open(summary_fname))

    report = open(fname).read()
    query = f"Below is an agriculture report published by the USDA:\n\n{report}\n\n"

    format_instruction = f"""Please write a detailed summary of the report.

You should format your response as a JSON object. The JSON object should contain the following keys:
    overview: a string that describes, in detail, the overview of the report. Your summary should focus on factors that affect the overall furuit and nut market.
    """
    for p in products:
        format_instruction += f"""
    {p}: a string that describes, in detail, information pertaining to {p} in the report. You should include information on {p} prices and production, as well as factors that affect them. 
        """
    query = format_query(query, format_instruction)
    response = inference(query, SUMMARY_PROMPT, temperature)
    try:
        response = json.loads(response.lower())
    except:
        response = response
    with open(summary_fname, "w") as f:
        json.dump(response, f, indent=4)
    return response
