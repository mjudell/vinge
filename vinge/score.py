"""
Score the blocked data
"""
from typing import Any, Dict
import json
import openai
import pandas as pd
import re
import time

import vinge.config as config


def parse_response(val: str) -> Dict[str, Any]:
    """
    Parse and validate a response from ChatGPT

    Parameters
    ----------
    val: str
        The raw response

    Returns
    -------
    Dict[str, Any]
        Keys are left_name, right_name, and probability
    """
    matcher = re.compile(r"```json(.*)```", re.DOTALL)
    match = matcher.search(val)
    if match is not None:
        val = match.groups()[0]
    res = json.loads(val)
    prob = float(res["probability"])
    assert 0. <= prob <= 1.
    return prob


def fetch_openai_match_probs(
    left_name: str,
    right_name: str,
    client: openai.OpenAI
) -> Dict[str, str]:
    """
    Fetch the open ai probability of name match

    Parameters
    ----------
    left_name: str
        The source name
    right_name: str
        The candidate target name
    """
    system = {
        "role": "system",
        "content": "I am going to show you two abbreviated company names. Please estimate the probability the companies are the same. Return the result as json with a single field called probability.",
    }
    
    user = {
        "role": "user",
        "content": f"company_1 is {left_name.upper()} company_2 is {right_name.upper()}"
    }

    completion = client.chat.completions.create(model=config.OPENAI_MODEL, messages=[system, user])

    return parse_response(completion.choices[0].message.content)


def create_openai_request(
    left_name: str,
    right_name: str,
    id: str,
) -> str:
    """
    Create the json request

    Parameters
    ----------
    left_name: str
        The source name
    right_name: str
        The candidate target name
    id: str
        Unique identifier for the query

    Returns
    -------
    str
        A json request
    """
    system = {
        "role": "system",
        "content": "I am going to show you two abbreviated company names. Please estimate the probability the companies are the same. Return the result as json with a single field called probability.",
    }
    
    user = {
        "role": "user",
        "content": f"company_1 is {left_name.upper()} company_2 is {right_name.upper()}"
    }

    request = {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": config.OPENAI_MODEL, "messages": [system, user], "max_tokens": config.OPENAI_MAX_TOKENS}
    }

    return json.dumps(request)


def create_openai_batch_request(blocked: pd.DataFrame) -> str:
    """
    Create the batch request jsonl file

    Parameters
    ----------
    blocked: pd.DataFrame
        Result of vinge.blocking.block

    Returns
    -------
    str
        The batch requests
    """
    requests = []
    for _, row in blocked.iterrows():
        requests.append(create_openai_request(row.l_name, row.r_name, row.uuid))
    return "\n".join(requests)
