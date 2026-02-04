import os
import asyncio
from typing import Optional, List, Dict, Any

from mistralai import Mistral
from mistralai.models import SystemMessage, UserMessage


client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


SYNTHESIZE_PROMPT = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:\n"

async def _run_llm(client: Mistral, agent: Dict[str, Any], user_prompt: str, system_prompt: Optional[str]) -> str:
    """Run a single LLM call with a model (async, per Mistral docs)."""
    sys = system_prompt or agent.get("system", "")

    messages = []
    if sys:
        messages.append(SystemMessage(content=sys))
    messages.append(UserMessage(content=user_prompt))

    response = await client.chat.complete_async(
        model=agent["model"],
        messages=messages,
        temperature=agent.get("temperature", 0.7),
        max_tokens=agent.get("max_tokens", 512),
    )
    return response.choices[0].message.content

def _pack(prev_responses: list[str]) -> str:
    """Pack previous responses into a single string."""
    return "\n".join([f"{i+1}. {resp}" for i, resp in enumerate(prev_responses)])

def _synthesize_responses(synthesize_prompt: str, prev_responses: list[str]) -> str:
    """Synthesize previous responses into a single system prompt for the next layer."""
    return synthesize_prompt + _pack(prev_responses)

async def run_layer(client: Mistral, agents: List[Dict[str, Any]], synthesized_message: Optional[str], user_prompt: str, sem: asyncio.Semaphore) -> List[str]:
    async def one(agent: Dict[str, Any]) -> str:
        async with sem:
            return await _run_llm(
                client=client,
                agent=agent,
                user_prompt=user_prompt,
                system_prompt=synthesized_message or agent.get("system", ""),
            )

    return await asyncio.gather(*(one(a) for a in agents))

async def run_moa(
    user_prompt: str,
    proposer_layers: List[List[Dict[str, Any]]],
    aggregator: Dict[str, Any],
    synthesize_prompt: str,
    concurrency: int = 8,
) -> str:
    sem = asyncio.Semaphore(concurrency)
    
    async def call(agent: Dict[str, Any], system_prompt: str, user_prompt: str) -> str:
        async with sem:
            return await _run_llm(client, agent, user_prompt, system_prompt)

    # Layer 1 
    results = await asyncio.gather(*[
        call(a, a.get("system", ""), user_prompt) for a in proposer_layers[0]
    ])
    
    print(f"\n[INFO] Layer 1 results:")
    for i, result in enumerate(results):
        print(f"Agent {i+1}: {result}\n")

    # Layers 2..k (each layer sees previous outputs + original question)
    for layer_agents in proposer_layers[1:]:
        layered_system = _synthesize_responses(synthesize_prompt, results)
        results = await asyncio.gather(*[
            call(a, layered_system, user_prompt) for a in layer_agents
        ])
        
        print(f"\n[INFO] Layer {proposer_layers.index(layer_agents) + 2} results:")
        for i, result in enumerate(results):
            print(f"Agent {i+1}: {result}\n")

    # Final aggregation
    final_system = _synthesize_responses(synthesize_prompt, results)
    final = await call(aggregator, final_system, user_prompt)
    print(f"\n[INFO] Final aggregated result: {final}")
    return final


if __name__ == "__main__":
    
    proposer_layers = [
        [   # Layer 1 
            {"model": "mistral-small-2506", "temperature": 0.4},
            {"model": "ministral-14b-2512", "temperature": 0.9},
            {"model": "ministral-8b-2512", "temperature": 0.7},
            {"model": "ministral-3b-2512", "temperature": 0.6},
        ],
    ]
    
    # Layer 2
    aggregator = {"model": "mistral-large-2512", "temperature": 0.0}
