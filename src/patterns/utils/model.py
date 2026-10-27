from dataclasses import dataclass
from time import perf_counter
@dataclass
class LLM_RESPONSE:
    resp: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def call_model(system: str, user: list[dict], client, model, max_tokens = 8000):
    
    s = perf_counter()
    r = client.messages.create(
        max_tokens=max_tokens,
        messages=user,
        system=system,
        model=model
    )
    
    latency = perf_counter() - s
    
    input = r.usage.input_tokens
    output = r.usage.output_tokens
    
    text = "".join(b.text for b in r.content if b.type=='text')
    
    return LLM_RESPONSE(resp=text, stop_reason=r.stop_reason, input_tokens=input, output_tokens=output, latency_s=latency)
    