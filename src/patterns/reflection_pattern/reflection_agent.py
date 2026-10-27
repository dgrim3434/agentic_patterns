from dataclasses import dataclass
from ..utils.model import call_model
from typing import Literal
BASE_GENERATION_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be imporved, output a list of recommendations and critiques.
If the user content is ok and there's nothing to change, output this: <OK>
"""

BASE_MODEL = 'claude-haiku-5-5'
MAX_TOKENS_CAP = 10_000

@dataclass
class ReflectionResult:
    output: str | None
    revisions: list[str]
    reflections: list[str]
    steps: int
    stop_reason: Literal['max_steps', 'no_change', 'max_token_cap']
    input_tokens: int
    output_tokens: int
    latency: float

class ReflectionAgent: 
    
    def __init__(self, client):
        
        self.client = client
    
    
    def run(
        self,
        user_msg: str,
        generation_system_prompt: str | None = None,
        reflection_system_prompt: str | None = None,
        model: str | None = None,
        n_steps: int = 4,
        max_tokens: int = 8000,
    ):
        
        if n_steps < 1:
            raise ValueError("ERROR: n_steps must be atleast 1")
        
        if len(user_msg) == 0:
            raise ValueError("ERROR: User message cannot be empty")
        
        generation_system_prompt = BASE_GENERATION_PROMPT if generation_system_prompt is None else generation_system_prompt
        reflection_system_prompt = BASE_REFLECTION_SYSTEM_PROMPT if reflection_system_prompt is None else reflection_system_prompt
        model = BASE_MODEL if model is None else model
        
        step = 1
        
        generation_history = [{'role': 'user', 'content': user_msg}]
        reflection_history = []
        total_latency = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        
        
        revisions = []
        reflections = []
        curr_max = max_tokens
        stop_sequence = False
        max_token_hits = 0
        
        while step < n_steps + 1:
            
            generation = call_model(system=generation_system_prompt, user=generation_history, client=self.client, max_tokens=curr_max, model=model)
            
            total_latency += generation.latency_s
            total_input_tokens += generation.input_tokens
            total_output_tokens += generation.output_tokens
            
            if generation.stop_reason == "max_tokens":
                max_token_hits +=1 
                if max_token_hits == 3:
                    return ReflectionResult(output=None, revisions=[], reflections=[], steps=step, stop_reason='max_token_cap', input_tokens=total_input_tokens, output_tokens=total_output_tokens, latency=total_latency)
                curr_max = int(min(curr_max * 1.5, MAX_TOKENS_CAP))
                continue
            
            output = generation.resp
            revisions.append(output)
            generation_history.append({'role': 'assistant', 'content': output})
            
            reflection_history.append({'role': 'user', 'content': output})
            
            reflection = call_model(system=reflection_system_prompt, user=reflection_history, client=self.client, max_tokens=curr_max, model=model)
            
            total_latency += reflection.latency_s
            total_input_tokens += reflection.input_tokens
            total_output_tokens += reflection.output_tokens
            
            if reflection.stop_reason == "max_tokens":
                max_token_hits += 1
                if max_token_hits == 3:
                    return ReflectionResult(output=None, revisions=[], reflections=[], steps=step, stop_reason='max_token_cap', input_tokens=total_input_tokens, output_tokens=total_output_tokens, latency=total_latency)
                curr_max = int(min(curr_max * 1.5, MAX_TOKENS_CAP))
                continue
            
            generation_history.append({'role': 'user', 'content': reflection.resp})
            reflection_history.append({'role': 'assistant', 'content': reflection.resp})
            
            reflections.append(reflection.resp)

            if "<OK>" in reflection.resp:
                stop_sequence = True
                break
            
            step += 1
        
        return ReflectionResult(output=revisions[-1], revisions=revisions, reflections=reflections, steps=step, stop_reason= 'no_change' if stop_sequence else 'max_steps',
                                input_tokens=total_input_tokens, output_tokens=total_output_tokens, latency=total_latency)