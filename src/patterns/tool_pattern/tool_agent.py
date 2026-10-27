from .tool import tool
from dataclasses import dataclass
from typing import Literal
from .tool import validate_arguements
from ..utils.model import call_model
import re
import json
from .tool import Tool

TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>

If a tool call isn't needed to answer the question you should answer the question directly.
"""


@dataclass
class ToolResult:
    resp: str
    stop_reason: Literal['end', 'max_tool_calls', 'tool_call_repeated']
    tools_used: list
    input_tokens: int
    output_tokens: int
    latency: float

class ToolAgent:
    
    def _tool_signatures(self):
        
        return "".join(t.fn_signature for t in self.tools)
    
    def __init__(self, client, model_name, tools: Tool | list[Tool], system_prompt: str | None = None, tag = 'tool_call'):
        
        
        self.tools = tools if isinstance(tools, list) else [tools]
        sys = TOOL_SYSTEM_PROMPT % self._tool_signatures()
            
        self.system_prompt = system_prompt if system_prompt is not None else sys
        self.client = client
        self.model_name = model_name
        self.tag = tag
    
    
    def _extract_call(self, resp: str) -> tuple[bool, list[dict] | None, list[str] | None]:
        
        result = re.findall(rf'<{self.tag}>(.*?)</{self.tag}>', resp, re.DOTALL)
        
        if len(result) == 0:
            return False, None, [f"Opening and closing brackets must be found for tool call. EX: <{self.tag}>TOOL CALL</{self.tag}>. Found: {resp}"]
        
        for r in result:
            errors = []
            valid = []
            try:
                t = json.loads(r.strip())
            
                valid.append(t)
        
            except json.JSONDecodeError as e:
            
                errors.append(f"Invalid JSON detected. Tool Call failed: {e}") 
        
        if len(valid) == 0:
            return False, None, errors
        
        return True, valid, errors
    
         
    def run(self, user_prompt: str, max_calls: int = 3, max_repeats: int = 3):
        
        user_history = [{'role': 'user', 'content': user_prompt}]
    
        tool_mapping = {t.name.lower(): t for t in self.tools}
        
        last_call = None
        repeated = 0
        tools_used = []
        t_input = 0
        t_output = 0
        t_latency = 0.0
        output_reason = None
        
        for i in range(1, max_calls + 1):
            
            r = call_model(system=self.system_prompt, user=user_history, client=self.client, model=self.model_name)
            t_input += r.input_tokens
            t_output += r.output_tokens
            t_latency += r.latency_s
            
            resp = r.resp
            user_history.append({'role': 'assistant', 'content': resp})
            
            if f'<{self.tag}>' in resp:
                tc = self._extract_call(resp)
                
                if not tc[0]:
                    user_history.append({'role': 'user', 'content': tc[2]})
                    
                else:
                    
                    errors = []
                    obsv = []
                    for tool in tc[1]:
                        tool_name = tool['name']
                    
                        t = tool_mapping.get(tool_name.lower())
                    
                        if t is None:
                            errors.append(f'Invalid tool name. Tool must be apart of the provided set. Got: {tool_name}.')
                            continue
                    
                        tool_call = validate_arguements(tool, json.loads(t.fn_signature))
                        args = tool_call['arguments']
                    
                        result = t.fn(**args)
                    
                        obsv.append(f'Observation: {result}.')
                    
                        tools_used.append(tool_name)
                    
                        if last_call is not None and last_call[0] == tool_name and last_call[1] == args:
                            repeated += 1
                        else:
                            repeated = 0
                    
                        if repeated >= max_repeats:
                            output_reason = "tool_call_repeated"
                            user_history.append({'role': 'user', 'content': "".join(obsv)})
                            break
                        
                        last_call = (tool_name, args)
                    
                    tot_errors = errors + tc[2]
                    
                    if len(tot_errors) != 0:
                        
                        for e in tot_errors:
                            obsv.append(e)
                    
                    user_history.append({'role': 'user', 'content': " ".join(obsv)})
            else:
                
                return ToolResult(resp=resp, stop_reason='end', tools_used=tools_used, input_tokens=t_input, output_tokens=t_output, latency=t_latency)
    
        final_prompt = self.system_prompt + "You have run out of tool calls. You must now provide your answer to the question."
        
        output = call_model(system=final_prompt, user=user_history, client=self.client, model=self.model_name)
        t_input += output.input_tokens
        t_output += output.output_tokens
        t_latency += output.latency_s
        
        return ToolResult(resp=output.resp, stop_reason='max_tool_calls' if output_reason is None else output_reason, tools_used=tools_used, input_tokens=t_input,
                          output_tokens=t_output, latency=t_latency)
            