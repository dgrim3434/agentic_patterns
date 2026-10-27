from ..tool_pattern.tool import Tool, validate_arguements
from ..utils.model import call_model
from dataclasses import dataclass
import json
import re

REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don' make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


@dataclass
class ReactStep:
    thought: str | None
    tool_call: str
    input_tokens: int
    output_tokens: int
    latency: float
    

@dataclass
class ReactResult:
    resp: str | None
    steps: list[ReactStep]
    input_tokens: int
    output_tokens: int
    latency: float

class ReactAgent:
    
    def _generate_prompt(self):
        
        peices = "\n".join(t.fn_signature for t in self.tools)
        
        return REACT_SYSTEM_PROMPT % peices
    
    def __init__(self, model_name: str, client, tools: Tool | list[Tool] = None, system_prompt: str | None=None):
        
        tools = tools if tools is not None else []
        
        self.tools = tools if isinstance(tools, list) else [tools]
        
        sys_prompt = self._generate_prompt()
        
        self.system = system_prompt if system_prompt is not None else sys_prompt
        self.model_name = model_name
        self.client = client
    
    def extract_tags(self,text, tag):
        
        result = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        
        if result is None:
            return None
        
        return result.group(1).strip()
    
    def run(self, user_prompt, max_iter=5, tool_tag = 'tool_call', result_tag='response'):
        
        tool_mapping = {t.name.lower(): t for t in self.tools}
        steps = []
        user_history = [{'role': 'user', 'content': f'<question>{user_prompt}</question>'}]
        
        for i in range(1, max_iter + 1):
            
            r = call_model(system=self.system, user=user_history, client=self.client, model=self.model_name)
            user_history.append({'role': 'assistant', 'content': r.resp})
            
            thought = self.extract_tags(r.resp, tag="thought")
            tool = self.extract_tags(r.resp, tag=tool_tag)
            result = self.extract_tags(r.resp, tag=result_tag)
            
            if tool is None:
                
                if result is None:
                    user_history.append({'role': 'user', 'content': f'Response did not contain <{tool_tag}></{tool_tag}> or <{result_tag}></{result_tag}>. A valid response must contain one of these tags.'})
                    continue
                
                step = ReactStep(thought=thought, tool_call=result, input_tokens=r.input_tokens, output_tokens=r.output_tokens, latency=r.latency_s)
                steps.append(step)
                input_toks = sum([s.input_tokens for s in steps])
                output_toks = sum([s.output_tokens for s in steps])
                latency = sum([s.latency for s in steps])
                
                return ReactResult(resp=result.strip(), steps=steps, input_tokens= input_toks, output_tokens=output_toks, latency=latency)
            else:
                
                try:
                    tool_call = json.loads(tool)
                except json.JSONDecodeError as e:
                    
                    user_history.append({'role': 'user', 'content': 'Tool call must be valid JSON. Load failed.'})
                    continue
                
                name = tool_call['name']
                
                if name.lower() not in tool_mapping:
                    user_history.append({'role': 'user', 'content': 'Tool name doesnt match any tools. To name must match exacted.'})
                    continue
                
                t = tool_mapping.get(name)
                
                tool_call = validate_arguements(tool_call, json.loads(t.fn_signature))
                
                tool_fn = t.fn
                
                args = tool_call['arguments']
                
                res = tool_fn(**args)
                
                step = ReactStep(thought=thought, tool_call=tool_call, input_tokens=r.input_tokens, output_tokens=r.output_tokens, latency=r.latency_s)
                steps.append(step)
                
                user_history.append({'role': 'user', 'content': f'<observation>{res}</observation>'})
        
        sys_prompt = self.system + f"You must provide your answer to the users question. All tools have been disabled. Wrap your answer in <{result_tag}></{result_tag}>"
        
        r = call_model(system=sys_prompt, user=user_history, client=self.client, model=self.model_name)
        
        result = self.extract_tags(r.resp, tag=result_tag)
        thought = self.extract_tags(r.resp, tag="thought")
        step = ReactStep(thought=thought, tool_call=result, input_tokens=r.input_tokens, output_tokens=r.output_tokens, latency=r.latency_s)
        steps.append(step)
        
        input_toks = sum([s.input_tokens for s in steps])
        output_toks = sum([s.output_tokens for s in steps])
        latency = sum([s.latency for s in steps])
        
        return ReactResult(resp=result, steps=steps, input_tokens=input_toks, output_tokens=output_toks, latency=latency)