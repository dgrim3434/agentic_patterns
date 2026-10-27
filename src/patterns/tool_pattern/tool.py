import json
from typing import Callable

def get_fn_signature(fn: Callable) -> dict:
    
    fn_signature = {
        'name': fn.__name__,
        'description': fn.__doc__,
        'parameters': {'properties': {}}
    }
    
    schema = {
        k: {'type': v.__name__} for k, v in fn.__annotations__.items() if k != 'return'
    }
    
    fn_signature['parameters']['properties'] = schema
    
    return fn_signature

def validate_arguements(tool_call: dict, tool_signiture:dict) -> dict:
    
    type_mapping = {
        'int': int,
        'str': str,
        'bool': bool,
        'float': float,
    }
    
    properties = tool_signiture['parameters']['properties']

    for arg_name, arg_value in tool_call['arguments'].items():
        
        expected_type = properties[arg_name].get('type')

        if not isinstance(arg_value, type_mapping[expected_type]):
            
            tool_call['arguments'][arg_name] = type_mapping[expected_type](arg_value)
        
        return tool_call

class Tool:
    
    def __init__(self, name:str, fn:Callable, fn_signature: str):
        
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature
    
    def __str__(self):
        
        return self.fn_signature
    
    def run(self, **kwargs):
        
        return self.fn(**kwargs)
    

def tool(fn: Callable):
    
    def wrapper():
        fn_signature = get_fn_signature(fn)
        
        return  Tool(
            name=fn_signature.get("name"), fn=fn, fn_signature=json.dumps(fn_signature)
        )
    
    return wrapper()