from ..tool_pattern.tool import Tool
from ..planning_pattern.react_agent import ReactAgent
from ..utils.model import call_model
from textwrap import dedent
from ..multiagent_pattern.crew import Crew

class Agent:
    
    def __init__(
        self,
        client,
        name: str,
        backstory: str,
        task_description: str,
        model_name: str ="claude-haiku-5-5",
        task_expected_outcome: str = "",
        tools: list[Tool] | None = None,
    ):
    
        self.client = client
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.model_name = model_name
        self.expected_outcome = task_expected_outcome
        self.tools = tools
        
        self.agent = ReactAgent(tools=tools, model_name=self.model_name, client=self.client)
        
        self.context = ""
        
        self.dependents = []
        self.dependencies = []

        Crew.register_agent(self)
        
    def get_name(self):
        
        return f'{self.name}'
    
    def __rshift__(self, other):
        self.add_dependents(other)
        return other
    
    def __lshift__(self, other):
        self.add_dependency(other)
        return other
    
    def __rrshift__(self, other):
        self.add_dependency(other)
        return self
    
    def add_dependency(self, other):
        
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise ValueError("ERROR: other must be of type Agent of list of Agents.")
    
    def add_dependents(self, other):
        
        if isinstance(other, Agent):
            self.dependents.append(other)
            other.dependencies.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            
            for item in other:
                self.dependents.append(item)
                item.dependencies.append(self)
        else:
            raise ValueError("ERROR: other must be of type Agent of list of Agents.")
    
    def add_context(self, text):
        
        self.context += f"{self.name} recieved context: \n{text}"
    
    def generate_prompt(self):

        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.expected_outcome}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt        
    
    def run(self):
        
        prompt = self.generate_prompt()
        
        output = self.agent.run(user_prompt=prompt)
        
        text_out = output.resp
        
        for dep in self.dependents:
            
            dep.add_context(text_out)
        
        return text_out