from collections import deque
from colorama import Fore
from graphviz import Digraph

class Crew:
    
    
    current_crew = None
    
    def __init__(self):
        self.agents = []
        
    def __enter__(self):
        
        Crew.current_crew = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        
        Crew.current_crew = None
    
    def add_agent(self, agent):
        
        self.agents.append(agent)
    
    @staticmethod
    def register_agent(agent):
        
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)
    
    def topological_sort(self):

        direct_connects = {a: len(a.dependencies) for a in self.agents}
        
        queue = deque([a for a in self.agents if direct_connects.get(a) == 0])
        
        sorted_agents = []
        
        while len(queue) != 0:
            
            curr = queue.popleft()
            sorted_agents.append(curr)
            
            for dep in curr.dependents:
                
                direct_connects[dep] -= 1
                
                if direct_connects.get(dep) == 0:
                    queue.append(dep)
        
        
        return sorted_agents

    def plot(self):
        
        dot = Digraph(format="png")
        
        for agent in self.agents:
            
            dot.node(agent.name)
            
            for dependency in agent.dependencies:
                
                dot.edge(dependency.name, agent.name)
        
        return dot
    
    def run(self):
        
        sorted_agents = self.topological_sort()
        
        for agent in sorted_agents:
            
            print(f"Running Agent: {agent.name}")
            
            out = agent.run()
            
            print(out)
