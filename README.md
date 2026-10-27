# Agentic Patterns from scratch

Within this repository the four main agentic patterns where created fully from scratch. These patterns included:
- **Reflection**: A Generate/critic loop where the critic provides feedback for how the generators response can be improved.
- **Tool**: Connecting the LLM to simple external tools allowing the LLM to access information which isn't baked within it's weights
- **ReAct**: Reason, act, observe loop where we force the LLM to reason about how to solve the problem then allow it to call tools which move the LLM closer to the answer.
- **Multi-agent**: Agents are grouped into a crew with each agent specialized for a specific task.

## Setup

- All notebooks require a Anthropic API key which should be set as ANTHROPIC_API_KEY if you would like to run the notebooks.
- If you would like to use a different API make sure to read it's specs for how the system prompt should be passed in since it might differ from the current Implementation.

## Pattern Deep Dive

### Reflection Pattern

The reflection pattern is extremely simple and runs in a loop where the generator LLM answers the users question. The critic LLM takes the generators response as an input and outputs feedback on how to make the answer better. This pattern can result in a high amount of tokens used since the we are passing unconstrained LLM outputs between LLM calls. In my opinion this patterns main use case is well delegating work to smaller less powerful LLMs with a larger grader LLM which provides feedback on the smaller models results which can be used to save compute cost.

### Tool Calling

Tool calling is an extremely important concept within working with language models. It's extremely important when you need the model to answer queries which require information which isn't baked within it's weights. An example of this is a query like: "What is the weather today in pennsylvania" where the model needs to access up to date weather information. It's important to understand that even though you are providing the model with tools it can still hallucinate mainly because calling the tool requires the model to understand that it cannot answer the query with it's own knowledge.

### ReAct Pattern

The ReAct pattern stands for Reason Action and describes a reason, action, observation loop where we force the model to first think about how it can solve the problem. Then take an action like calling a tool. and obvserving the results of the tool call. This pattern can be very good when the task contains smaller easier subtasks where the model can break the larger task into peices solving each one then combining the results to arrive at the actual solution.

### Multi-Agent Pattern

The multi-Agent pattern is widely used within agentic frameworks like CrewAI and Langchain. This pattern pretty much combines the previous patterns into one. Where we have specialized agents which work together to answer the larger task. These specialized agents can have specific tools to help them with the specifc task they are designed for. Although it has been found that Multi-Agent architecturese don't significantly outperform single agent architectures by a significant margin there seems to be a lot of room for growth given this setup allows you to employ specialized fine-tuned models for specific tasks which I believe can create extremely powerful workflows.