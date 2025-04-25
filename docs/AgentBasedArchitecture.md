# Agent-Based Architecture

This document describes the agent-based architecture implemented in this application, which follows the "100% agents, no hardcoded" approach.

## Overview

The application is built around a fully agent-driven architecture where all decisions and workflows are determined dynamically by specialized agents rather than hardcoded logic. This approach provides greater flexibility, adaptability, and extensibility compared to traditional hardcoded workflows.

## Core Principles

1. **Agent Autonomy**: Agents make decisions based on their specialized knowledge and context
2. **Dynamic Workflows**: Workflows are determined at runtime based on user queries and context
3. **Memory and Learning**: Agents maintain memory and learn from past interactions
4. **Modular Design**: Clear separation of concerns between agents, tools, and core functionality

## Architecture Components

### 1. Agent Registry

The Agent Registry is responsible for loading, registering, and managing all agents in the system. It provides a centralized registry for accessing agents by name.

```python
class AgentRegistry:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, name, agent):
        self.agents[name] = agent
        
    def get_agent(self, name):
        return self.agents.get(name)
```

### 2. Orchestration Engine

The Orchestration Engine manages the execution of agent-driven workflows. It coordinates the interactions between agents, tools, and the user.

```python
class OrchestrationEngine:
    def __init__(self, agent_registry, tool_registry):
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        
    async def execute_workflow(self, user_query, context=None):
        # Initialize workflow context
        workflow_context = context or {}
        workflow_context["user_query"] = user_query
        
        # Get the router agent
        router_agent = self.agent_registry.get_agent("WorkflowRouterAgent")
        
        # Execute the workflow
        while True:
            # Get the next action from the router agent
            action = await self.get_next_action(router_agent, workflow_context)
            
            # Execute the action
            if action["action"] == "call_agent":
                # Call another agent
                agent_name = action["details"]["agent_name"]
                agent = self.agent_registry.get_agent(agent_name)
                result = await self.execute_agent(agent, action["details"]["input"])
                workflow_context["agent_result"] = result
                
            elif action["action"] == "call_tool":
                # Call a tool
                tool_name = action["details"]["tool_name"]
                tool = self.tool_registry.get_tool(tool_name)
                result = await self.execute_tool(tool, action["details"]["parameters"])
                workflow_context["tool_result"] = result
                
            elif action["action"] == "return_to_user":
                # Return the final response to the user
                return action["details"]["final_response"]
                
            # Update workflow context with state updates
            workflow_context.update(action.get("state_update", {}))
```

### 3. Agent System

The agent system consists of specialized agents for different tasks:

1. **WorkflowRouterAgent**: Orchestrates the workflow by determining the next action
2. **ContentProcessorAgent**: Processes and summarizes knowledge base content
3. **QueryAnalyzerAgent**: Analyzes user queries to determine intent
4. **DataGatheringAgent**: Retrieves relevant information from various sources
5. **TemplatePopulatorAgent**: Populates templates with extracted data
6. **FinalSynthesizerAgent**: Creates final, polished responses

### 4. Tool Registry

The Tool Registry manages all tools available to agents, providing a centralized registry for accessing tools by name.

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        
    def register_tool(self, name, tool):
        self.tools[name] = tool
        
    def get_tool(self, name):
        return self.tools.get(name)
```

### 5. Memory System

The Memory System provides persistent memory for agents to learn from past interactions.

```python
class AgentMemory:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.memory = {}
        
    def get(self, key, default=None):
        return self.memory.get(key, default)
        
    def set(self, key, value):
        self.memory[key] = value
        
    def update(self, data):
        self.memory.update(data)
        
    def save(self):
        # Save memory to disk
        pass
        
    def load(self):
        # Load memory from disk
        pass
```

## Workflow Execution

The workflow execution process follows these steps:

1. User submits a query
2. OrchestrationEngine initializes the workflow context
3. WorkflowRouterAgent determines the next action
4. OrchestrationEngine executes the action (call agent, call tool, or return to user)
5. WorkflowRouterAgent receives the updated context and determines the next action
6. Process repeats until a final response is returned to the user

## Benefits of Agent-Based Architecture

1. **Flexibility**: Easily adapt to new requirements without changing core code
2. **Extensibility**: Add new agents and tools without modifying existing components
3. **Maintainability**: Clear separation of concerns makes the code easier to maintain
4. **Scalability**: Agents can be distributed across multiple servers for improved performance
5. **Learning**: Agents can learn from past interactions to improve over time

## Conclusion

The agent-based architecture implemented in this application provides a flexible, extensible, and maintainable approach to building complex applications. By following the "100% agents, no hardcoded" approach, the application can adapt to changing requirements and learn from past interactions to improve over time.
