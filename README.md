# Agent-Based RAG Application

This repository contains a 100% agent-based implementation of a RAG (Retrieval-Augmented Generation) application with no hardcoded workflows.

## Architecture

The application follows a fully agent-driven architecture where all decisions and workflows are determined dynamically by specialized agents rather than hardcoded logic.

### Key Components

- **WorkflowRouterAgent**: Orchestrates the workflow by determining the next action based on the user's query and context
- **ContentProcessorAgent**: Processes and summarizes knowledge base content
- **QueryAnalyzerAgent**: Analyzes user queries to determine intent
- **DataGatheringAgent**: Retrieves relevant information from various sources
- **OrchestrationEngine**: Manages the execution of agent-driven workflows

## Features

- Dynamic workflow determination based on query analysis
- Agent memory and learning capabilities
- Specialized content processing for knowledge base documents
- Improved document retrieval with filename-based search
- Knowledge base management and file uploads
- Template selection and document generation
- Chat creation and management

## Implementation Details

The application uses a modular design with clear separation of concerns:

- `agents/`: Contains agent definitions and implementations
- `core/`: Core functionality including orchestration and evaluation
- `tools/`: Tool implementations for various operations
- `data/`: Data storage and management

### UI Features

The user interface includes:

1. **Knowledge Bases Zone**
   - Dropdown selection for existing knowledge bases
   - "+" button to create new knowledge bases via modal popup
   - Upload files button that appears when a KB is selected

2. **Templates Zone**
   - Dropdown selection for document templates
   - "+" button to upload new templates via modal popup
   - Clear button to reset template selection

3. **Chats Zone**
   - List of recent chats with improved styling
   - "+" button to create new chats via modal popup
   - Enhanced delete buttons with trash icons

## Agent System

The agent system is the core of the application, with these key features:

1. **Agent Registry**: Centralized registry for all agents
2. **Agent Memory**: Persistent memory for agents to learn from past interactions
3. **Orchestration Engine**: Manages the execution of agent-driven workflows
4. **Tool Registry**: Centralized registry for all tools that agents can use

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the web interface at http://localhost:5001

## Dependencies
- OpenAI API
- Bootstrap 5.1.3
- Font Awesome 6.0.0
- Python backend (Flask)
