# ChatBase: Comprehensive Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
   - [Agent System](#agent-system)
   - [RAG Implementation](#rag-implementation)
   - [Template System](#template-system)
   - [Vector Store Integration](#vector-store-integration)
   - [Database System](#database-system)
4. [Workflows](#workflows)
   - [Standard RAG Workflow](#standard-rag-workflow)
   - [Template Population Workflow](#template-population-workflow)
   - [Template Analysis Workflow](#template-analysis-workflow)
5. [API Reference](#api-reference)
   - [Routes](#routes)
   - [Tools](#tools)
   - [Agents](#agents)
6. [Configuration](#configuration)
   - [Environment Variables](#environment-variables)
   - [Constants](#constants)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)
9. [Future Improvements](#future-improvements)

## Introduction

ChatBase is an advanced conversational AI application that combines Retrieval-Augmented Generation (RAG), document template processing, and multi-agent orchestration to provide intelligent responses based on knowledge bases and document templates. The system is designed to handle various user intents, from simple knowledge base queries to complex document template population and analysis.

The application is built using Flask as the web framework, OpenAI's API for language model capabilities, and a custom agent system for workflow orchestration. It supports multiple knowledge bases through Vector Stores and can process various document formats including PDF, TXT, and Markdown.

## System Architecture

ChatBase follows a modular architecture with several key components:

1. **Web Interface**: A Flask-based web application that provides the user interface and API endpoints.
2. **Agent System**: A multi-agent architecture that orchestrates different specialized agents for various tasks.
3. **RAG System**: A retrieval-augmented generation system that fetches relevant information from Vector Stores.
4. **Template System**: A document template processing system that can extract data from documents and populate templates.
5. **Database**: A SQLite database that stores chat history and other persistent data.

The application flow typically follows these steps:

1. User submits a query through the web interface
2. The query analyzer agent determines the user's intent
3. Based on the intent, the appropriate workflow is triggered
4. The workflow orchestrates multiple agents to fulfill the user's request
5. The final response is returned to the user and stored in the database

## Core Components

### Agent System

The agent system is built on OpenAI's Agents framework and consists of several specialized agents:

#### Query Analyzer Agent

```python
query_analyzer_agent = Agent(
    name="QueryAnalyzerAgent",
    instructions="Analyze the user's query, available templates, and temporary files to determine the true intent.",
    model=COMPLETION_MODEL
)
```

This agent analyzes the user's query and determines the intent, which can be one of:
- `kb_query`: User wants information from the knowledge base only
- `temp_context_query`: User wants information based on temporary files only
- `kb_query_with_temp_context`: User wants information that combines knowledge base and temporary files
- `populate_template`: User wants to fill a template with data
- `analyze_template`: User wants analysis or comparison related to a template

#### Data Gatherer Agent

```python
data_gatherer_agent = Agent(
    name="DataGathererAgent",
    instructions="You gather specific information using tools based on instructions.",
    tools=[get_kb_document_content, process_temporary_file, retrieve_template_content],
    model=COMPLETION_MODEL,
    output_type=Union[RetrievalSuccess, RetrievalError]
)
```

This agent is responsible for retrieving content from various sources:
- Knowledge base (Vector Store)
- Temporary files uploaded by the user
- Document templates

#### Data Extractor Agent

```python
data_extractor_agent = Agent(
    name="DataExtractorAgent",
    instructions="Use the 'extract_data_for_template' tool to extract required fields from the provided context sources.",
    tools=[extract_data_for_template],
    model=COMPLETION_MODEL,
    tool_use_behavior="stop_on_first_tool"
)
```

This agent extracts structured data from unstructured text, which is used for template population.

#### Template Populator Agent

```python
template_populator_agent = Agent(
    name="TemplatePopulatorAgent",
    instructions="Receive template text, JSON data, and user query. Analyze the user's request carefully to understand what they want.",
    model=COMPLETION_MODEL,
    output_type=FinalAnswer
)
```

This agent populates document templates with extracted data.

#### Template Analyzer Agent

```python
template_analyzer_agent = Agent(
    name="TemplateAnalyzerAgent",
    instructions="Analyze document templates and provide insights, comparisons, or explanations.",
    model=COMPLETION_MODEL,
    output_type=FinalAnswer
)
```

This agent analyzes document templates and provides insights or explanations.

#### Final Synthesizer Agent

```python
final_synthesizer_agent = Agent(
    name="FinalSynthesizerAgent",
    instructions="You are a helpful assistant that provides clear, concise, and accurate responses.",
    model=COMPLETION_MODEL,
    output_type=FinalAnswer
)
```

This agent synthesizes the final response to the user based on the information gathered by other agents.

### RAG Implementation

The RAG (Retrieval-Augmented Generation) implementation is a core component of the system that enables the retrieval of relevant information from knowledge bases to augment the language model's responses.

#### Vector Store Integration

The system integrates with OpenAI's Vector Stores for semantic search:

```python
@function_tool
async def get_kb_document_content(ctx: RunContextWrapper, document_type: str, query_or_identifier: str) -> Union[RetrievalSuccess, RetrievalError]:
    """Retrieves content from the knowledge base (Vector Store) based on document type and query/identifier."""
    # Implementation details...
```

The implementation includes:
- Flexible search with and without filters
- Fallback mechanisms when no results are found
- Multiple retrieval attempts with different queries

#### Standard RAG Workflow

```python
async def run_standard_agent_rag(user_query: str, history: List[Dict[str, str]], workflow_context: Dict, vs_id: Optional[str] = None) -> str:
    """Implements a simplified RAG workflow using the final_synthesizer_agent."""
    # Implementation details...
```

The standard RAG workflow:
1. Retrieves relevant content from the knowledge base
2. Creates a prompt that includes the query and KB content
3. Runs the final synthesizer agent with the prompt
4. Returns the final answer

### Template System

The template system enables the processing and population of document templates with extracted data.

#### Template Retrieval

```python
@function_tool
def retrieve_template_content(template_name: str) -> Union[RetrievalSuccess, RetrievalError]:
    """Retrieves the text content of a specified document template (txt, md, pdf)."""
    # Implementation details...
```

The system supports multiple template formats:
- Markdown (.md)
- Text (.txt)
- PDF (.pdf)

#### Field Detection

```python
def detect_required_fields_from_template(template_content: str, template_name: str) -> List[str]:
    """Dynamically detect required fields from a template based on content analysis."""
    # Implementation details...
```

The system can automatically detect required fields in templates using various patterns:
- `[Field Name]`
- `{Field Name}`
- `<Field Name>`
- `__Field Name__`
- `${Field Name}`
- `$FieldName`
- `FIELD: Field Name`

#### Data Extraction

```python
@function_tool
async def extract_data_for_template(ctx: RunContextWrapper, context_sources: List[str], required_fields: List[str]) -> ExtractedData:
    """Extracts specific data fields required for a template from provided text context sources."""
    # Implementation details...
```

The data extraction process:
1. Determines the document type based on required fields
2. Creates field-specific guidelines and mappings
3. Uses the language model to extract structured data from unstructured text
4. Returns the extracted data as a JSON object

#### Template Population

```python
# Template population process
populator_input_dict = {
    "template": template_content,
    "data": extracted_data,
    "user_query": user_query,
    "kb_content": kb_content
}
populator_res_raw = await Runner.run(template_populator_agent, input=json.dumps(populator_input_dict))
```

The template population process:
1. Fills the template with the provided data
2. Uses placeholders for missing fields
3. Adds compliance notes if relevant (e.g., for labor contracts)
4. Returns the completed document as a Markdown string

### Vector Store Integration

The system integrates with OpenAI's Vector Stores for semantic search and knowledge retrieval.

#### Vector Store Management

```python
async def get_vector_stores(cache_duration=DEFAULT_VS_CACHE_DURATION):
    """Retrieves and caches the list of vector stores from OpenAI."""
    # Implementation details...
```

The system maintains a cache of vector stores to reduce API calls and improve performance.

#### File Upload to Vector Store

```python
async def add_files_to_vector_store(vector_store_id, file_paths_with_names, all_metadata_dict):
    """Uploads files individually, associates them with the vector store, and adds attributes."""
    # Implementation details...
```

The file upload process:
1. Uploads the file to OpenAI
2. Associates the file with the vector store
3. Updates the file attributes with metadata

### Database System

The system uses SQLite for persistent storage of chat history and other data.

#### Chat History Database

```python
class ChatHistoryDB:
    """Database for storing chat history."""
    # Implementation details...
```

The database schema includes:
- `chats` table: Stores chat metadata (id, title, vector_store_id, created_at, updated_at)
- `messages` table: Stores chat messages (id, chat_id, role, content, created_at)

## Workflows

### Standard RAG Workflow

The standard RAG workflow is used for simple knowledge base queries:

```python
async def run_standard_agent_rag(user_query: str, history: List[Dict[str, str]], workflow_context: Dict, vs_id: Optional[str] = None) -> str:
    """Implements a simplified RAG workflow using the final_synthesizer_agent."""
    # Implementation details...
```

Flow:
1. User submits a query
2. Query analyzer determines the intent as `kb_query`
3. System retrieves relevant content from the knowledge base
4. Final synthesizer agent generates a response based on the query and retrieved content
5. Response is returned to the user

### Template Population Workflow

The template population workflow is used to fill document templates with extracted data:

```python
# Template population workflow
if intent == "populate_template":
    # Implementation details...
```

Flow:
1. User submits a query with a selected template
2. Query analyzer determines the intent as `populate_template`
3. System retrieves the template content
4. System extracts required fields from the template
5. System gathers context from temporary files and knowledge base
6. Data extractor agent extracts structured data from the context
7. Template populator agent fills the template with the extracted data
8. Populated template is returned to the user

### Template Analysis Workflow

The template analysis workflow is used to analyze document templates:

```python
# Template analysis workflow
elif intent == "analyze_template":
    # Implementation details...
```

Flow:
1. User submits a query with a selected template
2. Query analyzer determines the intent as `analyze_template`
3. System retrieves the template content
4. System gathers relevant content from the knowledge base
5. Template analyzer agent analyzes the template and generates insights
6. Analysis is returned to the user

## API Reference

### Routes

#### Main Routes

- `GET /`: Home page with recent chats and vector stores
- `GET /chat_view/<chat_id>`: Chat interface for a specific chat
- `POST /chat/<chat_id>`: API endpoint for sending messages in a chat
- `POST /new_chat`: Create a new chat with a selected vector store

#### Template Routes

- `GET /list_templates`: List available templates
- `POST /upload_template`: Upload a new template
- `GET /templates/<filename>`: Download a template

#### File Upload Routes

- `POST /upload_temp_file`: Upload a temporary file for the current session

### Tools

#### Knowledge Base Tools

- `get_kb_document_content`: Retrieves content from the knowledge base
- `process_temporary_file`: Reads and processes temporary files
- `retrieve_template_content`: Retrieves document templates

#### Data Extraction Tools

- `extract_data_for_template`: Extracts structured data from unstructured text

### Agents

- `QueryAnalyzerAgent`: Determines user intent
- `DataGathererAgent`: Retrieves content from various sources
- `DataExtractorAgent`: Extracts structured data
- `TemplatePopulatorAgent`: Populates templates with data
- `TemplateAnalyzerAgent`: Analyzes templates
- `FinalSynthesizerAgent`: Synthesizes final responses

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `FLASK_SECRET_KEY`: Secret key for Flask sessions
- `FLASK_DEBUG`: Enable debug mode (0 or 1)
- `COMPLETION_MODEL`: Language model to use (default: gpt-4o-mini)
- `MAX_COMPLETION_TOKENS`: Maximum tokens for completions (default: 8000)
- `COMPLETION_TEMPERATURE`: Temperature for completions (default: 0.3)
- `MAX_SEARCH_RESULTS_TOOL`: Maximum search results per tool call (default: 5)
- `SEARCH_RANKER`: Ranker for vector store search (default: auto)
- `DEFAULT_VS_CACHE_DURATION`: Cache duration for vector stores in seconds (default: 300)

### Constants

- `UPLOAD_FOLDER`: Folder for temporary file uploads
- `TEMPLATE_DIR`: Folder for document templates
- `ALLOWED_EXTENSIONS`: Allowed file extensions for uploads
- `ALLOWED_TEMPLATE_EXTENSIONS`: Allowed file extensions for templates
- `DATABASE_FILE`: SQLite database file

## Deployment

### Prerequisites

- Python 3.8+
- OpenAI API key
- Flask
- PyMuPDF (for PDF processing)
- SQLite

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see Configuration)
4. Create necessary directories:
   - `uploads`: For temporary file uploads
   - `templates/document_templates`: For document templates
5. Initialize the database: The application will automatically create the database on first run
6. Run the application: `python app.py`

### Docker Deployment

A Dockerfile is provided for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
```

Build and run the Docker container:

```bash
docker build -t chatbase .
docker run -p 5001:5001 -e OPENAI_API_KEY=your_api_key chatbase
```

## Troubleshooting

### Common Issues

#### Vector Store Search Returns No Results

If the vector store search returns no results, check:
- The vector store ID is correct
- The vector store contains relevant documents
- The query is specific enough
- The document_type filter is not too restrictive

Solution: The application now implements a fallback mechanism that tries without filters if no results are found with filters.

#### Template Field Detection Fails

If the template field detection fails, check:
- The template format is supported
- The template contains field placeholders in a recognized format
- The template is not corrupted

Solution: The application provides default fields for common document types (contracts, invoices) when field detection fails.

#### Data Extraction Returns Empty Results

If the data extraction returns empty results, check:
- The context sources contain the required information
- The required fields are correctly specified
- The document type is correctly determined

Solution: The application provides detailed field guidelines and mappings to improve extraction accuracy.

## Future Improvements

### Short-term Improvements

1. **Enhanced Vector Store Search**: Implement more sophisticated search strategies, such as hybrid search combining semantic and keyword search.
2. **Improved Template Field Detection**: Enhance the field detection algorithm to handle more complex templates and edge cases.
3. **Better Error Handling**: Implement more robust error handling and recovery mechanisms.

### Medium-term Improvements

1. **Multi-Modal Support**: Add support for image and audio processing.
2. **Advanced Template Population**: Implement more sophisticated template population with conditional logic and formatting options.
3. **User Authentication**: Add user authentication and authorization for multi-user support.

### Long-term Vision

1. **Workflow Customization**: Allow users to create and customize their own workflows.
2. **Integration with External Systems**: Add integrations with CRM, ERP, and other business systems.
3. **Advanced Analytics**: Implement analytics and insights on usage patterns and performance.

---

This documentation provides a comprehensive overview of the ChatBase application, its architecture, components, and workflows. It is intended for both developers and non-developers to understand how the system works and how to build it from scratch.
