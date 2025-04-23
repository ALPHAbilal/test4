# Knowledge Base UX Optimization Plan

## 1. Source Attribution Implementation

### Current Issue
- The system retrieves content from the knowledge base but doesn't provide source information in the final response
- Users cannot verify where information comes from, reducing trust

### Implementation Steps
1. **Modify `get_kb_document_content` function to include more metadata**
   - Enhance the `RetrievalSuccess` model to include additional source metadata:
     - File ID
     - File name
     - Page/section information (if available)
     - Confidence score

2. **Update the `final_synthesizer_agent` instructions**
   - Add explicit instructions to cite sources
   - Provide formatting guidelines for citations

3. **Implement source formatting in the response**
   - Add a "Sources" section at the end of responses
   - Format inline citations where appropriate

## 2. Agent-Driven Search Query Formulation

### Current Issue
- Raw user queries are passed directly to vector search
- No query reformulation or optimization for vector search

### Implementation Steps
1. **Create a `SearchQueryGeneratorAgent`**
   - Purpose: Transform user queries into optimal vector search queries
   - Input: User query, conversation history, intent
   - Output: Optimized search query or multiple search queries

2. **Define the agent model**
   ```python
   search_query_generator_agent = Agent(
       name="SearchQueryGeneratorAgent",
       instructions="""Generate optimal search queries for vector store retrieval.
       Your job is to transform the user's natural language query into the most effective search query.
       Consider:
       1. Extracting key entities and concepts
       2. Removing conversational elements
       3. Adding synonyms for important terms
       4. Breaking complex queries into multiple simpler queries
       5. Considering the conversation history for context
       
       Return a JSON object with:
       - primary_query: The main optimized query
       - alternative_queries: List of alternative queries to try if primary fails
       - document_type_filter: Suggested document type to filter by (or "general")
       """,
       model=COMPLETION_MODEL,
       output_type=SearchQueryPlan
   )
   ```

3. **Integrate into the workflow**
   - Modify `run_standard_agent_rag` to call this agent before retrieval
   - Use the generated queries for vector store search

## 3. Enhanced Retrieval Strategy

### Current Issue
- Basic similarity search with fixed k
- No re-ranking or hybrid search capabilities

### Implementation Steps
1. **Implement re-ranking approach**
   - Retrieve larger initial set (k=10-20)
   - Create a `ReRankAgent` to score and select the most relevant documents
   - Pass only the top N re-ranked documents to the synthesizer

2. **Define the ReRankAgent**
   ```python
   rerank_agent = Agent(
       name="ReRankAgent",
       instructions="""Re-rank retrieved documents based on relevance to the user query.
       You will receive a list of documents and the original user query.
       For each document:
       1. Assess how directly it answers the query
       2. Consider the specificity and completeness of information
       3. Evaluate the credibility and recency of the source
       4. Assign a relevance score from 0-10
       
       Return a JSON object with:
       - ranked_documents: List of document IDs sorted by relevance
       - relevance_scores: Dictionary mapping document IDs to scores
       - reasoning: Brief explanation of ranking decisions
       """,
       model=COMPLETION_MODEL,
       output_type=ReRankResult
   )
   ```

3. **Investigate hybrid search options**
   - Research OpenAI Vector Store API capabilities for hybrid search
   - Implement if available, otherwise focus on re-ranking

## 4. Streaming Implementation

### Current Issue
- Responses are sent only after complete generation
- Poor perceived performance for longer responses

### Implementation Steps
1. **Update Flask endpoint to support streaming**
   - Modify `/chat/<chat_id>` route to use server-sent events (SSE)
   - Create a streaming version of the response generation

2. **Implement streaming in the agent execution**
   - Update Runner.run to support streaming mode
   - Create a stream handler to process and forward tokens

3. **Update frontend to handle streaming responses**
   - Modify JavaScript to consume SSE stream
   - Implement progressive rendering of response

## 5. Vector Store Tool Encapsulation

### Current Issue
- Direct procedural calls to vector store client
- Violates "100% agents" philosophy for data access

### Implementation Steps
1. **Create a VectorStoreTool wrapper**
   ```python
   @function_tool(strict_mode=False)
   async def search_vector_store(ctx: RunContextWrapper, query: str, document_type: str = "general", 
                                max_results: int = 5, use_hybrid_search: bool = False) -> Union[VectorSearchSuccess, VectorSearchError]:
       """Searches the vector store for relevant documents matching the query.
       
       Args:
           query: The search query
           document_type: Optional document type to filter by
           max_results: Maximum number of results to return
           use_hybrid_search: Whether to use hybrid search (if available)
           
       Returns:
           Search results with content and metadata
       """
       # Implementation that wraps the vector store client
   ```

2. **Modify the workflow to use this tool**
   - Update `run_standard_agent_rag` to use an agent with this tool
   - Create a dedicated `KnowledgeRetrievalAgent` that uses this tool

3. **Define the KnowledgeRetrievalAgent**
   ```python
   knowledge_retrieval_agent = Agent(
       name="KnowledgeRetrievalAgent",
       instructions="""Retrieve relevant knowledge from the vector store.
       Use the search_vector_store tool to find information relevant to the user's query.
       Consider:
       1. Using the optimized search query provided
       2. Trying alternative queries if initial search yields no results
       3. Adjusting document type filters if needed
       4. Requesting more results for complex topics
       
       Return the retrieved information with source attribution.
       """,
       tools=[search_vector_store],
       model=COMPLETION_MODEL,
       output_type=Union[VectorSearchSuccess, VectorSearchError]
   )
   ```

## Implementation Timeline

1. **Week 1: Source Attribution**
   - Enhance RetrievalSuccess model
   - Update synthesizer instructions
   - Implement source formatting

2. **Week 2: Query Formulation**
   - Create SearchQueryGeneratorAgent
   - Integrate into workflow
   - Test with various query types

3. **Week 3: Re-ranking**
   - Implement larger initial retrieval
   - Create ReRankAgent
   - Integrate into workflow

4. **Week 4: Streaming**
   - Update Flask endpoint
   - Implement streaming in agent execution
   - Update frontend

5. **Week 5: Vector Store Tool**
   - Create VectorStoreTool wrapper
   - Create KnowledgeRetrievalAgent
   - Integrate into workflow

## Success Metrics

- **Source Attribution**: % of responses that include proper citations
- **Query Formulation**: Improvement in relevant document retrieval rate
- **Re-ranking**: Increase in user satisfaction with response relevance
- **Streaming**: Reduction in perceived response time
- **Vector Store Tool**: Successful encapsulation without performance degradation
