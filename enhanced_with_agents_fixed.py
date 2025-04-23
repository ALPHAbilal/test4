import os
import time
import logging
import uvicorn
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import openai
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Import our agent system
from agent_system_fixed import DocumentProcessingAgentSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the agent system
agent_system = DocumentProcessingAgentSystem()

# Processing jobs tracking
processing_jobs = {}

# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize advanced logging if available
    try:
        from advanced_tracing import register_advanced_tracing
        from performance_monitor import start_monitoring
        from workflow_visualization import start_workflow, end_workflow

        # Register advanced tracing
        logger.info("Initializing advanced tracing...")
        trace_processor = register_advanced_tracing()

        # Start performance monitoring
        logger.info("Starting performance monitoring...")
        perf_monitor = start_monitoring()

        # Start application workflow
        app_workflow_id = start_workflow("Application Lifecycle", {
            "start_time": time.time(),
            "app_version": "1.0.0"
        })

        logger.info("Advanced logging modules initialized successfully")
    except ImportError as e:
        logger.warning(f"Advanced logging modules not fully available: {e}")
        trace_processor = None
        perf_monitor = None
        app_workflow_id = None

    # Startup: Initialize the assistant
    ai_solution.create_assistant()
    logger.info(f"Assistant created with ID: {ai_solution.assistant_id}")
    logger.info("Agent system initialized")

    yield

    # Shutdown: Clean up resources
    ai_solution.cleanup()

    # Shutdown: Clean up advanced logging
    if trace_processor:
        logger.info("Shutting down advanced tracing...")
        trace_processor.shutdown()

    if perf_monitor:
        logger.info("Stopping performance monitoring...")
        try:
            from performance_monitor import stop_monitoring
            stop_monitoring()
        except ImportError:
            pass

    if app_workflow_id:
        logger.info("Ending application workflow...")
        try:
            end_workflow(app_workflow_id, "completed", {
                "end_time": time.time(),
                "shutdown_reason": "normal"
            })

            # Save workflow visualizations
            from workflow_visualization import save_workflow_diagram, save_workflow_timeline
            diagram_path = save_workflow_diagram()
            timeline_path = save_workflow_timeline()
            logger.info(f"Workflow visualizations saved to {diagram_path} and {timeline_path}")
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error saving workflow visualizations: {e}")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AISolution:
    def __init__(self):
        self.vector_store = None
        self.assistant_id = None
        self.thread_id = None
        self.file_paths = []
        # Add these for conversation support
        self.current_thread = None
        self.conversation_history = []
        # Add this for document content cache
        self.document_content_cache = {}

    def _chunk_file(self, file_path: str):
        """Split files into manageable chunks with PDF support"""
        try:
            text = ""
            file_extension = file_path.lower().split('.')[-1]

            # Handle PDFs
            if file_extension == 'pdf':
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text() or ""
                            text += page_text + "\n\n"
                        logger.info(f"Extracted {len(text)} characters from PDF {file_path}")
                except Exception as e:
                    logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
                    return []
            # Handle text files
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # Try with a different encoding if utf-8 fails
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                    except Exception as e:
                        logger.error(f"Error reading file with latin-1 encoding: {str(e)}")
                        return []

            # Cache the document content for agent processing
            self.document_content_cache[file_path] = text

            # If we got no text, return empty list
            if not text or text.strip() == "":
                logger.warning(f"No text content extracted from {file_path}")
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Split {file_path} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def create_knowledge_base(self, file_paths: list = None):
        """Create vector store from files"""
        if file_paths:
            self.file_paths = file_paths

        all_chunks = []
        for path in self.file_paths:
            chunks = self._chunk_file(path)
            all_chunks.extend(chunks)
            logger.info(f"Processed {path}: {len(chunks)} chunks")

        if not all_chunks:
            logger.warning("Warning: No text chunks extracted from files")
            return False

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = FAISS.from_texts(all_chunks, embeddings)
        return True

    def create_assistant(self):
        """Create OpenAI Assistant with file search"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Create Assistant", {
                "timestamp": start_time,
                "model": "gpt-4-turbo"
            })
            logger.info(f"Started create assistant workflow tracking (ID: {workflow_id})")

        try:
            # Create assistant with performance monitoring
            with MonitoredOperation("create_assistant") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info("Creating OpenAI Assistant with file search capability")

                assistant = openai.beta.assistants.create(
                    name="File Expert",
                    instructions="Use file search to answer questions accurately based on the provided context. Always reference the source when answering.",
                    tools=[{"type": "file_search"}],
                    model="gpt-4-turbo",
                )

                self.assistant_id = assistant.id
                processing_time = time.time() - start_time

                logger.info(f"Assistant created successfully with ID: {self.assistant_id} in {processing_time:.2f}s")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": processing_time * 1000,
                        "assistant_id": self.assistant_id,
                        "model": "gpt-4-turbo"
                    })

            # End workflow tracking if enabled
            if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                end_workflow(workflow_id, "completed", {
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "assistant_id": self.assistant_id
                })
                logger.info(f"Completed create assistant workflow (ID: {workflow_id})")

            return self.assistant_id
        except Exception as e:
            logger.error(f"Error creating assistant: {e}", exc_info=True)

            # Record failure if advanced logging is enabled
            if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                record_failure("create_assistant", e, {})

                # End workflow if it was started
                if workflow_id and 'end_workflow' in globals():
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__
                    })

            # Re-raise the exception
            raise

    def create_persistent_thread(self):
        """Create a persistent thread for conversation"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Create Thread", {
                "timestamp": start_time,
                "has_existing_thread": bool(self.current_thread)
            })
            logger.info(f"Started create thread workflow tracking (ID: {workflow_id})")

        try:
            # Only create a new thread if one doesn't exist
            if not self.current_thread:
                # Create thread with performance monitoring
                with MonitoredOperation("create_thread") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                    logger.info("Creating new persistent thread for conversation")

                    self.current_thread = openai.beta.threads.create()
                    thread_id = self.current_thread.id
                    processing_time = time.time() - start_time

                    logger.info(f"Thread created successfully with ID: {thread_id} in {processing_time:.2f}s")

                    if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": processing_time * 1000,
                            "thread_id": thread_id
                        })
            else:
                logger.info(f"Using existing thread: {self.current_thread.id}")
                thread_id = self.current_thread.id

            # End workflow tracking if enabled
            if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                end_workflow(workflow_id, "completed", {
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "thread_id": thread_id,
                    "created_new": not bool(self.current_thread)
                })
                logger.info(f"Completed create thread workflow (ID: {workflow_id})")

            return thread_id
        except Exception as e:
            logger.error(f"Error creating thread: {e}", exc_info=True)

            # Record failure if advanced logging is enabled
            if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                record_failure("create_thread", e, {})

                # End workflow if it was started
                if workflow_id and 'end_workflow' in globals():
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__
                    })

            # Re-raise the exception
            raise

    def process_query(self, user_query: str, new_conversation: bool = False):
        """Handle query with RAG + Assistant with conversation support"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Process Query", {
                "timestamp": start_time,
                "query": user_query[:100] + "..." if len(user_query) > 100 else user_query,
                "new_conversation": new_conversation,
                "has_vector_store": bool(self.vector_store),
                "has_thread": bool(self.current_thread)
            })
            logger.info(f"Started process query workflow tracking (ID: {workflow_id})")

        try:
            # Check if knowledge base exists
            if not self.vector_store:
                logger.warning("Knowledge base not created yet. Please upload files first.")

                # End workflow tracking if enabled
                if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                    end_workflow(workflow_id, "failed", {
                        "error": "Knowledge base not created",
                        "error_type": "MissingVectorStore"
                    })

                return "Error: Knowledge base not created yet. Please upload files first."

            # Search local knowledge base with performance monitoring
            with MonitoredOperation("similarity_search") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Performing similarity search for query: '{user_query[:50]}...'")
                search_start = time.time()

                docs = self.vector_store.similarity_search(user_query, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                search_time = time.time() - search_start
                logger.info(f"Similarity search completed in {search_time:.2f}s, found {len(docs)} documents")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": search_time * 1000,
                        "document_count": len(docs),
                        "context_length": len(context)
                    })

            # Create a new thread or use existing one
            with WorkflowStep("prepare_thread", workflow_id) if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as step:
                if new_conversation or not self.current_thread:
                    self.current_thread = openai.beta.threads.create()
                    thread_id = self.current_thread.id
                    self.conversation_history = []
                    logger.info(f"Starting new conversation with thread: {thread_id}")

                    if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING and step:
                        step.add_metadata({
                            "thread_id": thread_id,
                            "created_new": True
                        })
                else:
                    thread_id = self.current_thread.id
                    logger.info(f"Using existing thread: {thread_id}")

                    if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING and step:
                        step.add_metadata({
                            "thread_id": thread_id,
                            "created_new": False,
                            "history_length": len(self.conversation_history)
                        })

                # Format the message with context only for new conversations
                if new_conversation or len(self.conversation_history) == 0:
                    message_content = f"Context:\n{context}\n\nQuestion: {user_query}"
                    logger.info("Adding context to message for new conversation")
                else:
                    message_content = user_query

            # Add message to thread with performance monitoring
            with MonitoredOperation("add_message") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Adding user message to thread: {thread_id}")
                message_start = time.time()

                message = openai.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=message_content
                )

                # Store in history
                self.conversation_history.append({"role": "user", "content": user_query})

                message_time = time.time() - message_start
                logger.info(f"Message added in {message_time:.2f}s")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": message_time * 1000,
                        "message_id": message.id,
                        "content_length": len(message_content)
                    })

            # Run the assistant with performance monitoring
            with MonitoredOperation("run_assistant") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Running assistant on thread: {thread_id}")
                run_start = time.time()

                run = openai.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant_id
                )

                # Wait for completion with timeout and exponential backoff
                max_wait_time = 60  # 60 seconds timeout
                poll_interval = 1.0  # Start with 1 second polling interval
                max_poll_interval = 5.0  # Maximum polling interval
                backoff_factor = 1.5  # Exponential backoff factor

                logger.info(f"Waiting for run {run.id} to complete (timeout: {max_wait_time}s)")

                while run.status not in ["completed", "failed", "expired", "cancelled"]:
                    elapsed_time = time.time() - run_start
                    if elapsed_time > max_wait_time:
                        logger.warning(f"Run {run.id} timed out after {elapsed_time:.2f}s")

                        if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                            op.set_metadata({
                                "processing_time_ms": elapsed_time * 1000,
                                "run_id": run.id,
                                "status": "timeout",
                                "polls": int(elapsed_time / poll_interval)
                            })

                        # End workflow tracking if enabled
                        if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                            end_workflow(workflow_id, "failed", {
                                "error": "Request timed out",
                                "error_type": "Timeout",
                                "processing_time_ms": elapsed_time * 1000
                            })

                        return "Error: Request timed out"

                    # Sleep with exponential backoff
                    time.sleep(poll_interval)

                    # Update polling interval with exponential backoff
                    poll_interval = min(poll_interval * backoff_factor, max_poll_interval)

                    # Retrieve run status
                    run = openai.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id
                    )

                    logger.debug(f"Run {run.id} status: {run.status} (elapsed: {elapsed_time:.2f}s)")

                run_time = time.time() - run_start
                logger.info(f"Run completed in {run_time:.2f}s with status: {run.status}")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": run_time * 1000,
                        "run_id": run.id,
                        "status": run.status
                    })

                if run.status != "completed":
                    error_message = f"Error: Run ended with status {run.status}"
                    logger.error(error_message)

                    # End workflow tracking if enabled
                    if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                        end_workflow(workflow_id, "failed", {
                            "error": error_message,
                            "error_type": "RunFailed",
                            "run_status": run.status,
                            "processing_time_ms": run_time * 1000
                        })

                    return error_message

            # Get the latest message with performance monitoring
            with MonitoredOperation("get_messages") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Retrieving messages from thread: {thread_id}")
                messages_start = time.time()

                messages = openai.beta.threads.messages.list(thread_id)
                latest_message = messages.data[0].content[0].text.value

                # Store in history
                self.conversation_history.append({"role": "assistant", "content": latest_message})

                messages_time = time.time() - messages_start
                logger.info(f"Messages retrieved in {messages_time:.2f}s, response length: {len(latest_message)}")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": messages_time * 1000,
                        "message_count": len(messages.data),
                        "response_length": len(latest_message)
                    })

            # End workflow tracking if enabled
            total_time = time.time() - start_time
            if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                end_workflow(workflow_id, "completed", {
                    "processing_time_ms": total_time * 1000,
                    "thread_id": thread_id,
                    "response_length": len(latest_message),
                    "history_length": len(self.conversation_history)
                })
                logger.info(f"Completed process query workflow (ID: {workflow_id}) in {total_time:.2f}s")

            return latest_message
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)

            # Record failure if advanced logging is enabled
            if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                record_failure("process_query", e, {
                    "query": user_query,
                    "new_conversation": new_conversation,
                    "thread_id": getattr(self, 'current_thread', {}).get('id', None)
                })

                # End workflow if it was started
                if workflow_id and 'end_workflow' in globals():
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "processing_time_ms": (time.time() - start_time) * 1000
                    })

            # Return error message
            return f"Error processing query: {str(e)}"

    def process_query_stream(self, user_query: str, new_conversation: bool = False):
        """Stream responses from the assistant with conversation support"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Stream Query", {
                "timestamp": start_time,
                "query": user_query[:100] + "..." if len(user_query) > 100 else user_query,
                "new_conversation": new_conversation,
                "has_vector_store": bool(self.vector_store),
                "has_thread": bool(self.current_thread)
            })
            logger.info(f"Started stream query workflow tracking (ID: {workflow_id})")

        try:
            # Check if knowledge base exists
            if not self.vector_store:
                logger.warning("Knowledge base not created yet. Please upload files first.")

                # End workflow tracking if enabled
                if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                    end_workflow(workflow_id, "failed", {
                        "error": "Knowledge base not created",
                        "error_type": "MissingVectorStore"
                    })

                yield "Error: Knowledge base not created yet. Please upload files first."
                return

            # Search local knowledge base with performance monitoring
            with MonitoredOperation("similarity_search_stream") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Performing similarity search for stream query: '{user_query[:50]}...'")
                search_start = time.time()

                docs = self.vector_store.similarity_search(user_query, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                search_time = time.time() - search_start
                logger.info(f"Similarity search for stream completed in {search_time:.2f}s, found {len(docs)} documents")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": search_time * 1000,
                        "document_count": len(docs),
                        "context_length": len(context)
                    })

            # Create a new thread or use existing one
            with WorkflowStep("prepare_thread_stream", workflow_id) if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as step:
                if new_conversation or not self.current_thread:
                    self.current_thread = openai.beta.threads.create()
                    thread_id = self.current_thread.id
                    self.conversation_history = []
                    logger.info(f"Starting new conversation with thread for streaming: {thread_id}")

                    if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING and step:
                        step.add_metadata({
                            "thread_id": thread_id,
                            "created_new": True
                        })
                else:
                    thread_id = self.current_thread.id
                    logger.info(f"Using existing thread for streaming: {thread_id}")

                    if 'WorkflowStep' in globals() and ENABLE_ADVANCED_LOGGING and step:
                        step.add_metadata({
                            "thread_id": thread_id,
                            "created_new": False,
                            "history_length": len(self.conversation_history)
                        })

                # Format the message with context only for new conversations
                if new_conversation or len(self.conversation_history) == 0:
                    message_content = f"Context:\n{context}\n\nQuestion: {user_query}"
                    logger.info("Adding context to message for new streaming conversation")
                else:
                    message_content = user_query

            # Add to history
            self.conversation_history.append({"role": "user", "content": user_query})

            # Add message to thread with performance monitoring
            with MonitoredOperation("add_message_stream") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Adding user message to thread for streaming: {thread_id}")
                message_start = time.time()

                message = openai.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=message_content
                )

                message_time = time.time() - message_start
                logger.info(f"Message added for streaming in {message_time:.2f}s")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": message_time * 1000,
                        "message_id": message.id,
                        "content_length": len(message_content)
                    })

            # Run the assistant with streaming and performance monitoring
            with MonitoredOperation("run_assistant_stream") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Running assistant with streaming on thread: {thread_id}")
                run_start = time.time()

                # Run the assistant with streaming
                run = openai.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant_id,
                    stream=True
                )

                # Collect full response for history
                full_response = ""
                chunk_count = 0
                first_chunk_time = None

                # Stream the response with logging
                logger.info("Starting to stream response chunks")

                for event in run:
                    if event.event == "thread.message.delta":
                        if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                            for content in event.data.delta.content:
                                if content.type == 'text':
                                    text_chunk = content.text.value
                                    full_response += text_chunk
                                    chunk_count += 1

                                    # Record time of first chunk
                                    if chunk_count == 1:
                                        first_chunk_time = time.time()
                                        time_to_first_chunk = first_chunk_time - run_start
                                        logger.info(f"First chunk received after {time_to_first_chunk:.2f}s")

                                    # Log every 10th chunk to avoid excessive logging
                                    if chunk_count % 10 == 0:
                                        logger.debug(f"Streamed {chunk_count} chunks, current response length: {len(full_response)}")

                                    yield text_chunk

                # After streaming is complete, store in history
                self.conversation_history.append({"role": "assistant", "content": full_response})

                # Log streaming completion
                stream_time = time.time() - run_start
                logger.info(f"Streaming completed in {stream_time:.2f}s, sent {chunk_count} chunks, total response length: {len(full_response)}")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": stream_time * 1000,
                        "chunk_count": chunk_count,
                        "response_length": len(full_response),
                        "time_to_first_chunk_ms": (first_chunk_time - run_start) * 1000 if first_chunk_time else None
                    })

            # End workflow tracking if enabled
            total_time = time.time() - start_time
            if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                end_workflow(workflow_id, "completed", {
                    "processing_time_ms": total_time * 1000,
                    "thread_id": thread_id,
                    "response_length": len(full_response),
                    "chunk_count": chunk_count,
                    "history_length": len(self.conversation_history)
                })
                logger.info(f"Completed stream query workflow (ID: {workflow_id}) in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing stream query: {e}", exc_info=True)

            # Record failure if advanced logging is enabled
            if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                record_failure("process_query_stream", e, {
                    "query": user_query,
                    "new_conversation": new_conversation,
                    "thread_id": getattr(self, 'current_thread', {}).get('id', None)
                })

                # End workflow if it was started
                if workflow_id and 'end_workflow' in globals():
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "processing_time_ms": (time.time() - start_time) * 1000
                    })

            # Yield error message
            yield f"Error processing stream query: {str(e)}"

    def process_document_with_agents(self, file_path, user_query=None):
        """Process a document using the agent system"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Process Document", {
                "timestamp": start_time,
                "file_path": file_path,
                "has_query": bool(user_query),
                "query": user_query[:100] + "..." if user_query and len(user_query) > 100 else user_query
            })
            logger.info(f"Started document processing workflow tracking (ID: {workflow_id})")

        try:
            # Check if document is cached with performance monitoring
            with MonitoredOperation("check_document_cache") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Checking document cache for {file_path}")
                cache_start = time.time()

                # Check if document is cached
                is_cached = file_path in self.document_content_cache

                if not is_cached:
                    logger.info(f"Document {file_path} not in cache, chunking file")
                    self._chunk_file(file_path)  # This will cache the content
                else:
                    logger.info(f"Document {file_path} found in cache")

                cache_time = time.time() - cache_start

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": cache_time * 1000,
                        "was_cached": is_cached,
                        "file_path": file_path
                    })

            # Get document content from cache
            document_content = self.document_content_cache.get(file_path, "")
            if not document_content:
                error_message = f"Could not read content from {file_path}"
                logger.error(error_message)

                # End workflow tracking if enabled
                if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                    end_workflow(workflow_id, "failed", {
                        "error": error_message,
                        "error_type": "EmptyDocument",
                        "file_path": file_path
                    })

                return {"error": error_message}

            # Process with agent system with performance monitoring
            with MonitoredOperation("process_document") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Processing document {file_path} with agents" + (f" and query: '{user_query}'" if user_query else ""))
                process_start = time.time()

                result = agent_system.process_document(document_content, user_query)

                process_time = time.time() - process_start
                logger.info(f"Document processing completed successfully in {process_time:.2f}s")

                if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                    op.set_metadata({
                        "processing_time_ms": process_time * 1000,
                        "document_length": len(document_content),
                        "has_query": bool(user_query),
                        "result_type": type(result).__name__
                    })

            # End workflow tracking if enabled
            total_time = time.time() - start_time
            if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
                end_workflow(workflow_id, "completed", {
                    "processing_time_ms": total_time * 1000,
                    "file_path": file_path,
                    "document_length": len(document_content),
                    "result_type": type(result).__name__
                })
                logger.info(f"Completed document processing workflow (ID: {workflow_id}) in {total_time:.2f}s")

            return result
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}", exc_info=True)

            # Record failure if advanced logging is enabled
            if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                record_failure("process_document", e, {
                    "file_path": file_path,
                    "has_query": bool(user_query),
                    "document_length": len(self.document_content_cache.get(file_path, "")) if file_path in self.document_content_cache else 0
                })

                # End workflow if it was started
                if workflow_id and 'end_workflow' in globals():
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                        "file_path": file_path
                    })

            return {"status": "error", "message": str(e)}

    def cleanup(self):
        """Clean up resources"""
        start_time = time.time()

        # Use workflow tracking if available
        workflow_id = None
        if 'start_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            workflow_id = start_workflow("Cleanup Resources", {
                "timestamp": start_time,
                "has_assistant": bool(self.assistant_id),
                "has_thread": bool(self.thread_id)
            })
            logger.info(f"Started cleanup workflow tracking (ID: {workflow_id})")

        cleanup_success = True
        errors = []

        # Clean up assistant with performance monitoring
        if self.assistant_id:
            with MonitoredOperation("delete_assistant") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Deleting assistant: {self.assistant_id}")
                assistant_start = time.time()

                try:
                    openai.beta.assistants.delete(self.assistant_id)
                    assistant_time = time.time() - assistant_start
                    logger.info(f"Assistant {self.assistant_id} deleted in {assistant_time:.2f}s")

                    if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": assistant_time * 1000,
                            "assistant_id": self.assistant_id,
                            "success": True
                        })
                except Exception as e:
                    cleanup_success = False
                    error_message = f"Error deleting assistant: {e}"
                    errors.append(error_message)
                    logger.error(error_message, exc_info=True)

                    if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": (time.time() - assistant_start) * 1000,
                            "assistant_id": self.assistant_id,
                            "success": False,
                            "error": str(e)
                        })

                    # Record failure if advanced logging is enabled
                    if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                        record_failure("delete_assistant", e, {
                            "assistant_id": self.assistant_id
                        })

        # Clean up thread with performance monitoring
        if self.thread_id:
            with MonitoredOperation("delete_thread") if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                logger.info(f"Deleting thread: {self.thread_id}")
                thread_start = time.time()

                try:
                    openai.beta.threads.delete(self.thread_id)
                    thread_time = time.time() - thread_start
                    logger.info(f"Thread {self.thread_id} deleted in {thread_time:.2f}s")

                    if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": thread_time * 1000,
                            "thread_id": self.thread_id,
                            "success": True
                        })
                except Exception as e:
                    cleanup_success = False
                    error_message = f"Error deleting thread: {e}"
                    errors.append(error_message)
                    logger.error(error_message, exc_info=True)

                    if 'MonitoredOperation' in globals() and ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": (time.time() - thread_start) * 1000,
                            "thread_id": self.thread_id,
                            "success": False,
                            "error": str(e)
                        })

                    # Record failure if advanced logging is enabled
                    if 'record_failure' in globals() and ENABLE_ADVANCED_LOGGING:
                        record_failure("delete_thread", e, {
                            "thread_id": self.thread_id
                        })

        # End workflow tracking if enabled
        total_time = time.time() - start_time
        if workflow_id and 'end_workflow' in globals() and ENABLE_ADVANCED_LOGGING:
            end_workflow(workflow_id, "completed" if cleanup_success else "failed", {
                "processing_time_ms": total_time * 1000,
                "success": cleanup_success,
                "errors": errors
            })
            logger.info(f"Completed cleanup workflow (ID: {workflow_id}) in {total_time:.2f}s with status: {cleanup_success}")

        return {"success": cleanup_success, "errors": errors}

# Create a global instance
ai_solution = AISolution()

@app.get("/")
async def get_index():
    """Serve the HTML interface"""
    try:
        with open("agent_interface.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("agent_interface.html not found")
        return HTMLResponse(content="<h1>Error: Interface file not found</h1>")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to be processed with better error handling"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", file.filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved to {file_path}")

        # Check if the file is supported
        file_extension = file_path.lower().split('.')[-1]
        if file_extension not in ['txt', 'pdf', 'md', 'csv']:
            return {
                "status": "error",
                "message": f"Unsupported file type: .{file_extension}. Please upload txt, pdf, md, or csv files."
            }

        # Process just this file to see if it works
        chunks = ai_solution._chunk_file(file_path)
        if not chunks:
            return {
                "status": "error",
                "message": f"Could not extract text from {file.filename}. Make sure it's a valid document."
            }

        # Add to file paths and update knowledge base
        if file_path not in ai_solution.file_paths:
            ai_solution.file_paths.append(file_path)

        success = ai_solution.create_knowledge_base()

        if success:
            return {"status": "success", "filename": file.filename, "path": file_path}
        else:
            return {"status": "error", "message": "Failed to create knowledge base from uploaded file."}
    except Exception as e:
        logger.error(f"Exception in upload_file: {str(e)}")
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

@app.post("/process-with-agents")
async def process_with_agents(file_path: str = Form(...), query: str = Form(None), background_tasks: BackgroundTasks = None):
    """Process a document with the agent system"""
    # Generate a unique job ID
    job_id = f"job_{int(time.time())}"

    # Check if the file exists
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"File not found: {file_path}"}
        )

    # Start processing in background if requested
    if background_tasks:
        processing_jobs[job_id] = {"status": "processing", "file_path": file_path, "query": query}
        background_tasks.add_task(process_document_background, job_id, file_path, query)
        return {"status": "processing", "job_id": job_id}

    # Process synchronously
    try:
        logger.info(f"Processing document {file_path} with query: {query}")
        result = ai_solution.process_document_with_agents(file_path, query)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/agent-job/{job_id}")
async def get_agent_job_status(job_id: str):
    """Get the status of a background agent job"""
    if job_id not in processing_jobs:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"Job not found: {job_id}"}
        )

    return processing_jobs[job_id]

@app.get("/agent-activity")
async def get_agent_activity(limit: int = 10):
    """Get recent agent activity logs"""
    try:
        activity = agent_system.get_agent_activity(limit)
        return {"activity": activity}
    except Exception as e:
        logger.error(f"Error getting agent activity: {str(e)}")
        return {"status": "error", "message": str(e)}

async def process_document_background(job_id, file_path, query):
    """Process a document in the background"""
    try:
        result = ai_solution.process_document_with_agents(file_path, query)
        processing_jobs[job_id] = {
            "status": "completed",
            "file_path": file_path,
            "query": query,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        processing_jobs[job_id] = {
            "status": "error",
            "file_path": file_path,
            "query": query,
            "error": str(e)
        }

@app.post("/debug-upload")
async def debug_upload(request: Request):
    """Debug endpoint to see what's coming in the request"""
    form = await request.form()

    # Return information about what was received
    form_data = {}
    files_info = {}

    for key, value in form.items():
        if isinstance(value, UploadFile):
            files_info[key] = {
                "filename": value.filename,
                "content_type": value.content_type,
                "size": "unknown"  # we'd need to read the file to get the size
            }
        else:
            form_data[key] = str(value)

    return {
        "received_form_data": form_data,
        "received_files": files_info,
        "content_type": request.headers.get("content-type", "none")
    }

@app.post("/upload-text")
async def upload_text(content: str = Form(...), filename: str = Form(...)):
    """Alternative endpoint that accepts text content directly"""
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", filename)

    # Write the content to a file
    with open(file_path, "w") as f:
        f.write(content)

    # Add to file paths and update knowledge base
    ai_solution.file_paths.append(file_path)
    success = ai_solution.create_knowledge_base()

    if success:
        return {"status": "success", "filename": filename, "path": file_path}
    else:
        return {"status": "error", "message": "Failed to process file"}

@app.post("/upload-simple")
async def upload_simple(file: UploadFile = File(...)):
    """Simplified upload endpoint with better error handling"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", file.filename)

        # Save the file using a simpler method
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to file paths and update knowledge base
        ai_solution.file_paths.append(file_path)
        success = ai_solution.create_knowledge_base()

        if success:
            return {"status": "success", "filename": file.filename, "path": file_path}
        else:
            return {"status": "error", "message": "Failed to process file"}
    except Exception as e:
        logger.error(f"Error in upload_simple: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Import advanced logging modules if available
try:
    from advanced_tracing import wrap_with_logging
    from performance_monitor import MonitoredOperation
    from failure_analysis import handle_async_exceptions, record_failure
    from workflow_visualization import start_workflow, end_workflow, WorkflowStep
    ENABLE_ADVANCED_LOGGING = True
except ImportError:
    logger.warning("Advanced logging modules not available, using basic logging")
    ENABLE_ADVANCED_LOGGING = False

    # Define dummy context managers for when advanced logging is not available
    class DummyContextManager:
        def __enter__(self): return None
        def __exit__(self, *args): pass

    class DummyWorkflowStep(DummyContextManager):
        def __init__(self, *args, **kwargs): pass
        def add_metadata(self, metadata): pass

    # Define dummy functions
    def start_workflow(*args, **kwargs): return None
    def end_workflow(*args, **kwargs): pass
    def record_failure(*args, **kwargs): pass
    def handle_async_exceptions(component): return lambda f: f

    # Replace actual classes with dummies
    MonitoredOperation = DummyContextManager
    WorkflowStep = DummyWorkflowStep

@app.post("/query")
@handle_async_exceptions("query_endpoint") if ENABLE_ADVANCED_LOGGING else lambda f: f
async def query(question: str = Form(...), new_conversation: bool = Form(False)):
    """Query the assistant with conversation support"""
    # Start workflow tracking if enabled
    workflow_id = None
    start_time = time.time()

    if ENABLE_ADVANCED_LOGGING:
        workflow_id = start_workflow("Query Processing", {
            "endpoint": "/query",
            "method": "POST",
            "timestamp": start_time,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "new_conversation": new_conversation
        })
        logger.info(f"Started query workflow tracking (ID: {workflow_id})")

    try:
        # Initialize assistant if needed
        with WorkflowStep("initialize_assistant", workflow_id) if ENABLE_ADVANCED_LOGGING else DummyContextManager() as step:
            if not ai_solution.assistant_id:
                logger.info("Creating assistant for query processing")
                ai_solution.create_assistant()
                if ENABLE_ADVANCED_LOGGING and step:
                    step.add_metadata({"assistant_id": ai_solution.assistant_id})

        # Process the query with performance monitoring
        with MonitoredOperation("process_query") if ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
            logger.info(f"Processing query: '{question[:50]}...' (new_conversation={new_conversation})")
            response = ai_solution.process_query(question, new_conversation)
            processing_time = time.time() - start_time

            if ENABLE_ADVANCED_LOGGING and op:
                op.set_metadata({
                    "processing_time_ms": processing_time * 1000,
                    "response_length": len(response) if response else 0,
                    "new_conversation": new_conversation
                })
                logger.info(f"Query processed in {processing_time:.2f}s, response length: {len(response) if response else 0}")

        # End workflow tracking if enabled
        if ENABLE_ADVANCED_LOGGING and workflow_id:
            end_workflow(workflow_id, "completed", {
                "processing_time_ms": processing_time * 1000,
                "response_length": len(response) if response else 0,
                "thread_id": ai_solution.current_thread
            })
            logger.info(f"Completed query workflow (ID: {workflow_id})")

        return {"response": response}
    except Exception as e:
        # Log the error
        logger.error(f"Error processing query: {e}", exc_info=True)

        # Record failure if advanced logging is enabled
        if ENABLE_ADVANCED_LOGGING:
            record_failure("query_endpoint", e, {
                "question": question,
                "new_conversation": new_conversation,
                "thread_id": getattr(ai_solution, 'current_thread', None)
            })

            # End workflow if it was started
            if workflow_id:
                end_workflow(workflow_id, "failed", {
                    "error": str(e),
                    "error_type": e.__class__.__name__
                })

        # Return error response
        return {"response": f"Error processing your query: {str(e)}", "error": True}

@app.post("/stream")
@handle_async_exceptions("stream_endpoint") if ENABLE_ADVANCED_LOGGING else lambda f: f
async def stream_query(question: str = Form(...), new_conversation: bool = Form(False)):
    """Stream a response from the assistant with conversation support"""
    # Start workflow tracking if enabled
    workflow_id = None
    start_time = time.time()

    if ENABLE_ADVANCED_LOGGING:
        workflow_id = start_workflow("Stream Query Processing", {
            "endpoint": "/stream",
            "method": "POST",
            "timestamp": start_time,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "new_conversation": new_conversation
        })
        logger.info(f"Started stream query workflow tracking (ID: {workflow_id})")

    try:
        # Initialize assistant if needed
        with WorkflowStep("initialize_assistant", workflow_id) if ENABLE_ADVANCED_LOGGING else DummyContextManager() as step:
            if not ai_solution.assistant_id:
                logger.info("Creating assistant for stream query processing")
                ai_solution.create_assistant()
                if ENABLE_ADVANCED_LOGGING and step:
                    step.add_metadata({"assistant_id": ai_solution.assistant_id})

        # Create a wrapper generator to add logging around the streaming process
        async def logged_stream_generator(stream_generator):
            total_tokens = 0
            chunk_count = 0
            stream_start_time = time.time()

            try:
                with MonitoredOperation("process_stream_query") if ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
                    logger.info(f"Starting stream for query: '{question[:50]}...' (new_conversation={new_conversation})")

                    async for chunk in stream_generator:
                        chunk_count += 1
                        chunk_size = len(chunk)
                        total_tokens += chunk_size / 4  # Rough estimate of tokens

                        # Log every 10th chunk to avoid excessive logging
                        if chunk_count % 10 == 0 and ENABLE_ADVANCED_LOGGING:
                            current_duration = time.time() - stream_start_time
                            logger.debug(f"Streaming chunk {chunk_count}, total ~{total_tokens:.0f} tokens, duration: {current_duration:.2f}s")

                        yield chunk

                    # Log completion
                    stream_duration = time.time() - stream_start_time
                    logger.info(f"Completed streaming response: {chunk_count} chunks, ~{total_tokens:.0f} tokens, duration: {stream_duration:.2f}s")

                    if ENABLE_ADVANCED_LOGGING and op:
                        op.set_metadata({
                            "processing_time_ms": stream_duration * 1000,
                            "chunk_count": chunk_count,
                            "estimated_tokens": total_tokens,
                            "new_conversation": new_conversation
                        })

                # End workflow tracking if enabled
                if ENABLE_ADVANCED_LOGGING and workflow_id:
                    end_workflow(workflow_id, "completed", {
                        "processing_time_ms": (time.time() - start_time) * 1000,
                        "chunk_count": chunk_count,
                        "estimated_tokens": total_tokens,
                        "thread_id": ai_solution.current_thread
                    })
                    logger.info(f"Completed stream query workflow (ID: {workflow_id})")

            except Exception as e:
                # Log the error
                logger.error(f"Error during streaming: {e}", exc_info=True)

                # Record failure if advanced logging is enabled
                if ENABLE_ADVANCED_LOGGING:
                    record_failure("stream_generator", e, {
                        "question": question,
                        "new_conversation": new_conversation,
                        "thread_id": getattr(ai_solution, 'current_thread', None),
                        "chunks_sent": chunk_count
                    })

                # Yield an error message
                yield f"Error during streaming: {str(e)}".encode()

                # End workflow if it was started
                if ENABLE_ADVANCED_LOGGING and workflow_id:
                    end_workflow(workflow_id, "failed", {
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "chunks_sent": chunk_count
                    })

                # Re-raise the exception to be handled by the decorator
                raise

        # Get the stream generator from the AI solution
        stream_generator = ai_solution.process_query_stream(question, new_conversation)

        # Return the streaming response with our logging wrapper
        return StreamingResponse(
            logged_stream_generator(stream_generator),
            media_type="text/event-stream"
        )

    except Exception as e:
        # Log the error
        logger.error(f"Error setting up stream query: {e}", exc_info=True)

        # Record failure if advanced logging is enabled
        if ENABLE_ADVANCED_LOGGING:
            record_failure("stream_endpoint", e, {
                "question": question,
                "new_conversation": new_conversation,
                "thread_id": getattr(ai_solution, 'current_thread', None)
            })

            # End workflow if it was started
            if workflow_id:
                end_workflow(workflow_id, "failed", {
                    "error": str(e),
                    "error_type": e.__class__.__name__
                })

        # For setup errors, return a streaming response with just the error message
        async def error_generator():
            yield f"Error: {str(e)}".encode()

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )

@app.get("/conversation-history")
@handle_async_exceptions("conversation_history_endpoint") if ENABLE_ADVANCED_LOGGING else lambda f: f
async def get_conversation_history():
    """Get the current conversation history"""
    # Start workflow tracking if enabled
    workflow_id = None
    start_time = time.time()

    if ENABLE_ADVANCED_LOGGING:
        workflow_id = start_workflow("Get Conversation History", {
            "endpoint": "/conversation-history",
            "method": "GET",
            "timestamp": start_time
        })
        logger.info(f"Started conversation history workflow tracking (ID: {workflow_id})")

    try:
        # Get conversation history with performance monitoring
        with MonitoredOperation("get_conversation_history") if ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
            if not ai_solution.conversation_history:
                logger.info("No conversation history available")
                result = {"history": []}
            else:
                history_count = len(ai_solution.conversation_history)
                logger.info(f"Retrieved conversation history: {history_count} messages")
                result = {"history": ai_solution.conversation_history}

            processing_time = time.time() - start_time

            if ENABLE_ADVANCED_LOGGING and op:
                op.set_metadata({
                    "processing_time_ms": processing_time * 1000,
                    "message_count": len(ai_solution.conversation_history) if ai_solution.conversation_history else 0,
                    "thread_id": ai_solution.current_thread
                })

        # End workflow tracking if enabled
        if ENABLE_ADVANCED_LOGGING and workflow_id:
            end_workflow(workflow_id, "completed", {
                "processing_time_ms": processing_time * 1000,
                "message_count": len(ai_solution.conversation_history) if ai_solution.conversation_history else 0,
                "thread_id": ai_solution.current_thread
            })
            logger.info(f"Completed conversation history workflow (ID: {workflow_id})")

        return result
    except Exception as e:
        # Log the error
        logger.error(f"Error retrieving conversation history: {e}", exc_info=True)

        # Record failure if advanced logging is enabled
        if ENABLE_ADVANCED_LOGGING:
            record_failure("conversation_history_endpoint", e, {
                "thread_id": getattr(ai_solution, 'current_thread', None)
            })

            # End workflow if it was started
            if workflow_id:
                end_workflow(workflow_id, "failed", {
                    "error": str(e),
                    "error_type": e.__class__.__name__
                })

        # Return empty history on error
        return {"history": [], "error": str(e)}

@app.post("/clear-conversation")
@handle_async_exceptions("clear_conversation_endpoint") if ENABLE_ADVANCED_LOGGING else lambda f: f
async def clear_conversation():
    """Clear the current conversation history and start a new thread"""
    # Start workflow tracking if enabled
    workflow_id = None
    start_time = time.time()
    old_thread_id = getattr(ai_solution, 'current_thread', None)

    if ENABLE_ADVANCED_LOGGING:
        workflow_id = start_workflow("Clear Conversation", {
            "endpoint": "/clear-conversation",
            "method": "POST",
            "timestamp": start_time,
            "old_thread_id": old_thread_id
        })
        logger.info(f"Started clear conversation workflow tracking (ID: {workflow_id})")

    try:
        # Clear conversation with performance monitoring
        with MonitoredOperation("clear_conversation") if ENABLE_ADVANCED_LOGGING else DummyContextManager() as op:
            # Store the old thread ID and history count for logging
            old_history_count = len(ai_solution.conversation_history) if ai_solution.conversation_history else 0

            # Clear the conversation
            logger.info(f"Clearing conversation history and thread ID: {old_thread_id}")
            ai_solution.current_thread = None
            ai_solution.conversation_history = []

            # Create a new thread
            with WorkflowStep("create_new_thread", workflow_id) if ENABLE_ADVANCED_LOGGING else DummyContextManager() as step:
                ai_solution.create_persistent_thread()
                new_thread_id = ai_solution.current_thread
                logger.info(f"Created new thread ID: {new_thread_id}")

                if ENABLE_ADVANCED_LOGGING and step:
                    step.add_metadata({"new_thread_id": new_thread_id})

            processing_time = time.time() - start_time

            if ENABLE_ADVANCED_LOGGING and op:
                op.set_metadata({
                    "processing_time_ms": processing_time * 1000,
                    "old_thread_id": old_thread_id,
                    "new_thread_id": ai_solution.current_thread,
                    "cleared_message_count": old_history_count
                })

        # End workflow tracking if enabled
        if ENABLE_ADVANCED_LOGGING and workflow_id:
            end_workflow(workflow_id, "completed", {
                "processing_time_ms": processing_time * 1000,
                "old_thread_id": old_thread_id,
                "new_thread_id": ai_solution.current_thread
            })
            logger.info(f"Completed clear conversation workflow (ID: {workflow_id})")

        return {"status": "success", "message": "Conversation cleared", "new_thread_id": ai_solution.current_thread}
    except Exception as e:
        # Log the error
        logger.error(f"Error clearing conversation: {e}", exc_info=True)

        # Record failure if advanced logging is enabled
        if ENABLE_ADVANCED_LOGGING:
            record_failure("clear_conversation_endpoint", e, {
                "old_thread_id": old_thread_id
            })

            # End workflow if it was started
            if workflow_id:
                end_workflow(workflow_id, "failed", {
                    "error": str(e),
                    "error_type": e.__class__.__name__
                })

        # Return error response
        return {"status": "error", "message": f"Error clearing conversation: {str(e)}"}

# Example standalone usage
if __name__ == "__main__":
    # Try different ports if 8000 is in use
    port = 8000
    max_port = 8010  # Try ports 8000-8010

    while port <= max_port:
        try:
            logger.info(f"Trying to start server on port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port)
            break
        except OSError as e:
            if e.errno == 10048:  # Port already in use
                logger.warning(f"Port {port} is already in use, trying next port...")
                port += 1
            else:
                # If it's a different error, re-raise it
                raise

    if port > max_port:
        logger.error("Could not find an available port in the range 8000-8010.")