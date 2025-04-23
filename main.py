# doc_processing_system.py - WITH CHUNKING

import os
import json
import time
import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Set
from pydantic import BaseModel, Field, ValidationError
import math # For ceiling function

# Import Agents SDK components (or mocks)
try:
    from agents import Agent, Runner, function_tool, handoff, RunContextWrapper, ModelSettings, enable_verbose_stdout_logging
except ImportError:
    logging.warning("Agents SDK not found. Using dummy implementations.")
    # ... (Keep your dummy classes/functions here if needed for testing without the SDK) ...
    class Agent:
        def __init__(self, name, instructions, model, tools=None, handoffs=None, model_settings=None, tool_use_behavior=None):
            self.name = name; self.instructions = instructions; self.model = model; self.tools = tools; self.handoffs = handoffs; self.model_settings = model_settings; self.tool_use_behavior = tool_use_behavior
    class Runner:
        @staticmethod
        async def run(agent, input, context=None, max_turns=None):
            logging.info(f"Dummy Runner: Running {agent.name} on input: {input[:100]}...")
            output = {}
            # ... (Simulate outputs as before) ...
            if agent.name == "Classifier": output = {"document_type": "contract", "document_subtype": "employment", "confidence": 0.9}
            elif agent.name == "Extractor": output = {"extracted_fields": {f"field_chunk_{time.time()}": "value"}, "confidence": 0.9, "extraction_notes": f"Note from chunk."}
            elif agent.name == "Analyzer": output = {"key_insights": [f"Insight from chunk {time.time()}"], "summary": f"Summary of chunk {time.time()}", "sentiment": "neutral"}
            elif agent.name == "InitialQA": output = {"answer": "Initial based on chunks.", "confidence": 0.8, "sources": ["Synthesized Data"]}
            elif agent.name == "QARefiner": output = {"answer": "Refined based on chunks.", "confidence": 0.9, "sources": ["Synthesized Data"]}
            class DummyResult:
                 def __init__(self, output_dict): self.final_output = json.dumps(output_dict)
            await asyncio.sleep(0.1)
            return DummyResult(output)
    class ModelSettings:
        def __init__(self, tool_choice=None): self.tool_choice = tool_choice
    def enable_verbose_stdout_logging(): logging.info("Verbose logging enabled (dummy).")

# Import AgentMemory (or mock)
try:
    from agent_memory import AgentMemory
except ImportError:
    logging.warning("AgentMemory not found. Using dummy implementation.")
    class AgentMemory:
        def log_agent_action(self, *args, **kwargs): pass


# --- ENABLE VERBOSE LOGGING ---
# enable_verbose_stdout_logging()
# -----------------------------

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models (Keep as before) ---
class ClassificationOutput(BaseModel): # ...
    document_type: str; document_subtype: Optional[str] = None; confidence: float
class ExtractionOutput(BaseModel): # ...
    extracted_fields: Dict[str, Any]; confidence: float; extraction_notes: Optional[str] = None
class AnalysisOutput(BaseModel): # ...
    key_insights: List[str]; summary: str; sentiment: Optional[str] = None
class QAOutput(BaseModel): # ...
    answer: str; confidence: float; sources: Optional[List[str]] = None
    def model_dump_json(self, indent=None, **kwargs): # Compatibility method
        data = self.__dict__
        if hasattr(self, 'model_dump'): data = self.model_dump(**kwargs)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    @classmethod
    def model_validate_json(cls, json_str: str): # Compatibility method
        data = json.loads(json_str)
        instance = cls(**data)
        if hasattr(cls, 'model_validate'): instance = cls.model_validate(data)
        return instance

# --- DocumentContext (Keep as before) ---
@dataclass
class DocumentContext: # ...
    document_content: str; user_query: Optional[str] = None; memory: Optional[AgentMemory] = None


# --- Document Processing System ---
class DocumentProcessingSystem:
    def __init__(self, api_key=None):
        logger.info("DPS __init__: Creating AgentMemory instance.")
        self.memory = AgentMemory()
        if api_key: os.environ["OPENAI_API_KEY"] = api_key
        self.classifier_agent = self._create_classifier_agent()
        self.extractor_agent = self._create_extractor_agent()
        self.analyzer_agent = self._create_analyzer_agent()
        self.qa_agent = self._create_qa_agent()
        self.refiner_agent = self._create_refiner_agent()
        logger.info("DPS __init__: All agents created.")

    # --- Agent Creation Methods (Keep instructions strict, remove tools/tool_choice) ---
    def _create_classifier_agent(self): # ... (Keep definition as in previous correct version) ...
        return Agent(name="Classifier", instructions="...", model="gpt-4o", tools=[])
    def _create_extractor_agent(self): # ... (Keep definition as in previous correct version) ...
        return Agent(name="Extractor", instructions="...", model="gpt-4o", tools=[])
    def _create_analyzer_agent(self): # ... (Keep definition as in previous correct version) ...
        return Agent(name="Analyzer", instructions="...", model="gpt-4o", tools=[])
    def _create_qa_agent(self): # ... (Keep definition as in previous correct version) ...
        return Agent(name="InitialQA", instructions="...", model="gpt-4o", tools=[])
    def _create_refiner_agent(self): # ... (Keep definition as in previous correct version) ...
        return Agent(name="QARefiner", instructions="...", model="gpt-4o", tools=[])

    # --- Helper: Text Chunking ---
    def _chunk_text(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        """Splits text into overlapping chunks."""
        if not isinstance(text, str): return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
            if start >= len(text) or chunk_overlap >= chunk_size : # Prevent infinite loop if overlap >= size
                break
        # Ensure the last part isn't missed if the loop condition stops it early
        if start < len(text) and start > 0 and len(text) - start + chunk_overlap < chunk_size:
             last_chunk_start = max(0, len(text)-chunk_size) # Ensure last chunk is full size if possible
             last_chunk = text[last_chunk_start:]
             if not chunks or chunks[-1] != last_chunk: # Avoid duplicate last chunk
                   chunks.append(last_chunk)

        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        return chunks

    # --- Helper: JSON Cleaning and Parsing (Keep as before) ---
    def _clean_and_parse_json(self, raw_output: Optional[Any], pydantic_model: type[BaseModel], agent_name_for_log: str) -> Dict[str, Any]:
        # ... (implementation from previous correct response) ...
        # Returns dict: {"status": "success", "parsed": ..., "cleaned": ..., "raw": ...}
        # OR {"status": "error", "message": ..., "cleaned": ..., "raw": ...}
        step_result: Dict[str, Any] = {"raw": raw_output, "cleaned": None} # Initialize step result
        # ... (rest of the function as provided before) ...
        if not isinstance(raw_output, str): # ... etc ...
             step_result["status"] = "error"; step_result["message"] = f"Invalid input type..."; return step_result
        cleaned_output = raw_output.strip() # ... rest of cleaning ...
        step_result["cleaned"] = cleaned_output
        if not cleaned_output: step_result["status"] = "error"; step_result["message"] = "..."; return step_result
        try:
            parsed_data = pydantic_model.model_validate_json(cleaned_output)
            step_result["status"] = "success"; step_result["parsed"] = parsed_data; return step_result
        except Exception as parse_error:
             error_message = f"Failed to parse {pydantic_model.__name__}: {parse_error}"; step_result["status"] = "error"; step_result["message"] = error_message; return step_result


    # --- PROCESS DOCUMENT WITH CHUNKING ---
    async def process_document(self, document_content: str, user_query: Optional[str] = None):
        logger.info(f"Starting document processing with chunking. Query: {'Yes' if user_query else 'No'}")
        start_time = time.time()
        pipeline_steps = []
        final_status = "success"
        pipeline_error_message = None

        # --- Agent results storage ---
        classification: Optional[ClassificationOutput] = None
        all_extracted_fields: Dict[str, Any] = {}
        extractor_confidences: List[float] = []
        extractor_notes: List[str] = []
        all_key_insights: Set[str] = set() # Use set for uniqueness
        analyzer_summaries: List[str] = []
        analyzer_sentiments: List[str] = []
        final_answer_obj: Optional[QAOutput] = None

        # --- Chunking Parameters ---
        CHUNK_SIZE = 2500  # Characters - adjust based on model/token limits
        CHUNK_OVERLAP = 300 # Characters

        # --- Helper to run agent step (modified slightly for clarity) ---
        async def run_agent_step(agent, agent_name, input_content, output_model, step_metadata=None):
            nonlocal final_status, pipeline_error_message
            step_result = {"agent_name": agent_name, "status": "pending", **(step_metadata or {})}
            parsed_data = None
            try:
                logger.info(f"Running {agent_name} agent {step_result.get('chunk_info', '')}...")
                agent_run_result = await Runner.run(agent, input=input_content)
                parse_result = self._clean_and_parse_json(
                    agent_run_result.final_output, output_model, f"{agent_name} {step_result.get('chunk_info', '')}"
                )
                step_result.update(parse_result)
                if parse_result["status"] == "success":
                    parsed_data = parse_result.get("parsed")
                else:
                    final_status = "error"
                    if not pipeline_error_message: pipeline_error_message = step_result.get("message") or f"{agent_name} step failed."
            except Exception as run_error:
                error_message = f"Error during {agent_name} agent run: {run_error}"
                logger.error(error_message, exc_info=True)
                step_result.update({"status": "error", "message": error_message})
                final_status = "error"
                if not pipeline_error_message: pipeline_error_message = error_message
            pipeline_steps.append(step_result)
            return parsed_data
        # --- End Helper ---

        # --- Execute Pipeline ---
        try:
            # Step 1: Classifier (Run on first chunk or truncated)
            # To avoid chunking logic just for classifier, let's truncate simply
            MAX_CLASSIFIER_CHARS = 4000
            classifier_content = document_content[:MAX_CLASSIFIER_CHARS]
            if len(document_content) > MAX_CLASSIFIER_CHARS:
                 logger.warning(f"Using truncated content ({MAX_CLASSIFIER_CHARS} chars) for Classifier.")
                 classifier_content += "\n[CONTENT TRUNCATED FOR CLASSIFICATION]"

            classifier_input = f"Document Content:\n---\n{classifier_content}\n---\nClassify the above document content."
            classification = await run_agent_step(self.classifier_agent, "Classifier", classifier_input, ClassificationOutput)
            if not classification: raise Exception(pipeline_error_message or "Classifier failed")

            # --- Chunk the full document for subsequent steps ---
            chunks = self._chunk_text(document_content, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                 raise ValueError("Document content resulted in zero chunks.")

            # Step 2: Extractor (Loop through chunks)
            logger.info(f"Running Extractor agent on {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                chunk_info = f"[Chunk {i+1}/{len(chunks)}]"
                extractor_input = f"Document Type: {classification.document_type}\n\nDocument Chunk Content:\n---\n{chunk}\n---\nExtract information from this chunk."
                extraction_chunk_result = await run_agent_step(
                    self.extractor_agent, "Extractor", extractor_input, ExtractionOutput, {"chunk_info": chunk_info}
                )
                if extraction_chunk_result: # If chunk parsing succeeded
                    all_extracted_fields.update(extraction_chunk_result.extracted_fields) # Merge fields (later overwrite)
                    extractor_confidences.append(extraction_chunk_result.confidence)
                    if extraction_chunk_result.extraction_notes:
                        extractor_notes.append(f"Chunk {i+1}: {extraction_chunk_result.extraction_notes}")
                # Note: We continue even if one chunk fails, but overall status might be error

            # Synthesize Extractor results
            avg_extraction_confidence = sum(extractor_confidences) / len(extractor_confidences) if extractor_confidences else 0.0
            combined_extraction_notes = "\n".join(extractor_notes) if extractor_notes else None
            # Create a final ExtractionOutput structure (even if some chunks failed)
            final_extraction_data = ExtractionOutput(
                 extracted_fields=all_extracted_fields,
                 confidence=avg_extraction_confidence,
                 extraction_notes=combined_extraction_notes
            )


            # Step 3: Analyzer (Loop through chunks)
            logger.info(f"Running Analyzer agent on {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                chunk_info = f"[Chunk {i+1}/{len(chunks)}]"
                analyzer_input = f"Document Chunk Content:\n---\n{chunk}\n---\nAnalyze this chunk."
                analysis_chunk_result = await run_agent_step(
                    self.analyzer_agent, "Analyzer", analyzer_input, AnalysisOutput, {"chunk_info": chunk_info}
                )
                if analysis_chunk_result:
                    all_key_insights.update(analysis_chunk_result.key_insights) # Add unique insights
                    analyzer_summaries.append(analysis_chunk_result.summary)
                    if analysis_chunk_result.sentiment and analysis_chunk_result.sentiment != "neutral":
                         analyzer_sentiments.append(analysis_chunk_result.sentiment) # Collect non-neutral sentiments
                # Continue even if one chunk fails

            # Synthesize Analyzer results (Simplified)
            # A better approach would use another LLM call to synthesize summaries
            final_summary = analyzer_summaries[-1] if analyzer_summaries else "Analysis summary could not be generated."
            # Simple sentiment aggregation: take first non-neutral, or last neutral, or none
            final_sentiment = next((s for s in analyzer_sentiments if s), analyzer_sentiments[-1] if analyzer_sentiments else None)

            final_analysis_data = AnalysisOutput(
                key_insights=sorted(list(all_key_insights)), # Convert set back to sorted list
                summary=final_summary,
                sentiment=final_sentiment
            )

            # Step 4 & 5: QA + Refinement (Using SYNTHESIZED Context)
            final_answer_obj = None
            if user_query:
                # Use SYNTHESIZED extraction and analysis results for context
                insights_str = '; '.join(final_analysis_data.key_insights) if final_analysis_data else "N/A"
                # Use the merged extracted fields
                extracted_str = json.dumps(final_extraction_data.extracted_fields, indent=2, ensure_ascii=False) if final_extraction_data else "{}"
                doc_type_str = f"{classification.document_type} ({classification.document_subtype or 'N/A'})" if classification else "N/A"

                qa_input_context_str = (
                    f"User Query: {user_query}\n\n"
                    f"--- Relevant Context ---\n"
                    f"Document Type: {doc_type_str}\n"
                    f"Extracted Data (Synthesized): {extracted_str}\n"
                    f"Key Insights (Synthesized): {insights_str}\n"
                    f"Overall Summary (from last chunk): {final_analysis_data.summary}\n" # Add synthesized summary
                    f"---\n"
                    f"Based ONLY on the Relevant Context provided above, answer the User Query."
                )

                initial_qa_answer = await run_agent_step(self.qa_agent, "InitialQA", qa_input_context_str, QAOutput)
                final_answer_obj = initial_qa_answer

                if initial_qa_answer:
                     refiner_input = (
                         f"Original User Query: {user_query}\n\n"
                         f"--- Original Context ---\n{qa_input_context_str}\n---\n\n"
                         f"Initial Answer (JSON):\n{initial_qa_answer.model_dump_json(indent=2)}\n\n"
                         f"Evaluate the Initial Answer based ONLY on the Original Context and Query, then provide a final, possibly refined, JSON response following the required structure."
                     )
                     refined_answer = await run_agent_step(self.refiner_agent, "QARefiner", refiner_input, QAOutput)
                     if refined_answer: final_answer_obj = refined_answer
                     else: logger.warning("QA refinement step failed. Using initial QA answer (if available).")
                else:
                     logger.warning("Initial QA step failed. Skipping refinement.")
            else:
                 logger.info("No user query provided. Skipping QA steps.")


        except Exception as pipeline_error:
            logger.error(f"Pipeline processing halted: {pipeline_error}")
            # final_status should already be 'error'


        # --- Generate Final Result Dictionary ---
        total_time = time.time() - start_time
        logger.info(f"Document processing completed in {total_time:.2f} seconds with status: {final_status}")

        final_result_dict = {
            "status": final_status,
            "processing_time": total_time,
            "filename": "N/A", # Added by app.py
            "pipeline_steps": pipeline_steps,
            # --- Add top-level convenience fields from final state ---
            "document_type": getattr(classification, 'document_type', None),
            "document_subtype": getattr(classification, 'document_subtype', None),
            "extracted_information": getattr(final_extraction_data, 'extracted_fields', None), # Use synthesized data
            "key_insights": getattr(final_analysis_data, 'key_insights', None), # Use synthesized data
            "answer": getattr(final_answer_obj, 'answer', None),
            "answer_confidence": getattr(final_answer_obj, 'confidence', None),
            "answer_sources": getattr(final_answer_obj, 'sources', None),
        }
        if final_status == "error":
             final_result_dict["message"] = pipeline_error_message or "An unspecified error occurred in the pipeline."

        try: logger.debug(f"Final dictionary structure:\n{json.dumps(final_result_dict, indent=2, default=str)}")
        except TypeError: logger.debug("Could not serialize final dictionary for logging.")

        return final_result_dict

# --- Example Usage Block (Optional for direct testing) ---
# async def main(): ...
# if __name__ == "__main__": ...