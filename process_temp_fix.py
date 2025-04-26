from typing import Union
from function_tool import function_tool
from run_context_wrapper import RunContextWrapper
from retrieval_success import RetrievalSuccess
from retrieval_error import RetrievalError
from logger import logger

@function_tool(strict_mode=False)
async def process_temp_file(ctx: RunContextWrapper, file_id: str = None, filename: str = None) -> Union[RetrievalSuccess, RetrievalError]:
    """Alias for process_temporary_file that handles both file_id and filename parameters.
    Reads and returns the text content of a previously uploaded temporary file for use as context."""
    logger.info(f"[Tool Call] process_temp_file: file_id='{file_id}', filename='{filename}'")
    
    # Use file_id as filename if provided, otherwise use filename
    actual_filename = file_id if file_id is not None else filename
    
    if actual_filename is None:
        return RetrievalError(error_message="Either file_id or filename must be provided.")
    
    # Call the original function with the determined filename
    return await process_temporary_file(ctx, actual_filename) 