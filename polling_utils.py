"""
Polling Utilities Module

This module provides utilities for optimizing polling strategies when interacting
with external APIs, particularly the OpenAI API. It implements exponential backoff
and other strategies to reduce the frequency of status polling requests.
"""

import logging
import time
import random
from typing import Optional, Callable, TypeVar, Any, Dict

# Setup logging
logger = logging.getLogger(__name__)

# Type variable for generic function
T = TypeVar('T')

class ExponentialBackoff:
    """
    Implements an exponential backoff strategy with jitter for polling operations.
    
    This reduces API calls by increasing the wait time between successive polls,
    while adding randomness to prevent synchronized retries from multiple clients.
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        factor: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.25
    ):
        """
        Initialize the exponential backoff strategy.
        
        Args:
            initial_delay: Initial delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 60.0)
            factor: Multiplication factor for exponential growth (default: 2.0)
            jitter: Whether to add randomness to the delay (default: True)
            jitter_factor: How much jitter to add as a fraction of the delay (default: 0.25)
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        self.attempt = 0
        
    def reset(self) -> None:
        """Reset the attempt counter."""
        self.attempt = 0
        
    def get_delay(self) -> float:
        """
        Calculate the next delay based on the current attempt.
        
        Returns:
            The delay in seconds for the next attempt
        """
        # Calculate base delay using exponential backoff
        delay = min(self.initial_delay * (self.factor ** self.attempt), self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            
        # Ensure delay is not negative or exceeds max_delay
        delay = max(0, min(delay, self.max_delay))
        
        # Increment attempt counter for next call
        self.attempt += 1
        
        return delay
    
    def sleep(self) -> float:
        """
        Sleep for the calculated delay time.
        
        Returns:
            The actual time slept in seconds
        """
        delay = self.get_delay()
        logger.debug(f"Backing off for {delay:.2f} seconds (attempt {self.attempt})")
        time.sleep(delay)
        return delay


async def poll_with_exponential_backoff(
    check_function: Callable[[], T],
    is_complete: Callable[[T], bool],
    max_wait_time: float = 300.0,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    factor: float = 2.0,
    jitter: bool = True,
    on_poll: Optional[Callable[[T], None]] = None
) -> T:
    """
    Poll a function with exponential backoff until a condition is met or timeout occurs.
    
    Args:
        check_function: Function to call to check status
        is_complete: Function that takes the result of check_function and returns True if polling should stop
        max_wait_time: Maximum total wait time in seconds (default: 300.0 = 5 minutes)
        initial_delay: Initial delay between polls in seconds (default: 1.0)
        max_delay: Maximum delay between polls in seconds (default: 30.0)
        factor: Multiplication factor for exponential growth (default: 2.0)
        jitter: Whether to add randomness to the delay (default: True)
        on_poll: Optional callback function to call after each poll with the result
        
    Returns:
        The final result from check_function
        
    Raises:
        TimeoutError: If max_wait_time is exceeded
    """
    import asyncio
    
    backoff = ExponentialBackoff(
        initial_delay=initial_delay,
        max_delay=max_delay,
        factor=factor,
        jitter=jitter
    )
    
    start_time = time.time()
    total_polls = 0
    
    while True:
        # Check if we've exceeded the maximum wait time
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            logger.warning(f"Polling timed out after {elapsed:.2f} seconds and {total_polls} polls")
            raise TimeoutError(f"Polling timed out after {elapsed:.2f} seconds")
        
        # Call the check function
        result = check_function()
        total_polls += 1
        
        # Call the on_poll callback if provided
        if on_poll:
            on_poll(result)
        
        # Check if we're done
        if is_complete(result):
            logger.debug(f"Polling completed after {elapsed:.2f} seconds and {total_polls} polls")
            return result
        
        # Calculate delay using exponential backoff
        delay = backoff.get_delay()
        
        # Sleep asynchronously
        await asyncio.sleep(delay)


def poll_with_exponential_backoff_sync(
    check_function: Callable[[], T],
    is_complete: Callable[[T], bool],
    max_wait_time: float = 300.0,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    factor: float = 2.0,
    jitter: bool = True,
    on_poll: Optional[Callable[[T], None]] = None
) -> T:
    """
    Synchronous version of poll_with_exponential_backoff.
    
    Args:
        check_function: Function to call to check status
        is_complete: Function that takes the result of check_function and returns True if polling should stop
        max_wait_time: Maximum total wait time in seconds (default: 300.0 = 5 minutes)
        initial_delay: Initial delay between polls in seconds (default: 1.0)
        max_delay: Maximum delay between polls in seconds (default: 30.0)
        factor: Multiplication factor for exponential growth (default: 2.0)
        jitter: Whether to add randomness to the delay (default: True)
        on_poll: Optional callback function to call after each poll with the result
        
    Returns:
        The final result from check_function
        
    Raises:
        TimeoutError: If max_wait_time is exceeded
    """
    backoff = ExponentialBackoff(
        initial_delay=initial_delay,
        max_delay=max_delay,
        factor=factor,
        jitter=jitter
    )
    
    start_time = time.time()
    total_polls = 0
    
    while True:
        # Check if we've exceeded the maximum wait time
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            logger.warning(f"Polling timed out after {elapsed:.2f} seconds and {total_polls} polls")
            raise TimeoutError(f"Polling timed out after {elapsed:.2f} seconds")
        
        # Call the check function
        result = check_function()
        total_polls += 1
        
        # Call the on_poll callback if provided
        if on_poll:
            on_poll(result)
        
        # Check if we're done
        if is_complete(result):
            logger.debug(f"Polling completed after {elapsed:.2f} seconds and {total_polls} polls")
            return result
        
        # Sleep using exponential backoff
        backoff.sleep()


# Specific OpenAI API polling functions
def poll_openai_run_until_complete(
    client: Any,
    thread_id: str,
    run_id: str,
    max_wait_time: float = 300.0,
    initial_delay: float = 1.0,
    max_delay: float = 30.0
) -> Dict[str, Any]:
    """
    Poll an OpenAI run until it completes or fails, using exponential backoff.
    
    Args:
        client: OpenAI client instance
        thread_id: Thread ID
        run_id: Run ID
        max_wait_time: Maximum wait time in seconds (default: 300.0 = 5 minutes)
        initial_delay: Initial delay between polls in seconds (default: 1.0)
        max_delay: Maximum delay between polls in seconds (default: 30.0)
        
    Returns:
        The final run object
        
    Raises:
        TimeoutError: If max_wait_time is exceeded
        Exception: If the run fails
    """
    def check_run():
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        logger.info(f"Run status: {run.status}")
        return run
    
    def is_complete(run):
        if run.status == "completed":
            return True
        elif run.status in ["failed", "cancelled", "expired"]:
            error_message = getattr(run, 'last_error', 'No error details available')
            raise Exception(f"Run failed with status {run.status}: {error_message}")
        return False
    
    def on_poll(run):
        # Handle required_action if needed
        if run.status == "requires_action" and hasattr(run, 'required_action'):
            # This would need to be implemented based on your specific needs
            logger.info(f"Run requires action: {run.required_action.type}")
    
    return poll_with_exponential_backoff_sync(
        check_function=check_run,
        is_complete=is_complete,
        max_wait_time=max_wait_time,
        initial_delay=initial_delay,
        max_delay=max_delay,
        on_poll=on_poll
    )


async def poll_openai_run_until_complete_async(
    client: Any,
    thread_id: str,
    run_id: str,
    max_wait_time: float = 300.0,
    initial_delay: float = 1.0,
    max_delay: float = 30.0
) -> Dict[str, Any]:
    """
    Asynchronously poll an OpenAI run until it completes or fails, using exponential backoff.
    
    Args:
        client: OpenAI client instance
        thread_id: Thread ID
        run_id: Run ID
        max_wait_time: Maximum wait time in seconds (default: 300.0 = 5 minutes)
        initial_delay: Initial delay between polls in seconds (default: 1.0)
        max_delay: Maximum delay between polls in seconds (default: 30.0)
        
    Returns:
        The final run object
        
    Raises:
        TimeoutError: If max_wait_time is exceeded
        Exception: If the run fails
    """
    import asyncio
    
    def check_run():
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        logger.info(f"Run status: {run.status}")
        return run
    
    def is_complete(run):
        if run.status == "completed":
            return True
        elif run.status in ["failed", "cancelled", "expired"]:
            error_message = getattr(run, 'last_error', 'No error details available')
            raise Exception(f"Run failed with status {run.status}: {error_message}")
        return False
    
    def on_poll(run):
        # Handle required_action if needed
        if run.status == "requires_action" and hasattr(run, 'required_action'):
            # This would need to be implemented based on your specific needs
            logger.info(f"Run requires action: {run.required_action.type}")
    
    return await poll_with_exponential_backoff(
        check_function=check_run,
        is_complete=is_complete,
        max_wait_time=max_wait_time,
        initial_delay=initial_delay,
        max_delay=max_delay,
        on_poll=on_poll
    )
