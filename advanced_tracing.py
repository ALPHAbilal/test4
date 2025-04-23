"""
Advanced Tracing Module

This module provides enhanced tracing and logging capabilities for the agent workflow,
helping to identify performance bottlenecks and failure points.
"""

import logging
import time
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

# Import Agent SDK tracing components
from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.traces import Trace
from agents.tracing.spans import Span

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a file handler for detailed logs
os.makedirs('logs', exist_ok=True)
detailed_log_file = os.path.join('logs', f'agent_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(detailed_log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create a performance log file
performance_log_file = os.path.join('logs', f'agent_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
perf_handler = logging.FileHandler(performance_log_file)
perf_handler.setLevel(logging.INFO)
perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
perf_handler.setFormatter(perf_formatter)
perf_logger = logging.getLogger('agent_performance')
perf_logger.setLevel(logging.INFO)
perf_logger.addHandler(perf_handler)

@dataclass
class SpanMetrics:
    """Metrics for a single span in the agent workflow"""
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    status: str = "started"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class TraceMetrics:
    """Metrics for a complete trace (workflow execution)"""
    trace_id: str
    workflow_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    span_count: int = 0
    error_count: int = 0
    status: str = "started"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class MetricsCollector:
    """Collects and aggregates metrics from traces and spans"""
    
    def __init__(self):
        self.traces: Dict[str, TraceMetrics] = {}
        self.spans: Dict[str, SpanMetrics] = {}
        self.span_durations_by_name: Dict[str, List[float]] = {}
        self.error_counts_by_name: Dict[str, int] = {}
        self.lock = threading.Lock()
        
    def add_trace(self, trace: Trace) -> None:
        """Add a new trace to the metrics collector"""
        with self.lock:
            self.traces[trace.trace_id] = TraceMetrics(
                trace_id=trace.trace_id,
                workflow_name=trace.workflow_name or "unknown",
                start_time=time.time()
            )
    
    def finish_trace(self, trace: Trace) -> None:
        """Mark a trace as finished and calculate its metrics"""
        with self.lock:
            if trace.trace_id in self.traces:
                trace_metrics = self.traces[trace.trace_id]
                trace_metrics.end_time = time.time()
                trace_metrics.duration_ms = (trace_metrics.end_time - trace_metrics.start_time) * 1000
                trace_metrics.error = str(trace.error) if trace.error else None
                trace_metrics.status = "error" if trace.error else "completed"
                
                # Log the trace completion
                self._log_trace_completion(trace_metrics)
    
    def add_span(self, span: Span[Any]) -> None:
        """Add a new span to the metrics collector"""
        with self.lock:
            # Extract agent or tool name from span attributes
            agent_name = None
            tool_name = None
            
            if hasattr(span, 'attributes'):
                if 'agent_name' in span.attributes:
                    agent_name = span.attributes['agent_name']
                if 'tool_name' in span.attributes:
                    tool_name = span.attributes['tool_name']
            
            # Calculate input size if possible
            input_size = None
            if hasattr(span, 'input') and span.input is not None:
                try:
                    if isinstance(span.input, str):
                        input_size = len(span.input)
                    elif isinstance(span.input, dict):
                        input_size = len(json.dumps(span.input))
                except:
                    pass
            
            # Create span metrics
            self.spans[span.span_id] = SpanMetrics(
                span_id=span.span_id,
                name=span.name,
                start_time=time.time(),
                parent_id=span.parent_id,
                trace_id=span.trace_id,
                input_size=input_size,
                agent_name=agent_name,
                tool_name=tool_name
            )
            
            # Update trace span count
            if span.trace_id in self.traces:
                self.traces[span.trace_id].span_count += 1
    
    def finish_span(self, span: Span[Any]) -> None:
        """Mark a span as finished and calculate its metrics"""
        with self.lock:
            if span.span_id in self.spans:
                span_metrics = self.spans[span.span_id]
                span_metrics.end_time = time.time()
                span_metrics.duration_ms = (span_metrics.end_time - span_metrics.start_time) * 1000
                span_metrics.error = str(span.error) if span.error else None
                span_metrics.status = "error" if span.error else "completed"
                
                # Calculate output size if possible
                if hasattr(span, 'output') and span.output is not None:
                    try:
                        if isinstance(span.output, str):
                            span_metrics.output_size = len(span.output)
                        elif isinstance(span.output, dict):
                            span_metrics.output_size = len(json.dumps(span.output))
                    except:
                        pass
                
                # Update aggregated metrics
                if span_metrics.name not in self.span_durations_by_name:
                    self.span_durations_by_name[span_metrics.name] = []
                
                self.span_durations_by_name[span_metrics.name].append(span_metrics.duration_ms)
                
                if span_metrics.error:
                    if span_metrics.name not in self.error_counts_by_name:
                        self.error_counts_by_name[span_metrics.name] = 0
                    
                    self.error_counts_by_name[span_metrics.name] += 1
                    
                    # Update trace error count
                    if span_metrics.trace_id in self.traces:
                        self.traces[span_metrics.trace_id].error_count += 1
                
                # Log the span completion
                self._log_span_completion(span_metrics)
    
    def _log_trace_completion(self, trace_metrics: TraceMetrics) -> None:
        """Log detailed information about a completed trace"""
        perf_logger.info(
            f"TRACE_COMPLETE|{trace_metrics.trace_id}|{trace_metrics.workflow_name}|"
            f"{trace_metrics.duration_ms:.2f}ms|{trace_metrics.span_count} spans|"
            f"{trace_metrics.error_count} errors|{trace_metrics.status}"
        )
        
        if trace_metrics.error:
            logger.error(
                f"Trace {trace_metrics.trace_id} ({trace_metrics.workflow_name}) "
                f"completed with error: {trace_metrics.error}"
            )
        else:
            logger.info(
                f"Trace {trace_metrics.trace_id} ({trace_metrics.workflow_name}) "
                f"completed successfully in {trace_metrics.duration_ms:.2f}ms "
                f"with {trace_metrics.span_count} spans"
            )
    
    def _log_span_completion(self, span_metrics: SpanMetrics) -> None:
        """Log detailed information about a completed span"""
        # Log to performance log
        agent_tool_info = ""
        if span_metrics.agent_name:
            agent_tool_info += f"|agent:{span_metrics.agent_name}"
        if span_metrics.tool_name:
            agent_tool_info += f"|tool:{span_metrics.tool_name}"
            
        perf_logger.info(
            f"SPAN_COMPLETE|{span_metrics.span_id}|{span_metrics.name}|"
            f"{span_metrics.duration_ms:.2f}ms|{span_metrics.status}{agent_tool_info}"
        )
        
        # Log errors to main log
        if span_metrics.error:
            logger.error(
                f"Span {span_metrics.span_id} ({span_metrics.name}) "
                f"completed with error: {span_metrics.error}"
            )
        else:
            logger.debug(
                f"Span {span_metrics.span_id} ({span_metrics.name}) "
                f"completed successfully in {span_metrics.duration_ms:.2f}ms"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        with self.lock:
            summary = {
                "span_count": len(self.spans),
                "trace_count": len(self.traces),
                "error_count": sum(self.error_counts_by_name.values()),
                "avg_durations_ms": {},
                "max_durations_ms": {},
                "error_rates": {}
            }
            
            # Calculate average and max durations
            for name, durations in self.span_durations_by_name.items():
                if durations:
                    summary["avg_durations_ms"][name] = sum(durations) / len(durations)
                    summary["max_durations_ms"][name] = max(durations)
            
            # Calculate error rates
            for name, count in self.error_counts_by_name.items():
                total = len([s for s in self.spans.values() if s.name == name])
                if total > 0:
                    summary["error_rates"][name] = count / total
            
            return summary

class AdvancedTraceProcessor(TracingProcessor):
    """
    Advanced trace processor that collects detailed metrics and logs
    performance information about the agent workflow.
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.slow_threshold_ms = 1000  # Log spans taking longer than 1 second as slow
        
        # Start a background thread to periodically log performance summaries
        self.shutdown_event = threading.Event()
        self.summary_thread = threading.Thread(target=self._log_periodic_summaries, daemon=True)
        self.summary_thread.start()
    
    def on_trace_start(self, trace: Trace) -> None:
        """Called when a trace is started"""
        logger.info(f"Starting trace: {trace.trace_id} ({trace.workflow_name})")
        self.metrics.add_trace(trace)
    
    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace is finished"""
        self.metrics.finish_trace(trace)
    
    def on_span_start(self, span: Span[Any]) -> None:
        """Called when a span is started"""
        logger.debug(f"Starting span: {span.span_id} ({span.name})")
        self.metrics.add_span(span)
    
    def on_span_end(self, span: Span[Any]) -> None:
        """Called when a span is finished"""
        self.metrics.finish_span(span)
        
        # Check for slow spans
        span_metrics = self.metrics.spans.get(span.span_id)
        if span_metrics and span_metrics.duration_ms and span_metrics.duration_ms > self.slow_threshold_ms:
            logger.warning(
                f"SLOW SPAN: {span_metrics.name} took {span_metrics.duration_ms:.2f}ms "
                f"(threshold: {self.slow_threshold_ms}ms)"
            )
    
    def shutdown(self) -> None:
        """Called when the application stops"""
        logger.info("Shutting down AdvancedTraceProcessor")
        self.shutdown_event.set()
        
        # Log final summary
        summary = self.metrics.get_performance_summary()
        logger.info(f"Final performance summary: {json.dumps(summary, indent=2)}")
    
    def force_flush(self) -> None:
        """Forces an immediate flush of all queued spans/traces"""
        # This processor doesn't queue anything, so nothing to flush
        pass
    
    def _log_periodic_summaries(self) -> None:
        """Background thread to periodically log performance summaries"""
        interval_seconds = 60  # Log summary every minute
        
        while not self.shutdown_event.is_set():
            # Sleep for the interval, but check for shutdown every second
            for _ in range(interval_seconds):
                if self.shutdown_event.is_set():
                    return
                time.sleep(1)
            
            # Log summary
            try:
                summary = self.metrics.get_performance_summary()
                if summary["span_count"] > 0:
                    logger.info(f"Performance summary: {json.dumps(summary, indent=2)}")
            except Exception as e:
                logger.error(f"Error logging performance summary: {e}")

class DetailedLoggingMiddleware:
    """
    Middleware to add detailed logging to agent functions.
    This can be used to wrap agent functions to add detailed logging.
    """
    
    def __init__(self, func, func_name=None):
        self.func = func
        self.func_name = func_name or func.__name__
    
    async def __call__(self, *args, **kwargs):
        start_time = time.time()
        
        # Log the function call
        arg_str = ", ".join([str(a) for a in args])
        kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        params = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
        logger.info(f"CALL_START: {self.func_name}({params})")
        
        try:
            # Call the original function
            result = await self.func(*args, **kwargs)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log the result
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            
            logger.info(f"CALL_END: {self.func_name} completed in {duration_ms:.2f}ms with result: {result_str}")
            
            # Log slow functions
            if duration_ms > 1000:
                logger.warning(f"SLOW_FUNCTION: {self.func_name} took {duration_ms:.2f}ms")
            
            return result
        except Exception as e:
            # Log the error
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"CALL_ERROR: {self.func_name} failed after {duration_ms:.2f}ms with error: {str(e)}")
            
            # Re-raise the exception
            raise

def wrap_with_logging(obj, methods=None):
    """
    Wrap methods of an object with detailed logging.
    
    Args:
        obj: The object whose methods to wrap
        methods: List of method names to wrap, or None to wrap all methods
        
    Returns:
        The original object with wrapped methods
    """
    if methods is None:
        # Get all callable attributes that don't start with _
        methods = [name for name in dir(obj) if not name.startswith('_') and callable(getattr(obj, name))]
    
    for method_name in methods:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                wrapped = DetailedLoggingMiddleware(method, f"{obj.__class__.__name__}.{method_name}")
                setattr(obj, method_name, wrapped)
    
    return obj

# Helper function to register the advanced trace processor
def register_advanced_tracing():
    """Register the advanced trace processor with the Agent SDK"""
    from agents.tracing import add_trace_processor
    
    processor = AdvancedTraceProcessor()
    add_trace_processor(processor)
    logger.info("Registered AdvancedTraceProcessor for detailed agent workflow tracing")
    return processor
