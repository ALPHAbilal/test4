"""
Failure Analysis Module

This module provides tools for analyzing and reporting failures in the agent workflow,
helping to identify common failure patterns and root causes.
"""

import os
import json
import logging
import traceback
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a failure log file
failure_log_file = os.path.join('logs', f'agent_failures_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
failure_handler = logging.FileHandler(failure_log_file)
failure_handler.setLevel(logging.ERROR)
failure_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
failure_handler.setFormatter(failure_formatter)
failure_logger = logging.getLogger('agent_failures')
failure_logger.setLevel(logging.ERROR)
failure_logger.addHandler(failure_handler)

@dataclass
class FailureRecord:
    """Record of a failure in the agent workflow"""
    timestamp: float
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "context": self.context
        }

class FailureAnalyzer:
    """
    Analyzes and reports failures in the agent workflow.
    """
    
    def __init__(self):
        self.failures: List[FailureRecord] = []
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.component_failure_counts: Dict[str, int] = defaultdict(int)
        self.error_type_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
        # Create JSON file for failures
        self.json_file = os.path.join('logs', f'failures_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(self.json_file, 'w') as f:
            json.dump([], f)
    
    def record_failure(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """
        Record a failure in the agent workflow.
        
        Args:
            component: The component where the failure occurred
            error: The exception that was raised
            context: Additional context about the failure
        """
        with self.lock:
            # Get stack trace
            stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            
            # Create failure record
            failure = FailureRecord(
                timestamp=time.time(),
                component=component,
                error_type=error.__class__.__name__,
                error_message=str(error),
                stack_trace=stack_trace,
                context=context or {}
            )
            
            # Add to list
            self.failures.append(failure)
            
            # Update counts
            self.failure_counts[f"{component}:{error.__class__.__name__}"] += 1
            self.component_failure_counts[component] += 1
            self.error_type_counts[error.__class__.__name__] += 1
            
            # Log the failure
            failure_logger.error(
                f"FAILURE|{component}|{error.__class__.__name__}|{str(error)}"
            )
            
            # Write to JSON file
            self._append_to_json(failure)
            
            # Log to console
            logger.error(
                f"Failure in {component}: {error.__class__.__name__}: {str(error)}"
            )
    
    def _append_to_json(self, failure: FailureRecord):
        """Append a failure record to the JSON file"""
        try:
            with open(self.json_file, 'r+') as f:
                try:
                    failures = json.load(f)
                except json.JSONDecodeError:
                    failures = []
                
                failures.append(failure.to_dict())
                
                f.seek(0)
                f.truncate()
                json.dump(failures, f, indent=2)
        except Exception as e:
            logger.error(f"Error appending failure to JSON file: {e}")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of failures"""
        with self.lock:
            summary = {
                "total_failures": len(self.failures),
                "failures_by_component": dict(self.component_failure_counts),
                "failures_by_error_type": dict(self.error_type_counts),
                "top_failures": self._get_top_failures(5),
                "recent_failures": self._get_recent_failures(5)
            }
            
            return summary
    
    def _get_top_failures(self, limit: int) -> List[Dict[str, Any]]:
        """Get the top failures by count"""
        top_failures = sorted(
            self.failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [{"failure": k, "count": v} for k, v in top_failures]
    
    def _get_recent_failures(self, limit: int) -> List[Dict[str, Any]]:
        """Get the most recent failures"""
        recent_failures = sorted(
            self.failures,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "component": f.component,
                "error_type": f.error_type,
                "error_message": f.error_message,
                "timestamp": datetime.fromtimestamp(f.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            }
            for f in recent_failures
        ]
    
    def generate_report(self) -> str:
        """Generate a detailed failure report"""
        summary = self.get_failure_summary()
        
        report = [
            "# Agent Workflow Failure Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Summary",
            f"Total Failures: {summary['total_failures']}",
            "",
            "## Failures by Component",
        ]
        
        for component, count in sorted(summary['failures_by_component'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {component}: {count}")
        
        report.extend([
            "",
            "## Failures by Error Type",
        ])
        
        for error_type, count in sorted(summary['failures_by_error_type'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {error_type}: {count}")
        
        report.extend([
            "",
            "## Top Failures",
        ])
        
        for failure in summary['top_failures']:
            report.append(f"- {failure['failure']}: {failure['count']}")
        
        report.extend([
            "",
            "## Recent Failures",
        ])
        
        for failure in summary['recent_failures']:
            report.append(f"- {failure['timestamp']} - {failure['component']}: {failure['error_type']}: {failure['error_message']}")
        
        report.extend([
            "",
            "## Detailed Failure Records",
        ])
        
        for i, failure in enumerate(self.failures):
            report.extend([
                f"### Failure {i+1}",
                f"- Component: {failure.component}",
                f"- Error Type: {failure.error_type}",
                f"- Error Message: {failure.error_message}",
                f"- Timestamp: {datetime.fromtimestamp(failure.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "#### Stack Trace",
                "```",
                failure.stack_trace,
                "```",
                "",
                "#### Context",
                "```json",
                json.dumps(failure.context, indent=2),
                "```",
                ""
            ])
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None) -> str:
        """Generate and save a failure report"""
        if filename is None:
            filename = os.path.join('logs', f'failure_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Failure report saved to {filename}")
        return filename

# Global instance
failure_analyzer = FailureAnalyzer()

def record_failure(component: str, error: Exception, context: Dict[str, Any] = None):
    """Record a failure in the agent workflow"""
    failure_analyzer.record_failure(component, error, context)

def get_failure_summary() -> Dict[str, Any]:
    """Get a summary of failures"""
    return failure_analyzer.get_failure_summary()

def generate_failure_report() -> str:
    """Generate a detailed failure report"""
    return failure_analyzer.generate_report()

def save_failure_report(filename: str = None) -> str:
    """Generate and save a failure report"""
    return failure_analyzer.save_report(filename)

# Exception handler decorator
def handle_exceptions(component: str):
    """
    Decorator to handle exceptions and record failures.
    
    Example:
        @handle_exceptions("document_processor")
        def process_document(document):
            # ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Record the failure
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                record_failure(component, e, context)
                
                # Re-raise the exception
                raise
        return wrapper
    return decorator

# Async exception handler decorator
def handle_async_exceptions(component: str):
    """
    Decorator to handle exceptions in async functions and record failures.
    
    Example:
        @handle_async_exceptions("document_processor")
        async def process_document(document):
            # ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Record the failure
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                record_failure(component, e, context)
                
                # Re-raise the exception
                raise
        return wrapper
    return decorator
