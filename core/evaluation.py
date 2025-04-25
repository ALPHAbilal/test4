"""
Evaluation Module

This module provides functionality for evaluating the effectiveness of agent learning.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import os
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LearningMetrics:
    """
    Class for tracking and evaluating agent learning metrics.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None):
        """
        Initialize the learning metrics tracker.
        
        Args:
            metrics_dir: Directory to store metrics data. If None, metrics are not persisted.
        """
        self.metrics = {}
        self.metrics_dir = metrics_dir
        
        # Create metrics directory if it doesn't exist
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            logger.info(f"Created metrics directory: {metrics_dir}")
        
        # Load persisted metrics if available
        if metrics_dir:
            self._load_persisted_metrics()
    
    def _get_metrics_path(self, metric_key: str) -> str:
        """
        Get the path for storing metrics data.
        
        Args:
            metric_key: Metric key
            
        Returns:
            Path to the metrics file
        """
        if not self.metrics_dir:
            return None
        
        # Replace any characters that are not allowed in filenames
        safe_key = metric_key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return os.path.join(self.metrics_dir, f"{safe_key}.json")
    
    def _load_persisted_metrics(self) -> None:
        """
        Load persisted metrics from disk.
        """
        if not self.metrics_dir:
            return
        
        try:
            # Get all JSON files in the metrics directory
            for filename in os.listdir(self.metrics_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.metrics_dir, filename)
                    
                    try:
                        # Load the metrics from the file
                        with open(file_path, "r") as f:
                            metrics_data = json.load(f)
                        
                        # Get the metric key from the filename
                        metric_key = os.path.splitext(filename)[0].replace("_", ":")
                        
                        # Store the metrics
                        self.metrics[metric_key] = metrics_data
                        logger.info(f"Loaded persisted metrics for {metric_key}")
                    except Exception as e:
                        logger.error(f"Error loading persisted metrics from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading persisted metrics: {e}")
    
    def _persist_metrics(self, metric_key: str) -> None:
        """
        Persist metrics to disk.
        
        Args:
            metric_key: Metric key
        """
        if not self.metrics_dir:
            return
        
        try:
            # Get the persistence path
            persistence_path = self._get_metrics_path(metric_key)
            
            # Get the metrics
            metrics_data = self.metrics.get(metric_key)
            
            # Persist the metrics
            with open(persistence_path, "w") as f:
                json.dump(metrics_data, f)
            
            logger.info(f"Persisted metrics for {metric_key}")
        except Exception as e:
            logger.error(f"Error persisting metrics for {metric_key}: {e}")
    
    def record_workflow_execution(
        self,
        query_type: str,
        pattern_used: bool,
        pattern_name: Optional[str] = None,
        step_count: int = 0,
        success: bool = True,
        execution_time: float = 0.0,
        session_id: Optional[str] = None
    ) -> None:
        """
        Record a workflow execution.
        
        Args:
            query_type: Type of query
            pattern_used: Whether a learned pattern was used
            pattern_name: Name of the pattern used (if any)
            step_count: Number of steps in the workflow
            success: Whether the workflow was successful
            execution_time: Time taken to execute the workflow (in seconds)
            session_id: Session ID
        """
        # Create the metric key
        metric_key = f"workflow_execution:{query_type}"
        
        # Get existing metrics or create new ones
        metrics_data = self.metrics.get(metric_key, {
            "executions": [],
            "pattern_usage": {
                "used": 0,
                "not_used": 0
            },
            "success_rate": {
                "with_pattern": {
                    "success": 0,
                    "failure": 0
                },
                "without_pattern": {
                    "success": 0,
                    "failure": 0
                }
            },
            "step_counts": [],
            "execution_times": [],
            "last_updated": time.time()
        })
        
        # Add execution data
        execution_data = {
            "timestamp": time.time(),
            "pattern_used": pattern_used,
            "pattern_name": pattern_name,
            "step_count": step_count,
            "success": success,
            "execution_time": execution_time,
            "session_id": session_id
        }
        
        metrics_data["executions"].append(execution_data)
        
        # Update pattern usage
        if pattern_used:
            metrics_data["pattern_usage"]["used"] += 1
        else:
            metrics_data["pattern_usage"]["not_used"] += 1
        
        # Update success rate
        if pattern_used:
            if success:
                metrics_data["success_rate"]["with_pattern"]["success"] += 1
            else:
                metrics_data["success_rate"]["with_pattern"]["failure"] += 1
        else:
            if success:
                metrics_data["success_rate"]["without_pattern"]["success"] += 1
            else:
                metrics_data["success_rate"]["without_pattern"]["failure"] += 1
        
        # Update step counts and execution times
        metrics_data["step_counts"].append(step_count)
        metrics_data["execution_times"].append(execution_time)
        
        # Update last updated timestamp
        metrics_data["last_updated"] = time.time()
        
        # Store the metrics
        self.metrics[metric_key] = metrics_data
        
        # Persist the metrics
        self._persist_metrics(metric_key)
    
    def record_pattern_learning(
        self,
        query_type: str,
        pattern_name: str,
        pattern_steps: List[str],
        success_count: int = 1,
        session_id: Optional[str] = None
    ) -> None:
        """
        Record a pattern learning event.
        
        Args:
            query_type: Type of query
            pattern_name: Name of the pattern
            pattern_steps: Steps in the pattern
            success_count: Success count for the pattern
            session_id: Session ID
        """
        # Create the metric key
        metric_key = f"pattern_learning:{query_type}"
        
        # Get existing metrics or create new ones
        metrics_data = self.metrics.get(metric_key, {
            "patterns": {},
            "learning_events": [],
            "last_updated": time.time()
        })
        
        # Add pattern data
        pattern_data = {
            "steps": pattern_steps,
            "success_count": success_count,
            "first_learned": time.time(),
            "last_updated": time.time()
        }
        
        # Check if pattern already exists
        if pattern_name in metrics_data["patterns"]:
            # Update existing pattern
            existing_pattern = metrics_data["patterns"][pattern_name]
            existing_pattern["success_count"] = success_count
            existing_pattern["last_updated"] = time.time()
            
            # Only update steps if they've changed
            if existing_pattern["steps"] != pattern_steps:
                existing_pattern["steps"] = pattern_steps
        else:
            # Add new pattern
            metrics_data["patterns"][pattern_name] = pattern_data
        
        # Add learning event
        learning_event = {
            "timestamp": time.time(),
            "pattern_name": pattern_name,
            "success_count": success_count,
            "session_id": session_id
        }
        
        metrics_data["learning_events"].append(learning_event)
        
        # Update last updated timestamp
        metrics_data["last_updated"] = time.time()
        
        # Store the metrics
        self.metrics[metric_key] = metrics_data
        
        # Persist the metrics
        self._persist_metrics(metric_key)
    
    def get_workflow_metrics(self, query_type: str) -> Dict[str, Any]:
        """
        Get metrics for a specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Metrics for the query type
        """
        # Create the metric key
        metric_key = f"workflow_execution:{query_type}"
        
        # Get the metrics
        metrics_data = self.metrics.get(metric_key, {})
        
        # If no metrics, return empty dict
        if not metrics_data:
            return {}
        
        # Calculate additional metrics
        metrics = {
            "total_executions": len(metrics_data.get("executions", [])),
            "pattern_usage_ratio": 0.0,
            "success_rate_with_pattern": 0.0,
            "success_rate_without_pattern": 0.0,
            "average_step_count": 0.0,
            "average_execution_time": 0.0
        }
        
        # Calculate pattern usage ratio
        pattern_usage = metrics_data.get("pattern_usage", {})
        used = pattern_usage.get("used", 0)
        not_used = pattern_usage.get("not_used", 0)
        total = used + not_used
        
        if total > 0:
            metrics["pattern_usage_ratio"] = used / total
        
        # Calculate success rates
        success_rate = metrics_data.get("success_rate", {})
        with_pattern = success_rate.get("with_pattern", {})
        without_pattern = success_rate.get("without_pattern", {})
        
        with_pattern_success = with_pattern.get("success", 0)
        with_pattern_failure = with_pattern.get("failure", 0)
        with_pattern_total = with_pattern_success + with_pattern_failure
        
        without_pattern_success = without_pattern.get("success", 0)
        without_pattern_failure = without_pattern.get("failure", 0)
        without_pattern_total = without_pattern_success + without_pattern_failure
        
        if with_pattern_total > 0:
            metrics["success_rate_with_pattern"] = with_pattern_success / with_pattern_total
        
        if without_pattern_total > 0:
            metrics["success_rate_without_pattern"] = without_pattern_success / without_pattern_total
        
        # Calculate average step count and execution time
        step_counts = metrics_data.get("step_counts", [])
        execution_times = metrics_data.get("execution_times", [])
        
        if step_counts:
            metrics["average_step_count"] = sum(step_counts) / len(step_counts)
            metrics["median_step_count"] = statistics.median(step_counts) if len(step_counts) > 0 else 0
            metrics["min_step_count"] = min(step_counts) if step_counts else 0
            metrics["max_step_count"] = max(step_counts) if step_counts else 0
        
        if execution_times:
            metrics["average_execution_time"] = sum(execution_times) / len(execution_times)
            metrics["median_execution_time"] = statistics.median(execution_times) if len(execution_times) > 0 else 0
            metrics["min_execution_time"] = min(execution_times) if execution_times else 0
            metrics["max_execution_time"] = max(execution_times) if execution_times else 0
        
        return metrics
    
    def get_pattern_metrics(self, query_type: str) -> Dict[str, Any]:
        """
        Get pattern metrics for a specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Pattern metrics for the query type
        """
        # Create the metric key
        metric_key = f"pattern_learning:{query_type}"
        
        # Get the metrics
        metrics_data = self.metrics.get(metric_key, {})
        
        # If no metrics, return empty dict
        if not metrics_data:
            return {}
        
        # Calculate additional metrics
        metrics = {
            "total_patterns": len(metrics_data.get("patterns", {})),
            "total_learning_events": len(metrics_data.get("learning_events", [])),
            "patterns": metrics_data.get("patterns", {}),
            "most_successful_pattern": None,
            "most_recent_pattern": None
        }
        
        # Find most successful pattern
        patterns = metrics_data.get("patterns", {})
        if patterns:
            most_successful_pattern = max(
                patterns.items(),
                key=lambda x: x[1].get("success_count", 0)
            )
            metrics["most_successful_pattern"] = {
                "name": most_successful_pattern[0],
                "data": most_successful_pattern[1]
            }
        
        # Find most recent pattern
        learning_events = metrics_data.get("learning_events", [])
        if learning_events:
            most_recent_event = max(
                learning_events,
                key=lambda x: x.get("timestamp", 0)
            )
            metrics["most_recent_pattern"] = most_recent_event.get("pattern_name")
        
        return metrics
    
    def get_learning_effectiveness(self, query_type: str) -> Dict[str, Any]:
        """
        Get learning effectiveness metrics for a specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Learning effectiveness metrics for the query type
        """
        # Get workflow metrics
        workflow_metrics = self.get_workflow_metrics(query_type)
        
        # Get pattern metrics
        pattern_metrics = self.get_pattern_metrics(query_type)
        
        # Calculate learning effectiveness
        effectiveness = {
            "query_type": query_type,
            "pattern_usage_ratio": workflow_metrics.get("pattern_usage_ratio", 0.0),
            "success_rate_improvement": 0.0,
            "step_count_improvement": 0.0,
            "execution_time_improvement": 0.0,
            "learning_score": 0.0
        }
        
        # Calculate success rate improvement
        success_rate_with_pattern = workflow_metrics.get("success_rate_with_pattern", 0.0)
        success_rate_without_pattern = workflow_metrics.get("success_rate_without_pattern", 0.0)
        
        if success_rate_without_pattern > 0:
            effectiveness["success_rate_improvement"] = (success_rate_with_pattern - success_rate_without_pattern) / success_rate_without_pattern
        
        # Calculate step count improvement
        # This requires more detailed analysis of executions
        workflow_executions = self.metrics.get(f"workflow_execution:{query_type}", {}).get("executions", [])
        
        if workflow_executions:
            # Get step counts for executions with and without patterns
            with_pattern_steps = [e.get("step_count", 0) for e in workflow_executions if e.get("pattern_used")]
            without_pattern_steps = [e.get("step_count", 0) for e in workflow_executions if not e.get("pattern_used")]
            
            # Calculate average step counts
            avg_with_pattern = sum(with_pattern_steps) / len(with_pattern_steps) if with_pattern_steps else 0
            avg_without_pattern = sum(without_pattern_steps) / len(without_pattern_steps) if without_pattern_steps else 0
            
            if avg_without_pattern > 0:
                effectiveness["step_count_improvement"] = (avg_without_pattern - avg_with_pattern) / avg_without_pattern
        
        # Calculate execution time improvement
        # This requires more detailed analysis of executions
        if workflow_executions:
            # Get execution times for executions with and without patterns
            with_pattern_times = [e.get("execution_time", 0) for e in workflow_executions if e.get("pattern_used")]
            without_pattern_times = [e.get("execution_time", 0) for e in workflow_executions if not e.get("pattern_used")]
            
            # Calculate average execution times
            avg_with_pattern = sum(with_pattern_times) / len(with_pattern_times) if with_pattern_times else 0
            avg_without_pattern = sum(without_pattern_times) / len(without_pattern_times) if without_pattern_times else 0
            
            if avg_without_pattern > 0:
                effectiveness["execution_time_improvement"] = (avg_without_pattern - avg_with_pattern) / avg_without_pattern
        
        # Calculate learning score
        # This is a weighted combination of the other metrics
        effectiveness["learning_score"] = (
            0.3 * effectiveness["pattern_usage_ratio"] +
            0.3 * max(0, effectiveness["success_rate_improvement"]) +
            0.2 * max(0, effectiveness["step_count_improvement"]) +
            0.2 * max(0, effectiveness["execution_time_improvement"])
        )
        
        return effectiveness
    
    def get_all_query_types(self) -> List[str]:
        """
        Get all query types with metrics.
        
        Returns:
            List of query types
        """
        query_types = set()
        
        for metric_key in self.metrics:
            if metric_key.startswith("workflow_execution:"):
                query_type = metric_key.split(":", 1)[1]
                query_types.add(query_type)
        
        return list(query_types)
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive learning report.
        
        Returns:
            Learning report
        """
        report = {
            "timestamp": time.time(),
            "query_types": [],
            "overall_metrics": {
                "total_executions": 0,
                "total_patterns": 0,
                "average_pattern_usage_ratio": 0.0,
                "average_success_rate_improvement": 0.0,
                "average_step_count_improvement": 0.0,
                "average_execution_time_improvement": 0.0,
                "average_learning_score": 0.0
            }
        }
        
        # Get all query types
        query_types = self.get_all_query_types()
        
        # Calculate metrics for each query type
        query_type_metrics = []
        
        for query_type in query_types:
            # Get learning effectiveness
            effectiveness = self.get_learning_effectiveness(query_type)
            
            # Get workflow metrics
            workflow_metrics = self.get_workflow_metrics(query_type)
            
            # Get pattern metrics
            pattern_metrics = self.get_pattern_metrics(query_type)
            
            # Combine metrics
            combined_metrics = {
                "query_type": query_type,
                "total_executions": workflow_metrics.get("total_executions", 0),
                "total_patterns": pattern_metrics.get("total_patterns", 0),
                "pattern_usage_ratio": effectiveness.get("pattern_usage_ratio", 0.0),
                "success_rate_improvement": effectiveness.get("success_rate_improvement", 0.0),
                "step_count_improvement": effectiveness.get("step_count_improvement", 0.0),
                "execution_time_improvement": effectiveness.get("execution_time_improvement", 0.0),
                "learning_score": effectiveness.get("learning_score", 0.0),
                "most_successful_pattern": pattern_metrics.get("most_successful_pattern", {}).get("name") if pattern_metrics.get("most_successful_pattern") else None
            }
            
            query_type_metrics.append(combined_metrics)
        
        # Sort query types by learning score
        query_type_metrics.sort(key=lambda x: x.get("learning_score", 0.0), reverse=True)
        
        # Add query type metrics to report
        report["query_types"] = query_type_metrics
        
        # Calculate overall metrics
        if query_type_metrics:
            report["overall_metrics"]["total_executions"] = sum(m.get("total_executions", 0) for m in query_type_metrics)
            report["overall_metrics"]["total_patterns"] = sum(m.get("total_patterns", 0) for m in query_type_metrics)
            report["overall_metrics"]["average_pattern_usage_ratio"] = sum(m.get("pattern_usage_ratio", 0.0) for m in query_type_metrics) / len(query_type_metrics)
            report["overall_metrics"]["average_success_rate_improvement"] = sum(m.get("success_rate_improvement", 0.0) for m in query_type_metrics) / len(query_type_metrics)
            report["overall_metrics"]["average_step_count_improvement"] = sum(m.get("step_count_improvement", 0.0) for m in query_type_metrics) / len(query_type_metrics)
            report["overall_metrics"]["average_execution_time_improvement"] = sum(m.get("execution_time_improvement", 0.0) for m in query_type_metrics) / len(query_type_metrics)
            report["overall_metrics"]["average_learning_score"] = sum(m.get("learning_score", 0.0) for m in query_type_metrics) / len(query_type_metrics)
        
        return report


# Create a singleton instance of the learning metrics tracker
learning_metrics = LearningMetrics(metrics_dir=os.path.join(os.getcwd(), "data", "metrics"))


def get_learning_metrics() -> LearningMetrics:
    """
    Get the learning metrics tracker instance.
    
    Returns:
        Learning metrics tracker instance
    """
    return learning_metrics
