"""
Learning Report Tool

This module provides tools for generating learning reports.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from agents import function_tool, RunContextWrapper
from core.evaluation import get_learning_metrics

logger = logging.getLogger(__name__)

@function_tool(strict_mode=False)
async def generate_learning_report(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Generate a comprehensive learning report.
    
    Returns:
        A comprehensive learning report
    """
    logger.info("[Tool Call] generate_learning_report")
    
    try:
        # Get the learning metrics
        learning_metrics = get_learning_metrics()
        
        # Generate the report
        report = learning_metrics.generate_learning_report()
        
        return report
    except Exception as e:
        logger.error(f"Error generating learning report: {e}")
        return {
            "error": f"Error generating learning report: {str(e)}"
        }

@function_tool(strict_mode=False)
async def get_query_type_metrics(ctx: RunContextWrapper, query_type: str) -> Dict[str, Any]:
    """
    Get metrics for a specific query type.
    
    Args:
        query_type: Type of query
        
    Returns:
        Metrics for the query type
    """
    logger.info(f"[Tool Call] get_query_type_metrics: query_type='{query_type}'")
    
    try:
        # Get the learning metrics
        learning_metrics = get_learning_metrics()
        
        # Get workflow metrics
        workflow_metrics = learning_metrics.get_workflow_metrics(query_type)
        
        # Get pattern metrics
        pattern_metrics = learning_metrics.get_pattern_metrics(query_type)
        
        # Get learning effectiveness
        effectiveness = learning_metrics.get_learning_effectiveness(query_type)
        
        # Combine metrics
        combined_metrics = {
            "query_type": query_type,
            "workflow_metrics": workflow_metrics,
            "pattern_metrics": pattern_metrics,
            "learning_effectiveness": effectiveness
        }
        
        return combined_metrics
    except Exception as e:
        logger.error(f"Error getting query type metrics: {e}")
        return {
            "error": f"Error getting query type metrics: {str(e)}"
        }

@function_tool(strict_mode=False)
async def get_all_query_types(ctx: RunContextWrapper) -> List[str]:
    """
    Get all query types with metrics.
    
    Returns:
        List of query types
    """
    logger.info("[Tool Call] get_all_query_types")
    
    try:
        # Get the learning metrics
        learning_metrics = get_learning_metrics()
        
        # Get all query types
        query_types = learning_metrics.get_all_query_types()
        
        return query_types
    except Exception as e:
        logger.error(f"Error getting all query types: {e}")
        return {
            "error": f"Error getting all query types: {str(e)}"
        }

@function_tool(strict_mode=False)
async def get_learning_effectiveness_summary(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Get a summary of learning effectiveness across all query types.
    
    Returns:
        Summary of learning effectiveness
    """
    logger.info("[Tool Call] get_learning_effectiveness_summary")
    
    try:
        # Get the learning metrics
        learning_metrics = get_learning_metrics()
        
        # Get all query types
        query_types = learning_metrics.get_all_query_types()
        
        # Calculate effectiveness for each query type
        effectiveness_metrics = []
        
        for query_type in query_types:
            effectiveness = learning_metrics.get_learning_effectiveness(query_type)
            effectiveness_metrics.append(effectiveness)
        
        # Sort by learning score
        effectiveness_metrics.sort(key=lambda x: x.get("learning_score", 0.0), reverse=True)
        
        # Calculate overall metrics
        overall_metrics = {
            "average_pattern_usage_ratio": 0.0,
            "average_success_rate_improvement": 0.0,
            "average_step_count_improvement": 0.0,
            "average_execution_time_improvement": 0.0,
            "average_learning_score": 0.0
        }
        
        if effectiveness_metrics:
            overall_metrics["average_pattern_usage_ratio"] = sum(m.get("pattern_usage_ratio", 0.0) for m in effectiveness_metrics) / len(effectiveness_metrics)
            overall_metrics["average_success_rate_improvement"] = sum(m.get("success_rate_improvement", 0.0) for m in effectiveness_metrics) / len(effectiveness_metrics)
            overall_metrics["average_step_count_improvement"] = sum(m.get("step_count_improvement", 0.0) for m in effectiveness_metrics) / len(effectiveness_metrics)
            overall_metrics["average_execution_time_improvement"] = sum(m.get("execution_time_improvement", 0.0) for m in effectiveness_metrics) / len(effectiveness_metrics)
            overall_metrics["average_learning_score"] = sum(m.get("learning_score", 0.0) for m in effectiveness_metrics) / len(effectiveness_metrics)
        
        # Create summary
        summary = {
            "query_types": effectiveness_metrics,
            "overall_metrics": overall_metrics
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error getting learning effectiveness summary: {e}")
        return {
            "error": f"Error getting learning effectiveness summary: {str(e)}"
        }
