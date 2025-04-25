"""
Test Evaluation Module

This module contains tests for the evaluation module.
"""

import unittest
import os
import json
import tempfile
import shutil
from core.evaluation import LearningMetrics

class TestLearningMetrics(unittest.TestCase):
    """Test the LearningMetrics class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()
        self.metrics = LearningMetrics(metrics_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_record_workflow_execution(self):
        """Test recording a workflow execution."""
        # Record a workflow execution
        self.metrics.record_workflow_execution(
            query_type="test_query",
            pattern_used=True,
            pattern_name="test_pattern",
            step_count=5,
            success=True,
            execution_time=1.5,
            session_id="test_session"
        )
        
        # Check that the metrics were recorded
        workflow_metrics = self.metrics.get_workflow_metrics("test_query")
        self.assertEqual(workflow_metrics["total_executions"], 1)
        self.assertEqual(workflow_metrics["pattern_usage_ratio"], 1.0)
        self.assertEqual(workflow_metrics["success_rate_with_pattern"], 1.0)
        self.assertEqual(workflow_metrics["average_step_count"], 5.0)
        self.assertEqual(workflow_metrics["average_execution_time"], 1.5)
    
    def test_record_pattern_learning(self):
        """Test recording pattern learning."""
        # Record pattern learning
        self.metrics.record_pattern_learning(
            query_type="test_query",
            pattern_name="test_pattern",
            pattern_steps=["step1", "step2", "step3"],
            success_count=2,
            session_id="test_session"
        )
        
        # Check that the metrics were recorded
        pattern_metrics = self.metrics.get_pattern_metrics("test_query")
        self.assertEqual(pattern_metrics["total_patterns"], 1)
        self.assertEqual(pattern_metrics["total_learning_events"], 1)
        self.assertEqual(pattern_metrics["most_successful_pattern"]["name"], "test_pattern")
        self.assertEqual(pattern_metrics["most_successful_pattern"]["data"]["success_count"], 2)
        self.assertEqual(pattern_metrics["most_successful_pattern"]["data"]["steps"], ["step1", "step2", "step3"])
    
    def test_get_learning_effectiveness(self):
        """Test getting learning effectiveness."""
        # Record workflow executions
        self.metrics.record_workflow_execution(
            query_type="test_query",
            pattern_used=True,
            pattern_name="test_pattern",
            step_count=5,
            success=True,
            execution_time=1.5,
            session_id="test_session"
        )
        
        self.metrics.record_workflow_execution(
            query_type="test_query",
            pattern_used=False,
            pattern_name=None,
            step_count=8,
            success=False,
            execution_time=2.5,
            session_id="test_session"
        )
        
        # Record pattern learning
        self.metrics.record_pattern_learning(
            query_type="test_query",
            pattern_name="test_pattern",
            pattern_steps=["step1", "step2", "step3"],
            success_count=2,
            session_id="test_session"
        )
        
        # Check learning effectiveness
        effectiveness = self.metrics.get_learning_effectiveness("test_query")
        self.assertEqual(effectiveness["query_type"], "test_query")
        self.assertEqual(effectiveness["pattern_usage_ratio"], 0.5)
        self.assertGreater(effectiveness["success_rate_improvement"], 0)
        self.assertGreater(effectiveness["step_count_improvement"], 0)
        self.assertGreater(effectiveness["execution_time_improvement"], 0)
        self.assertGreater(effectiveness["learning_score"], 0)
    
    def test_generate_learning_report(self):
        """Test generating a learning report."""
        # Record workflow executions for multiple query types
        self.metrics.record_workflow_execution(
            query_type="query_type_1",
            pattern_used=True,
            pattern_name="pattern_1",
            step_count=5,
            success=True,
            execution_time=1.5,
            session_id="test_session"
        )
        
        self.metrics.record_workflow_execution(
            query_type="query_type_2",
            pattern_used=False,
            pattern_name=None,
            step_count=8,
            success=False,
            execution_time=2.5,
            session_id="test_session"
        )
        
        # Record pattern learning
        self.metrics.record_pattern_learning(
            query_type="query_type_1",
            pattern_name="pattern_1",
            pattern_steps=["step1", "step2", "step3"],
            success_count=2,
            session_id="test_session"
        )
        
        # Generate report
        report = self.metrics.generate_learning_report()
        
        # Check report
        self.assertIn("query_types", report)
        self.assertIn("overall_metrics", report)
        self.assertEqual(len(report["query_types"]), 2)
        
        # Check that query types are sorted by learning score
        self.assertEqual(report["query_types"][0]["query_type"], "query_type_1")
        self.assertEqual(report["query_types"][1]["query_type"], "query_type_2")
    
    def test_persistence(self):
        """Test that metrics are persisted to disk."""
        # Record a workflow execution
        self.metrics.record_workflow_execution(
            query_type="test_query",
            pattern_used=True,
            pattern_name="test_pattern",
            step_count=5,
            success=True,
            execution_time=1.5,
            session_id="test_session"
        )
        
        # Check that the metrics file was created
        metrics_file = os.path.join(self.temp_dir, "workflow_execution_test_query.json")
        self.assertTrue(os.path.exists(metrics_file))
        
        # Check that the metrics file contains the expected data
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        
        self.assertIn("executions", metrics_data)
        self.assertEqual(len(metrics_data["executions"]), 1)
        self.assertEqual(metrics_data["executions"][0]["pattern_name"], "test_pattern")
        self.assertEqual(metrics_data["executions"][0]["step_count"], 5)
        self.assertEqual(metrics_data["executions"][0]["success"], True)
        
        # Create a new metrics instance and check that it loads the persisted metrics
        new_metrics = LearningMetrics(metrics_dir=self.temp_dir)
        workflow_metrics = new_metrics.get_workflow_metrics("test_query")
        self.assertEqual(workflow_metrics["total_executions"], 1)
        self.assertEqual(workflow_metrics["pattern_usage_ratio"], 1.0)
        self.assertEqual(workflow_metrics["success_rate_with_pattern"], 1.0)
        self.assertEqual(workflow_metrics["average_step_count"], 5.0)
        self.assertEqual(workflow_metrics["average_execution_time"], 1.5)


if __name__ == "__main__":
    unittest.main()
