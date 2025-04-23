"""
Workflow Visualization Module

This module provides tools for visualizing the agent workflow, helping to understand
the flow of data and the relationships between different components.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

@dataclass
class WorkflowStep:
    """A step in the agent workflow"""
    step_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    status: str = "started"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "parent_id": self.parent_id,
            "children": self.children,
            "status": self.status,
            "metadata": self.metadata
        }

class WorkflowTracker:
    """
    Tracks and visualizes the agent workflow.
    """
    
    def __init__(self):
        self.steps: Dict[str, WorkflowStep] = {}
        self.root_steps: List[str] = []
        self.current_workflow_id: Optional[str] = None
        self.lock = threading.Lock()
        
        # Create JSON file for workflow
        self.json_file = os.path.join('logs', f'workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(self.json_file, 'w') as f:
            json.dump({"steps": []}, f)
    
    def start_workflow(self, name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new workflow.
        
        Args:
            name: The name of the workflow
            metadata: Additional metadata about the workflow
            
        Returns:
            The ID of the workflow
        """
        with self.lock:
            workflow_id = f"workflow_{int(time.time() * 1000)}"
            
            # Create workflow step
            step = WorkflowStep(
                step_id=workflow_id,
                name=name,
                start_time=time.time(),
                metadata=metadata or {}
            )
            
            # Add to steps
            self.steps[workflow_id] = step
            self.root_steps.append(workflow_id)
            self.current_workflow_id = workflow_id
            
            # Log the workflow start
            logger.info(f"Started workflow: {name} (ID: {workflow_id})")
            
            # Update JSON file
            self._update_json()
            
            return workflow_id
    
    def end_workflow(self, workflow_id: str, status: str = "completed", metadata: Dict[str, Any] = None) -> None:
        """
        End a workflow.
        
        Args:
            workflow_id: The ID of the workflow
            status: The status of the workflow (completed, failed, etc.)
            metadata: Additional metadata about the workflow
        """
        with self.lock:
            if workflow_id not in self.steps:
                logger.warning(f"Workflow {workflow_id} not found")
                return
            
            # Update workflow step
            step = self.steps[workflow_id]
            step.end_time = time.time()
            step.status = status
            
            if metadata:
                step.metadata.update(metadata)
            
            # Log the workflow end
            duration_ms = (step.end_time - step.start_time) * 1000
            logger.info(f"Ended workflow: {step.name} (ID: {workflow_id}) - Status: {status} - Duration: {duration_ms:.2f}ms")
            
            # Update JSON file
            self._update_json()
            
            # If this is the current workflow, clear it
            if self.current_workflow_id == workflow_id:
                self.current_workflow_id = None
    
    def start_step(self, name: str, parent_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new step in the workflow.
        
        Args:
            name: The name of the step
            parent_id: The ID of the parent step (or None for root steps)
            metadata: Additional metadata about the step
            
        Returns:
            The ID of the step
        """
        with self.lock:
            # If parent_id is None, use current workflow
            if parent_id is None:
                parent_id = self.current_workflow_id
            
            # Generate step ID
            step_id = f"step_{int(time.time() * 1000)}_{len(self.steps)}"
            
            # Create step
            step = WorkflowStep(
                step_id=step_id,
                name=name,
                start_time=time.time(),
                parent_id=parent_id,
                metadata=metadata or {}
            )
            
            # Add to steps
            self.steps[step_id] = step
            
            # Add to parent's children if parent exists
            if parent_id and parent_id in self.steps:
                self.steps[parent_id].children.append(step_id)
            elif parent_id:
                logger.warning(f"Parent step {parent_id} not found for step {name}")
            else:
                # No parent, add to root steps
                self.root_steps.append(step_id)
            
            # Log the step start
            parent_info = f" (parent: {parent_id})" if parent_id else ""
            logger.info(f"Started step: {name} (ID: {step_id}){parent_info}")
            
            # Update JSON file
            self._update_json()
            
            return step_id
    
    def end_step(self, step_id: str, status: str = "completed", metadata: Dict[str, Any] = None) -> None:
        """
        End a step in the workflow.
        
        Args:
            step_id: The ID of the step
            status: The status of the step (completed, failed, etc.)
            metadata: Additional metadata about the step
        """
        with self.lock:
            if step_id not in self.steps:
                logger.warning(f"Step {step_id} not found")
                return
            
            # Update step
            step = self.steps[step_id]
            step.end_time = time.time()
            step.status = status
            
            if metadata:
                step.metadata.update(metadata)
            
            # Log the step end
            duration_ms = (step.end_time - step.start_time) * 1000
            logger.info(f"Ended step: {step.name} (ID: {step_id}) - Status: {status} - Duration: {duration_ms:.2f}ms")
            
            # Update JSON file
            self._update_json()
    
    def _update_json(self):
        """Update the JSON file with the current workflow state"""
        try:
            with open(self.json_file, 'w') as f:
                json.dump({
                    "steps": [step.to_dict() for step in self.steps.values()],
                    "root_steps": self.root_steps
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating workflow JSON file: {e}")
    
    def generate_mermaid_diagram(self) -> str:
        """
        Generate a Mermaid.js diagram of the workflow.
        
        Returns:
            A Mermaid.js diagram as a string
        """
        with self.lock:
            # Start with the diagram header
            diagram = ["graph TD;"]
            
            # Add nodes
            for step_id, step in self.steps.items():
                # Format the node label
                duration = ""
                if step.end_time:
                    duration_ms = (step.end_time - step.start_time) * 1000
                    duration = f"<br/>{duration_ms:.0f}ms"
                
                # Set node style based on status
                style = ""
                if step.status == "completed":
                    style = "style {step_id} fill:#d4edda,stroke:#28a745"
                elif step.status == "failed":
                    style = "style {step_id} fill:#f8d7da,stroke:#dc3545"
                elif step.status == "started":
                    style = "style {step_id} fill:#fff3cd,stroke:#ffc107"
                
                # Add the node
                diagram.append(f'    {step_id}["{step.name}{duration}"];')
                if style:
                    diagram.append(f'    {style.format(step_id=step_id)};')
            
            # Add edges
            for step_id, step in self.steps.items():
                if step.parent_id:
                    diagram.append(f'    {step.parent_id} --> {step_id};')
            
            return "\n".join(diagram)
    
    def save_mermaid_diagram(self, filename: str = None) -> str:
        """
        Generate and save a Mermaid.js diagram of the workflow.
        
        Args:
            filename: The filename to save the diagram to (or None for default)
            
        Returns:
            The path to the saved file
        """
        if filename is None:
            filename = os.path.join('logs', f'workflow_diagram_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        
        diagram = self.generate_mermaid_diagram()
        
        with open(filename, 'w') as f:
            f.write("# Workflow Diagram\n\n")
            f.write("```mermaid\n")
            f.write(diagram)
            f.write("\n```\n")
        
        logger.info(f"Workflow diagram saved to {filename}")
        return filename
    
    def generate_timeline(self) -> str:
        """
        Generate a timeline of the workflow as HTML.
        
        Returns:
            An HTML timeline as a string
        """
        with self.lock:
            # Start with the HTML header
            html = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Workflow Timeline</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        .timeline { position: relative; margin: 20px 0; }",
                "        .timeline-item { position: relative; margin-bottom: 10px; padding-left: 50px; }",
                "        .timeline-item::before { content: ''; position: absolute; left: 20px; top: 0; bottom: 0; width: 2px; background: #ccc; }",
                "        .timeline-item::after { content: ''; position: absolute; left: 16px; top: 10px; width: 10px; height: 10px; border-radius: 50%; background: #007bff; }",
                "        .timeline-item.completed::after { background: #28a745; }",
                "        .timeline-item.failed::after { background: #dc3545; }",
                "        .timeline-item.started::after { background: #ffc107; }",
                "        .timeline-content { padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
                "        .timeline-content h3 { margin-top: 0; }",
                "        .timeline-content p { margin: 5px 0; }",
                "        .timeline-content.completed { border-left: 4px solid #28a745; }",
                "        .timeline-content.failed { border-left: 4px solid #dc3545; }",
                "        .timeline-content.started { border-left: 4px solid #ffc107; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>Workflow Timeline</h1>",
                "    <div class=\"timeline\">"
            ]
            
            # Sort steps by start time
            sorted_steps = sorted(self.steps.values(), key=lambda s: s.start_time)
            
            # Add timeline items
            for step in sorted_steps:
                # Format timestamps
                start_time_str = datetime.fromtimestamp(step.start_time).strftime("%H:%M:%S.%f")[:-3]
                end_time_str = datetime.fromtimestamp(step.end_time).strftime("%H:%M:%S.%f")[:-3] if step.end_time else "In Progress"
                
                # Calculate duration
                duration = ""
                if step.end_time:
                    duration_ms = (step.end_time - step.start_time) * 1000
                    duration = f"{duration_ms:.2f}ms"
                
                # Add the timeline item
                html.extend([
                    f"        <div class=\"timeline-item {step.status}\">",
                    f"            <div class=\"timeline-content {step.status}\">",
                    f"                <h3>{step.name}</h3>",
                    f"                <p><strong>ID:</strong> {step.step_id}</p>",
                    f"                <p><strong>Start:</strong> {start_time_str}</p>",
                    f"                <p><strong>End:</strong> {end_time_str}</p>",
                    f"                <p><strong>Duration:</strong> {duration}</p>",
                    f"                <p><strong>Status:</strong> {step.status}</p>"
                ])
                
                # Add parent info if available
                if step.parent_id:
                    parent_name = self.steps[step.parent_id].name if step.parent_id in self.steps else "Unknown"
                    html.append(f"                <p><strong>Parent:</strong> {parent_name} ({step.parent_id})</p>")
                
                # Add metadata if available
                if step.metadata:
                    html.append(f"                <p><strong>Metadata:</strong></p>")
                    html.append(f"                <pre>{json.dumps(step.metadata, indent=2)}</pre>")
                
                # Close the timeline item
                html.extend([
                    "            </div>",
                    "        </div>"
                ])
            
            # Close the HTML
            html.extend([
                "    </div>",
                "</body>",
                "</html>"
            ])
            
            return "\n".join(html)
    
    def save_timeline(self, filename: str = None) -> str:
        """
        Generate and save a timeline of the workflow.
        
        Args:
            filename: The filename to save the timeline to (or None for default)
            
        Returns:
            The path to the saved file
        """
        if filename is None:
            filename = os.path.join('logs', f'workflow_timeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        
        timeline = self.generate_timeline()
        
        with open(filename, 'w') as f:
            f.write(timeline)
        
        logger.info(f"Workflow timeline saved to {filename}")
        return filename

# Global instance
workflow_tracker = WorkflowTracker()

def start_workflow(name: str, metadata: Dict[str, Any] = None) -> str:
    """Start a new workflow"""
    return workflow_tracker.start_workflow(name, metadata)

def end_workflow(workflow_id: str, status: str = "completed", metadata: Dict[str, Any] = None) -> None:
    """End a workflow"""
    workflow_tracker.end_workflow(workflow_id, status, metadata)

def start_step(name: str, parent_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
    """Start a new step in the workflow"""
    return workflow_tracker.start_step(name, parent_id, metadata)

def end_step(step_id: str, status: str = "completed", metadata: Dict[str, Any] = None) -> None:
    """End a step in the workflow"""
    workflow_tracker.end_step(step_id, status, metadata)

def save_workflow_diagram() -> str:
    """Generate and save a diagram of the workflow"""
    return workflow_tracker.save_mermaid_diagram()

def save_workflow_timeline() -> str:
    """Generate and save a timeline of the workflow"""
    return workflow_tracker.save_timeline()

# Context manager for workflow steps
class WorkflowStep:
    """
    Context manager for workflow steps.
    
    Example:
        with WorkflowStep("process_document", parent_id=workflow_id) as step:
            result = process_document(...)
            step.add_metadata({"document_size": len(document)})
    """
    
    def __init__(self, name: str, parent_id: Optional[str] = None, metadata: Dict[str, Any] = None):
        self.name = name
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.step_id = None
    
    def __enter__(self):
        self.step_id = start_step(self.name, self.parent_id, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Step failed
            end_step(self.step_id, "failed", {"error": str(exc_val)})
        else:
            # Step completed
            end_step(self.step_id, "completed")
    
    def add_metadata(self, metadata: Dict[str, Any]):
        """Add metadata to the step"""
        self.metadata.update(metadata)
        # Update the step in the tracker
        if self.step_id:
            workflow_tracker.steps[self.step_id].metadata.update(metadata)
