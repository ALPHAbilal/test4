"""
Performance Monitoring Module

This module provides tools for monitoring and analyzing the performance of
the agent workflow, helping to identify bottlenecks and optimize performance.
"""

import os
import time
import json
import logging
import threading
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a performance log file
performance_log_file = os.path.join('logs', f'system_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
perf_handler = logging.FileHandler(performance_log_file)
perf_handler.setLevel(logging.INFO)
perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
perf_handler.setFormatter(perf_formatter)
perf_logger = logging.getLogger('system_performance')
perf_logger.setLevel(logging.INFO)
perf_logger.addHandler(perf_handler)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_cpu_percent: float
    process_memory_percent: float
    process_memory_mb: float
    process_threads: int
    
    def to_csv(self) -> str:
        """Convert to CSV format"""
        return (
            f"{self.timestamp},{self.cpu_percent:.2f},{self.memory_percent:.2f},"
            f"{self.memory_used_mb:.2f},{self.memory_available_mb:.2f},"
            f"{self.disk_io_read_mb:.2f},{self.disk_io_write_mb:.2f},"
            f"{self.network_sent_mb:.2f},{self.network_recv_mb:.2f},"
            f"{self.process_cpu_percent:.2f},{self.process_memory_percent:.2f},"
            f"{self.process_memory_mb:.2f},{self.process_threads}"
        )
    
    @staticmethod
    def csv_header() -> str:
        """Get CSV header"""
        return (
            "timestamp,cpu_percent,memory_percent,memory_used_mb,memory_available_mb,"
            "disk_io_read_mb,disk_io_write_mb,network_sent_mb,network_recv_mb,"
            "process_cpu_percent,process_memory_percent,process_memory_mb,process_threads"
        )

class PerformanceMonitor:
    """
    Monitors system and process performance metrics.
    """
    
    def __init__(self, interval_seconds: int = 5):
        self.interval_seconds = interval_seconds
        self.metrics: List[SystemMetrics] = []
        self.shutdown_event = threading.Event()
        self.monitoring_thread = None
        self.process = psutil.Process()
        
        # Initialize previous IO counters for delta calculations
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_net_io = psutil.net_io_counters()
        self.prev_timestamp = time.time()
        
        # Create CSV file for metrics
        self.csv_file = os.path.join('logs', f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with open(self.csv_file, 'w') as f:
            f.write(f"{SystemMetrics.csv_header()}\n")
    
    def start(self):
        """Start monitoring in a background thread"""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Performance monitoring is already running")
            return
        
        self.shutdown_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started performance monitoring (interval: {self.interval_seconds}s)")
    
    def stop(self):
        """Stop monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Performance monitoring is not running")
            return
        
        self.shutdown_event.set()
        self.monitoring_thread.join(timeout=10)
        logger.info("Stopped performance monitoring")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _monitor_loop(self):
        """Background thread to collect metrics"""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Log metrics
                perf_logger.info(
                    f"SYSTEM_METRICS|CPU:{metrics.cpu_percent:.1f}%|MEM:{metrics.memory_percent:.1f}%|"
                    f"PROC_CPU:{metrics.process_cpu_percent:.1f}%|PROC_MEM:{metrics.process_memory_mb:.1f}MB"
                )
                
                # Write to CSV
                with open(self.csv_file, 'a') as f:
                    f.write(f"{metrics.to_csv()}\n")
                
                # Check for high resource usage
                self._check_resource_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
            
            # Sleep for the interval, but check for shutdown every second
            for _ in range(self.interval_seconds):
                if self.shutdown_event.is_set():
                    return
                time.sleep(1)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect system and process metrics"""
        # Get current timestamp
        current_time = time.time()
        time_delta = current_time - self.prev_timestamp
        
        # System CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk and network IO (calculate deltas)
        current_disk_io = psutil.disk_io_counters()
        current_net_io = psutil.net_io_counters()
        
        disk_read_mb = (current_disk_io.read_bytes - self.prev_disk_io.read_bytes) / (1024 * 1024)
        disk_write_mb = (current_disk_io.write_bytes - self.prev_disk_io.write_bytes) / (1024 * 1024)
        net_sent_mb = (current_net_io.bytes_sent - self.prev_net_io.bytes_sent) / (1024 * 1024)
        net_recv_mb = (current_net_io.bytes_recv - self.prev_net_io.bytes_recv) / (1024 * 1024)
        
        # Normalize to per-second values
        if time_delta > 0:
            disk_read_mb /= time_delta
            disk_write_mb /= time_delta
            net_sent_mb /= time_delta
            net_recv_mb /= time_delta
        
        # Update previous values
        self.prev_disk_io = current_disk_io
        self.prev_net_io = current_net_io
        self.prev_timestamp = current_time
        
        # Process metrics
        try:
            process_cpu_percent = self.process.cpu_percent()
            process_memory_info = self.process.memory_info()
            process_memory_mb = process_memory_info.rss / (1024 * 1024)
            process_memory_percent = self.process.memory_percent()
            process_threads = self.process.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_cpu_percent = 0
            process_memory_mb = 0
            process_memory_percent = 0
            process_threads = 0
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            process_cpu_percent=process_cpu_percent,
            process_memory_percent=process_memory_percent,
            process_memory_mb=process_memory_mb,
            process_threads=process_threads
        )
    
    def _check_resource_alerts(self, metrics: SystemMetrics):
        """Check for high resource usage and log alerts"""
        # CPU alerts
        if metrics.cpu_percent > 90:
            logger.warning(f"HIGH CPU USAGE: System CPU at {metrics.cpu_percent:.1f}%")
        
        if metrics.process_cpu_percent > 80:
            logger.warning(f"HIGH PROCESS CPU USAGE: Process CPU at {metrics.process_cpu_percent:.1f}%")
        
        # Memory alerts
        if metrics.memory_percent > 90:
            logger.warning(f"HIGH MEMORY USAGE: System memory at {metrics.memory_percent:.1f}%")
        
        if metrics.process_memory_percent > 70:
            logger.warning(f"HIGH PROCESS MEMORY USAGE: Process using {metrics.process_memory_percent:.1f}% of system memory")
    
    def _generate_summary_report(self):
        """Generate a summary report of collected metrics"""
        if not self.metrics:
            logger.warning("No metrics collected, cannot generate summary report")
            return
        
        # Calculate summary statistics
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)
        max_cpu = max(m.cpu_percent for m in self.metrics)
        avg_memory = sum(m.memory_percent for m in self.metrics) / len(self.metrics)
        max_memory = max(m.memory_percent for m in self.metrics)
        avg_process_cpu = sum(m.process_cpu_percent for m in self.metrics) / len(self.metrics)
        max_process_cpu = max(m.process_cpu_percent for m in self.metrics)
        avg_process_memory = sum(m.process_memory_mb for m in self.metrics) / len(self.metrics)
        max_process_memory = max(m.process_memory_mb for m in self.metrics)
        
        # Log summary
        summary = {
            "metrics_count": len(self.metrics),
            "duration_seconds": self.metrics[-1].timestamp - self.metrics[0].timestamp,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "avg_memory_percent": avg_memory,
            "max_memory_percent": max_memory,
            "avg_process_cpu_percent": avg_process_cpu,
            "max_process_cpu_percent": max_process_cpu,
            "avg_process_memory_mb": avg_process_memory,
            "max_process_memory_mb": max_process_memory
        }
        
        logger.info(f"Performance monitoring summary: {json.dumps(summary, indent=2)}")
        
        # Save summary to file
        summary_file = os.path.join('logs', f'performance_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Performance summary saved to {summary_file}")
        logger.info(f"Detailed metrics saved to {self.csv_file}")

# Global instance
performance_monitor = PerformanceMonitor()

def start_monitoring(interval_seconds: int = 5):
    """Start performance monitoring"""
    performance_monitor.interval_seconds = interval_seconds
    performance_monitor.start()

def stop_monitoring():
    """Stop performance monitoring"""
    performance_monitor.stop()

# Context manager for monitoring specific operations
class MonitoredOperation:
    """
    Context manager for monitoring specific operations.
    
    Example:
        with MonitoredOperation("process_document") as op:
            result = process_document(...)
            op.set_metadata({"document_size": len(document)})
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if exc_type is not None:
            # Operation failed
            logger.error(
                f"Operation {self.operation_name} failed after {duration_ms:.2f}ms: {exc_val}"
            )
        else:
            # Operation succeeded
            logger.info(
                f"Operation {self.operation_name} completed in {duration_ms:.2f}ms"
            )
        
        # Log with metadata
        metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        perf_logger.info(
            f"OPERATION|{self.operation_name}|{duration_ms:.2f}ms|"
            f"{'ERROR' if exc_type else 'SUCCESS'}|{metadata_str}"
        )
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set metadata for the operation"""
        self.metadata.update(metadata)
