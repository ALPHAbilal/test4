import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMemory:
    """Knowledge store for agents to learn from interactions"""
    
    def __init__(self, db_path="agent_memory.db"):
        self.db_path = db_path
        self._initialize_db_if_needed()
    
    @contextmanager
    def _get_conn(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _initialize_db_if_needed(self):
        """Create tables if they don't exist"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_logs (
                    id INTEGER PRIMARY KEY,
                    agent_id TEXT,
                    action TEXT,
                    inputs TEXT,
                    outputs TEXT,
                    execution_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_strategies (
                    id INTEGER PRIMARY KEY,
                    agent_id TEXT,
                    strategy_name TEXT,
                    strategy_definition TEXT,
                    performance_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY,
                    agent_id TEXT,
                    task_type TEXT,
                    confidence REAL,
                    execution_time REAL,
                    success_rate REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                conn.commit()
                logger.info("Database schema initialized/verified.")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise
    
    def log_agent_action(self, agent_id, action, inputs, outputs, execution_time):
        """Log agent actions with execution time"""
        sql = "INSERT INTO agent_logs (agent_id, action, inputs, outputs, execution_time) VALUES (?, ?, ?, ?, ?)"
        params = (agent_id, action, json.dumps(inputs), json.dumps(outputs), execution_time)
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                logger.debug(f"Logged action '{action}' for agent '{agent_id}'")
        except Exception as e:
            logger.error(f"Error logging agent action: {e}", exc_info=True)
            raise
    
    def update_strategy(self, agent_id, strategy_name, strategy_definition, performance_score):
        """Update strategies based on performance"""
        sql = "INSERT INTO agent_strategies (agent_id, strategy_name, strategy_definition, performance_score) VALUES (?, ?, ?, ?)"
        params = (agent_id, strategy_name, json.dumps(strategy_definition), performance_score)
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating strategy: {e}", exc_info=True)
            raise
    
    def get_best_strategy(self, agent_id):
        """Get the best performing strategy for an agent"""
        sql = "SELECT strategy_definition FROM agent_strategies WHERE agent_id = ? ORDER BY performance_score DESC LIMIT 1"
        params = (agent_id,)
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                result = cursor.fetchone()
                return json.loads(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}", exc_info=True)
            return None
    
    def log_performance(self, agent_id, task_type, confidence, execution_time, success_rate=None):
        """Log performance metrics for self-assessment"""
        sql = "INSERT INTO agent_performance (agent_id, task_type, confidence, execution_time, success_rate) VALUES (?, ?, ?, ?, ?)"
        params = (agent_id, task_type, confidence, execution_time, success_rate)
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging performance: {e}", exc_info=True)
            raise
            
    def get_agent_performance(self, agent_id, task_type=None, limit=5):
        """Get recent performance metrics for an agent"""
        query = "SELECT task_type, confidence, execution_time, success_rate FROM agent_performance WHERE agent_id = ?"
        params = [agent_id]
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting agent performance: {e}", exc_info=True)
            return []