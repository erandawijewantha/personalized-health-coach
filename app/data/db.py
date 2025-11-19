"""
Database operations for SQLite.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.utils.logger import logger
from app.utils.exceptions import DatabaseException
from app.models.schemas import UserLog, UserProfile, Suggestion


class Database:
    """SQLite database handler."""
    
    def __init__(self, db_path: str = "data/health_coach.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        activity_minutes INTEGER,
                        sleep_hours REAL,
                        water_intake_ml INTEGER,
                        calories INTEGER,
                        heart_rate INTEGER,
                        steps INTEGER,
                        mood TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        age INTEGER,
                        weight_kg REAL,
                        height_cm REAL,
                        health_goals TEXT,
                        medical_conditions TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS suggestions (
                        suggestion_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        category TEXT,
                        text TEXT,
                        reasoning TEXT,
                        confidence_score REAL,
                        source TEXT
                    )
                """)
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise DatabaseException(f"Failed to initialize database: {e}")
    
    def insert_user_log(self, log: UserLog) -> bool:
        """Insert user health log."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_logs (
                        user_id, timestamp, activity_minutes, sleep_hours,
                        water_intake_ml, calories, heart_rate, steps, mood
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log.user_id, log.timestamp.isoformat(),
                    log.activity_minutes, log.sleep_hours, log.water_intake_ml,
                    log.calories, log.heart_rate, log.steps, log.mood
                ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting user log: {e}")
            raise DatabaseException(f"Failed to insert log: {e}")
    
    def get_user_logs(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieve user logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_logs 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (user_id, limit))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user logs: {e}")
            raise DatabaseException(f"Failed to retrieve logs: {e}")
    
    def upsert_user_profile(self, profile: UserProfile) -> bool:
        """Insert or update user profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_profiles (
                        user_id, age, weight_kg, height_cm, health_goals, medical_conditions
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id, profile.age, profile.weight_kg, profile.height_cm,
                    json.dumps(profile.health_goals), json.dumps(profile.medical_conditions)
                ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error upserting user profile: {e}")
            raise DatabaseException(f"Failed to upsert profile: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if row:
                    profile = dict(row)
                    profile['health_goals'] = json.loads(profile['health_goals'])
                    profile['medical_conditions'] = json.loads(profile['medical_conditions'])
                    return profile
                return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user profile: {e}")
            raise DatabaseException(f"Failed to retrieve profile: {e}")
    
    def insert_suggestion(self, suggestion: Suggestion) -> bool:
        """Insert suggestion."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO suggestions (
                        suggestion_id, user_id, timestamp, category, text,
                        reasoning, confidence_score, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    suggestion.suggestion_id, suggestion.user_id,
                    suggestion.timestamp.isoformat(), suggestion.category,
                    suggestion.text, suggestion.reasoning,
                    suggestion.confidence_score, suggestion.source
                ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting suggestion: {e}")
            raise DatabaseException(f"Failed to insert suggestion: {e}")