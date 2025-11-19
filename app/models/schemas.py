"""
Pydantic models for data validation and API schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserLog(BaseModel):
    """User health data log entry."""
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    activity_minutes: Optional[int] = None
    sleep_hours: Optional[float] = None
    water_intake_ml: Optional[int] = None
    calories: Optional[int] = None
    heart_rate: Optional[int] = None
    steps: Optional[int] = None
    mood: Optional[str] = None


class UserProfile(BaseModel):
    """User profile with health metrics."""
    user_id: str
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    health_goals: Optional[List[str]] = []
    medical_conditions: Optional[List[str]] = []


class Suggestion(BaseModel):
    """Health suggestion/recommendation."""
    suggestion_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    category: str
    text: str
    reasoning: str
    confidence_score: float
    source: str


class AgentState(BaseModel):
    """State for LangGraph agent workflow."""
    user_id: str
    query: str
    user_data: Optional[Dict[str, Any]] = {}
    retrieved_context: Optional[List[str]] = []
    analysis: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = []
    reasoning_trace: List[str] = []
    next_agent: Optional[str] = None


class SuggestionRequest(BaseModel):
    """Request for getting suggestions."""
    user_id: str
    query: Optional[str] = "Give me personalized health recommendations"


class SuggestionResponse(BaseModel):
    """Response with suggestions."""
    suggestions: List[Suggestion]
    reasoning: List[str]