"""
Custom exceptions for the health coach application.
"""


class HealthCoachException(Exception):
    """Base exception for all health coach errors."""
    pass


class DatabaseException(HealthCoachException):
    """Database operation errors."""
    pass


class AgentException(HealthCoachException):
    """Agent execution errors."""
    pass


class RetrievalException(HealthCoachException):
    """RAG/ontology retrieval errors."""
    pass


class LLMException(HealthCoachException):
    """LLM API call errors."""
    pass


class RecommendationException(HealthCoachException):
    """Recommendation generation errors."""
    pass


class ValidationException(HealthCoachException):
    """Data validation errors."""
    pass