"""
FastAPI application for health coach service.
"""

import os
import time
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.utils.logger import logger
from app.utils.exceptions import DatabaseException, AgentException
from app.models.schemas import UserLog, UserProfile, Suggestion, SuggestionRequest, SuggestionResponse
from app.data.db import Database
from app.agents.supervisor import SupervisorWorkflow

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app."""
    logger.info("Starting Health Coach API")
    app.state.db = Database(os.getenv("DATABASE_PATH", "data/health_coach.db"))
    app.state.supervisor = SupervisorWorkflow()
    yield
    logger.info("Shutting down Health Coach API")


app = FastAPI(
    title="Personalized Health Coach API",
    description="Agentic AI system for personalized health recommendations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Personalized Health Coach"}


@app.post("/log_data", status_code=status.HTTP_201_CREATED)
async def log_user_data(log: UserLog):
    """Log user health data."""
    try:
        start_time = time.time()
        
        app.state.db.insert_user_log(log)
        
        latency = time.time() - start_time
        logger.info(f"Data logged for user {log.user_id} - Latency: {latency:.3f}s")
        
        return {"status": "success", "message": "Data logged successfully", "latency_ms": int(latency * 1000)}
    
    except DatabaseException as e:
        logger.error(f"Database error logging data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error logging data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/profile", status_code=status.HTTP_201_CREATED)
async def create_or_update_profile(profile: UserProfile):
    """Create or update user profile."""
    try:
        app.state.db.upsert_user_profile(profile)
        return {"status": "success", "message": "Profile updated successfully"}
    
    except DatabaseException as e:
        logger.error(f"Database error updating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile."""
    try:
        profile = app.state.db.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    
    except DatabaseException as e:
        logger.error(f"Database error retrieving profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/get_suggestion", response_model=SuggestionResponse)
async def get_suggestions(request: SuggestionRequest):
    """Get personalized health suggestions using agentic workflow."""
    try:
        start_time = time.time()
        logger.info(f"Suggestion request for user {request.user_id}")
        
        # Gather user data
        user_logs = app.state.db.get_user_logs(request.user_id, limit=30)
        user_profile = app.state.db.get_user_profile(request.user_id)
        
        user_data = {
            "logs": user_logs,
            "profile": user_profile
        }
        
        # Execute agentic workflow
        result = app.state.supervisor.execute(
            user_id=request.user_id,
            query=request.query,
            user_data=user_data
        )
        
        # Convert to Suggestion objects
        suggestions = []
        for rec in result.get("recommendations", []):
            suggestion = Suggestion(
                suggestion_id=rec["suggestion_id"],
                user_id=rec["user_id"],
                timestamp=rec["timestamp"],
                category=rec["category"],
                text=rec["text"],
                reasoning=rec["reasoning"],
                confidence_score=rec["confidence_score"],
                source=rec["source"]
            )
            suggestions.append(suggestion)
            
            # Store in database
            try:
                app.state.db.insert_suggestion(suggestion)
            except Exception as e:
                logger.error(f"Error storing suggestion: {e}")
        
        latency = time.time() - start_time
        logger.info(f"Suggestions generated for user {request.user_id} - Count: {len(suggestions)} - Latency: {latency:.3f}s")
        
        return SuggestionResponse(
            suggestions=suggestions,
            reasoning=result.get("reasoning_trace", [])
        )
    
    except AgentException as e:
        logger.error(f"Agent error generating suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/logs/{user_id}")
async def get_user_logs(user_id: str, limit: int = 30):
    """Get user health logs."""
    try:
        logs = app.state.db.get_user_logs(user_id, limit=limit)
        return {"logs": logs, "count": len(logs)}
    
    except DatabaseException as e:
        logger.error(f"Database error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)