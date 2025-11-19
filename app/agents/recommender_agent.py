"""
Recommender Agent: Generates personalized health recommendations.
Uses ReAct pattern (Reason-Act-Observe).
"""

import os
import uuid
from typing import List, Dict, Any
from datetime import datetime
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import logger
from app.utils.exceptions import AgentException, RecommendationException
from app.utils.tools import RecommenderTool
from app.models.schemas import AgentState


class RecommenderAgent:
    """Generates personalized health recommendations."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048"))
        )
        self.recommender_tool = RecommenderTool()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def recommend(self, state: AgentState) -> AgentState:
        """
        Generate recommendations using ReAct pattern.
        Reason -> Act (generate recs) -> Observe (validate/format).
        """
        logger.info(f"Recommender agent started for user {state.user_id}")
        
        try:
            # REASON: Determine recommendation strategy
            strategy = self._reason(state)
            state.reasoning_trace.append(f"[Recommender-Reason] {strategy}")
            
            # ACT: Generate recommendations
            raw_recommendations = self._act(state)
            state.reasoning_trace.append(f"[Recommender-Act] Generated {len(raw_recommendations)} recommendations")
            
            # OBSERVE: Validate and format
            state.recommendations = self._observe(raw_recommendations, state)
            state.reasoning_trace.append(f"[Recommender-Observe] Finalized {len(state.recommendations)} recommendations")
            
            state.next_agent = None  # End of workflow
            
            logger.info("Recommender agent completed")
            return state
            
        except Exception as e:
            logger.error(f"Recommender agent error: {e}")
            raise AgentException(f"Recommendation generation failed: {e}")
    
    def _reason(self, state: AgentState) -> str:
        """Reason about recommendation strategy."""
        context_summary = "\n".join(state.retrieved_context[:5])
        
        prompt = f"""Based on this health analysis and context, what recommendation strategy should be used?

Analysis: {state.analysis[:300]}
Context: {context_summary[:300]}

In 2 sentences, state the recommendation approach (e.g., focus on hydration and sleep)."""
        
        try:
            strategy = self._call_llm(prompt)
            return strategy.strip()
        except Exception as e:
            return f"Provide balanced recommendations across key health areas: {str(e)}"
    
    def _act(self, state: AgentState) -> List[Dict[str, Any]]:
        """Generate personalized recommendations."""
        # Build user context for embedding-based matching
        user_context = self._build_user_context(state)
        
        # Get embedding-based recommendations
        try:
            base_recommendations = self.recommender_tool.get_personalized_recommendations(
                user_context=user_context,
                top_n=5
            )
        except RecommendationException as e:
            logger.error(f"Recommender tool error: {e}")
            base_recommendations = []
        
        # Enhance with LLM reasoning
        recommendations = []
        context_str = "\n".join(state.retrieved_context[:5])
        
        for rec_text, score in base_recommendations[:3]:
            prompt = f"""Personalize this health recommendation for the user.

Base Recommendation: {rec_text}
User Analysis: {state.analysis[:400]}
Health Context: {context_str[:400]}

Provide:
1. Personalized recommendation (1-2 sentences)
2. Reasoning (why this helps, 1-2 sentences)
3. Category (one of: hydration, sleep, exercise, nutrition, stress)

Format:
RECOMMENDATION: [text]
REASONING: [text]
CATEGORY: [category]"""
            
            try:
                response = self._call_llm(prompt)
                parsed = self._parse_llm_response(response)
                
                recommendations.append({
                    "text": parsed.get("recommendation", rec_text),
                    "reasoning": parsed.get("reasoning", "Based on your health patterns"),
                    "category": parsed.get("category", "general"),
                    "confidence_score": score,
                    "source": "embedding+llm"
                })
            except Exception as e:
                logger.error(f"Error enhancing recommendation: {e}")
                recommendations.append({
                    "text": rec_text,
                    "reasoning": "Based on health best practices",
                    "category": "general",
                    "confidence_score": score,
                    "source": "embedding"
                })
        
        return recommendations
    
    def _observe(self, raw_recommendations: List[Dict[str, Any]], state: AgentState) -> List[Dict[str, Any]]:
        """Validate and format final recommendations."""
        final_recommendations = []
        
        for rec in raw_recommendations:
            # Add metadata
            rec["suggestion_id"] = str(uuid.uuid4())
            rec["user_id"] = state.user_id
            rec["timestamp"] = datetime.now().isoformat()
            
            # Ensure required fields
            if not rec.get("text"):
                continue
            if not rec.get("reasoning"):
                rec["reasoning"] = "Based on your health data"
            if not rec.get("category"):
                rec["category"] = "general"
            if not rec.get("confidence_score"):
                rec["confidence_score"] = 0.7
            if not rec.get("source"):
                rec["source"] = "system"
            
            final_recommendations.append(rec)
        
        return final_recommendations
    
    def _build_user_context(self, state: AgentState) -> str:
        """Build context string for embedding-based matching."""
        context_parts = [state.query, state.analysis[:300]]
        
        if state.retrieved_context:
            context_parts.extend(state.retrieved_context[:3])
        
        return " ".join(context_parts)
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse structured LLM response."""
        parsed = {}
        
        lines = response.strip().split("\n")
        for line in lines:
            if "RECOMMENDATION:" in line:
                parsed["recommendation"] = line.split("RECOMMENDATION:", 1)[1].strip()
            elif "REASONING:" in line:
                parsed["reasoning"] = line.split("REASONING:", 1)[1].strip()
            elif "CATEGORY:" in line:
                parsed["category"] = line.split("CATEGORY:", 1)[1].strip().lower()
        
        return parsed