"""
Analyzer Agent: Processes user data and identifies patterns.
Uses ReAct pattern (Reason-Act-Observe).
"""

import os
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import logger
from app.utils.exceptions import AgentException
from app.models.schemas import AgentState


class AnalyzerAgent:
    """Analyzes user health data and identifies patterns."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048"))
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def analyze(self, state: AgentState) -> AgentState:
        """
        Analyze user data using ReAct pattern.
        Reason -> Act (analyze data) -> Observe (extract insights).
        """
        logger.info(f"Analyzer agent started for user {state.user_id}")
        
        try:
            # REASON: Determine what analysis is needed
            reasoning = self._reason(state)
            state.reasoning_trace.append(f"[Analyzer-Reason] {reasoning}")
            
            # ACT: Perform analysis
            analysis_result = self._act(state)
            state.reasoning_trace.append(f"[Analyzer-Act] Analyzed user data")
            
            # OBSERVE: Extract insights
            state.analysis = self._observe(analysis_result)
            state.reasoning_trace.append(f"[Analyzer-Observe] {state.analysis[:100]}...")
            
            state.next_agent = "retriever"
            
            logger.info("Analyzer agent completed")
            return state
            
        except Exception as e:
            logger.error(f"Analyzer agent error: {e}")
            raise AgentException(f"Analysis failed: {e}")
    
    def _reason(self, state: AgentState) -> str:
        """Reason about what analysis is needed."""
        user_data_summary = self._summarize_user_data(state.user_data)
        
        prompt = f"""You are analyzing health data. Reason about what patterns or issues to look for.

User Query: {state.query}
User Data Summary: {user_data_summary}

In 2-3 sentences, state what health patterns you need to identify."""
        
        try:
            reasoning = self._call_llm(prompt)
            return reasoning.strip()
        except Exception as e:
            return f"Analyze general health patterns from available data: {str(e)}"
    
    def _act(self, state: AgentState) -> str:
        """Perform data analysis."""
        user_data_summary = self._summarize_user_data(state.user_data)
        
        prompt = f"""Analyze this health data and identify key patterns, issues, or areas of concern.

User Data: {user_data_summary}
User Query: {state.query}

Provide:
1. Key metrics (averages, trends)
2. Potential health concerns
3. Positive patterns
4. Areas needing improvement

Be concise (under 300 words)."""
        
        analysis = self._call_llm(prompt)
        return analysis
    
    def _observe(self, analysis_result: str) -> str:
        """Extract and structure insights from analysis."""
        return analysis_result.strip()
    
    def _summarize_user_data(self, user_data: Dict[str, Any]) -> str:
        """Create concise summary of user data."""
        if not user_data:
            return "No user data available"
        
        summary_parts = []
        
        if "logs" in user_data and user_data["logs"]:
            logs = user_data["logs"][:7]  # Last 7 entries
            
            metrics = {}
            for log in logs:
                for key in ["activity_minutes", "sleep_hours", "water_intake_ml", "steps", "heart_rate"]:
                    if log.get(key):
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(log[key])
            
            for metric, values in metrics.items():
                avg = sum(values) / len(values)
                summary_parts.append(f"Avg {metric}: {avg:.1f}")
        
        if "profile" in user_data and user_data["profile"]:
            profile = user_data["profile"]
            if profile.get("age"):
                summary_parts.append(f"Age: {profile['age']}")
            if profile.get("health_goals"):
                summary_parts.append(f"Goals: {', '.join(profile['health_goals'][:3])}")
        
        return "; ".join(summary_parts) if summary_parts else "Limited data available"