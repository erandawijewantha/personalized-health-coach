"""
Supervisor Agent: Orchestrates workflow using LangGraph.
Routes between Analyzer, Retriever, and Recommender agents.
"""

import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from app.utils.logger import logger
from app.utils.exceptions import AgentException
from app.models.schemas import AgentState
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.recommender_agent import RecommenderAgent


class SupervisorWorkflow:
    """Supervisor coordinating health coach agent workflow."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7"))
        )
        
        self.analyzer = AnalyzerAgent()
        self.retriever = RetrieverAgent()
        self.recommender = RecommenderAgent()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with supervisor pattern."""
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("recommender", self._recommender_node)
        
        # Set entry point
        workflow.set_entry_point("analyzer")
        
        # Add edges based on routing logic
        workflow.add_conditional_edges(
            "analyzer",
            self._route_from_analyzer,
            {
                "retriever": "retriever",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "retriever",
            self._route_from_retriever,
            {
                "recommender": "recommender",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "recommender",
            self._route_from_recommender,
            {
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _analyzer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analyzer agent."""
        logger.info("Supervisor: Routing to Analyzer")
        agent_state = AgentState(**state)
        result = self.analyzer.analyze(agent_state)
        return result.model_dump()
    
    def _retriever_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retriever agent."""
        logger.info("Supervisor: Routing to Retriever")
        agent_state = AgentState(**state)
        result = self.retriever.retrieve(agent_state)
        return result.model_dump()
    
    def _recommender_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recommender agent."""
        logger.info("Supervisor: Routing to Recommender")
        agent_state = AgentState(**state)
        result = self.recommender.recommend(agent_state)
        return result.model_dump()
    
    def _route_from_analyzer(self, state: Dict[str, Any]) -> str:
        """Route after analyzer completes."""
        next_agent = state.get("next_agent")
        if next_agent == "retriever":
            return "retriever"
        return "end"
    
    def _route_from_retriever(self, state: Dict[str, Any]) -> str:
        """Route after retriever completes."""
        next_agent = state.get("next_agent")
        if next_agent == "recommender":
            return "recommender"
        return "end"
    
    def _route_from_recommender(self, state: Dict[str, Any]) -> str:
        """Route after recommender completes (always end)."""
        return "end"
    
    def execute(self, user_id: str, query: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full workflow and return results."""
        logger.info(f"Supervisor: Starting workflow for user {user_id}")
        
        try:
            initial_state = {
                "user_id": user_id,
                "query": query,
                "user_data": user_data,
                "retrieved_context": [],
                "analysis": None,
                "recommendations": [],
                "reasoning_trace": [],
                "next_agent": None
            }
            
            result = self.workflow.invoke(initial_state)
            
            logger.info(f"Supervisor: Workflow completed with {len(result.get('recommendations', []))} recommendations")
            
            return result
            
        except Exception as e:
            logger.error(f"Supervisor workflow error: {e}")
            raise AgentException(f"Workflow execution failed: {e}")