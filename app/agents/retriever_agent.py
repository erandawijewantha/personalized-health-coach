"""
Retriever Agent: Queries RAG and ontology for relevant health information.
Uses ReAct pattern (Reason-Act-Observe).
"""

import os
from typing import List
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.logger import logger
from app.utils.exceptions import AgentException, RetrievalException
from app.utils.tools import RAGTool, OntologyTool
from app.models.schemas import AgentState


class RetrieverAgent:
    """Retrieves relevant health information from RAG and ontology."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048"))
        )
        self.rag_tool = RAGTool()
        self.ontology_tool = OntologyTool()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def retrieve(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant information using ReAct pattern.
        Reason -> Act (query RAG/ontology) -> Observe (filter results).
        """
        logger.info(f"Retriever agent started for user {state.user_id}")
        
        try:
            # REASON: Determine what information to retrieve
            query_terms = self._reason(state)
            state.reasoning_trace.append(f"[Retriever-Reason] Query terms: {query_terms}")
            
            # ACT: Query RAG and ontology
            rag_results, ontology_results = self._act(query_terms, state)
            state.reasoning_trace.append(f"[Retriever-Act] Retrieved {len(rag_results)} RAG docs, ontology data")
            
            # OBSERVE: Filter and structure results
            state.retrieved_context = self._observe(rag_results, ontology_results, state)
            state.reasoning_trace.append(f"[Retriever-Observe] Filtered to {len(state.retrieved_context)} relevant items")
            
            state.next_agent = "recommender"
            
            logger.info("Retriever agent completed")
            return state
            
        except Exception as e:
            logger.error(f"Retriever agent error: {e}")
            raise AgentException(f"Retrieval failed: {e}")
    
    def _reason(self, state: AgentState) -> List[str]:
        """Reason about what information to retrieve."""
        prompt = f"""Extract 3-5 key health concepts/terms to search for based on this analysis.

Analysis: {state.analysis}
User Query: {state.query}

Return ONLY a comma-separated list of search terms (e.g., "hydration, sleep, exercise")."""
        
        try:
            response = self._call_llm(prompt)
            terms = [term.strip().lower() for term in response.split(",")]
            return terms[:5]
        except Exception as e:
            logger.error(f"Error extracting query terms: {e}")
            return ["hydration", "sleep", "exercise"]
    
    def _act(self, query_terms: List[str], state: AgentState) -> tuple:
        """Execute retrieval from RAG and ontology."""
        # RAG retrieval
        rag_results = []
        try:
            query = " ".join(query_terms)
            rag_results = self.rag_tool.retrieve(query)
        except RetrievalException as e:
            logger.error(f"RAG retrieval error: {e}")
        
        # Ontology query
        ontology_results = {}
        try:
            ontology_results = self.ontology_tool.query(query_terms)
        except Exception as e:
            logger.error(f"Ontology query error: {e}")
        
        return rag_results, ontology_results
    
    def _observe(self, rag_results: List[str], ontology_results: dict, state: AgentState) -> List[str]:
        """Filter and structure retrieved information."""
        context = []
        
        # Add RAG results
        for doc in rag_results:
            context.append(f"[Knowledge] {doc}")
        
        # Add ontology insights
        if ontology_results:
            for concept, relations in ontology_results.items():
                if relations.get("influences"):
                    context.append(f"[Ontology] {concept} affects: {', '.join(relations['influences'][:3])}")
                if relations.get("influenced_by"):
                    context.append(f"[Ontology] {concept} influenced by: {', '.join(relations['influenced_by'][:3])}")
        
        return context[:10]  # Limit context to avoid token overflow