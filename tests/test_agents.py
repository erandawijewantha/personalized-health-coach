"""
Unit tests for agents.
"""

import pytest
from unittest.mock import Mock, patch
from app.models.schemas import AgentState
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.recommender_agent import RecommenderAgent


@pytest.fixture
def sample_state():
    """Sample agent state for testing."""
    return AgentState(
        user_id="test_user",
        query="Give me health recommendations",
        user_data={
            "logs": [
                {"activity_minutes": 30, "sleep_hours": 6.5, "water_intake_ml": 1500},
                {"activity_minutes": 45, "sleep_hours": 7.0, "water_intake_ml": 2000}
            ],
            "profile": {"age": 30, "health_goals": ["weight_loss", "better_sleep"]}
        },
        retrieved_context=[],
        analysis=None,
        recommendations=[],
        reasoning_trace=[],
        next_agent=None
    )


class TestAnalyzerAgent:
    """Tests for AnalyzerAgent."""
    
    @patch('app.agents.analyzer_agent.ChatGroq')
    def test_analyzer_initializes(self, mock_groq):
        """Test analyzer agent initialization."""
        agent = AnalyzerAgent()
        assert agent is not None
        assert agent.llm is not None
    
    @patch('app.agents.analyzer_agent.ChatGroq')
    def test_analyzer_analyze(self, mock_groq, sample_state):
        """Test analyzer analyze method."""
        mock_response = Mock()
        mock_response.content = "User shows low sleep and moderate activity patterns."
        mock_groq.return_value.invoke.return_value = mock_response
        
        agent = AnalyzerAgent()
        result = agent.analyze(sample_state)
        
        assert result.analysis is not None
        assert len(result.reasoning_trace) > 0
        assert result.next_agent == "retriever"
    
    @patch('app.agents.analyzer_agent.ChatGroq')
    def test_analyzer_handles_empty_data(self, mock_groq):
        """Test analyzer with empty user data."""
        mock_response = Mock()
        mock_response.content = "Limited data available for analysis."
        mock_groq.return_value.invoke.return_value = mock_response
        
        empty_state = AgentState(
            user_id="test_user",
            query="Give recommendations",
            user_data={},
            reasoning_trace=[]
        )
        
        agent = AnalyzerAgent()
        result = agent.analyze(empty_state)
        
        assert result.analysis is not None


class TestRetrieverAgent:
    """Tests for RetrieverAgent."""
    
    @patch('app.agents.retriever_agent.RAGTool')
    @patch('app.agents.retriever_agent.OntologyTool')
    @patch('app.agents.retriever_agent.ChatGroq')
    def test_retriever_initializes(self, mock_groq, mock_ontology, mock_rag):
        """Test retriever agent initialization."""
        agent = RetrieverAgent()
        assert agent is not None
        assert agent.rag_tool is not None
        assert agent.ontology_tool is not None
    
    @patch('app.agents.retriever_agent.RAGTool')
    @patch('app.agents.retriever_agent.OntologyTool')
    @patch('app.agents.retriever_agent.ChatGroq')
    def test_retriever_retrieve(self, mock_groq, mock_ontology, mock_rag, sample_state):
        """Test retriever retrieve method."""
        sample_state.analysis = "User needs better sleep and hydration"
        
        mock_response = Mock()
        mock_response.content = "sleep, hydration, exercise"
        mock_groq.return_value.invoke.return_value = mock_response
        
        mock_rag.return_value.retrieve.return_value = [
            "Sleep 7-9 hours for health",
            "Drink 8 glasses of water daily"
        ]
        
        mock_ontology.return_value.query.return_value = {
            "sleep": {"influences": ["energy", "mood"]}
        }
        
        agent = RetrieverAgent()
        result = agent.retrieve(sample_state)
        
        assert len(result.retrieved_context) > 0
        assert result.next_agent == "recommender"


class TestRecommenderAgent:
    """Tests for RecommenderAgent."""
    
    @patch('app.agents.recommender_agent.RecommenderTool')
    @patch('app.agents.recommender_agent.ChatGroq')
    def test_recommender_initializes(self, mock_groq, mock_rec_tool):
        """Test recommender agent initialization."""
        agent = RecommenderAgent()
        assert agent is not None
        assert agent.recommender_tool is not None
    
    @patch('app.agents.recommender_agent.RecommenderTool')
    @patch('app.agents.recommender_agent.ChatGroq')
    def test_recommender_recommend(self, mock_groq, mock_rec_tool, sample_state):
        """Test recommender recommend method."""
        sample_state.analysis = "User needs better sleep"
        sample_state.retrieved_context = ["Sleep is important for health"]
        
        mock_response = Mock()
        mock_response.content = """RECOMMENDATION: Aim for 7-9 hours of sleep nightly
REASONING: Better sleep improves energy and mood
CATEGORY: sleep"""
        mock_groq.return_value.invoke.return_value = mock_response
        
        mock_rec_tool.return_value.get_personalized_recommendations.return_value = [
            ("Aim for 7-9 hours of sleep", 0.85)
        ]
        
        agent = RecommenderAgent()
        result = agent.recommend(sample_state)
        
        assert len(result.recommendations) > 0
        assert result.recommendations[0]["text"] is not None
        assert result.recommendations[0]["reasoning"] is not None
        assert result.next_agent is None
    
    def test_parse_llm_response(self):
        """Test LLM response parsing."""
        agent = RecommenderAgent()
        
        response = """RECOMMENDATION: Drink more water
REASONING: Prevents dehydration
CATEGORY: hydration"""
        
        parsed = agent._parse_llm_response(response)
        
        assert parsed["recommendation"] == "Drink more water"
        assert parsed["reasoning"] == "Prevents dehydration"
        assert parsed["category"] == "hydration"