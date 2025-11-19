# Personalized Digital Health Coach

An agentic AI system providing personalized health recommendations using RAG, ontology-based reasoning, and ML-powered recommendation engines.

## 🎯 Overview

This proof-of-concept implements a multi-agent AI health coach that:
- Analyzes user health data (activity, sleep, nutrition, etc.)
- Retrieves trustworthy health information via RAG
- Uses health ontology for relationship-based reasoning
- Generates personalized, explainable recommendations

## 🏗️ Architecture (30-Second Sketch)
```
User Data (Logs/Dataset) 
    ↓
Ingestion (SQLite)
    ↓
Agentic Workflow (LangGraph)
    ├─ Analyzer Agent → Process user data
    ├─ Retriever Agent → Query RAG (FAISS) + Ontology (NetworkX)
    └─ Recommender Agent → Generate personalized recs (SentenceTransformers)
    ↓
Supervisor (Orchestration)
    ↓
Inference (Groq LLM API)
    ↓
API (FastAPI)
    ↓
Frontend (Streamlit)
```

**Key Components:**
- **Data Layer**: SQLite database with synthetic health data
- **Retrieval**: FAISS vector store + NetworkX health ontology
- **Agents**: LangGraph supervisor pattern with ReAct reasoning
- **ML**: SentenceTransformers for embeddings, cosine similarity for recommendations
- **Serving**: Docker Compose, FastAPI, Streamlit

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 16GB RAM (CPU-only, no GPU required)
- Groq API key (free tier)

### Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd personalized-health-coach
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

5. **Initialize data directories**
```bash
mkdir -p data/datasets data/vector_store logs
```

### Running the Application

**Option 1: Local Development**

Start API:
```bash
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

Start Frontend:
```bash
streamlit run app/frontend/app.py
```

Access:
- API: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

**Option 2: Docker Compose**
```bash
cd docker
docker-compose up --build
```

Access:
- API: http://localhost:8000
- Frontend: http://localhost:8501

## 📊 Usage Example

### 1. Log Health Data
```python
import requests

response = requests.post("http://localhost:8000/log_data", json={
    "user_id": "user_001",
    "activity_minutes": 45,
    "sleep_hours": 7.5,
    "water_intake_ml": 2000,
    "steps": 10000,
    "heart_rate": 72,
    "calories": 2100,
    "mood": "Energetic"
})
```

### 2. Get Personalized Suggestions
```python
response = requests.post("http://localhost:8000/get_suggestion", json={
    "user_id": "user_001",
    "query": "How can I improve my energy levels?"
})

suggestions = response.json()["suggestions"]
for s in suggestions:
    print(f"{s['category']}: {s['text']}")
    print(f"Reasoning: {s['reasoning']}\n")
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

Lint code:
```bash
flake8 app/ --max-line-length=127
```

## 🔧 Configuration

Key environment variables in `.env`:
```bash
# LLM
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Retrieval
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
TOP_K_RESULTS=3

# Recommendation
SIMILARITY_THRESHOLD=0.7
```

## 📁 Project Structure
```
personalized-health-coach/
├── app/
│   ├── agents/          # LangGraph agents
│   ├── api/             # FastAPI endpoints
│   ├── frontend/        # Streamlit UI
│   ├── models/          # Pydantic schemas
│   ├── utils/           # Tools (RAG, ontology, recommender)
│   ├── data/            # Database operations
│   └── ontology/        # Health ontology
├── data/                # Local data storage
├── docker/              # Docker configuration
├── tests/               # Unit tests
└── .github/             # CI/CD workflows
```

## 🎓 Key Features

### ReAct Agent Pattern
Each agent follows Reason-Act-Observe:
```python
# REASON: Determine what to do
reasoning = agent._reason(state)

# ACT: Execute action
result = agent._act(state)

# OBSERVE: Extract insights
insights = agent._observe(result)
```

### RAG with FAISS
- CPU-optimized vector store
- 512-token chunks for context management
- Top-k retrieval (k=3) for relevance

### Health Ontology
- NetworkX graph: 17 nodes, 25 relationships
- Concepts: hydration, sleep, exercise, stress, etc.
- Queries: influences, influenced_by, related concepts

### Personalized Recommendations
- SentenceTransformers embeddings
- Cosine similarity matching
- Diversity filtering (avoid >85% similar recs)

## 📈 Performance Metrics

Tracked metrics:
- **Latency**: p95 response time (~2-5s per suggestion request)
- **Cost**: Groq free tier usage (estimated <$0.001/request)
- **Quality**: Cosine similarity scores (threshold: 0.7)

## 🐛 Known Issues & Postmortem

### Issue 1: Token Overflow in Long Contexts
**Problem**: Initial prompts exceeded 4K tokens with full user history.
**Fix**: Implemented context compression (last 7 logs, truncate to 300 chars).

### Issue 2: Rate Limiting on Groq API
**Problem**: Sequential agent calls hit rate limits.
**Fix**: Added tenacity retry with exponential backoff (3 attempts).

### Issue 3: Vector Store Initialization Delay
**Problem**: First-time FAISS creation took 10+ seconds.
**Fix**: Pre-build vector store with default health docs, persist to disk.

## 🚀 Cloud Deployment (Stub)

For production deployment:

**AWS ECS**:
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-uri>
docker build -t health-coach .
docker tag health-coach:latest <ecr-uri>/health-coach:latest
docker push <ecr-uri>/health-coach:latest

# Deploy to ECS
aws ecs update-service --cluster health-coach --service api --force-new-deployment
```

**GCP Cloud Run**:
```bash
gcloud builds submit --tag gcr.io/<project-id>/health-coach
gcloud run deploy health-coach --image gcr.io/<project-id>/health-coach --platform managed
```

## 🔒 Privacy & Ethics

- **Data**: Only synthetic/public datasets (Faker, Kaggle)
- **Storage**: Local SQLite, no external transmission
- **Guardrails**: Planned integration with llama-guard-3-8b
- **Explainability**: All recommendations include reasoning traces

## 📚 References

- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Groq API: https://console.groq.com/
- FAISS: https://github.com/facebookresearch/faiss
- SentenceTransformers: https://www.sbert.net/

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Run tests and linting
4. Submit pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.