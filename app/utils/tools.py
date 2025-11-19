"""
Tools for RAG, ontology queries, and recommendations.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.logger import logger
from app.utils.exceptions import RetrievalException, RecommendationException
from app.ontology.health_ontology import HealthOntology


class RAGTool:
    """RAG retrieval using FAISS vector store."""
    
    def __init__(self, vector_store_path: str = None, embedding_model: str = None):
        self.vector_store_path = vector_store_path or os.getenv("VECTOR_STORE_PATH", "data/vector_store/")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.top_k = int(os.getenv("TOP_K_RESULTS", "3"))
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = None
        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create with default health documents."""
        try:
            if Path(self.vector_store_path).exists() and Path(self.vector_store_path + "/index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded")
            else:
                self._create_default_vector_store()
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self._create_default_vector_store()
    
    def _create_default_vector_store(self):
        """Create vector store with default health knowledge."""
        default_docs = [
            "Proper hydration is essential for maintaining energy levels. Adults should drink 8-10 glasses of water daily.",
            "Regular sleep of 7-9 hours improves mood, cognitive function, and immune system health.",
            "Exercise for at least 150 minutes per week helps reduce cardiovascular disease risk and improves mental health.",
            "Balanced nutrition with fruits, vegetables, whole grains, and lean proteins supports overall health.",
            "Chronic stress negatively impacts sleep quality, heart health, and mental wellbeing. Stress management is crucial.",
            "Adequate sleep helps with muscle recovery and athletic performance. Sleep deprivation reduces endurance.",
            "Dehydration can cause fatigue, headaches, and reduced mental clarity. Monitor fluid intake during exercise.",
            "High blood pressure is linked to poor diet, lack of exercise, and high stress levels. Lifestyle changes help.",
            "Weight management requires balanced calorie intake and regular physical activity. Sustainable habits matter.",
            "Mental health affects physical health. Regular exercise and good sleep improve mood and reduce anxiety."
        ]
        
        try:
            Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=50
            )
            
            chunks = []
            for doc in default_docs:
                chunks.extend(text_splitter.split_text(doc))
            
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            self.vector_store.save_local(self.vector_store_path)
            logger.info("Default vector store created")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise RetrievalException(f"Failed to create vector store: {e}")
    
    def retrieve(self, query: str, k: int = None) -> List[str]:
        """Retrieve top-k relevant documents."""
        if not self.vector_store:
            raise RetrievalException("Vector store not initialized")
        
        try:
            k = k or self.top_k
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise RetrievalException(f"Failed to retrieve documents: {e}")
    
    def add_documents(self, documents: List[str]):
        """Add new documents to vector store."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=50
            )
            
            chunks = []
            for doc in documents:
                chunks.extend(text_splitter.split_text(doc))
            
            if self.vector_store:
                self.vector_store.add_texts(chunks)
            else:
                self.vector_store = FAISS.from_texts(chunks, self.embeddings)
            
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise RetrievalException(f"Failed to add documents: {e}")


class OntologyTool:
    """Ontology query tool."""
    
    def __init__(self):
        self.ontology = HealthOntology()
    
    def query(self, concepts: List[str]) -> Dict[str, Any]:
        """Query ontology for concept relationships."""
        try:
            return self.ontology.query_ontology(concepts)
        except Exception as e:
            logger.error(f"Ontology query error: {e}")
            return {}


class RecommenderTool:
    """Recommendation tool using embeddings and cosine similarity."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name, device='cpu')
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        self.recommendation_templates = [
            "Increase water intake to 2-3 liters daily to boost energy and reduce fatigue",
            "Aim for 7-9 hours of sleep to improve mood and cognitive function",
            "Add 30 minutes of moderate exercise 5 days per week to enhance cardiovascular health",
            "Include more whole grains and vegetables in your diet for better nutrition",
            "Practice stress-reduction techniques like meditation or deep breathing daily",
            "Monitor heart rate during exercise to stay in optimal training zones",
            "Take rest days between intense workouts for proper muscle recovery",
            "Stay hydrated before, during, and after physical activity",
            "Maintain consistent sleep schedule to regulate circadian rhythm",
            "Balance cardio and strength training for comprehensive fitness"
        ]
    
    def get_personalized_recommendations(
        self,
        user_context: str,
        candidate_recommendations: List[str] = None,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Generate personalized recommendations using embedding similarity."""
        try:
            candidates = candidate_recommendations or self.recommendation_templates
            
            user_embedding = self.model.encode(user_context, convert_to_numpy=True)
            candidate_embeddings = self.model.encode(candidates, convert_to_numpy=True)
            
            similarities = np.dot(candidate_embeddings, user_embedding) / (
                np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(user_embedding)
            )
            
            filtered_indices = np.where(similarities >= self.similarity_threshold)[0]
            
            if len(filtered_indices) == 0:
                top_indices = np.argsort(similarities)[-top_n:][::-1]
            else:
                sorted_filtered = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)
                top_indices = sorted_filtered[:top_n]
            
            # Add diversity: avoid too similar recommendations
            selected_recs = []
            for idx in top_indices:
                rec = candidates[idx]
                score = float(similarities[idx])
                
                # Check diversity
                is_diverse = True
                for existing_rec, _ in selected_recs:
                    similarity = self._compute_similarity(rec, existing_rec)
                    if similarity > 0.85:
                        is_diverse = False
                        break
                
                if is_diverse:
                    selected_recs.append((rec, score))
                
                if len(selected_recs) >= top_n:
                    break
            
            return selected_recs
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            raise RecommendationException(f"Failed to generate recommendations: {e}")
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))