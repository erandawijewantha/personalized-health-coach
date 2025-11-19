"""
Streamlit frontend for Personalized Health Coach.
"""

import os
import requests
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Personalized Health Coach",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Personalized Health Coach")
st.markdown("Your AI-powered health companion for personalized wellness recommendations")

# Sidebar for user selection
st.sidebar.header("User Settings")
user_id = st.sidebar.text_input("User ID", value="user_001")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Log Health Data", "💡 Get Suggestions", "📈 View History"])

# Tab 1: Log Health Data
with tab1:
    st.header("Log Your Health Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        activity_minutes = st.number_input("Activity Minutes", min_value=0, max_value=300, value=30)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.5, step=0.5)
        water_intake = st.number_input("Water Intake (ml)", min_value=0, max_value=5000, value=2000, step=100)
    
    with col2:
        steps = st.number_input("Steps", min_value=0, max_value=50000, value=8000, step=500)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
        calories = st.number_input("Calories", min_value=0, max_value=5000, value=2000, step=100)
    
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Stressed", "Tired", "Energetic"])
    
    if st.button("Log Data", type="primary"):
        with st.spinner("Logging data..."):
            try:
                payload = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "activity_minutes": activity_minutes,
                    "sleep_hours": sleep_hours,
                    "water_intake_ml": water_intake,
                    "calories": calories,
                    "heart_rate": heart_rate,
                    "steps": steps,
                    "mood": mood
                }
                
                response = requests.post(f"{API_BASE_URL}/log_data", json=payload)
                
                if response.status_code == 201:
                    st.success(f"✅ Data logged successfully! (Latency: {response.json().get('latency_ms', 0)}ms)")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Tab 2: Get Suggestions
with tab2:
    st.header("Get Personalized Recommendations")
    
    query = st.text_area(
        "What would you like help with?",
        value="Give me personalized health recommendations based on my recent data",
        height=100
    )
    
    if st.button("Get Suggestions", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            try:
                payload = {
                    "user_id": user_id,
                    "query": query
                }
                
                response = requests.post(f"{API_BASE_URL}/get_suggestion", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get("suggestions", [])
                    reasoning = data.get("reasoning", [])
                    
                    if suggestions:
                        st.success(f"✅ Generated {len(suggestions)} recommendations")
                        
                        for idx, suggestion in enumerate(suggestions, 1):
                            with st.expander(f"💡 Recommendation {idx}: {suggestion['category'].upper()}", expanded=True):
                                st.markdown(f"**{suggestion['text']}**")
                                st.markdown(f"*Reasoning:* {suggestion['reasoning']}")
                                st.progress(suggestion['confidence_score'])
                                st.caption(f"Confidence: {suggestion['confidence_score']:.2%} | Source: {suggestion['source']}")
                        
                        # Show reasoning trace
                        with st.expander("🔍 View Agent Reasoning"):
                            for step in reasoning:
                                st.text(step)
                    else:
                        st.warning("No recommendations generated. Try logging more health data.")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Tab 3: View History
with tab3:
    st.header("Health Data History")
    
    if st.button("Load History"):
        with st.spinner("Loading history..."):
            try:
                response = requests.get(f"{API_BASE_URL}/logs/{user_id}?limit=30")
                
                if response.status_code == 200:
                    data = response.json()
                    logs = data.get("logs", [])
                    
                    if logs:
                        st.success(f"✅ Loaded {len(logs)} entries")
                        
                        # Display logs
                        st.dataframe(
                            logs,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Simple charts
                        if len(logs) > 1:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Sleep Trend")
                                sleep_data = [log.get("sleep_hours", 0) for log in reversed(logs) if log.get("sleep_hours")]
                                if sleep_data:
                                    st.line_chart(sleep_data)
                            
                            with col2:
                                st.subheader("Activity Trend")
                                activity_data = [log.get("activity_minutes", 0) for log in reversed(logs) if log.get("activity_minutes")]
                                if activity_data:
                                    st.line_chart(activity_data)
                    else:
                        st.info("No health data logged yet. Start logging in the first tab!")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About**  
Personalized Health Coach uses:
- 🤖 Agentic AI (LangGraph)
- 📚 RAG with FAISS
- 🧠 Health Ontology
- 🎯 ML-based Recommendations
""")