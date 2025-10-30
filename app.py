"""
Brain Dump Sanctuary - Day 2 Complete System
Includes: Streamlit UI + LangGraph Agent + Multi-Perspective Analysis

File structure:
braindump_sanctuary/
├── app.py (THIS FILE - run with: streamlit run app.py)
├── agents.py (LangGraph workflows)
├── braindump_core.py (from Day 1)
└── requirements.txt
"""

# ============== app.py - MAIN STREAMLIT APP ==============
import streamlit as st
import sys
from datetime import datetime
import plotly.graph_objects as go
import numpy as np

# Import Day 1 components
from braindump_core import BrainDumpDB, EmbeddingEngine, ClusterEngine, create_cluster_graph

# Import Day 2 components (defined below)
from agents import QuestionAgent, PerspectiveAgent, SearchAgent

# Page config
st.set_page_config(
    page_title="Brain Dump Sanctuary",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = BrainDumpDB("braindump.db")
if 'embedder' not in st.session_state:
    st.session_state.embedder = EmbeddingEngine()
if 'clusterer' not in st.session_state:
    st.session_state.clusterer = ClusterEngine(min_cluster_size=2)
if 'question_agent' not in st.session_state:
    st.session_state.question_agent = QuestionAgent()
if 'perspective_agent' not in st.session_state:
    st.session_state.perspective_agent = PerspectiveAgent()
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = SearchAgent()

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("Transform stale lists into actionable curiosity")
    
    st.divider()
    
    # Add new brain dump
    st.subheader("💭 New Brain Dump")
    new_dump = st.text_area(
        "What's on your mind?",
        placeholder="Why do dreams feel so real?",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("Add to Sanctuary", type="primary", use_container_width=True):
        if new_dump.strip():
            st.session_state.db.add_dump(new_dump.strip())
            st.success("Added! Refresh clusters below.")
            st.rerun()
        else:
            st.error("Can't add empty thought!")
    
    st.divider()
    
    # Actions
    st.subheader("⚙️ Actions")
    
    if st.button("🔄 Refresh Clusters", use_container_width=True):
        st.rerun()
    
    if st.button("🗑️ Clear All Dumps", use_container_width=True):
        st.session_state.db.conn.execute("DELETE FROM dumps")
        st.session_state.db.conn.commit()
        st.success("All dumps cleared!")
        st.rerun()
    
    # Stats
    st.divider()
    dumps = st.session_state.db.get_all_dumps()
    st.metric("Total Brain Dumps", len(dumps))

# Main content
st.title("🧠 Brain Dump Sanctuary")
st.markdown("*Where racing thoughts become structured curiosity*")

# Get all dumps
dumps = st.session_state.db.get_all_dumps()

if len(dumps) == 0:
    st.info("👋 Welcome! Add your first brain dump using the sidebar.")
    
    with st.expander("🎯 Try these example dumps"):
        examples = [
            "Why do dreams feel so real but fade so quickly?",
            "How does quantum entanglement actually work?",
            "Are LLMs actually understanding or just pattern matching?",
            "What causes the smell of rain on dry ground?",
            "Why does time feel faster as we age?",
        ]
        for ex in examples:
            if st.button(f"Add: {ex}", key=ex):
                st.session_state.db.add_dump(ex)
                st.rerun()
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Cluster Map", 
        "❓ Question Expander", 
        "🔍 Deep Dive (Search)",
        "⚖️ Multi-Perspective"
    ])
    
    # TAB 1: CLUSTER VISUALIZATION
    with tab1:
        st.subheader("Semantic Clustering of Your Brain Dumps")
        
        if len(dumps) < 3:
            st.warning("Add at least 3 brain dumps to see meaningful clusters")
        else:
            with st.spinner("Generating embeddings and clustering..."):
                # Generate embeddings
                texts = [d[1] for d in dumps]
                embeddings = st.session_state.embedder.embed(texts)
                
                # Update embeddings in DB
                for i, (dump_id, _, _) in enumerate(dumps):
                    st.session_state.db.update_embedding(dump_id, embeddings[i])
                
                # Cluster
                clusters, coords_2d = st.session_state.clusterer.fit_predict(embeddings)
                
                # Update clusters in DB
                for i, (dump_id, _, _) in enumerate(dumps):
                    st.session_state.db.update_cluster(dump_id, clusters[i])
                
                # Visualize
                fig = create_cluster_graph(dumps, coords_2d, clusters)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster details
                unique_clusters = sorted(set(clusters) - {-1})
                if unique_clusters:
                    st.subheader("📊 Cluster Details")
                    for cluster_id in unique_clusters:
                        cluster_dumps = [dumps[i] for i in range(len(dumps)) if clusters[i] == cluster_id]
                        with st.expander(f"Cluster {cluster_id} ({len(cluster_dumps)} items)"):
                            for dump_id, text, _ in cluster_dumps:
                                st.markdown(f"- {text}")
    
    # TAB 2: QUESTION GENERATION
    with tab2:
        st.subheader("Expand Vague Ideas into Structured Questions")
        
        # Select a dump
        dump_texts = [f"{d[0]}: {d[1][:60]}..." if len(d[1]) > 60 else f"{d[0]}: {d[1]}" for d in dumps]
        selected = st.selectbox("Choose a brain dump to expand:", dump_texts)
        
        if selected:
            dump_id = int(selected.split(":")[0])
            original_text = next(d[1] for d in dumps if d[0] == dump_id)
            
            st.markdown("**Original thought:**")
            st.info(original_text)
            
            if st.button("🎯 Generate Socratic Questions", type="primary"):
                with st.spinner("Thinking deeply..."):
                    questions = st.session_state.question_agent.generate_questions(original_text)
                    
                    st.markdown("**Exploration questions:**")
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"{i}. {q}")
    
    # TAB 3: SEARCH-ENHANCED DEEP DIVE
    with tab3:
        st.subheader("Web-Enhanced Deep Dive")
        
        dump_texts = [f"{d[0]}: {d[1][:60]}..." if len(d[1]) > 60 else f"{d[0]}: {d[1]}" for d in dumps]
        selected = st.selectbox("Choose a brain dump to research:", dump_texts, key="search_select")
        
        if selected:
            dump_id = int(selected.split(":")[0])
            original_text = next(d[1] for d in dumps if d[0] == dump_id)
            
            st.markdown("**Research topic:**")
            st.info(original_text)
            
            if st.button("🔍 Deep Dive with Web Search", type="primary"):
                with st.spinner("Searching the web and synthesizing..."):
                    result = st.session_state.search_agent.deep_dive(original_text)
                    
                    st.markdown("**Key Findings:**")
                    st.markdown(result['summary'])
                    
                    if result['sources']:
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                st.markdown(f"- [{source['title']}]({source['url']})")
    
    # TAB 4: MULTI-PERSPECTIVE ANALYSIS
    with tab4:
        st.subheader("Controversial Topics: Multiple Perspectives")
        st.markdown("*Best for topics with debate/nuance*")
        
        dump_texts = [f"{d[0]}: {d[1][:60]}..." if len(d[1]) > 60 else f"{d[0]}: {d[1]}" for d in dumps]
        selected = st.selectbox("Choose a controversial topic:", dump_texts, key="perspective_select")
        
        if selected:
            dump_id = int(selected.split(":")[0])
            original_text = next(d[1] for d in dumps if d[0] == dump_id)
            
            st.markdown("**Topic:**")
            st.info(original_text)
            
            if st.button("⚖️ Generate Multiple Perspectives", type="primary"):
                with st.spinner("Analyzing from different angles..."):
                    perspectives = st.session_state.perspective_agent.analyze(original_text)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🔴 Skeptical View")
                        st.markdown(perspectives['skeptical'])
                    
                    with col2:
                        st.markdown("### 🟢 Optimistic View")
                        st.markdown(perspectives['optimistic'])
                    
                    with col3:
                        st.markdown("### 🟡 Nuanced View")
                        st.markdown(perspectives['nuanced'])

# Footer
st.divider()
st.markdown("*Built with LangGraph, Claude, and Streamlit*")