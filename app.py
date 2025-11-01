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
import pandas as pd

# Import Day 1 components
from braindump_core import BrainDumpDB, EmbeddingEngine, ClusterEngine, create_knowledge_graph

# Import Day 2 components
from agents import QuestionAgent, SearchAgent

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
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = SearchAgent()
if 'question_agent' not in st.session_state:
    st.session_state.question_agent = QuestionAgent()

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("Transform stale lists into actionable curiosity")
    
    st.divider()
    
    # Tab Switcher
    tab_selection = st.radio(
        "Navigate",
        ["Home", "Feed"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Add new brain dump
    st.subheader("💭 New Brain Dump")
    
    # Use a unique key that changes when we clear
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    new_dump = st.text_area(
        "What's on your mind?",
        placeholder="Why do dreams feel so real?",
        height=100,
        label_visibility="collapsed",
        key=f"dump_input_{st.session_state.input_key}"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        add_clicked = st.button("Add to Sanctuary", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True)
    
    if add_clicked:
        if new_dump.strip():
            if not st.session_state.get('adding', False):
                st.session_state.adding = True
                st.session_state.db.add_dump(new_dump.strip())
                st.session_state.input_key += 1
                st.success("Added! ✨")
                st.rerun()
            else:
                st.session_state.adding = False
        else:
            st.error("Can't add empty thought!")
    
    if clear_clicked:
        if not st.session_state.get('clearing_input', False):
            st.session_state.clearing_input = True
            st.session_state.input_key += 1
            st.rerun()
        else:
            st.session_state.clearing_input = False
    
    st.divider()
    
    # Actions
    st.subheader("⚙️ Actions")
    
    if st.button("🔄 Refresh Clusters", use_container_width=True):
        st.info("Clusters will refresh on the Home tab")
    
    if st.button("🗑️ Clear All Dumps", use_container_width=True):
        if not st.session_state.get('clearing', False):
            st.session_state.clearing = True
            st.session_state.db.conn.execute("DELETE FROM dumps")
            st.session_state.db.conn.execute("DELETE FROM cluster_labels")
            st.session_state.db.conn.execute("DELETE FROM sqlite_sequence WHERE name='dumps'")
            st.session_state.db.conn.commit()
            st.session_state.input_key = 0
            st.success("All dumps cleared!")
            st.rerun()
        else:
            st.session_state.clearing = False
    
    # Stats
    st.divider()
    dumps = st.session_state.db.get_all_dumps()
    st.metric("Total Brain Dumps", len(dumps))

# ============== HELPER FUNCTIONS ==============

def render_brain_dump_table(dumps, cluster_labels_map):
    """Render brain dumps as a table with cluster labels and timestamps"""
    if not dumps:
        st.info("No brain dumps yet.")
        return
    
    # Prepare data for table
    table_data = []
    for dump_id, text, cluster_id in reversed(dumps):  # Most recent first
        cluster_label = "Unclustered"
        if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
            cluster_label = cluster_labels_map[cluster_id]['label']
        
        # Get created_at from DB
        cursor = st.session_state.db.conn.execute(
            "SELECT created_at FROM dumps WHERE id = ?",
            (dump_id,)
        )
        created_at = cursor.fetchone()
        created_at_str = created_at[0] if created_at else "Unknown"
        
        table_data.append({
            "Brain Dump": text,
            "Cluster Label": cluster_label,
            "Created At": created_at_str
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_feed_card(dump_id, text, cluster_id, cluster_labels_map):
    """Render a single blog-style card for a brain dump with agent outputs"""
    with st.container(border=True):
        # Title
        st.markdown(f"### 💭 {text}")
        
        # Cluster label badge
        if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
            cluster_label = cluster_labels_map[cluster_id]['label']
            st.markdown(f"🏷️ **{cluster_label}**")
        
        st.divider()
        
        # Summary from Search Agent
        with st.spinner("Generating summary..."):
            result = st.session_state.search_agent.deep_dive(text)
            st.markdown("**Summary:**")
            st.write(result['summary'])
        
        st.divider()
        
        # Questions for Reflection from Question Agent
        with st.spinner("Generating reflection questions..."):
            questions = st.session_state.question_agent.generate_questions(text)
            st.markdown("**Questions for Reflection:**")
            for i, q in enumerate(questions, 1):
                st.markdown(f"{i}. {q}")


def render_home():
    """Render the Home tab with cluster map, text input, and brain dump table"""
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("*Where racing thoughts become structured curiosity*")
    
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
        # Cluster Map Section
        st.subheader("🗺️ Semantic Cluster Map")
        
        if len(dumps) < 3:
            st.warning("⚠️ Add at least 3 brain dumps to see meaningful clusters")
        else:
            try:
                col1, col2 = st.columns([4, 1])
                with col2:
                    force_refresh = st.button("🔄 Refresh", help="Recalculate embeddings and clusters")
                
                # Check if embeddings already exist for all dumps
                existing_embeddings = st.session_state.db.get_embeddings()
                existing_ids = {emb[0] for emb in existing_embeddings}
                all_dump_ids = {d[0] for d in dumps}
                
                # Check if cluster labels already exist
                existing_labels = st.session_state.db.get_all_cluster_labels()
                
                # Only recalculate if we have new dumps without embeddings OR force refresh
                need_recalculation = force_refresh or not (existing_ids >= all_dump_ids)
                
                if need_recalculation:
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
                        
                        # Auto-generate cluster labels for new clusters
                        cluster_labels_dict = {}
                        unique_clusters = set(clusters) - {-1}
                        
                        if unique_clusters:
                            with st.spinner("Generating cluster labels with Gemini..."):
                                for cluster_id in unique_clusters:
                                    if cluster_id not in existing_labels:
                                        cluster_dumps = [dumps[i][1] for i in range(len(dumps)) if clusters[i] == cluster_id]
                                        label = st.session_state.clusterer.generate_cluster_label(cluster_dumps)
                                        cluster_labels_dict[cluster_id] = label
                                        st.session_state.db.save_cluster_label(cluster_id, label)
                                    else:
                                        cluster_labels_dict[cluster_id] = existing_labels[cluster_id]['label']
                else:
                    # Use existing embeddings and clusters
                    st.info("📦 Using cached embeddings and clusters")
                    
                    # Load existing embeddings
                    embeddings_list = []
                    for dump_id, _, _ in dumps:
                        matching_emb = next((emb[1] for emb in existing_embeddings if emb[0] == dump_id), None)
                        if matching_emb is not None:
                            embeddings_list.append(matching_emb)
                    
                    embeddings = np.array(embeddings_list)
                    
                    # Get clusters from database
                    clusters = np.array([d[2] if d[2] is not None else -1 for d in dumps])
                    
                    # Get 2D coordinates for visualization
                    coords_2d = st.session_state.clusterer.reducer.fit_transform(embeddings)
                    
                    # Load existing cluster labels
                    cluster_labels_dict = {k: v['label'] for k, v in existing_labels.items()}
                    unique_clusters = set(clusters) - {-1}
                
                # Visualize with labels
                fig = create_knowledge_graph(dumps, coords_2d, clusters, cluster_labels_dict, embeddings)
                st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
                if st.checkbox("Show technical details"):
                    st.exception(e)
        
        st.divider()
        
        # Brain Dump Table Section
        st.subheader("📋 All Brain Dumps")
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        render_brain_dump_table(dumps, cluster_labels_map)


def render_feed():
    """Render the Feed tab with blog-style cards for 5 most recent brain dumps"""
    st.title("📰 Feed")
    st.markdown("*Agent-generated insights for your most recent thoughts*")
    
    dumps = st.session_state.db.get_all_dumps()
    
    if len(dumps) == 0:
        st.info("👋 No brain dumps yet. Add one using the sidebar to get started!")
    else:
        # Get 5 most recent dumps
        recent_dumps = list(reversed(dumps[-5:]))
        
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        
        st.markdown(f"Showing **{len(recent_dumps)}** most recent brain dumps")
        st.divider()
        
        for dump_id, text, cluster_id in recent_dumps:
            render_feed_card(dump_id, text, cluster_id, cluster_labels_map)
            st.markdown("")  # Spacing between cards


# ============== MAIN APP ==============
# Main content
st.markdown("")  # Spacing

# Get all dumps
dumps = st.session_state.db.get_all_dumps()

# Render selected tab
if tab_selection == "Home":
    render_home()
else:
    render_feed()

# Footer
st.divider()
st.markdown("*Built with Google Gemini, Tavily Search, and Streamlit*")