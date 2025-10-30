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
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = SearchAgent()
if 'question_agent' not in st.session_state:
    st.session_state.question_agent = QuestionAgent()
if 'perspective_agent' not in st.session_state:
    # Pass search_agent to PerspectiveAgent for web-enhanced analysis
    st.session_state.perspective_agent = PerspectiveAgent(search_agent=st.session_state.search_agent)

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Dump Sanctuary")
    st.markdown("Transform stale lists into actionable curiosity")
    
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
            # Prevent infinite loop with flag
            if not st.session_state.get('adding', False):
                st.session_state.adding = True
                st.session_state.db.add_dump(new_dump.strip())
                # Increment key to force text area to reset
                st.session_state.input_key += 1
                st.success("Added! Refresh clusters below.")
                st.rerun()
            else:
                st.session_state.adding = False
        else:
            st.error("Can't add empty thought!")
    
    if clear_clicked:
        # Prevent infinite loop with flag
        if not st.session_state.get('clearing_input', False):
            st.session_state.clearing_input = True
            # Increment key to clear the text area
            st.session_state.input_key += 1
            st.rerun()
        else:
            st.session_state.clearing_input = False
    
    st.divider()
    
    # Actions
    st.subheader("⚙️ Actions")
    
    # Refresh button - no rerun needed, just display message
    if st.button("🔄 Refresh Clusters", use_container_width=True):
        st.info("Clusters will refresh when you view the Cluster Map tab")
    
    # Clear all dumps button
    if st.button("🗑️ Clear All Dumps", use_container_width=True):
        # Use a flag to prevent infinite loop
        if not st.session_state.get('clearing', False):
            st.session_state.clearing = True
            st.session_state.db.conn.execute("DELETE FROM dumps")
            st.session_state.db.conn.execute("DELETE FROM cluster_labels")
            st.session_state.db.conn.execute("DELETE FROM sqlite_sequence WHERE name='dumps'")
            st.session_state.db.conn.commit()
            # Reset the input key counter to start fresh
            st.session_state.input_key = 0
            st.success("All dumps cleared!")
            st.rerun()
        else:
            st.session_state.clearing = False
    
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
            st.warning("⚠️ Add at least 3 brain dumps to see meaningful clusters")
            st.info("💡 Try adding more diverse thoughts to discover hidden patterns!")
        else:
            try:
                # Force refresh button
                col1, col2 = st.columns([4, 1])
                with col2:
                    force_refresh = st.button("🔄 Force Refresh", help="Recalculate embeddings and clusters")
                
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
                                    # Only generate label if it doesn't exist
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
                
                # Visualize with labels (common for both paths)
                fig = create_cluster_graph(dumps, coords_2d, clusters, cluster_labels_dict)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster details with labels (common for both paths)
                if unique_clusters:
                    st.subheader("📊 Cluster Details")
                    for cluster_id in sorted(unique_clusters):
                        cluster_dumps = [dumps[i] for i in range(len(dumps)) if clusters[i] == cluster_id]
                        label = cluster_labels_dict.get(cluster_id, f"Cluster {cluster_id}")
                        with st.expander(f"🏷️ {label} ({len(cluster_dumps)} items)"):
                            for dump_id, text, _ in cluster_dumps:
                                st.markdown(f"- {text}")
                else:
                    st.info("No distinct clusters found yet. Add more varied brain dumps!")
                        
            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")
                st.info("Try adding more brain dumps or refresh the page.")
                if st.checkbox("Show technical details"):
                    st.exception(e)
    
    # TAB 2: QUESTION GENERATION
    with tab2:
        st.subheader("Expand Vague Ideas into Structured Questions")
        
        # Get cluster labels for better display
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        
        # Select a dump with cluster labels
        dump_options = []
        for d in dumps:
            dump_id, text, cluster_id = d
            cluster_label = ""
            if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
                cluster_label = f" [{cluster_labels_map[cluster_id]['label']}]"
            display_text = f"{dump_id}: {text[:50]}...{cluster_label}" if len(text) > 50 else f"{dump_id}: {text}{cluster_label}"
            dump_options.append(display_text)
        
        selected = st.selectbox("Choose a brain dump to expand:", dump_options)
        
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
        
        # API Status indicator
        if st.session_state.search_agent.use_real_search:
            st.success("✅ Tavily API Connected - Real web search enabled")
        else:
            st.warning("⚠️ Using mock search - Add TAVILY_API_KEY to .env for real results")
        
        # Get cluster labels for better display
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        
        # Select a dump with cluster labels
        dump_options = []
        dump_id_map = {}  # Map display text to dump_id
        for d in dumps:
            dump_id, text, cluster_id = d
            cluster_label = ""
            if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
                cluster_label = f" [{cluster_labels_map[cluster_id]['label']}]"
            display_text = f"{dump_id}: {text[:50]}...{cluster_label}" if len(text) > 50 else f"{dump_id}: {text}{cluster_label}"
            dump_options.append(display_text)
            dump_id_map[display_text] = dump_id
        
        # Initialize session state for selected dump
        if 'search_selected_dump_id' not in st.session_state:
            st.session_state.search_selected_dump_id = None
        
        # Find the index to use based on stored dump_id
        default_index = 0
        if st.session_state.search_selected_dump_id is not None:
            for i, option in enumerate(dump_options):
                if dump_id_map[option] == st.session_state.search_selected_dump_id:
                    default_index = i
                    break
        
        selected = st.selectbox(
            "Choose a brain dump to research:", 
            dump_options, 
            key="search_select",
            index=default_index
        )
        
        if selected:
            # Store the selected dump_id
            st.session_state.search_selected_dump_id = dump_id_map[selected]
            
            dump_id = dump_id_map[selected]
            original_text = next(d[1] for d in dumps if d[0] == dump_id)
            
            st.markdown("**Research topic:**")
            st.info(original_text)
            
            if st.button("🔍 Deep Dive with Web Search", type="primary"):
                with st.spinner("Searching the web and synthesizing..."):
                    result = st.session_state.search_agent.deep_dive(original_text)
                    
                    st.markdown("### 📊 Key Findings")
                    st.markdown(result['summary'])
                    
                    if result['sources']:
                        st.markdown("### 📚 Sources & Details")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"� {i}. {source['title']}"):
                                st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                                if source.get('snippet'):
                                    st.markdown(f"**Snippet:** {source['snippet']}")
                                if source.get('score'):
                                    st.caption(f"Relevance: {source['score']:.2f}")
                    else:
                        st.info("No sources available (using mock search mode)")
    
    # TAB 4: MULTI-PERSPECTIVE ANALYSIS
    with tab4:
        st.subheader("Controversial Topics: Multiple Perspectives")
        st.markdown("*Best for topics with debate/nuance*")
        
        # API Status indicator
        if st.session_state.search_agent.use_real_search:
            st.success("✅ Web-enhanced perspectives enabled")
        else:
            st.info("ℹ️ Add TAVILY_API_KEY to .env for web-context-enhanced perspectives")
        
        # Get cluster labels for better display
        cluster_labels_map = st.session_state.db.get_all_cluster_labels()
        
        # Select a dump with cluster labels
        dump_options = []
        dump_id_map = {}  # Map display text to dump_id
        for d in dumps:
            dump_id, text, cluster_id = d
            cluster_label = ""
            if cluster_id is not None and cluster_id != -1 and cluster_id in cluster_labels_map:
                cluster_label = f" [{cluster_labels_map[cluster_id]['label']}]"
            display_text = f"{dump_id}: {text[:50]}...{cluster_label}" if len(text) > 50 else f"{dump_id}: {text}{cluster_label}"
            dump_options.append(display_text)
            dump_id_map[display_text] = dump_id
        
        # Initialize session state for selected dump
        if 'perspective_selected_dump_id' not in st.session_state:
            st.session_state.perspective_selected_dump_id = None
        
        # Find the index to use based on stored dump_id
        default_index = 0
        if st.session_state.perspective_selected_dump_id is not None:
            for i, option in enumerate(dump_options):
                if dump_id_map[option] == st.session_state.perspective_selected_dump_id:
                    default_index = i
                    break
        
        selected = st.selectbox(
            "Choose a controversial topic:", 
            dump_options, 
            key="perspective_select",
            index=default_index
        )
        
        if selected:
            # Store the selected dump_id
            st.session_state.perspective_selected_dump_id = dump_id_map[selected]
            
            dump_id = dump_id_map[selected]
            original_text = next(d[1] for d in dumps if d[0] == dump_id)
            
            st.markdown("**Topic:**")
            st.info(original_text)
            
            if st.button("⚖️ Generate Multiple Perspectives", type="primary"):
                with st.spinner("Analyzing from different angles..."):
                    perspectives = st.session_state.perspective_agent.analyze(original_text)
                    
                    st.markdown("### 🎭 Perspectives")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 🔴 Skeptical View")
                        st.markdown(perspectives['skeptical'])
                    
                    with col2:
                        st.markdown("#### 🟢 Optimistic View")
                        st.markdown(perspectives['optimistic'])
                    
                    with col3:
                        st.markdown("#### 🟡 Nuanced View")
                        st.markdown(perspectives['nuanced'])
                    
                    # Display referenced articles if available
                    if 'sources' in perspectives and perspectives['sources']:
                        st.markdown("---")
                        st.markdown("### 📚 Referenced Articles")
                        
                        cols = st.columns(3)
                        for i, article in enumerate(perspectives['sources']):
                            with cols[i % 3]:
                                with st.expander(f"📄 {article['title'][:50]}..."):
                                    st.markdown(f"**[Read full article]({article['url']})**")
                                    st.caption(article['snippet'])

# Footer
st.divider()
st.markdown("*Built with Google Gemini, Tavily Search, and Streamlit*")