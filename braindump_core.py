"""
Brain Dump Sanctuary - Core Pipeline (Day 1 MVP + LangChain Integration)
Runs entirely on CPU - no GPU needed
"""

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports for cluster labeling with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithFallbacks

# ============== 1. DATABASE LAYER ==============
class BrainDumpDB:
    def __init__(self, db_path="braindump.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dumps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB,
                cluster_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add cluster labels table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_labels (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add_dump(self, text):
        cursor = self.conn.execute(
            "INSERT INTO dumps (text) VALUES (?)", 
            (text,)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_dumps(self):
        cursor = self.conn.execute("SELECT id, text, cluster_id FROM dumps")
        return cursor.fetchall()
    
    def update_embedding(self, dump_id, embedding):
        embedding_blob = embedding.astype(np.float32).tobytes()
        self.conn.execute(
            "UPDATE dumps SET embedding = ? WHERE id = ?",
            (embedding_blob, dump_id)
        )
        self.conn.commit()
    
    def update_cluster(self, dump_id, cluster_id):
        self.conn.execute(
            "UPDATE dumps SET cluster_id = ? WHERE id = ?",
            (int(cluster_id), dump_id)
        )
        self.conn.commit()
    
    def get_embeddings(self):
        cursor = self.conn.execute("SELECT id, embedding FROM dumps WHERE embedding IS NOT NULL")
        results = []
        for row in cursor.fetchall():
            dump_id, embedding_blob = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            results.append((dump_id, embedding))
        return results
    
    def save_cluster_label(self, cluster_id, label, description=None):
        """Save or update a cluster label"""
        self.conn.execute(
            "INSERT OR REPLACE INTO cluster_labels (cluster_id, label, description) VALUES (?, ?, ?)",
            (int(cluster_id), label, description)
        )
        self.conn.commit()
    
    def get_cluster_label(self, cluster_id):
        """Get label for a specific cluster"""
        cursor = self.conn.execute(
            "SELECT label, description FROM cluster_labels WHERE cluster_id = ?",
            (int(cluster_id),)
        )
        result = cursor.fetchone()
        return result if result else (None, None)
    
    def get_all_cluster_labels(self):
        """Get all cluster labels"""
        cursor = self.conn.execute("SELECT cluster_id, label, description FROM cluster_labels")
        return {row[0]: {"label": row[1], "description": row[2]} for row in cursor.fetchall()}
    
    def get_cluster_dumps(self, cluster_id):
        """Get all dumps in a specific cluster"""
        cursor = self.conn.execute(
            "SELECT id, text FROM dumps WHERE cluster_id = ?",
            (int(cluster_id),)
        )
        return cursor.fetchall()


# ============== 2. EMBEDDING ENGINE ==============
class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded (CPU mode)")
    
    def embed(self, texts):
        """Fast CPU inference - 5-10ms per text"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings


# ============== 3. CLUSTERING ENGINE ==============
class ClusterEngine:
    def __init__(self, min_cluster_size=3):
        self.min_cluster_size = min_cluster_size
        self.clusterer = None
        self.reducer = None
        self.llm = None
        self.labeling_chain = None
        
        # Initialize LangChain for cluster labeling
        self._init_labeling_chain()
    
    def _init_labeling_chain(self):
        """Initialize LangChain chain for automatic cluster labeling with Google Gemini"""
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("⚠️ GOOGLE_API_KEY not found. Cluster labeling will be disabled.")
                print("   Get your API key from: https://aistudio.google.com/app/apikey")
                return
            
            # Create primary Gemini LLM
            primary_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=api_key
            )
            
            # Create fallback Gemini LLM (using same model but could use gemini-pro)
            fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.5,
                google_api_key=api_key
            )
            
            # LLM with fallback support
            self.llm = primary_llm.with_fallbacks([fallback_llm])
            
            # Create labeling prompt
            prompt = ChatPromptTemplate.from_template(
                """You are an expert at identifying themes in clusters of related thoughts.

Given these brain dump entries from a cluster:
{cluster_texts}

Task: Generate a short, descriptive label (2-4 words) that captures the unifying theme.

Rules:
- Be specific and insightful
- Use natural language, not generic terms
- Focus on the underlying curiosity or topic
- Be specific if entries are less.
- Be broad if entries are more.

Label:"""
            )
            
            # Create the chain
            self.labeling_chain = (
                prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            print("✓ LangChain cluster labeling initialized with Google Gemini")
            
        except Exception as e:
            print(f"⚠️ Could not initialize cluster labeling: {e}")
            self.llm = None
            self.labeling_chain = None
    
    def generate_cluster_label(self, cluster_texts):
        """Generate a label for a cluster using LangChain"""
        if not self.labeling_chain:
            return "Cluster"
        
        try:
            # Take up to 5 representative texts
            sample_texts = cluster_texts[:5]
            texts_str = "\n".join([f"- {text}" for text in sample_texts])
            
            label = self.labeling_chain.invoke({"cluster_texts": texts_str})
            return label.strip()
            
        except Exception as e:
            print(f"⚠️ Cluster labeling failed: {e}")
            return "Cluster"
    
    def fit_predict(self, embeddings):
        """HDBSCAN clustering - works great on CPU"""
        print(f"Clustering {len(embeddings)} brain dumps...")
        
        n_samples = len(embeddings)
        
        # Adjust min_cluster_size for small datasets
        effective_min_cluster_size = min(self.min_cluster_size, max(2, n_samples // 2))
        
        # HDBSCAN for semantic clustering
        self.clusterer = HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            min_samples=1  # More lenient clustering
        )
        clusters = self.clusterer.fit_predict(embeddings)
        
        # UMAP for 2D visualization with better separation parameters
        # n_neighbors controls local vs global structure (lower = tighter clusters)
        n_neighbors = max(2, min(10, n_samples - 1))  # Reduced from 15 for tighter clusters
        
        # For very small datasets, use simpler initialization
        init = 'spectral' if n_samples > 10 else 'random'
        
        self.reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.3,  # Increased from 0.1 for better separation between clusters
            spread=1.5,     # Controls how clumped embeddings are (higher = more spread)
            metric='euclidean',
            init=init,
            negative_sample_rate=10,  # Helps with separation
            repulsion_strength=1.2    # Pushes dissimilar points apart
        )
        coords_2d = self.reducer.fit_transform(embeddings)
        
        print(f"✓ Found {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters")
        return clusters, coords_2d


# ============== 4. VISUALIZATION ==============
def build_graph_edges(embeddings, clusters, max_edges_per_node=5, similarity_threshold=0.7):
    """Build edges between semantically similar items (Obsidian knowledge graph style)"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    edges = []
    similarities = cosine_similarity(embeddings)
    
    # Add edges between items in same cluster
    unique_clusters = set(clusters)
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        # Connect items within cluster (up to max_edges_per_node each)
        for i in cluster_indices:
            # Find most similar items in same cluster
            cluster_similarities = [
                (j, similarities[i][j]) for j in cluster_indices if i != j
            ]
            cluster_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for j, sim in cluster_similarities[:max_edges_per_node]:
                if i < j:  # Avoid duplicates
                    edges.append((i, j, sim))
    
    return edges


def create_knowledge_graph(dump_data, coords_2d, clusters, cluster_labels=None, embeddings=None):
    """
    Obsidian-style knowledge graph visualization with force-directed layout.
    Shows connections between related brain dumps in a network style.
    """
    
    fig = go.Figure()
    
    # Color palette for clusters
    cluster_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Light Blue
        '#F8B739',  # Orange
        '#52BE80',  # Green
        '#EC7063',  # Coral
        '#AF7AC5',  # Lavender
        '#5DADE2',  # Ocean Blue
        '#48C9B0',  # Turquoise
        '#F1948A',  # Pink
        '#85929E',  # Gray Blue
        '#F39C12',  # Dark Orange
        '#3498DB',  # Bright Blue
        '#E74C3C',  # Bright Red
        '#9B59B6'   # Violet
    ]
    
    # Color map for clusters
    unique_clusters = set(clusters)
    colors = {-1: '#95A5A6'}  # Gray for noise points
    
    for i, c in enumerate([c for c in unique_clusters if c != -1]):
        colors[c] = cluster_colors[i % len(cluster_colors)]
    
    # Build edges between related items
    edges = []
    if embeddings is not None:
        edges = build_graph_edges(embeddings, clusters)
    
    # Draw edges first (so they appear behind nodes)
    for i, j, similarity in edges:
        x0, y0 = coords_2d[i]
        x1, y1 = coords_2d[j]
        
        # Edge opacity based on similarity
        edge_opacity = 0.2 + (similarity - 0.7) * 0.4  # Range: 0.2-0.6
        edge_opacity = max(0.1, min(0.6, edge_opacity))
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=1.5,
                color=f'rgba(200, 200, 200, {edge_opacity})',
            ),
            hoverinfo='none',
            showlegend=False,
            name='',
        ))
    
    # Draw nodes (clusters)
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_coords = coords_2d[mask]
        cluster_texts = [dump_data[i][1] for i in range(len(dump_data)) if clusters[i] == cluster_id]
        
        # Get cluster label if available
        if cluster_labels and cluster_id in cluster_labels:
            label = cluster_labels[cluster_id]
        else:
            label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unclustered"
        
        # Determine node size (larger for bigger clusters)
        node_size = min(24, 14 + len(cluster_texts) // 2)
        
        fig.add_trace(go.Scatter(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            mode='markers',
            name=label,
            marker=dict(
                size=node_size,
                color=colors[cluster_id],
                line=dict(width=2, color='rgba(255, 255, 255, 0.8)'),
                opacity=0.95,
                symbol='circle',
            ),
            text=cluster_texts,
            hovertext=[f"<b>{text}</b>" for text in cluster_texts],
            hoverinfo='text',
            showlegend=True,
        ))
    
    fig.update_layout(
        title={
            'text': "🧠 Brain Dump Sanctuary - Knowledge Graph",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        showlegend=True,
        hovermode='closest',
        width=1400,
        height=800,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='white', family='monospace'),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),
        margin=dict(l=0, r=200, t=50, b=0),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(13, 17, 23, 0.9)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    
    return fig


def create_cluster_graph(dump_data, coords_2d, clusters, cluster_labels=None):
    """Interactive Plotly visualization with cluster labels (Legacy - use create_knowledge_graph instead)"""
    
    fig = go.Figure()
    
    # Expanded color palette with 20 distinct colors
    cluster_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Light Blue
        '#F8B739',  # Orange
        '#52BE80',  # Green
        '#EC7063',  # Coral
        '#AF7AC5',  # Lavender
        '#5DADE2',  # Ocean Blue
        '#48C9B0',  # Turquoise
        '#F1948A',  # Pink
        '#85929E',  # Gray Blue
        '#F39C12',  # Dark Orange
        '#3498DB',  # Bright Blue
        '#E74C3C',  # Bright Red
        '#9B59B6'   # Violet
    ]
    
    # Color map for clusters
    unique_clusters = set(clusters)
    colors = {-1: '#95A5A6'}  # Gray for noise points
    
    for i, c in enumerate([c for c in unique_clusters if c != -1]):
        colors[c] = cluster_colors[i % len(cluster_colors)]
    
    # Plot each cluster
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_coords = coords_2d[mask]
        cluster_texts = [dump_data[i][1] for i in range(len(dump_data)) if clusters[i] == cluster_id]
        
        # Get cluster label if available
        if cluster_labels and cluster_id in cluster_labels:
            label = cluster_labels[cluster_id]
        else:
            label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unclustered"
        
        fig.add_trace(go.Scatter(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            mode='markers+text',
            name=label,
            marker=dict(
                size=14,  # Slightly larger for visibility
                color=colors[cluster_id],
                line=dict(width=2, color='white'),  # Thicker white border
                opacity=0.9  # Slight transparency to see overlaps
            ),
            text=[f"{i+1}" for i in range(len(cluster_texts))],
            textposition="top center",
            textfont=dict(size=10, color='white'),
            hovertext=cluster_texts,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title="Brain Dump Sanctuary - Semantic Clusters",
        showlegend=True,
        hovermode='closest',
        width=1200,  # Wider for better separation
        height=700,  # Taller for better separation
        plot_bgcolor='#1a1a1a',  # Dark background
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='#30363d',
            borderwidth=1
        )
    )
    
    return fig


# ============== 5. DEMO PIPELINE ==============
def run_demo():
    """Complete pipeline demo"""
    
    # Sample brain dumps
    sample_dumps = [
        # Consciousness & Mind (8 items)
        "Why do dreams feel so real but fade so quickly?",
        "Is consciousness an emergent property?",
        "Why do we forget things we just read?",
        "What causes earworms (songs stuck in head)?",
        "Why does time feel faster as we age?",
        "How does anesthesia actually work?",
        "What makes us self-aware?",
        "Can machines ever truly be conscious?",
        
        # Physics & Quantum (8 items)
        "How does quantum entanglement actually work?",
        "Is math discovered or invented?",
        "What happens at the event horizon of a black hole?",
        "Why is the speed of light constant?",
        "What is dark matter made of?",
        "How can particles be in two places at once?",
        "What came before the Big Bang?",
        "Why does time only move forward?",
        
        # AI & Technology (8 items)
        "Are LLMs actually understanding or just pattern matching?",
        "How do neural networks learn representations?",
        "What makes a question 'good' vs 'bad'?",
        "Can AI ever be truly creative?",
        "How do transformers attend to context?",
        "What is the halting problem and why does it matter?",
        "Will we ever achieve AGI?",
        "How do computers generate random numbers?",
        
        # Biology & Nature (8 items)
        "How do birds navigate during migration?",
        "What causes the smell of rain on dry ground?",
        "What's the connection between gut bacteria and mood?",
        "How do whales communicate across oceans?",
        "Why do cats purr?",
        "How do fireflies synchronize their flashing?",
        "Why do we yawn when others yawn?",
        "How do octopuses change color instantly?",
        
        # Food & Chemistry (8 items)
        "What makes sourdough bread different from regular bread?",
        "Why does coffee smell better than it tastes?",
        "What makes food spicy and why do we like it?",
        "How does fermentation preserve food?",
        "Why does chocolate melt at body temperature?",
        "What causes that metallic taste when you bite foil?",
        "Why do onions make us cry?",
        "How do flavors combine to create umami?",
        
        # Music & Art (8 items)
        "Why do some songs give me chills?",
        "What makes a melody memorable?",
        "How does rhythm affect our emotions?",
        "Why do major keys sound happy and minor keys sad?",
        "What makes abstract art 'good'?",
        "How does color theory influence mood?",
        "Why do we find symmetry beautiful?",
        "What is the golden ratio in design?",
        
        # Psychology & Society (8 items)
        "Why do we procrastinate even when we know better?",
        "How does confirmation bias shape our beliefs?",
        "What causes impostor syndrome?",
        "Why are first impressions so lasting?",
        "How do echo chambers form online?",
        "What makes some ideas go viral?",
        "Why is it hard to change someone's mind?",
        "How does groupthink override individual judgment?",
        
        # Language & Communication (4 items)
        "Why do different languages have different sounds?",
        "How did writing systems evolve independently?",
        "What makes a joke funny across cultures?",
        "Why do babies learn language so easily?"
    ]
    
    print("=== BRAIN DUMP SANCTUARY - DEMO ===\n")
    
    # 1. Initialize
    db = BrainDumpDB("braindump.db")
    embedder = EmbeddingEngine()
    clusterer = ClusterEngine(min_cluster_size=2)
    
    # 2. Add dumps to database
    print("\n[1/4] Adding brain dumps to database...")
    for dump in sample_dumps:
        db.add_dump(dump)
    
    # 3. Generate embeddings
    print("\n[2/4] Generating embeddings (CPU)...")
    dumps = db.get_all_dumps()
    texts = [d[1] for d in dumps]
    embeddings = embedder.embed(texts)
    
    for i, (dump_id, text, _) in enumerate(dumps):
        db.update_embedding(dump_id, embeddings[i])
    
    # 4. Cluster
    print("\n[3/5] Performing semantic clustering...")
    clusters, coords_2d = clusterer.fit_predict(embeddings)
    
    for i, (dump_id, _, _) in enumerate(dumps):
        db.update_cluster(dump_id, clusters[i])
    
    # 5. Auto-generate cluster labels using LangChain
    print("\n[4/5] Generating cluster labels with LLM...")
    cluster_labels = {}
    unique_clusters = set(clusters) - {-1}
    
    for cluster_id in unique_clusters:
        cluster_dumps = [dumps[i][1] for i in range(len(dumps)) if clusters[i] == cluster_id]
        label = clusterer.generate_cluster_label(cluster_dumps)
        cluster_labels[cluster_id] = label
        db.save_cluster_label(cluster_id, label)
        print(f"  Cluster {cluster_id}: '{label}'")
    
    # 6. Visualize
    print("\n[5/5] Creating knowledge graph visualization...")
    fig = create_knowledge_graph(dumps, coords_2d, clusters, cluster_labels, embeddings)
    fig.write_html("brain_dump_knowledge_graph.html")
    print("✓ Saved to: brain_dump_knowledge_graph.html")
    
    # 7. Show cluster insights
    print("\n=== CLUSTER INSIGHTS ===")
    for cluster_id in sorted(unique_clusters):
        cluster_dumps = [dumps[i][1] for i in range(len(dumps)) if clusters[i] == cluster_id]
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        print(f"\n{label} ({len(cluster_dumps)} items):")
        for dump in cluster_dumps[:3]:  # Show first 3
            print(f"  - {dump}")


if __name__ == "__main__":
    run_demo()