"""
Brain Dump Sanctuary - Core Pipeline (Day 1 MVP + LangChain Integration)
Now with Neo4j Graph Database Backend
Runs entirely on CPU - no GPU needed
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Load environment variables from .env file
load_dotenv()

# LangChain imports for cluster labeling with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithFallbacks

# ============== 1. DATABASE LAYER - Neo4j ==============
class BrainDumpDB:
    """
    Neo4j-based graph database for Brain Dump Sanctuary.
    
    Graph Schema:
    - Node: Dump {id, text, embedding[], cluster_id, created_at}
    - Node: Cluster {id, label, description, created_at}
    - Relationship: Dump -[:IN_CLUSTER]-> Cluster
    - Relationship: Dump -[:SIMILAR_TO {weight}]-> Dump
    """
    
    def __init__(self):
        """Initialize Neo4j connection from environment variables."""
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            print("  Make sure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set in .env")
            raise
        
        self._init_schema()
    
    def _init_schema(self):
        """Initialize graph schema with nodes and indexes."""
        with self.driver.session() as session:
            # Create Dump nodes with indexes
            session.run("""
                CREATE INDEX dump_id_index IF NOT EXISTS 
                FOR (d:Dump) ON (d.id)
            """)
            session.run("""
                CREATE INDEX dump_created_index IF NOT EXISTS 
                FOR (d:Dump) ON (d.created_at)
            """)
            
            # Create Cluster nodes with indexes
            session.run("""
                CREATE INDEX cluster_id_index IF NOT EXISTS 
                FOR (c:Cluster) ON (c.id)
            """)
            
            print("✓ Graph schema initialized")
    
    def add_dump(self, text):
        """Add a new brain dump to the database."""
        with self.driver.session() as session:
            result = session.run("""
                CREATE (d:Dump {
                    id: randomUuid(),
                    text: $text,
                    created_at: datetime()
                })
                RETURN d.id as dump_id
            """, text=text)
            dump_id = result.single()["dump_id"]
            return dump_id
    
    def get_all_dumps(self):
        """Get all dumps with their cluster assignments and timestamps."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Dump)
                OPTIONAL MATCH (d)-[:IN_CLUSTER]->(c:Cluster)
                RETURN d.id as id, d.text as text, c.id as cluster_id, d.created_at as created_at
                ORDER BY d.created_at DESC
            """)
            return [(row["id"], row["text"], row["cluster_id"], row["created_at"]) for row in result]
    
    def update_embedding(self, dump_id, embedding):
        """Store embedding vector for a dump."""
        # Convert numpy array to list for Neo4j storage
        embedding_list = embedding.astype(np.float32).tolist()
        
        with self.driver.session() as session:
            session.run("""
                MATCH (d:Dump {id: $dump_id})
                SET d.embedding = $embedding
            """, dump_id=dump_id, embedding=embedding_list)
    
    def update_cluster(self, dump_id, cluster_id):
        """Assign dump to a cluster."""
        with self.driver.session() as session:
            # Remove existing cluster relationship
            session.run("""
                MATCH (d:Dump {id: $dump_id})-[r:IN_CLUSTER]->()
                DELETE r
            """, dump_id=dump_id)
            
            # Create or match cluster node and add relationship
            session.run("""
                MATCH (d:Dump {id: $dump_id})
                MERGE (c:Cluster {id: $cluster_id})
                ON CREATE SET c.created_at = datetime()
                CREATE (d)-[:IN_CLUSTER]->(c)
            """, dump_id=dump_id, cluster_id=int(cluster_id))
    
    def get_embeddings(self):
        """Retrieve all embeddings from database."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Dump)
                WHERE d.embedding IS NOT NULL
                RETURN d.id as dump_id, d.embedding as embedding
            """)
            embeddings = []
            for row in result:
                dump_id = row["dump_id"]
                embedding_list = row["embedding"]
                # Convert list back to numpy array
                embedding = np.array(embedding_list, dtype=np.float32)
                embeddings.append((dump_id, embedding))
            return embeddings
    
    def save_cluster_label(self, cluster_id, label, description=None):
        """Save or update a cluster label."""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Cluster {id: $cluster_id})
                ON CREATE SET c.created_at = datetime()
                SET c.label = $label, c.description = $description
            """, cluster_id=int(cluster_id), label=label, description=description)
    
    def get_cluster_label(self, cluster_id):
        """Get label for a specific cluster."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster {id: $cluster_id})
                RETURN c.label as label, c.description as description
            """, cluster_id=int(cluster_id))
            row = result.single()
            if row:
                return (row["label"], row["description"])
            return (None, None)
    
    def get_all_cluster_labels(self):
        """Get all cluster labels."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)
                RETURN c.id as cluster_id, c.label as label, c.description as description
            """)
            return {
                row["cluster_id"]: {
                    "label": row["label"],
                    "description": row["description"]
                }
                for row in result
            }
    
    def get_cluster_dumps(self, cluster_id):
        """Get all dumps in a specific cluster."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Dump)-[:IN_CLUSTER]->(c:Cluster {id: $cluster_id})
                RETURN d.id as dump_id, d.text as text
            """, cluster_id=int(cluster_id))
            return [(row["dump_id"], row["text"]) for row in result]
    
    def add_generated_dump(self, text, cluster_id):
        """
        Add a generated braindump directly to a specific cluster.
        Returns the new dump_id.
        """
        with self.driver.session() as session:
            result = session.run("""
                CREATE (d:Dump {
                    id: randomUuid(),
                    text: $text,
                    created_at: datetime()
                })
                WITH d
                MATCH (c:Cluster {id: $cluster_id})
                CREATE (d)-[:IN_CLUSTER]->(c)
                RETURN d.id as dump_id
            """, text=text, cluster_id=int(cluster_id))
            dump_id = result.single()["dump_id"]
            return dump_id
    
    def save_feed_cache(self, dump_id, summary, questions, image_urls=None):
        """
        Store cached feed data (summary, questions, and image URLs) for a dump.
        
        Args:
            dump_id: ID of the dump
            summary: Summary text from FeedAgent
            questions: List of reflection questions
            image_urls: List of image URLs related to the brain dump
        """
        with self.driver.session() as session:
            # Store as JSON strings for Neo4j compatibility
            questions_json = json.dumps(questions) if isinstance(questions, list) else questions
            image_urls_json = json.dumps(image_urls) if image_urls else json.dumps([])
            
            session.run("""
                MATCH (d:Dump {id: $dump_id})
                SET d.summary = $summary,
                    d.questions = $questions,
                    d.image_urls = $image_urls,
                    d.feed_cache_generated_at = datetime()
            """, dump_id=dump_id, summary=summary, questions=questions_json, image_urls=image_urls_json)
    
    def get_feed_cache(self, dump_id):
        """
        Retrieve cached feed data for a dump.
        Returns: {"summary": str, "questions": list, "image_urls": list} or None if not cached
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Dump {id: $dump_id})
                RETURN d.summary as summary, d.questions as questions, d.image_urls as image_urls
            """, dump_id=dump_id)
            
            row = result.single()
            if row and row["summary"]:
                questions = json.loads(row["questions"]) if row["questions"] else []
                image_urls = json.loads(row["image_urls"]) if row["image_urls"] else []
                return {
                    "summary": row["summary"],
                    "questions": questions,
                    "image_urls": image_urls
                }
            return None
    
    def close(self):
        """Close database connection."""
        self.driver.close()


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
    db = BrainDumpDB()
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
    
    for i, (dump_id, text, _, created_at) in enumerate(dumps):
        db.update_embedding(dump_id, embeddings[i])
    
    # 4. Cluster
    print("\n[3/5] Performing semantic clustering...")
    clusters, coords_2d = clusterer.fit_predict(embeddings)
    
    for i, (dump_id, _, _, _) in enumerate(dumps):
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