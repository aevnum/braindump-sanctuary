"""
Brain Dump Sanctuary - Core Pipeline (Day 1 MVP)
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
    
    def fit_predict(self, embeddings):
        """HDBSCAN clustering - works great on CPU"""
        print(f"Clustering {len(embeddings)} brain dumps...")
        
        # HDBSCAN for semantic clustering
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        clusters = self.clusterer.fit_predict(embeddings)
        
        # UMAP for 2D visualization
        self.reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(embeddings)-1)
        )
        coords_2d = self.reducer.fit_transform(embeddings)
        
        print(f"✓ Found {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters")
        return clusters, coords_2d


# ============== 4. VISUALIZATION ==============
def create_cluster_graph(dump_data, coords_2d, clusters):
    """Interactive Plotly visualization"""
    
    fig = go.Figure()
    
    # Color map for clusters
    unique_clusters = set(clusters)
    colors = {-1: 'gray'}  # Noise points
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan']
    for i, c in enumerate([c for c in unique_clusters if c != -1]):
        colors[c] = cluster_colors[i % len(cluster_colors)]
    
    # Plot each cluster
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_coords = coords_2d[mask]
        cluster_texts = [dump_data[i][1] for i in range(len(dump_data)) if clusters[i] == cluster_id]
        
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unclustered"
        
        fig.add_trace(go.Scatter(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            mode='markers+text',
            name=label,
            marker=dict(
                size=12,
                color=colors[cluster_id],
                line=dict(width=1, color='white')
            ),
            text=[f"{i+1}" for i in range(len(cluster_texts))],
            textposition="top center",
            hovertext=cluster_texts,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title="Brain Dump Sanctuary - Semantic Clusters",
        showlegend=True,
        hovermode='closest',
        width=900,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


# ============== 5. DEMO PIPELINE ==============
def run_demo():
    """Complete pipeline demo"""
    
    # Sample brain dumps
    sample_dumps = [
        "Why do dreams feel so real but fade so quickly?",
        "How does quantum entanglement actually work?",
        "What makes sourdough bread different from regular bread?",
        "Are LLMs actually understanding or just pattern matching?",
        "Why do some songs give me chills?",
        "How do birds navigate during migration?",
        "What causes the smell of rain on dry ground?",
        "Is consciousness an emergent property?",
        "Why do we forget things we just read?",
        "How do neural networks learn representations?",
        "What's the connection between gut bacteria and mood?",
        "Why does time feel faster as we age?",
        "How do whales communicate across oceans?",
        "What makes a question 'good' vs 'bad'?",
        "Why do cats purr?",
        "How does anesthesia actually work?",
        "What causes earworms (songs stuck in head)?",
        "Is math discovered or invented?",
        "How do fireflies synchronize their flashing?",
        "Why do we yawn when others yawn?"
    ]
    
    print("=== BRAIN DUMP SANCTUARY - DEMO ===\n")
    
    # 1. Initialize
    db = BrainDumpDB("demo.db")
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
    print("\n[3/4] Performing semantic clustering...")
    clusters, coords_2d = clusterer.fit_predict(embeddings)
    
    for i, (dump_id, _, _) in enumerate(dumps):
        db.update_cluster(dump_id, clusters[i])
    
    # 5. Visualize
    print("\n[4/4] Creating visualization...")
    fig = create_cluster_graph(dumps, coords_2d, clusters)
    fig.write_html("brain_dump_clusters.html")
    print("✓ Saved to: brain_dump_clusters.html")
    
    # 6. Show cluster insights
    print("\n=== CLUSTER INSIGHTS ===")
    unique_clusters = set(clusters) - {-1}
    for cluster_id in sorted(unique_clusters):
        cluster_dumps = [dumps[i][1] for i in range(len(dumps)) if clusters[i] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_dumps)} items):")
        for dump in cluster_dumps[:3]:  # Show first 3
            print(f"  - {dump}")


if __name__ == "__main__":
    run_demo()