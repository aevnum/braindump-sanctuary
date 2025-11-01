# 🧠 Brain Dump Sanctuary

> **Capture, cluster, and reflect on your thoughts in real-time.**

A full-stack NLP application that transforms scattered thoughts into organized, semantically meaningful clusters with AI-powered insights. Built with Streamlit, LangGraph agents, and Neo4j.

---

## ✨ Features

### Core Capabilities
- **💭 Brain Dump Capture**: Write down thoughts, ideas, and questions instantly
- **🔗 Semantic Clustering**: Automatically group related thoughts using embeddings (sentence-transformers) and HDBSCAN
- **📊 Interactive Knowledge Graph**: Visualize all thoughts as an interactive Plotly graph with color-coded clusters
- **🏷️ AI-Generated Labels**: Auto-label clusters using Google Gemini LLM (e.g., "Personal Growth", "Technical Concepts")
- **🤖 Multi-Agent Intelligence**:
  - **SearchAgent**: Generate contextualized summaries with optional web search (Tavily)
  - **QuestionAgent**: Generate Socratic questions for deeper reflection
  - **GenerationAgent**: Synthesize insights across related thoughts
  - **FeedAgent**: Curate a blog-style feed of your 5 most recent thoughts with agent insights
- **📝 Persistent Storage**: All thoughts stored in Neo4j graph database with full history
- **🔄 Real-time Updates**: Refresh embeddings and recalculate clusters on demand

### Two-View Interface
1. **Home Tab** 🏠: Semantic cluster map + comprehensive table of all thoughts
2. **Feed Tab** 📰: Blog-style cards with summaries and reflection questions

---

## 🏗️ Architecture

### Tech Stack
```
Frontend:          Streamlit (single-page Python web app)
Backend:           Python 3.10+ with LangChain/LangGraph
Database:          Neo4j Aura (graph database)
NLP Pipeline:      
  - Embeddings:    sentence-transformers (all-MiniLM-L6-v2)
  - Clustering:    HDBSCAN + UMAP dimensionality reduction
  - LLM:           Google Gemini 2.5 Flash (via LangChain)
Search:            Tavily API (optional, with mock fallback)
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                         │
│  (app.py - Home Tab | Feed Tab | Sidebar Controls)     │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    ┌───▼────┐         ┌────▼──────┐
    │ AGENTS │         │ EMBEDDINGS │
    │ (Day2) │         │ & CLUSTERS │
    └───┬────┘         │ (Day 1)    │
        │              └────┬───────┘
        │                   │
    ┌───┴───────────────────▼────┐
    │    BRAINDUMP_CORE.PY       │
    │  - BrainDumpDB (Neo4j)     │
    │  - EmbeddingEngine         │
    │  - ClusterEngine           │
    │  - ClusterLabelEngine      │
    └───┬────────────────────────┘
        │
    ┌───▼──────────────────┐
    │   NEO4J GRAPH DB     │
    │  (Aura Cloud)        │
    │  - Dump Nodes        │
    │  - Cluster Nodes     │
    │  - Relationships     │
    └──────────────────────┘
```

### Data Flow

**Input → Processing → Storage → Visualization**

1. **User enters brain dump** → `app.py` sidebar
2. **Generate embedding** → `EmbeddingEngine` (sentence-transformer)
3. **Store in Neo4j** → `BrainDumpDB.add_dump()`
4. **Re-cluster on demand** → `ClusterEngine.cluster()` + `ClusterLabelEngine.label_clusters()`
5. **Visualize clusters** → Plotly interactive graph (Home tab)
6. **Enrich with AI** → Agents analyze for Feed tab

### Neo4j Graph Schema

```cypher
Node: Dump
├── id: UUID
├── text: String (the brain dump)
├── embedding: Vector[384] (from sentence-transformer)
├── created_at: DateTime
└── cluster_id: Reference to Cluster

Node: Cluster
├── id: UUID
├── label: String (e.g., "Personal Growth")
├── description: String (optional)
├── created_at: DateTime

Relationships:
├── Dump -[:IN_CLUSTER]-> Cluster
└── Dump -[:SIMILAR_TO {weight: 0.0-1.0}]-> Dump
```

### Component Breakdown

| File | Purpose | Key Classes |
|------|---------|------------|
| `app.py` | Streamlit UI & session management | Main app logic, page layout |
| `braindump_core.py` | NLP pipeline & database | `BrainDumpDB`, `EmbeddingEngine`, `ClusterEngine`, `ClusterLabelEngine` |
| `agents.py` | AI agents for insights | `QuestionAgent`, `SearchAgent`, `GenerationAgent`, `FeedAgent` |
| `neo4j_maintenance.py` | Database utilities | Neo4j query helpers, debugging |
| `cleanup_db.py` | Reset database | Wipe all data (for testing) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Neo4j Aura account (free tier available)
- API keys for Google Gemini (free) and optionally Tavily

### 1. Setup Neo4j (Required)

Create a free Neo4j Aura instance:
1. Go to https://console.neo4j.io/
2. Sign up for a free account
3. Create a new "Free" Aura instance
4. Copy your connection details:
   - **URI**: `neo4j+ssc://xxxxx.databases.neo4j.io`
   - **Username**: `neo4j`
   - **Password**: (set during creation)

### 2. Installation

```bash
# Clone repository
git clone <repo-url>
cd braindump-sanctuary

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env  # or create manually

# Edit .env with your credentials
```

**.env Template:**
```env
# Neo4j (Required)
NEO4J_URI=neo4j+ssc://your-instance-id.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here

# Google Gemini (Required for AI features)
GOOGLE_API_KEY=your_google_api_key_here

# Web Search (Optional - uses mock data if not provided)
TAVILY_API_KEY=your_tavily_api_key_here

# Alternative: Perplexity (Optional)
PERPLEXITY_API_KEY=your_perplexity_key_here
```

**Get API Keys:**
- 🔑 **Google Gemini** (free): https://aistudio.google.com/app/apikey
- 🔍 **Tavily Search** (optional): https://app.tavily.com
- 🧠 **Perplexity** (optional): https://www.perplexity.ai/

### 3. Run the App

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📱 User Interface Guide

### Sidebar Controls

**Left Sidebar** - Main interaction hub:
- 🗺️ **Navigation**: Radio buttons to toggle between **Home** and **Feed** tabs
- 📝 **Text Area**: Input field for new brain dumps
- ✅ **Add to Sanctuary**: Save the thought to Neo4j
- 🗑️ **Clear**: Empty the input field
- 🔄 **Refresh Clusters**: Recalculate all embeddings and clusters (computationally intensive, 10-30 seconds)
- ⚠️ **Clear All Dumps**: Permanently delete all thoughts from the database
- 📊 **Total Brain Dumps**: Counter showing total stored thoughts

### Home Tab 🏠 - Knowledge Graph View

**Top Section: Interactive Cluster Map**
- **Visualization**: 2D interactive Plotly graph
- **Nodes**: Each point = one brain dump
- **Colors**: Each color = a semantic cluster (similar ideas grouped together)
- **Layout**: UMAP dimensionality reduction for spatial meaning
- **Interactions**: Hover to see full text, zoom/pan to explore
- **Relationships**: Edges show semantic similarity between thoughts

**Bottom Section: Comprehensive Table**
- **Columns**: Brain Dump | Cluster Label | Created At
- **Sorting**: Most recent first (newest at top)
- **Format**: Plain text, easy to copy

**Actions**:
- Click "🔄 **Refresh Clusters**" to recalculate (when you add many new thoughts)
- Spinner animations show progress of embedding generation and labeling

### Feed Tab 📰 - Blog-Style Feed

**What You See**:
1. **Up to 5 Recent Cards** in reverse chronological order
2. **Each Card Contains**:
   - 💭 **Title**: Your full brain dump text
   - 🏷️ **Cluster Badge**: Category label (generated by LLM)
   - 📄 **Summary**: AI-generated context
     - If Tavily API set: Real web search + synthesis
     - If not: Mock contextual data (demo mode)
   - ❓ **Reflection Questions**: 5 Socratic questions (from QuestionAgent)

**Why Use Feed?**
- Quick morning review without table scrolling
- AI-generated insights help you think deeper
- Guided reflection via questions
- Patterns become visible across multiple cards
- Perfect for journaling workflow

---

## 🔄 Typical Workflows

### Workflow 1: Daily Brain Dumping + Review
```
Morning:
1. Open app at http://localhost:8501
2. Sidebar: Type 3-5 quick thoughts
3. Click "Add to Sanctuary" each time
4. Switch to Feed tab
5. Read summaries and reflect on questions
6. (Takes 5-10 minutes)

Evening:
1. Go to Home tab
2. Click "Refresh Clusters"
3. Observe how new thoughts clustered with existing ones
4. Spot emerging themes and connections
```

### Workflow 2: Deep Dive Analysis
```
1. Accumulate 20+ thoughts over several days
2. Go to Home tab
3. Click "Refresh Clusters"
4. Examine the visualization:
   - Which clusters are densest?
   - What themes emerge?
   - Any surprising connections?
5. Switch to Feed tab
6. Read the agent-generated insights
7. Use for next creative session
```

### Workflow 3: Topic-Specific Exploration
```
1. Add 10+ thoughts about a specific topic
2. They should cluster together automatically
3. Hover over the cluster in Home tab
4. Read the AI-generated cluster label
5. Check Feed tab for synthesis across related thoughts
6. Use the questions to drill deeper
```

---

## 🤖 Agent Capabilities

### QuestionAgent
**Role**: Socratic questioning for deeper reflection

**Input**: Brain dump text
**Output**: 5 open-ended questions

**Example**:
```
Brain Dump: "Why do I procrastinate on important tasks?"

Generated Questions:
1. What task are you procrastinating on right now?
2. What emotion arises when you think about starting it?
3. What would you do instead if you didn't procrastinate?
4. What is the smallest first step you could take?
5. What would happen if you started today?
```

### SearchAgent
**Role**: Web search + synthesis or mock data generation

**Input**: Brain dump text
**Output**: Relevant context or summary

**With Tavily API**:
- Searches the web for related information
- Synthesizes findings with LLM
- Provides citations and context

**Without Tavily API** (Demo Mode):
- Generates plausible contextual information
- Maintains consistency with the thought
- Allows testing without API key

### GenerationAgent
**Role**: Cross-thought synthesis

**Input**: Multiple related brain dumps (from same cluster)
**Output**: Synthesized insight combining themes

**Example**:
```
Related Dumps:
- "Neural networks learn patterns"
- "Our brains also learn from patterns"
- "What if consciousness is pattern recognition?"

Synthesis: These thoughts suggest that consciousness might emerge 
from the brain's pattern recognition capabilities...
```

### FeedAgent
**Role**: Orchestrates other agents for feed generation

**Process**:
1. Selects 5 most recent thoughts
2. For each thought:
   - Gets cluster assignment
   - Calls SearchAgent for summary
   - Calls QuestionAgent for reflection questions
3. Formats as blog-style cards

---

## ⚙️ Configuration & Customization

### Environment Variables (.env)

Create `.env` file in project root:

```bash
# ============ REQUIRED ============

# Neo4j Aura Connection
NEO4J_URI=neo4j+ssc://your-instance-id.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-super-secure-password

# Google Gemini LLM API
GOOGLE_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxx

# ============ OPTIONAL ============

# Tavily Web Search (for better summaries)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxx

# Perplexity API (alternative search)
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxx
```

### Optional: Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4ECDC4"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#161b22"
textColor = "#c9d1d9"

[client]
showErrorDetails = true

[logger]
level = "info"

[server]
port = 8501
```

---

## 🔧 Development & Debugging

### Running Locally
```bash
streamlit run app.py
```

### Clearing Database
```bash
python cleanup_db.py
```

### Neo4j Maintenance
```bash
python neo4j_maintenance.py
```

### Viewing Logs
```bash
# Terminal shows Streamlit logs
# Check sidebar for spinner status during refresh
```

### Common Issues

**Issue**: "Failed to connect to Neo4j"
- Verify `.env` has correct URI, username, password
- Check Neo4j Aura instance is running (console.neo4j.io)
- Firewall: Neo4j needs outbound HTTPS (port 7687)

**Issue**: "GOOGLE_API_KEY not found"
- Ensure key is in `.env` file
- Restart streamlit: `streamlit run app.py`

**Issue**: Clusters not showing
- Click "Refresh Clusters" in sidebar
- Wait for spinner to finish (10-30 seconds)
- Needs at least 2-3 thoughts for clustering

**Issue**: Tavily search not working
- Mock data is used if API key missing (expected)
- Optional feature; not required for core functionality

---

## 📊 Performance Notes

### Computational Complexity
- **Adding 1 thought**: ~1 second
- **Clustering N thoughts**: ~O(N) to O(N log N) depending on N
  - 10 thoughts: ~2 seconds
  - 100 thoughts: ~5-10 seconds
  - 1000 thoughts: ~30-60 seconds
- **LLM labeling**: ~2-5 seconds per cluster (depends on Gemini API latency)

### Storage
- Neo4j Free Tier: ~5 GB storage
- Typical thought: ~500 bytes
- Can store ~10 million thoughts theoretically

### Scaling Recommendations
- **Local testing**: ✅ Recommended
- **Shared team use**: Consider dedicated Neo4j instance
- **Large scale (10k+ thoughts)**: May need performance tuning (indexing, batching)

---

## 🎓 Learning Resources

### Concepts Explained

**Semantic Embeddings**
- sentence-transformers model converts text to 384-dimensional vectors
- Similar texts → similar vectors → can cluster together
- Distance in vector space ≈ semantic similarity

**HDBSCAN Clustering**
- Density-based clustering (unlike K-means which needs K)
- Automatically finds clusters of any shape
- Robust to outliers (marks them as "noise")

**Neo4j Graph Database**
- Stores relationships as first-class citizens
- Fast for relationship queries (unlike SQL)
- Perfect for knowledge graphs and recommendations

**LLMs for Labeling**
- Google Gemini generates cluster labels from examples
- LangChain chains handle prompt + LLM + parsing
- Enables semantic understanding of cluster themes

---

## 📝 License

See `LICENSE` file for details.

---

## 🤝 Contributing

Found a bug or have a feature idea? 

1. Check existing issues
2. Create new issue with clear description
3. (Optional) Submit PR with fix

---

## ❓ FAQ

**Q**: Can I use this with a local Neo4j instance?
**A**: Yes! Update `NEO4J_URI` to `neo4j://localhost:7687` in `.env`

**Q**: What if I don't have a Tavily API key?
**A**: That's fine! SearchAgent will generate mock data (demo mode)

**Q**: Can I export all my thoughts?
**A**: Use the table in Home tab or query Neo4j directly. Export feature coming soon.

**Q**: How often should I click "Refresh Clusters"?
**A**: After adding multiple new thoughts (5+), or when you want to see updated organization.

**Q**: Is my data secure?
**A**: Only you have your Neo4j password. Data is encrypted in transit and at rest on Neo4j Aura.

---

## 🚀 Roadmap

- [ ] Export functionality (CSV, JSON, markdown)
- [ ] Thought search & filtering
- [ ] Collaborative mode (multiple users)
- [ ] Email digest of weekly summaries
- [ ] Browser extension for quick capture
- [ ] Mobile app
- [ ] Advanced analytics dashboard

[client]
showErrorDetails = true
```

---

## 🔧 Database Management

### Re-calculate Clusters
1. Go to Home tab
2. Click 🔄 **Refresh** button
3. Wait for spinner to complete

### Clear Everything and Start Fresh
1. Click 🗑️ **Clear All Dumps** in sidebar
2. Confirm action
3. Add new thoughts

### Neo4j Maintenance
Use the `neo4j_maintenance.py` script:
```bash
# Show database statistics
python neo4j_maintenance.py stats

# Remove duplicate brain dumps
python neo4j_maintenance.py dedup

# Clear all data (WARNING: Irreversible)
python neo4j_maintenance.py clear
```

---

## 📊 Performance Tips

### Clustering
- **Fast**: 3-20 dumps (< 5 seconds)
- **Moderate**: 20-100 dumps (5-30 seconds)
- **Slow**: 100+ dumps (> 30 seconds)
- Use caching to avoid recomputing

### API Usage
- **Google Gemini**: Free tier allows ~60 requests/min
- **Tavily Search**: Free tier allows ~100 searches/month
- Set environment variables to enable features

### Neo4j Query Performance
- Dump and cluster lookups are indexed for fast retrieval
- Embedding vectors cached in graph nodes
- Knowledge graph relationships enable fast similarity searches

---

## 🐛 Troubleshooting

### Neo4j Connection Error
→ Verify your Aura instance is running: https://console.neo4j.io/
→ Check `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`
→ Ensure your IP is allowlisted in Aura instance settings

### "API Key not found" error
→ Make sure `.env` file exists with `GOOGLE_API_KEY`
→ Restart streamlit: `streamlit run app.py`

### Clustering takes forever
→ You might have 100+ dumps
→ Only happens on first run or "Refresh"
→ Consider batching cluster operations for large datasets

### Feed cards showing mock data
→ Your `TAVILY_API_KEY` isn't set
→ Add it to `.env` and restart

### "ModuleNotFoundError" for imports
→ Missing dependencies: `pip install -r requirements.txt`
→ Wrong Python version? Try `python3 -m pip install ...`

---

## 📚 Architecture Overview

```
braindump_core.py
├── BrainDumpDB (Neo4j Graph Database)
├── EmbeddingEngine (sentence-transformers)
├── ClusterEngine (HDBSCAN + UMAP)
└── Visualization (Plotly)

agents.py
├── QuestionAgent (Socratic questions)
└── SearchAgent (Web search + synthesis)

app.py
├── Sidebar (Input + Navigation)
├── Home Tab (Knowledge Graph + Table)
├── Feed Tab (Blog-style cards)
└── Helper Functions (Rendering)

neo4j_maintenance.py
├── Database statistics
├── Deduplication
└── Data cleanup
```

---

## 🎯 Next Steps

1. **Set up Neo4j Aura** instance at https://console.neo4j.io/
2. **Add your first thought** via sidebar
3. **Switch between Home and Feed** to see different views
4. **Check back daily** to build your idea garden
5. **Monitor performance** using `neo4j_maintenance.py stats`

---

## 💡 Tips

- **Best for**: Capturing fleeting thoughts, finding patterns, exploring ideas
- **Start small**: 5-10 thoughts to see clustering in action
- **Use specificity**: "Why do dreams fade?" works better than "dreams"
- **Review regularly**: Feed tab is great for morning/evening review
- **Share clusters**: Screenshot Home tab to share your thinking

---

## 📝 Notes

- All data stored locally in `braindump.db` (no cloud sync)
- Embeddings cached in database (fast retrieval)
- Cluster labels generated once and reused
- Each brain dump timestamped automatically

---

**Ready to catch your thoughts before they slip away!** 🧠✨

For issues or questions, check REFACTOR_SUMMARY.md or ARCHITECTURE.md
