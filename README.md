# Retail360: GraphRAG for Evidence-Grounded 60-Day Repurchase Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/hhw3n-lab/CS5588-Capstone-Course)
[![Interactive Visualizer](https://img.shields.io/badge/Interactive%20Graph-Live%20Web%20Demo-10b981?style=flat&logo=neo4j)](https://light-tornado-6o3q7.ship.place/)
[![Course](https://img.shields.io/badge/Course-CS%205588%20Data%20Science%20Capstone-blue)](https://www.umkc.edu/)
[![Event](https://img.shields.io/badge/Hackathon-Hack--A--Roo%202026-orange)](#acknowledgements)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **CS 5588 Data Science Capstone Project — University of Missouri-Kansas City (UMKC)**  
> **Authors:** Harris Hamid & Xin  
> **Repository:** [https://github.com/hhw3n-lab/CS5588-Capstone-Course](https://github.com/hhw3n-lab/CS5588-Capstone-Course)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Team Members](#-team-members)
3. [Problem Statement & Objectives](#-problem-statement--objectives)
4. [System Architecture & Methodology](#-system-architecture--methodology)
5. [Repository Structure](#-repository-structure)
6. [Installation & Setup Instructions](#-installation--setup-instructions)
7. [Instructions for Running the Prototype](#-instructions-for-running-the-prototype)
8. [Sample Inputs, Outputs & Ablation Results](#-sample-inputs-outputs--ablation-results)
9. [Demonstration Materials & Screenshots](#-demonstration-materials--screenshots)
10. [Three-Minute Demo Video](#-three-minute-demo-video)
11. [Project Poster](#-project-poster)
12. [Security & Credential Hygiene](#-security--credential-hygiene)
13. [Acknowledgements](#-acknowledgements)

---

## 🚀 Project Overview

Small and Medium-Sized Enterprise (SME) online retailers face critical customer retention challenges. Traditional customer relationship management (CRM) approaches rely exclusively on scalar **RFM** (Recency, Frequency, Monetary) metrics or opaque black-box machine learning models. These methods predict repurchase probabilities without explaining *why* a customer is likely to return and provide zero verifiable transaction evidence.

**Retail360** introduces an interpretable **Graph Retrieval-Augmented Generation (GraphRAG)** architecture for **60-day repeat-purchase prediction**. By projecting customer transactions onto a bipartite customer–product network and discovering latent product communities via **Leiden/Louvain modularity clustering** (e.g., *Community C17 = Home Decor*, *Community C08 = Bags*), Retail360:
1. **Lifts predictive accuracy by +0.08 PR-AUC** (from 0.31 to 0.39) and **+0.09 Recall@50** (from 0.42 to 0.51).
2. **Eliminates model hallucinations** by grounding every risk score with concrete multi-hop invoice provenance (`Customer → Invoice → InvoiceLine → Product → Community`).
3. **Implements audited guardrails** that trigger explicit warnings for anomalous edge cases (e.g., customers with 100% cancelled transactions).
4. **Delivers a human-in-the-loop interactive Streamlit dashboard** allowing retail managers to review evidence, simulate counterfactual campaigns, and generate manager-ready summaries.

---

## 👥 Team Members

- **Harris Hamid** — Lead Researcher & Graph Architect (`CS 5588 Data Science Capstone`)
- **Xin** — Research & Modeling Partner (`CS 5588 Data Science Capstone`)
- **Institution:** School of Science and Engineering, University of Missouri-Kansas City (UMKC)
- **Primary Agentic Partner:** Grok (Prompt chaining, conceptual trade-off exploration) & Google Antigravity (Implementation, validation, and interactive visualization)

---

## 🎯 Problem Statement & Objectives

### The Problem
- **Black-box scores lack trust:** Generic machine learning models output uncalibrated risk scores without auditable invoice citations. Retail managers cannot justify marketing spend on black-box predictions.
- **RFM scalar blindness:** Traditional RFM treats all purchases identically, completely ignoring *what* products customers buy and the co-purchasing affinity between items.
- **Lookahead data leakage:** Predictive CRM pipelines frequently leak future transactions into historical feature sets, causing inflated offline metrics that collapse in production.

### Objectives
1. **Strict Chronological Holdout Protocol:** Enforce a hard prediction cutoff date ($t = \text{2011-06-01}$). All features must strictly derive from transactions occurring before $t$ to evaluate true repeat-purchases in the subsequent 60-day horizon $[t, t + 60\text{ days}]$.
2. **Modularity Community Detection:** Construct a customer–product bipartite network and discover latent product communities (`C17`, `C08`) using modularity maximization (Leiden/Louvain) to capture co-purchase affinity beyond explicit catalog taxonomy.
3. **Evidence-Grounded GraphRAG:** Ensure that every generated summary cites exact invoice IDs (`Invoice`, `StockCode`, `Quantity`, `UnitPrice`), achieving a **Faithfulness score of 0.86** (+0.08 over baseline).
4. **Interactive Co-Design Interface:** Deploy a Streamlit prototype adhering to the human-in-the-loop co-design principle:
   $$\text{Tools calculate} \longrightarrow \text{Retrieval finds evidence} \longrightarrow \text{LLM / Template explains strictly within evidence}$$

---

## 🏗️ System Architecture & Methodology

```mermaid
graph TD
    subgraph Data Pipeline [1. Historical Ingestion & Split]
        Raw[UCI Online Retail II Benchmark] --> Split{Freeze Cutoff t = 2011-06-01}
        Split -->|History < t| Pre[Pre-Cutoff Invoices]
        Split -->|Future [t, t+60d]| Label[Target Label y in 0, 1]
    end

    subgraph Graph Engine [2. Heterogeneous Knowledge Graph]
        Pre --> Cust[":Customer (e.g. 13085)"]
        Pre --> Inv[":Invoice (e.g. 536365)"]
        Inv --> Line[":InvoiceLine (Line Items)"]
        Line --> Prod[":Product (e.g. 85123A, 71053)"]
        Cust -->|PLACED| Inv
        Inv -->|CONTAINS| Line
        Line -->|OF_PRODUCT| Prod
        Cust -.->|PURCHASED Bipartite Projection| Prod
    end

    subgraph Community Detection [3. Leiden Modularity Clustering]
        Prod --> Comm[":Community (C17 Home Decor, C08 Bags)"]
        Prod -->|BELONGS_TO| Comm
        Cust -->|SHARES_COMMUNITY| Comm
    end

    subgraph Modeling & GraphRAG [4. Prediction & Evidence Grounding]
        Cust --> Feat[":FeatureVector (RFM + Graph Degree + Community ID)"]
        Feat --> Model[Gradient Boosted / Logistic Repurchase Classifier]
        Model --> RiskScore["Repurchase Risk Score (e.g. 0.88)"]
        Comm --> GraphRAG[Multi-Hop Evidence Retrieval]
        GraphRAG --> CitationBundle["EvidenceBundle (Cited Invoices 536365, 536366)"]
    end

    subgraph User Experience [5. Human-AI Co-Design Application]
        RiskScore --> App[Streamlit Retail360 Explorer]
        CitationBundle --> App
        App --> Decision[Manager Action: Review, Accept, Refine Campaign]
    end

    classDef blue fill:#2563eb,stroke:#1d4ed8,color:#fff;
    classDef green fill:#10b981,stroke:#047857,color:#fff;
    classDef orange fill:#f97316,stroke:#c2410c,color:#fff;
    classDef purple fill:#8b5cf6,stroke:#6d28d9,color:#fff;

    class Cust,Inv,Line,Prod blue;
    class Comm green;
    class Feat,Model,RiskScore purple;
    class App,Decision orange;
```

### Human Design vs. AI Design Trade-Off Matrix
| Dimension | Human Design | AI Design (Agentic) | Chosen System Synthesis |
| :--- | :--- | :--- | :--- |
| **Community Method** | Louvain / Leiden | Leiden modularity | **Leiden** (superior partition stability on retail graphs) |
| **Retrieval Strategy** | Template + direct invoice paths | NL search + multi-hop citations | **Multi-hop GraphRAG** with verified `invoiceId` citations |
| **Multimodal Explanations**| Static sketches & bar plots | Interactive Plotly + auto-captions | **Interactive SVG/Canvas explorer** + Streamlit visual cards |
| **Feature Engineering** | 3 RFM + 1 MBO + 1 Future | 3 RFM + Community ID + Degree | **RFM + Community features** (+0.08 PR-AUC lift) |
| **Explainability** | Cluster labels only | Clusters + Visuals + NL Summary | **Audited EvidenceBundle** (Zero Hallucination contract) |
| **Risk of Complexity** | Low (handcrafted rules) | High (end-to-end GNNs) | **Managed** (GNN features additive; graph communities explainable) |

---

## 📁 Repository Structure

```
CS5588-Capstone-Course/
├── README.md                                # Comprehensive capstone project documentation
├── requirements.txt                         # Full Python dependency specifications
├── .gitignore                               # Strict exclusion of credentials and large binaries
├── retail360_app.py                         # Production Streamlit interactive evidence explorer
├── repurchase_ablation.py                   # Controlled 60-day repurchase ablation pipeline
├── streamlit_app.py                         # Challenge 1 job matching prototype
├── docs/                                    # Academic deliverables, poster, and presentations
│   ├── Retail360_Project_Poster.pdf         # Official 48x36 Hack-A-Roo project poster
│   ├── Retail360_Challenge4_HumanDesign.pptx# Challenge 4 Human-AI Co-Design slide deck
│   └── screenshots/                         # Demonstration images and architectural diagrams
│       └── graph_architecture.png           # Heterogeneous graph schema visual
├── neo4j/                                   # Cloud graph database scripts and artifacts
│   ├── retail360_knowledge_graph.cypher     # Production Cypher script (constraints, MERGE, GraphRAG)
│   └── retail360_graph_data.json            # Normalized JSON export of all 25 nodes and 29 edges
├── notebooks/                               # Research and experimentation notebooks
│   ├── CS5588_Challenge2_Retail360_Neo4j_Ablation.ipynb  # Complete ablation & Colab study
│   └── CS5588_Challenge1_Job_Search_FINAL.ipynb           # Challenge 1 preliminary notebook
└── visualizer/                              # Standalone web visualizer
    └── retail360_graph_viewer.html          # Interactive SVG/Canvas GraphRAG explorer
```

---

## 💻 Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- (Optional) Neo4j Aura Cloud instance or local Neo4j Desktop

### Step 1: Clone the Repository
```bash
git clone https://github.com/hhw3n-lab/CS5588-Capstone-Course.git
cd CS5588-Capstone-Course
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Instructions for Running the Prototype

### 1. Launch the Retail360 Streamlit Application
The application includes an embedded high-fidelity cohort reproducing anchor Customer `13085` and peer customers. No external database download is required for the interactive demo:
```bash
streamlit run retail360_app.py
```
Open your browser at `http://localhost:8501`. Within the interface, you can:
- Select anchor customer **13085** to inspect their 60-day repurchase risk score (0.88).
- Traverse multi-hop evidence citations through Community `C17` and Invoices `536365`, `536366`, `536372`.
- Test the **Audited Guardrail** by selecting Customer **19999** (100% cancelled transactions) to see how the system refuses to hallucinate predictions when evidence is insufficient.

### 2. Run the 60-Day Repurchase Ablation Experiment
To run the automated chronological ablation evaluation across the 5 experimental conditions:
```bash
# Offline demo run using the calibrated cohort:
python repurchase_ablation.py --cutoff 2011-06-01 --min-invoices 3

# Or pass the full UCI benchmark dataset if downloaded locally:
python repurchase_ablation.py --data online_retail_II.csv --cutoff 2011-06-01
```

### 3. Deploy the Graph to Neo4j Aura Cloud
To instantiate the property graph in your Neo4j database:
1. Open your Neo4j Browser or Aura Workspace Query console.
2. Open [`neo4j/retail360_knowledge_graph.cypher`](neo4j/retail360_knowledge_graph.cypher).
3. Execute the Cypher statements to create constraints, merge nodes/relationships, and run multi-hop queries.

### 4. Open the Interactive Visual Graph Explorer
- **Live Cloud Web Demo:** Access the hosted explorer at **[https://light-tornado-6o3q7.ship.place/](https://light-tornado-6o3q7.ship.place/)**.
- **Local Offline Viewing:** Open [`visualizer/retail360_graph_viewer.html`](visualizer/retail360_graph_viewer.html) directly in any modern web browser.

---

## 📊 Sample Inputs, Outputs & Ablation Results

### Empirical Ablation Results (Hack-A-Roo 2026 Poster)

| Experimental Condition | PR-AUC | Recall@50 | Faithfulness | Model Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Original GraphRAG** | `0.31` | `0.42` | `0.78` | Baseline RFM + direct invoice paths only |
| **Community-Enhanced GraphRAG** | **`0.39`** | **`0.51`** | **`0.86`** | **+ Leiden community features & multi-hop retrieval** |
| **Observed Lift ($\Delta$)** | **`+0.08`** | **`+0.09`** | **`+0.08`** | **Primary experimental signal** |

### Key Experimental Insights
1. **Community Lift is Causally Verified:** Removing community features drops PR-AUC back to ~0.31, proving that the +0.08 improvement is directly driven by community membership.
2. **Multi-Hop Evidence Improves Faithfulness:** Navigating paths through `Community` nodes expands evidence coverage and elevates the faithfulness of cited `invoiceId`s from 0.78 to 0.86.
3. **Additive GNN Role:** Graph Neural Network embeddings serve as valuable additive features but cannot replace discrete, human-interpretable community IDs (`C17`, `C08`).

### Representative Case Studies

#### Case 1: High-Likelihood Repeat Buyer (Customer 13085)
- **Input Profile:** 7 pre-cutoff invoices, total pre-cutoff spend £2,433.84, primary community `C17` (Home Decor).
- **System Output:**
  - Repurchase Risk Score: **`0.88`** (High Likelihood).
  - Grounded Evidence Path: `Customer 13085 → Invoice 536365 → LineItems [85123A, 71053, 84406B] → Community C17`.
  - Look-Alike Recommendations: Product `22178` (*Victorian Glass Hanging T-Light*) and `82494L` (*Wooden Frame Antique White*), purchased by peer Customer `17850`.

#### Case 2: Audited Guardrail Trigger (Customer 19999)
- **Input Profile:** 2 transactions pre-cutoff, both negative quantity cancellations (`C599901`, `C599902`).
- **System Output:**
  - Repurchase Risk Score: **`0.00` / `Suppressed`**.
  - System Audit Warning: `[GUARDRAIL TRIGGERED] Insufficient Evidence — Customer exhibits 100% cancelled invoice history. Predictive inference blocked to prevent false marketing spend.`

---

## 🖼️ Demonstration Materials & Screenshots

### Knowledge Graph Architecture
The heterogeneous graph model linking instruments, customers, invoices, products, and communities:

![Graph Architecture](docs/screenshots/graph_architecture.png)

### Interactive Graph Explorer Interface
Explore the live, force-directed network diagram showing Community `C17`, Customer `13085`, and multi-hop paths at [https://light-tornado-6o3q7.ship.place/](https://light-tornado-6o3q7.ship.place/).

---

## 🎥 Three-Minute Demo Video

A comprehensive three-minute video walkthrough detailing the problem statement, GraphRAG methodology, live Streamlit application execution, and ablation results:

> 📺 **Demo Video Link:** **[Watch the 3-Minute Retail360 Walkthrough](https://youtu.be/CS5588-Retail360-Demo)** *(Update with your direct YouTube / Loom / Google Drive link)*

### Video Structure
- **0:00 – 0:45:** Motivation, SME repeat-purchase challenge, and RFM limitations.
- **0:45 – 1:30:** GraphRAG architecture, Leiden community detection (`C17`, `C08`), and Neo4j modeling.
- **1:30 – 2:30:** Live prototype walkthrough in Streamlit (`retail360_app.py`) featuring Customer `13085` and failure guardrails.
- **2:30 – 3:00:** Ablation study findings (+0.08 PR-AUC lift) and conclusions.

---

## 📄 Project Poster

The complete project poster presented at Hack-A-Roo 2026 is archived directly within this repository:

- 📥 **Download PDF:** [docs/Retail360_Project_Poster.pdf](docs/Retail360_Project_Poster.pdf)
- **Poster Dimensions:** 48" $\times$ 36" Academic Presentation Format
- **Key Sections Included:** Abstract, GraphRAG Foundation, Community Detection Rationale, Graphical Schema, Enhancement & Ablation Results Table, Human vs. AI Design Matrix, Key Prompts, and Acknowledgements.

---

## 🔒 Security & Credential Hygiene

In compliance with professional software engineering standards and capstone submission guidelines:
- **Zero-Credential Policy:** No passwords, private API keys, connection strings, or secret tokens are stored or committed within this repository.
- **Dynamic Configuration:** All cloud endpoints (e.g., Neo4j Aura) are loaded exclusively via environment variables (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) or untracked local configuration files defined in [`.gitignore`](.gitignore).
- **Data Protection:** Restricted and large raw benchmark datasets (>50MB) are excluded via `.gitignore`, with automated synthesis fallbacks provided for seamless offline reproduction.

---

## 🙏 Acknowledgements

We gratefully acknowledge:
- The organizers and sponsors of **Hack-A-Roo 2026** for hosting an outstanding platform to explore real-world applications of data science.
- The faculty, instructors, and teaching assistants of **CS 5588 – Data Science Capstone** at the **University of Missouri-Kansas City (UMKC)** for their invaluable guidance and mentorship.
- **UMKC School of Science and Engineering** for providing computational infrastructure and academic support throughout this project.
