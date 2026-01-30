# Search Types Documentation - Sequential Guide

This document describes the three search strategies used in the Agent's hybrid GraphRAG retrieval system, explained sequentially from initiation through execution.

---

## Overview: Hybrid GraphRAG Retrieval Pipeline

The Agent performs search through a sequential three-stage pipeline:

```
User Query
    ↓
[STAGE 1] Keyword Search + Vector/Semantic Search (Parallel Execution)
    ↓
[STAGE 2] Graph Expansion (via Neo4j Relationships)
    ↓
[STAGE 3] Re-Ranking & Result Return
```

All three search types work **together sequentially** in every query.

---

## Search Type 1: Keyword Search

### Purpose

Extract and match exact terms from the query against indexed keywords stored on document chunks.

### Sequential Process

**Step 1: Keyword Extraction**

```python
# If keywords not provided, extract from query automatically
query = "What is Alexis Torres' Python experience?"
# Auto-extracted: ["Alexis", "Torres", "Python"]

# OR use explicitly provided keywords (e.g., from person name extraction)
person_names = extract_person_names(query)  # ["Alexis", "Torres"]
keywords = person_names
```

**Step 2: Keyword Matching in Neo4j**

```
For each Chunk node in database:
  1. Access chunk.keywords property (stored as list)
  2. Compare against search keywords
  3. Apply match type logic:
     - "any" (OR): Chunk matches if ANY keyword appears
     - "all" (AND): Chunk matches if ALL keywords appear
  4. Assign keyword_score (0.0 to 1.0)
```

**Step 3: Scoring**

- Chunks matching keywords receive a `keyword_score`
- Formula: `keyword_score = (keywords_matched / total_keywords) * 1.0`
- Default match type: `"any"` (broader results)

### Configuration Parameters

```python
graphrag_search(
    query="...",
    use_keyword_search=True,        # Enable/disable keyword matching
    keywords=None,                   # Auto-extract if None
    keyword_match_type="any",        # "any" (OR) or "all" (AND)
    keyword_boost=0.3                # Weight 0.0-1.0 in hybrid score
)
```

### Example Sequence

```
Query: "Show me Alexis Torres' cloud projects"
↓
Keywords extracted: ["Alexis", "Torres", "cloud", "projects"]
↓
Database search:
  - Chunk A: has ["Alexis", "Torres", "AWS", "cloud"] → matches 3/4 keywords
  - Chunk B: has ["cloud", "DevOps", "deployment"] → matches 1/4 keywords
  - Chunk C: has ["Alexis", "resume", "skills"] → matches 1/4 keywords
↓
Results ranked by keyword score:
  1. Chunk A: keyword_score = 0.75
  2. Chunk B: keyword_score = 0.25
  3. Chunk C: keyword_score = 0.25
```

---

## Search Type 2: Semantic Search (Vector Similarity)

### Purpose

Understand the _meaning_ and _context_ of the query by comparing embeddings. Finds semantically similar chunks even if keywords don't match exactly.

### Sequential Process

**Step 1: Query Embedding**

```python
# Convert query text to 1536-dimensional vector
query = "What are their cloud platform experiences?"
query_embedding = embed_query(query)  # 1536-dim vector

# Using Azure OpenAI text-embedding-3-small
# Configuration from .env:
# EMBEDDING_MODE=foundry
# AZURE_OPENAI_EMBED_DEPLOYMENT_NAME=text-embedding-3-small
# EMBEDDING_VECTOR_DIM=1536
```

**Step 2: Vector Similarity Search in Neo4j**

```cypher
MATCH (c:Chunk)
WHERE c.embedding IS NOT NULL
WITH c,
     gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
WHERE similarity >= $similarity_threshold
RETURN c
ORDER BY similarity DESC
LIMIT $top_k
```

**Step 3: Similarity Calculation**

- Metric: Cosine similarity (0.0 to 1.0)
- Formula: `similarity = dot_product(query_vec, chunk_vec) / (||query_vec|| * ||chunk_vec||)`
- Threshold filtering: Only chunks with `similarity >= 0.7` (configurable)

**Step 4: Score Assignment**

- Each chunk receives a `vector_similarity_score` (0.0-1.0)
- Higher score = more semantically similar to query

### Configuration Parameters

```python
graphrag_search(
    query="...",
    top_k=10,                           # Return top 10 chunks
    similarity_threshold=0.7,           # Minimum similarity 0.0-1.0
    initial_fetch_limit=200             # Pre-filter limit before ranking
)
```

### Example Sequence

```
Query: "What are their cloud platform experiences?"
↓
Embedding: query_embedding = [0.234, -0.156, 0.892, ..., 0.445]  (1536 dims)
↓
Vector search against all chunks:
  - Chunk A (AWS infrastructure): similarity = 0.89
  - Chunk B (Azure DevOps): similarity = 0.87
  - Chunk C (GCP migration): similarity = 0.85
  - Chunk D (Python projects): similarity = 0.45 → filtered out (< 0.7)
  - Chunk E (Docker container): similarity = 0.72
↓
Results ranked by similarity:
  1. Chunk A: 0.89
  2. Chunk B: 0.87
  3. Chunk C: 0.85
  4. Chunk E: 0.72
```

---

## Search Type 3: Graph Search (Relationship Expansion)

### Purpose

Starting from seed chunks (found via keyword + semantic search), traverse Neo4j relationships to discover connected context. Expands search results through document structure and semantic relationships.

### Sequential Process

**Step 1: Seed Chunk Identification**

```python
# Results from Stage 1 (keyword + semantic) become seeds
seed_chunks = [
    {id: chunk_1, text: "AWS infrastructure...", similarity: 0.89},
    {id: chunk_2, text: "Azure DevOps...", similarity: 0.87},
    {id: chunk_3, text: "GCP migration...", similarity: 0.85},
]
```

**Step 2: 1-Hop Graph Expansion**

```cypher
# From each seed chunk, find connected chunks
MATCH (seed:Chunk)-[r]-(neighbor:Chunk)
WHERE r.type IN ['SEMANTICALLY_SIMILAR', 'NEXT_CHUNK']
  AND (r.similarity >= $similarity_threshold OR r.type = 'NEXT_CHUNK')
RETURN neighbor, r.type as relationship_type, r.similarity
```

**Relationship Types**:

- `SEMANTICALLY_SIMILAR`: Chunks with high embedding similarity (edges created during ingestion)
- `NEXT_CHUNK`: Sequential chunks from same document (preserve document flow)

**Step 3: Multi-Hop Expansion** (optional, 2-hop for comprehensive context)

```cypher
# From 1-hop neighbors, find 2-hop neighbors
MATCH (seed:Chunk)-[*1..2]-(neighbor:Chunk)
WHERE depth(path) <= 2
RETURN neighbor, path_score
```

**Step 4: Path Scoring**

```python
# Score decreases with each hop (closer = more relevant)
hop_0_score = 1.0    # Direct seed match
hop_1_score = 0.8    # One relationship away
hop_2_score = 0.6    # Two relationships away
```

**Step 5: Result Expansion**

```python
expanded_results = seed_chunks + hop_1_neighbors + hop_2_neighbors
# Remove duplicates, apply expansion limits
# semantic_expansion_limit = 100 (max from SEMANTICALLY_SIMILAR)
# hierarchical_expansion_limit = 50 (max from NEXT_CHUNK)
```

### Configuration Parameters

```python
graphrag_search(
    query="...",
    max_hops=1,                             # 1 or 2 hops (Agent default: 1)
    similarity_threshold=0.7,               # Min similarity for SEMANTICALLY_SIMILAR
    semantic_expansion_limit=100,           # Max chunks from semantic edges
    hierarchical_expansion_limit=50         # Max chunks from sequential edges
)
```

### Example Sequence

```
Seed Chunks from Stage 1:
  - Chunk A (AWS infrastructure)
  - Chunk B (Azure DevOps)
  - Chunk C (GCP migration)

↓ [EXPAND 1-HOP]

From Chunk A:
  - Via NEXT_CHUNK → Chunk A1 (AWS details continued)
  - Via NEXT_CHUNK → Chunk A2 (AWS performance metrics)
  - Via SEMANTICALLY_SIMILAR → Chunk X (Cloud infrastructure patterns)

From Chunk B:
  - Via NEXT_CHUNK → Chunk B1 (Azure setup)
  - Via SEMANTICALLY_SIMILAR → Chunk Y (DevOps best practices)

From Chunk C:
  - Via NEXT_CHUNK → Chunk C1 (GCP configuration)

↓ [ALL EXPANDED RESULTS]

Total chunks now: 3 (seeds) + 7 (1-hop) = 10 chunks
```

### Graph Structure in Neo4j

The Neo4j knowledge graph is organized as a document-centric structure with both chunk relationships and entity relationships:

```
Document Node
    ├── file_name: "Resume.pdf"
    ├── file_id: "uuid_123"
    ├── file_path: "/uploads/resumes/Resume.pdf"
    └── directory_name: "resumes"
         │
         ├─── [HAS_CHUNK] ──→ Chunk[0]
         │                       ├── text: "Alexis Torres works at TechCorp..."
         │                       ├── embedding: [1536 dims]
         │                       ├── keywords: ["Alexis", "Torres", "TechCorp", "experience"]
         │                       ├── chunk_index: 0
         │                       ├── [NEXT_CHUNK] ──→ Chunk[1]
         │                       │
         │                       ├── [MENTIONS] ──→ Entity(Person: Alexis Torres)
         │                       │                   ├── id: "entity:Person:Alexis_Torres"
         │                       │                   ├── name: "Alexis Torres"
         │                       │                   ├── type: "Person"
         │                       │                   ├── [WORKS_FOR] ──→ Entity(Org: TechCorp)
         │                       │                   │                    ├── id: "entity:Organization:TechCorp"
         │                       │                   │                    ├── name: "TechCorp"
         │                       │                   │                    ├── type: "Organization"
         │                       │                   │                    └── [LOCATED_IN] ──→ Entity(Location: San Francisco)
         │                       │                   │
         │                       │                   ├── [WORKED_WITH] ──→ Entity(Person: Sarah Chen)
         │                       │                   │                      └── [WORKS_FOR] ──→ Entity(Org: TechCorp)
         │                       │                   │
         │                       │                   └── [EXPERIENCED_WITH] ──→ Entity(Tech: Python)
         │                       │                                               ├── id: "entity:Technology:Python"
         │                       │                                               ├── name: "Python"
         │                       │                                               └── type: "Technology"
         │                       │
         │                       └── [MENTIONS] ──→ Entity(Org: TechCorp)
         │                                          └── [LOCATED_IN] ──→ Entity(Location: San Francisco)
         │
         ├─── [HAS_CHUNK] ──→ Chunk[1]
         │                       ├── text: "Python and AWS projects include..."
         │                       ├── embedding: [1536 dims]
         │                       ├── keywords: ["Python", "AWS", "projects", "backend"]
         │                       ├── chunk_index: 1
         │                       ├── [PREV_CHUNK] ──→ Chunk[0]
         │                       ├── [NEXT_CHUNK] ──→ Chunk[2]
         │                       ├── [SEMANTICALLY_SIMILAR] ──→ Chunk[X] (from another doc)
         │                       │                              └── similarity: 0.85
         │                       │
         │                       ├── [MENTIONS] ──→ Entity(Tech: Python)
         │                       │                   └── [MENTIONS_BY] ← Chunk[0], Chunk[1], Chunk[2]
         │                       │
         │                       ├── [MENTIONS] ──→ Entity(Tech: AWS)
         │                       │                   ├── id: "entity:Technology:AWS"
         │                       │                   ├── name: "AWS"
         │                       │                   ├── type: "Technology"
         │                       │                   └── [EXPERIENCED_WITH] ← Entity(Person: Alexis Torres)
         │                       │
         │                       └── [MENTIONS] ──→ Entity(Person: Alexis Torres)
         │
         └─── [HAS_CHUNK] ──→ Chunk[2]
                              ├── text: "Led AWS migration with Sarah Chen..."
                              ├── keywords: ["AWS", "migration", "cloud", "Sarah", "Chen"]
                              ├── [PREV_CHUNK] ──→ Chunk[1]
                              │
                              ├── [MENTIONS] ──→ Entity(Tech: AWS)
                              │
                              ├── [MENTIONS] ──→ Entity(Person: Sarah Chen)
                              │                   ├── [WORKS_FOR] ──→ Entity(Org: TechCorp)
                              │                   ├── [WORKED_WITH] ──→ Entity(Person: Alexis Torres)
                              │                   └── [COLLABORATED_WITH] ──→ Entity(Person: Alexis Torres)
                              │                                               └── project: "AWS migration"
                              │
                              └── [MENTIONS] ──→ Entity(Org: TechCorp)
```

**Key Entity Relationships Shown**:

1. **WORKS_FOR**: Alexis Torres → TechCorp, Sarah Chen → TechCorp
2. **WORKED_WITH / COLLABORATED_WITH**: Alexis Torres ↔ Sarah Chen
3. **EXPERIENCED_WITH**: Alexis Torres → Python, Alexis Torres → AWS
4. **LOCATED_IN**: TechCorp → San Francisco
5. **MENTIONS**: Chunks → All mentioned entities
6. **MENTIONS_BY**: Reverse relationship showing which chunks mention an entity

This visualization now shows:

- **Complete entity network**: Person → Organization → Location
- **Skill relationships**: Person → Technology via EXPERIENCED_WITH
- **Collaboration links**: Person ↔ Person via WORKED_WITH/COLLABORATED_WITH
- **Multi-directional relationships**: Both MENTIONS (chunk→entity) and MENTIONS_BY (entity→chunks)
- **Cross-document entity linking**: Same entity (AWS, TechCorp) mentioned across multiple chunks
- **Hierarchical structure**: Document → Chunks → Entities → Entity-to-Entity relationships

### Relationship Types in Detail

#### 1. MENTIONS

**Purpose**: Connect Chunk nodes to Entity nodes they mention/reference

**Creation**: Created during entity extraction phase (LLMGraphTransformer)

- NER identifies entities (people, organizations, technologies, locations)
- Each entity mention in a chunk creates a MENTIONS relationship
- Multiple chunks can mention the same entity
- Same entity mentioned multiple times = multiple MENTIONS edges (one per mention location)

**Properties**:

```python
{
  "extraction_source": "llmgraph",        # How entity was extracted
  "confidence": 0.92,                     # Confidence score for mention
  "mention_context": "working with",      # Context of the mention
  "createdAt": datetime()
}
```

**Example**:

```
Chunk[0]: "Alexis Torres worked at TechCorp"
  ├── [MENTIONS] → Entity(Person: Alexis_Torres) confidence: 0.98
  └── [MENTIONS] → Entity(Org: TechCorp) confidence: 0.95

Chunk[1]: "Using AWS and Docker for infrastructure"
  ├── [MENTIONS] → Entity(Tech: AWS) confidence: 0.99
  └── [MENTIONS] → Entity(Tech: Docker) confidence: 0.97

Chunk[2]: "Collaborated with Sarah Chen on AWS migration"
  ├── [MENTIONS] → Entity(Person: Sarah_Chen) confidence: 0.96
  └── [MENTIONS] → Entity(Tech: AWS) confidence: 0.99 (second mention!)
```

**Why it matters**:

- Links textual content (chunks) to structured knowledge (entities)
- Enables entity-centric search starting from documents
- Tracks entity mentions across document (how many times mentioned?)
- Provides confidence scores for extraction accuracy

**Query Example**:

```cypher
# Find all chunks mentioning "Alexis Torres"
MATCH (e:Entity {name: "Alexis Torres"})
MATCH (c:Chunk)-[r:MENTIONS]->(e)
RETURN c.text, r.confidence
ORDER BY r.confidence DESC
```

---

#### 2. SEMANTICALLY_SIMILAR

**Purpose**: Connect chunks with high embedding similarity across _different_ documents

**Creation**: Built during document ingestion

- Compares embeddings of all chunks
- Creates edge if cosine similarity > threshold (typically 0.7-0.8)
- Edge weight = similarity score (0.0-1.0)

**Example**:

```
Chunk from Resume.pdf:
  "AWS infrastructure design and deployment"

Connected via SEMANTICALLY_SIMILAR to:
  - Chunk from CloudGuide.pdf: "AWS best practices" (similarity: 0.88)
  - Chunk from DevOps.pdf: "Infrastructure as Code" (similarity: 0.81)
  - Chunk from Python.pdf: "Backend services architecture" (similarity: 0.75)
```

**Why it matters**:

- Finds conceptually related content across documents
- Expands search beyond exact matches
- Discovers alternative perspectives on the same topic

#### 2. NEXT_CHUNK / PREV_CHUNK

**Purpose**: Link sequential chunks within the _same document_

**Creation**: Automatically created during document chunking

- Chunk[0] --[NEXT_CHUNK]--> Chunk[1] --[NEXT_CHUNK]--> Chunk[2]
- Preserves document flow and context
- No weight/similarity score (always valid sequential links)

**Example**:

```
Document: "Alexis_Torres_Resume.pdf"

Chunk[0]: "Professional Summary: Alexis Torres is a software engineer..."
   ↓ [NEXT_CHUNK]
Chunk[1]: "...with 8 years of experience in Python and cloud infrastructure..."
   ↓ [NEXT_CHUNK]
Chunk[2]: "Skills: Python, AWS, Docker, Kubernetes, Azure, etc."
   ↓ [NEXT_CHUNK]
Chunk[3]: "Experience: Senior Engineer at TechCorp (2020-2024)..."
   ↓ [NEXT_CHUNK]
Chunk[4]: "...Led team of 5 engineers. Managed AWS migration project..."
```

**Why it matters**:

- Provides **document context** and flow
- Ensures related content isn't fragmented
- Sequential reading finds information split across chunks
- Prevents breaking semantic meaning at chunk boundaries

#### 3. HAS_CHUNK

**Purpose**: Links Document nodes to their constituent Chunk nodes

**Direction**: Document --[HAS_CHUNK]--> Chunk

**Example**:

```
Document(file_id="doc_1", file_name="Resume.pdf")
   --[HAS_CHUNK]→ Chunk[0]
   --[HAS_CHUNK]→ Chunk[1]
   --[HAS_CHUNK]→ Chunk[2]
   --[HAS_CHUNK]→ Chunk[3]
```

#### 3. HAS_CHUNK

**Purpose**: Links Document nodes to their constituent Chunk nodes

**Direction**: Document --[HAS_CHUNK]--> Chunk

**Example**:

```
Document(file_id="doc_1", file_name="Resume.pdf")
   --[HAS_CHUNK]→ Chunk[0]
   --[HAS_CHUNK]→ Chunk[1]
   --[HAS_CHUNK]→ Chunk[2]
   --[HAS_CHUNK]→ Chunk[3]
```

**Why it matters**:

- Enables document-level filtering and scoping
- Preserves file metadata with chunks
- Allows directory/machine scoping of searches

---

## Entity Relationship Creation During Document Ingestion

During the document processing phase, the system extracts not just chunks, but also **named entities** (people, organizations, locations) and creates relationships between them. This enables advanced entity-based graph traversal and discovery.

### Entity Extraction Process

**Step 1: Named Entity Recognition (NER)**

When a document is uploaded and processed, an LLM-based entity extraction runs:

```
Input Document: "Alexis Torres worked at TechCorp from 2020-2024.
                  He led the AWS migration with Sarah Chen."

Extracted Entities:
  - Person: Alexis Torres
  - Organization: TechCorp
  - Technology: AWS
  - Person: Sarah Chen
  - Time Period: 2020-2024
```

**Step 2: Entity Node Creation**

Each unique entity becomes a Neo4j Entity node:

```cypher
CREATE (e:Entity {
  id: "entity:Person:Alexis_Torres",
  name: "Alexis Torres",
  type: "Person",
  extraction_mode: "graphrag",
  createdAt: datetime()
})
```

**Step 3: Relationship Extraction Between Entities**

The LLM also extracts relationships between entities:

```
Entity Relationships Found:
  - Alexis Torres --[WORKED_FOR]--> TechCorp (2020-2024)
  - Alexis Torres --[WORKED_WITH]--> Sarah Chen
  - Sarah Chen --[WORKED_FOR]--> TechCorp
  - Alexis Torres --[EXPERIENCED_WITH]--> AWS
```

### Entity Relationships Created

The system creates multiple types of entity-to-entity relationships:

#### 1. WORK_FOR / WORKS_FOR

**Purpose**: Link Person entities to Organization entities they work/worked for

**Properties**:

```python
{
  "start_date": "2020",
  "end_date": "2024",
  "position": "Senior Engineer",
  "description": "Worked as Senior Engineer",
  "extraction_source": "llmgraph",
  "confidence": 0.92
}
```

**Example**:

```
Person(Alexis Torres) --[WORKS_FOR]→ Organization(TechCorp)
  ├── start_date: "2020"
  ├── end_date: "2024"
  ├── position: "Senior Engineer"
  └── confidence: 0.92
```

**Impact on Graph Expansion**:

- When searching for "TechCorp", graph expansion discovers all WORKS_FOR relationships
- Finds all people who worked at TechCorp
- Creates connections between colleagues who worked at same organization

#### 2. WORKED_WITH / COLLABORATED_WITH

**Purpose**: Link Person entities who worked together on projects

**Properties**:

```python
{
  "project": "AWS migration",
  "role_A": "Project Lead",
  "role_B": "Team Member",
  "collaboration_type": "direct",
  "description": "Collaborated on AWS migration project",
  "extraction_source": "llmgraph"
}
```

**Example**:

```
Person(Alexis Torres) --[WORKED_WITH]→ Person(Sarah Chen)
  ├── project: "AWS migration"
  ├── role_A: "Project Lead"
  ├── role_B: "Team Member"
  └── collaboration_type: "direct"
```

**Impact on Graph Expansion**:

- When searching for "Alexis Torres projects", discovers Sarah Chen as collaborator
- Retrieves information about Sarah's perspective on shared projects
- Creates multi-person narrative around initiatives

#### 3. EXPERIENCED_WITH / PROFICIENT_IN

**Purpose**: Link Person entities to Technology entities they have experience with

**Properties**:

```python
{
  "proficiency_level": "expert",
  "years_of_experience": "8",
  "context": "Cloud infrastructure and deployment",
  "extraction_source": "llmgraph"
}
```

**Example**:

```
Person(Alexis Torres) --[EXPERIENCED_WITH]→ Technology(AWS)
  ├── proficiency_level: "expert"
  ├── years_of_experience: "8"
  ├── context: "Cloud infrastructure"
  └── extraction_source: "llmgraph"
```

**Impact on Graph Expansion**:

- When searching for "AWS experts", discovers all people EXPERIENCED_WITH AWS
- Connects skill-based queries to relevant people
- Enables technology-driven discovery

#### 4. LOCATED_IN / BASED_IN

**Purpose**: Link Person/Organization entities to Location entities

**Properties**:

```python
{
  "location_type": "primary_office",
  "start_date": optional,
  "end_date": optional,
  "extraction_source": "llmgraph"
}
```

**Example**:

```
Organization(TechCorp) --[LOCATED_IN]→ Location(San Francisco, CA)
Person(Alexis Torres) --[BASED_IN]→ Location(San Francisco, CA)
```

#### 5. FOUNDED / LEADS / MANAGES

**Purpose**: Link organizational hierarchy and leadership

**Example**:

```
Person(Alexis Torres) --[LEADS]→ Team(Cloud Infrastructure Team)
Team(Cloud Infrastructure Team) --[PART_OF]→ Organization(TechCorp)
```

#### 6. AFFILIATED_WITH

**Purpose**: Generic affiliation relationships (certificates, memberships, partnerships)

**Example**:

```
Person(Alexis Torres) --[AFFILIATED_WITH]→ Certification(AWS Solutions Architect)
Organization(TechCorp) --[AFFILIATED_WITH]→ Organization(AWS Partner Network)
```

### Entity Relationship Graph Structure

The complete entity relationship graph looks like:

```
                    ┌─────────────────┐
                    │   TechCorp      │
                    │ (Organization)  │
                    └────────┬────────┘
                             │
          ┌──────[WORKS_FOR]──┼──────[WORKS_FOR]──┐
          │                   │                   │
     ┌────▼────┐         ┌────▼────┐         ┌───▼────┐
     │  Alexis │         │   Sarah │         │  James │
     │  Torres │         │   Chen  │         │  Smith │
     │ (Person)│         │(Person) │         │(Person)│
     └────┬────┘         └────┬────┘         └───┬────┘
          │                   │                  │
          └──[WORKED_WITH]────┴──[WORKED_WITH]──┘
          │                   │                  │
          └──[EXPERIENCED_WITH]─ AWS, Docker, Kubernetes
                              │
                              └──[BASED_IN]──→ San Francisco, CA
```

### How Entity Relationships Affect Search

**Query**: "Show me all people who worked with Alexis Torres at TechCorp"

**Execution**:

1. Find Entity: `Person(Alexis Torres)`
2. Traverse WORKED_FOR: → `Organization(TechCorp)`
3. Traverse WORKS_FOR: → Other Person entities at TechCorp
4. Traverse WORKED_WITH: → People who collaborated with Alexis
5. Retrieve chunks mentioning these people and entities

**Result**:

- Alexis Torres' colleagues (via WORKS_FOR to same org)
- Sarah Chen, James Smith (via WORKED_WITH)
- Their combined experience and projects
- Comprehensive team narrative

### Integration with Chunk Search

Entity relationships exist alongside chunk relationships in the same graph:

```
Document (Resume.pdf)
    ├── [HAS_CHUNK] → Chunk[0] "Professional Summary"
    │                    └── [MENTIONS] → Entity(Alexis Torres)
    │                                      └── [WORKS_FOR] → Entity(TechCorp)
    │
    ├── [HAS_CHUNK] → Chunk[1] "Work Experience"
    │                    ├── [MENTIONS] → Entity(TechCorp)
    │                    ├── [MENTIONS] → Entity(AWS)
    │                    └── [MENTIONS] → Entity(Sarah Chen)
    │
    └── [HAS_CHUNK] → Chunk[2] "Technical Skills"
                          ├── [MENTIONS] → Entity(AWS)
                          ├── [MENTIONS] → Entity(Docker)
                          └── [MENTIONS] → Entity(Kubernetes)
```

**Hybrid Retrieval Including Entities**:

1. **Chunk-focused search**: "Tell me about Alexis' projects" → Find chunks mentioning Alexis
2. **Entity-focused search**: "Who did Alexis work with?" → Find entities via WORKED_WITH
3. **Hybrid search**: "Show Alexis' AWS projects with team members" →
   - Find chunks about AWS projects (chunk search)
   - Find team members via WORKED_WITH (entity search)
   - Combine results with relationship context

### Configuration for Entity-Based Search

Entity relationships are created automatically during document ingestion with LLMGraphTransformer (when `ENABLE_LLMGRAPH_TRANSFORMER=true`):

```bash
# .env configuration
ENABLE_LLMGRAPH_TRANSFORMER=true        # Enable entity extraction
LLMGRAPH_MODEL=gpt-4o-mini              # Model for extraction
AZURE_OPENAI_LLM_API_VERSION=2024-12-01-preview
```

Entity relationships can be queried via specialized endpoints:

```python
# Retrieve by relationship type
POST /api/graph/retrieve-by-relationship
{
  "source_entity_id": "entity:Person:Alexis_Torres",
  "relationship_types": ["WORKED_WITH", "COLLABORATED_WITH"],
  "max_chunks": 15,
  "max_hops": 2
}
```

---

## Graph Expansion Deep Dive

### How Graph Expansion Works

**Goal**: Starting from seed chunks found via keyword/semantic search, discover related content by following relationships.

**Process**:

1. **Identify Seed Chunks**
   - Results from Stage 1 (keyword + semantic search)
   - These are the starting points for relationship traversal

2. **1-Hop Expansion** (Primary)

   ```
   For each seed chunk:

     a) Follow SEMANTICALLY_SIMILAR edges
        - Find all connected chunks
        - Each edge has a weight (similarity score)
        - Include if weight >= similarity_threshold (0.7 default)
        - Add to results with path_score = 0.8 (1-hop penalty)

     b) Follow NEXT_CHUNK edges
        - Find next chunk in same document
        - Always included (no threshold)
        - Add to results with path_score = 0.8

     c) Follow PREV_CHUNK edges (optional)
        - Find previous chunk in same document
        - Provides backward context
        - Add with path_score = 0.8
   ```

3. **2-Hop Expansion** (Optional, when `max_hops=2`)

   ```
   For each 1-hop result:

     a) Follow SEMANTICALLY_SIMILAR from 1-hop neighbors
        - Connect to new chunks not yet seen
        - Include if similarity >= threshold
        - Add with path_score = 0.6 (2-hop penalty)

     b) Follow NEXT_CHUNK from 1-hop chunks
        - Continue document flow further
        - Add with path_score = 0.6
   ```

4. **Apply Expansion Limits**

   ```
   semantic_expansion_limit = 100
   - Maximum chunks from SEMANTICALLY_SIMILAR paths
   - Prevents excessive cross-document expansion

   hierarchical_expansion_limit = 50
   - Maximum chunks from NEXT_CHUNK/PREV_CHUNK paths
   - Prevents excessive within-document expansion
   ```

5. **Deduplicate & Compile**
   - Remove duplicate chunks (same chunk_id)
   - Combine all expansion results
   - Retain path information for scoring

### Graph Expansion Example

```
Initial Query: "Tell me about Alexis' Python and cloud projects"

Stage 1 Results (Seed Chunks):
├── Chunk A: "Alexis Torres - Python backend projects"
│   (similarity: 0.92, keyword_score: 0.8)
├── Chunk B: "AWS cloud infrastructure experience"
│   (similarity: 0.88, keyword_score: 0.6)
└── Chunk C: "Professional summary - Alexis"
    (similarity: 0.85, keyword_score: 0.7)

Stage 2: 1-Hop Graph Expansion from Chunk A
├── Via NEXT_CHUNK → Chunk A1: "Details of Flask application project"
│   (path_score: 0.8, from sequential document flow)
├── Via NEXT_CHUNK → Chunk A2: "Django REST API implementation"
│   (path_score: 0.8)
├── Via SEMANTICALLY_SIMILAR → Chunk D: "Python best practices guide"
│   (path_score: 0.8, similarity: 0.82, from different doc)
└── Via SEMANTICALLY_SIMILAR → Chunk E: "Microservices with Python"
    (path_score: 0.8, similarity: 0.79)

Stage 2: 1-Hop Graph Expansion from Chunk B
├── Via NEXT_CHUNK → Chunk B1: "AWS services used - EC2, RDS, Lambda"
│   (path_score: 0.8)
├── Via NEXT_CHUNK → Chunk B2: "Cost optimization and performance tuning"
│   (path_score: 0.8)
├── Via SEMANTICALLY_SIMILAR → Chunk F: "Azure cloud solutions"
│   (path_score: 0.8, similarity: 0.84, competitor tech)
└── Via SEMANTICALLY_SIMILAR → Chunk G: "Infrastructure as Code with Terraform"
    (path_score: 0.8, similarity: 0.81)

Stage 2: 1-Hop Graph Expansion from Chunk C
├── Via NEXT_CHUNK → Chunk C1: "Work experience section"
│   (path_score: 0.8)
└── Via SEMANTICALLY_SIMILAR → Chunk H: "Senior engineer profile"
    (path_score: 0.8, similarity: 0.76)

Total After 1-Hop: 3 seeds + 11 neighbors = 14 chunks
```

### Why Graph Expansion Matters

1. **Document Context Preservation**
   - NEXT_CHUNK ensures information split across chunks stays together
   - Example: "Alexis Torres" in Chunk 0 connects to "Python experience" in Chunk 1

2. **Cross-Document Discovery**
   - SEMANTICALLY_SIMILAR finds related content you wouldn't think to search for
   - Example: Query about "Python projects" finds "Microservices architecture" guide

3. **Relationship Signal**
   - Chunks directly related to seed results get a boost
   - Path score reflects proximity (closer = higher score)
   - Explains _why_ results are relevant

4. **Reduced Context Switching**
   - All related content retrieved in one pass
   - Avoids multiple sequential queries
   - Comprehensive answer with all supporting context

### Path Score Calculation

```
Base score components:
  vector_similarity = 0.92    (from semantic search)
  keyword_score = 0.80         (from keyword matching)

For each chunk type:
  Seed chunk (0-hop):
    path_score = 1.0
    hop_count = 0

  1-hop neighbor:
    path_score = 0.8
    hop_count = 1

  2-hop neighbor:
    path_score = 0.6
    hop_count = 2

Final hybrid score = (0.7 * vector_similarity) +
                     (0.3 * keyword_score) +
                     (path_score * 0.5)
```

---

## Sequential Execution: Complete Flow

### Stage 1: Initialize & Execute Keyword + Semantic Search

```python
def graphrag_search(query="...", top_k=10):
    # 1. Extract or receive keywords
    keywords = extract_or_use_provided_keywords(query)

    # 2. Embed query
    query_embedding = embed(query)

    # 3. Parallel execution:
    # Thread 1: Keyword search
    keyword_results = neo4j_keyword_search(
        keywords=keywords,
        match_type="any",
        top_k=top_k
    )

    # Thread 2: Vector similarity search
    vector_results = neo4j_vector_search(
        embedding=query_embedding,
        top_k=top_k,
        similarity_threshold=0.7
    )

    # 4. Merge results (deduplicate by chunk ID)
    seed_chunks = merge_results(keyword_results, vector_results)
```

### Stage 2: Graph Expansion

```python
    # 5. Expand via graph relationships
    expanded_chunks = graph_expand(
        seed_chunks=seed_chunks,
        max_hops=1,
        similarity_threshold=0.7,
        semantic_expansion_limit=100,
        hierarchical_expansion_limit=50
    )
```

### Stage 3: Re-Ranking & Return

```python
    # 6. Re-rank with hybrid scoring
    hybrid_scores = []
    for chunk in expanded_chunks:
        score = (
            chunk.vector_similarity * (1 - keyword_boost) +      # Semantic
            chunk.keyword_score * keyword_boost +                 # Keyword
            chunk.path_score +                                    # Graph proximity
            chunk.proximity_boost                                 # Document context
        )
        hybrid_scores.append((chunk, score))

    # 7. Sort and return top-k
    ranked_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    final_results = ranked_results[:top_k]

    return final_results
```

---

## Execution Example: Complete Sequential Flow

### Query

```
"Show me all Python and cloud projects by Alexis Torres"
```

### Step-by-Step Execution

**Step 1: Keyword Extraction**

```
Input: "Show me all Python and cloud projects by Alexis Torres"
Output: ["Python", "cloud", "projects", "Alexis", "Torres"]
```

**Step 2: Query Embedding**

```
Input: Full query text
Output: query_embedding (1536 dimensions)
```

**Step 3: Keyword Search (Thread 1)**

```
Looking for chunks with ANY of: ["Python", "cloud", "projects", "Alexis", "Torres"]
Results:
  - Chunk A: has ["Alexis", "Torres", "Python"] → keyword_score = 0.6
  - Chunk B: has ["cloud", "projects", "infrastructure"] → keyword_score = 0.6
  - Chunk C: has ["Python", "AWS", "cloud"] → keyword_score = 0.6
```

**Step 4: Vector Search (Thread 2)**

```
Comparing query_embedding against all chunk embeddings:
Results (similarity >= 0.7):
  - Chunk D: similarity = 0.88
  - Chunk E: similarity = 0.85
  - Chunk A: similarity = 0.82
  - Chunk F: similarity = 0.72
```

**Step 5: Merge Results**

```
Combined seed chunks (deduped):
  - Chunk A: keyword_score=0.6, vector_score=0.82
  - Chunk B: keyword_score=0.6, vector_score=N/A
  - Chunk C: keyword_score=0.6, vector_score=N/A
  - Chunk D: keyword_score=N/A, vector_score=0.88
  - Chunk E: keyword_score=N/A, vector_score=0.85
  - Chunk F: keyword_score=N/A, vector_score=0.72
```

**Step 6: Graph Expansion (1-hop)**

```
From Chunk A:
  - NEXT_CHUNK → Chunk A1 (Python project details)
  - SEMANTICALLY_SIMILAR → Chunk G (Python best practices)

From Chunk D:
  - NEXT_CHUNK → Chunk D1 (Cloud architecture)
  - SEMANTICALLY_SIMILAR → Chunk H (Cloud security)

Total expanded: 6 seeds + 4 neighbors = 10 chunks
```

**Step 7: Re-Ranking with Hybrid Score**

```
Hybrid Formula: 0.7 * vector_score + 0.3 * keyword_score + path_bonus

Final scores:
  1. Chunk D: (0.7 * 0.88) + (0.3 * 0) + 0.0 = 0.616
  2. Chunk A: (0.7 * 0.82) + (0.3 * 0.6) + 0.0 = 0.754
  3. Chunk G: (0.7 * 0.79) + (0.3 * 0) + 0.1 = 0.653 (1-hop bonus)
  4. Chunk E: (0.7 * 0.85) + (0.3 * 0) + 0.0 = 0.595
  5. Chunk B: (0.7 * 0) + (0.3 * 0.6) + 0.0 = 0.180
  ...

Sorted (descending):
  1. Chunk A: 0.754
  2. Chunk G: 0.653
  3. Chunk D: 0.616
  4. Chunk E: 0.595
  5. Chunk H: 0.550
  ... (return top 10)
```

**Step 8: Return Results**

```python
results = [
    {
        "text": "Chunk A: Python project implementation...",
        "file_name": "Alexis_Torres_Resume.pdf",
        "similarity": 0.82,
        "hybrid_score": 0.754,
        "keywords_matched": 3
    },
    {
        "text": "Chunk G: Python best practices...",
        "file_name": "Python_Guidelines.pdf",
        "similarity": 0.79,
        "hybrid_score": 0.653,
        "keywords_matched": 1
    },
    ...
]
```

---

## Configuration Reference

### Default Settings

```python
# Basic search
graphrag_search(
    query="user query",
    top_k=10,                          # Results to return
    similarity_threshold=0.7,          # Min semantic similarity
    use_keyword_search=True,           # Enable keyword matching
    keyword_match_type="any",          # "any" (OR) or "all" (AND)
    keyword_boost=0.3,                 # Weight for keyword score (0-1)
    max_hops=1,                        # Graph expansion hops (1 or 2)
    semantic_expansion_limit=100,      # Max semantic expansion results
    hierarchical_expansion_limit=50,   # Max hierarchical expansion results
)
```

### Environment Variables (.env)

```bash
# Vector Embeddings
EMBEDDING_MODE=foundry
AZURE_OPENAI_EMBED_DEPLOYMENT_NAME=text-embedding-3-small
EMBEDDING_VECTOR_DIM=1536

# Neo4j Connection
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=dxcneo4j

# Docling Configuration
USE_DOCLING=true
DOCLING_LLM_PROVIDER=docling
DOCLING_USE_MARKDOWN_FALLBACK=false
```

---

## Common Search Issues & Solutions

### Issue: Searching by First Name Only Returns Few Results

**Problem**: When you search with only a first name (e.g., "Alexis" without "Torres"), the search has difficulty finding relevant results, especially for keyword matching.

**Root Causes**:

1. **Person Name Extraction Filtering**
   - The `extract_person_names()` function applies filters to extracted names:
     - **Minimum length**: Only extracts words with 3+ characters
     - **Capitalization**: Only extracts capitalized words (proper nouns)
     - **Common word exclusion**: Filters out common English words
   - Example: Single names like "Alexis" (6 chars) are extracted, but "Bob" (3 chars) might be marginal

2. **Low Keyword Match Scoring**
   - First name alone creates a weak keyword signal:

     ```
     Query: "Alexis"
     Extracted keywords: ["Alexis"]

     Chunk with ["Alexis", "Torres", "Python"]:
       - Keywords matched: 1/1 = 1.0 score ✓ (good)

     But if chunk has ONLY last name ["Torres", "Python"]:
       - Keywords matched: 0/1 = 0.0 score ✗ (poor match)
     ```

3. **Semantic Search Compensation**
   - Vector search alone may not find the person without context:

     ```
     Query: "Alexis"
     Query embedding focuses on: first name, person reference

     Chunks with "Alexis" → high semantic similarity ✓
     But chunks with only "Torres" or "Alexis Torres" mentioned
     differently → lower similarity ✗
     ```

4. **Keyword Boost Impact**
   - Default `keyword_boost=0.3` means only 30% of score comes from keywords
   - With single-name queries, keyword signal is weak, so semantic search dominates
   - If chunks don't contain the exact first name, vector similarity becomes the only signal

**Example Scenario**:

```
Query: "Find Alexis' resume"
↓
Extracted keywords: ["Find", "Alexis", "resume"]
- "Find" filtered as common word → removed
- "Alexis" kept
- "resume" kept
Result: keywords = ["Alexis", "resume"]

Database chunks for Alexis Torres:
- Chunk A: keywords = ["Alexis", "Torres", "experience", "skills"]
  - Keyword match: 1/2 = 0.5 ✗ (only "Alexis", missing "resume")
  - Semantic match: 0.75 (good, but not excellent)
  - Hybrid score: (0.75 * 0.7) + (0.5 * 0.3) = 0.675

- Chunk B: keywords = ["resume", "summary", "professional"]
  - Keyword match: 1/2 = 0.5 ✗ (only "resume", missing "Alexis")
  - Semantic match: 0.65 (moderate, word "resume" present but no "Alexis")
  - Hybrid score: (0.65 * 0.7) + (0.5 * 0.3) = 0.605
```

**Solutions**:

1. **Use Full Names When Possible**

   ```
   ❌ Poor: "Find Alexis"
   ✅ Better: "Find Alexis Torres"
   ✅ Best: "Show me Alexis Torres' projects"
   ```

   - Full names create stronger keyword signals
   - Combines first + last name for keyword matching
   - Provides context keywords for semantic search

2. **Add Context Around First Name**

   ```
   ❌ Weak: "Alexis"
   ✅ Better: "Alexis experience"
   ✅ Better: "Alexis skills"
   ✅ Better: "Alexis project manager"
   ```

   - Additional context words boost semantic similarity
   - Creates more comprehensive keyword set

3. **Increase Keyword Boost** (if multiple names not available)
   - For first-name-only queries, can programmatically adjust:

   ```python
   if query_contains_only_firstname():
       graphrag_search(
           query=query,
           keywords=["Alexis"],
           keyword_boost=0.6  # Increase from default 0.3
       )
   ```

   - This gives 60% weight to keyword matching instead of 30%

4. **Rely on Semantic Search**
   - If first name alone is necessary, the semantic search will still work:
   - Query "Alexis" will embed to a person-focused vector
   - Chunks containing "Alexis" will match with high similarity
   - Results may be slower to retrieve but will still appear

5. **Use Explicit Keywords Parameter**
   - Bypass extraction filters by providing keywords explicitly:
   ```python
   graphrag_search(
       query="Alexis",
       keywords=["Alexis"],  # Explicit keywords skip extraction filters
       keyword_boost=0.5
   )
   ```

**Why This Happens**:

The keyword extraction is designed to be selective to avoid matching common words and improve precision. However, this means:

- Single first names create weak keyword signals (only 1 keyword)
- Without supporting context, keyword matching has low confidence
- The system relies more on semantic similarity, which may miss variations
- Name variations (nicknames, alternative spellings) won't match

**Best Practice**:

For name-based searches, always include both first and last names when possible. This enables both keyword matching (exact name match) and semantic matching (person recognition). The hybrid approach with dual signals is much more effective than single-name queries.

---

## Performance Notes

### Execution Time

- Keyword search: ~10-50ms (database index lookup)
- Vector search: ~50-200ms (embedding + similarity calculation)
- Graph expansion: ~100-500ms (1-2 hop traversal)
- Re-ranking: ~10-50ms (scoring and sorting)
- **Total average: 200-800ms per query**

### Optimization Tips

1. **Faster results**: Set `max_hops=1` and lower `semantic_expansion_limit`
2. **Better relevance**: Set `max_hops=2` and higher expansion limits
3. **Keyword precision**: Use `keyword_match_type="all"` for stricter matching
4. **Semantic focus**: Increase `similarity_threshold` to 0.80+
5. **First-name queries**: Include context words or increase `keyword_boost`

---

## Summary

The sequential search flow is:

1. **Keyword Search**: Extract and match exact terms
2. **Semantic Search**: Find similar content via embeddings
3. **Graph Expansion**: Traverse relationships for context
4. **Re-Ranking**: Combine all signals into hybrid score
5. **Return**: Top-k results sorted by relevance

All three search types execute together in every query to provide comprehensive, relevant results.
