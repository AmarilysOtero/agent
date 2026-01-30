# Graph Relationship Discovery Implementation Plan

## Problem Statement

**Current Issue:**
Query: "Is there any relationship between Kevin and Alexis?"

- ❌ Returns: "No relationship found"
- ✅ Graph has: `(Kevin)-[:AT_ORGANIZATION]->(DXC)<-[:AT_ORGANIZATION]-(Alexis)`

**Root Cause:**
Vector search filters results by `similarity >= 0.3` BEFORE graph expansion, removing low-similarity results that could reveal graph relationships.

---

## Solution Overview

**Generic, Domain-Agnostic Approach:**

1. **High-recall retrieval** (no early filtering)
2. **Detect multiple entities** in results
3. **Graph expansion** to find connections
4. **Then filter/rank** with graph results preserved

**Key Principle:** Never let embedding similarity decide if relationships exist.

---

## Implementation Steps

### **Phase 1: Entity Seeding from Graph Mentions (Generic)**

**File:** `neo4j_backend/services/graphrag_retrieval.py`

**Goal:** Extract entity seeds from what the retrieved chunks actually mention in the graph, not from filenames or person-specific metadata.

**Why Generic:** Every document type can produce `Chunk → MENTIONS → Entity`. Works for resumes, financials, policies, contracts, etc.

**Add Function:**

```python
def extract_seed_entities_from_chunks(
    chunk_ids: List[str],
    max_entities: int = 15,
) -> List[Dict]:
    """
    Generic entity seeding:
    Given top retrieved chunk_ids, return the most salient Entity nodes
    mentioned by those chunks.

    Args:
        chunk_ids: List of chunk IDs from hybrid retrieval
        max_entities: Maximum number of entities to return

    Returns:
        List of dicts: [{entity_id, name, type, mention_count, avg_conf}]
    """
    if not chunk_ids:
        return []

    cypher = """
    MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
    WHERE c.id IN $chunk_ids OR c.chunk_id IN $chunk_ids
    WITH e,
         count(*) AS mention_count,
         avg(coalesce(m.confidence, 1.0)) AS avg_conf
    RETURN
      coalesce(e.id, e.entity_id, toString(id(e))) AS entity_id,
      e.name AS name,
      coalesce(e.type, head(labels(e))) AS type,
      mention_count,
      avg_conf
    ORDER BY mention_count DESC, avg_conf DESC
    LIMIT $max_entities
    """

    with driver.session() as session:
        rows = session.run(cypher, chunk_ids=chunk_ids, max_entities=max_entities)
        return [dict(r) for r in rows]
```

**Notes:**

- Uses `c.id OR c.chunk_id` because chunk IDs may be stored under either property
- Uses `coalesce(e.id, e.entity_id, internal id)` because entity IDs vary
- Uses `MENTIONS` relationship which exists in the graph
- Avoids brittle filename parsing

**Purpose:** Domain-agnostic entity extraction via graph relationships.

---

### **Phase 2: Graph Connection Discovery (Generic + Safe)**

**File:** `neo4j_backend/services/graphrag_retrieval.py`

**Goal:** Given seed entities, discover real connections via bounded paths, and gather supporting chunks for citations.

**Key Improvements:**

- No filename matching
- No broad `MATCH (e)` over all nodes
- No direction assumptions (uses undirected paths)
- Bounded hops to avoid graph explosions
- Returns paths + evidence chunks (citations), not "synthetic chunks with similarity 0.9"

**Add Function:**

```python
def discover_graph_connections_between_entities(
    seed_entities: List[Dict],
    max_pairs: int = 20,
    max_hops: int = 3,
    max_paths: int = 10,
) -> List[Dict]:
    """
    Generic connection discovery:
    - picks top entity pairs from seed_entities
    - finds bounded shortest paths between them
    - returns structured path objects (not fake chunks)

    Args:
        seed_entities: List of entity dicts from extract_seed_entities_from_chunks
        max_pairs: Maximum entity pairs to test
        max_hops: Maximum path length (default 3 hops)
        max_paths: Maximum paths to return

    Returns:
        List of path dicts with structure information
    """
    if len(seed_entities) < 2:
        return []

    # Take top N entities and form bounded pairs (avoid O(n^2) blowups)
    top = seed_entities[: min(len(seed_entities), 10)]
    entity_ids = [e["entity_id"] for e in top]

    cypher = """
    // Resolve entity nodes by stable id
    MATCH (e:Entity)
    WHERE coalesce(e.id, e.entity_id, toString(id(e))) IN $entity_ids
    WITH collect(e) AS ents
    UNWIND ents AS a
    UNWIND ents AS b
    WITH a, b
    WHERE id(a) < id(b)
    WITH a, b
    LIMIT $max_pairs

    // Find bounded connection (shortestPath, undirected)
    MATCH p = shortestPath((a)-[*..$max_hops]-(b))
    RETURN
      coalesce(a.id, a.entity_id, toString(id(a))) AS a_id,
      a.name AS a_name,
      coalesce(a.type, head(labels(a))) AS a_type,
      coalesce(b.id, b.entity_id, toString(id(b))) AS b_id,
      b.name AS b_name,
      coalesce(b.type, head(labels(b))) AS b_type,
      [n IN nodes(p) | coalesce(n.name, n.id, n.entity_id, toString(id(n)))] AS path_nodes,
      [r IN relationships(p) | type(r)] AS path_rels,
      length(p) AS path_len
    ORDER BY path_len ASC
    LIMIT $max_paths
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            entity_ids=entity_ids,
            max_pairs=max_pairs,
            max_hops=max_hops,
            max_paths=max_paths,
        )
        return [dict(r) for r in rows]
```

**Phase 2b: Pull Supporting Chunks for Citations**

This ensures your fGraph-Aware Reranking\*\*

**File:** `neo4j_backend/services/graphrag_retrieval.py`

**Add Function:**

```python
def rerank_with_graph_awareness(
    results: List[Dict],
    path_entities: List[str],
    min_similarity: float = 0.3
) -> List[Dict]:
    """Rank results - boost chunks mentioning path entities, preserve supporting evidence

    Args:
        results: Combined list of vector and graph-supporting chunks
        path_entities: Entity names from discovered graph paths
        min_similarity: Threshold for vector results (not applied to graph-supporting chunks)

    Returns:
        Filtered and ranked results
    """
    filtered = []
    path_entities_lower = [e.lower() for e in path_entities]

    for result in results:
        # Always keep graph-supporting chunks (from Phase 2b)
        if result.get('source') == 'graph_supporting_evidence':
            filtered.append(result)
            continue

        # For vector/hybrid results, apply threshold
        if result.get('similarity', 0) >= min_similarity:
            # Boost chunks that mention path entities (graph-aware boosting)
            text_lower = result.get('text', '').lower()
            if any(entity in text_lower for entity in path_entities_lower):
                result['hybrid_score'] = result.get('hybrid_score', 0) * 1.15
                result['metadata'] = result.get('metadata', {})
                result['metadata']['graph_boosted'] = True

            filtered.append(result)

    # Sort by hybrid_score descending
    filtered.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

    return filtered
```

**Key Changes:**

- Preserves `graph_supporting_evidence` chunks (real text, citable)
- Boosts vector chunks that mention path entities (15% boost)
- Ensures at least some top chunks even if similarity < 0.3 for relationship queries
- No synthetic chunks with artificial 0.9 scores

**Purpose:** Graph-aware reranking that preserves citations and boosts relevant evidence
with driver.session() as session:
rows = session.run(cypher, names=path_entity_names, max_chunks=max_chunks)
chunks = []
for row in rows:
chunks.append({
'id': row['chunk_id'],
'text': row['text'],
'file_id': row['doc_id'],
'similarity': row['similarity'],
'hybrid_score': row['hybrid_score'],
'keyword_score': row['keyword_score'],
'source': 'graph_supporting_evidence',
'metadata': {
'entity_name': row['entity_name'],
'type': 'supporting_chunk'
}
})
return chunks

````

**Purpose:**
- Generic graph traversal - works for ANY entities, ANY relationships
- Uses undirected shortest paths (no direction assumptions)
- Bounded by max_hops to prevent performance issues
- Returns actual chunks for citations (not fake high-scoring synthetic chunks)

---

### **Phase 3: Add Reranking Function**

**File:** `neo4j_backend/services/graphrag_retrieval.py`

**Add Function:**
```python
def rerank_results(results: List[Dict], min_similarity: float = 0.3) -> List[Dict]:
    """Rank results - preserve graph discoveries, filter vector results

    Args:
        results: Combined list of vector and graph results
        min_similarity: Threshold for vector results (not applied to graph results)

    Returns:
        Filtered and ranked results
    """
    filtered = []

    for result in results:
        # Always keep graph traversal results (high-confidence facts)
        if result.get('source') == 'graph_traversal':
            filtered.append(result)
            continue

        # For vector/hybrid results, apply threshold
        if result.get('similarity', 0) >= min_similarity:
            filtered.append(result)

    # Sort by hybrid_score descending
    filtered.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

    return filtered
````

**Purpose:** Preserve graph facts, filter only vector results.

---

### **Phase 4: Modify Main Retrieval Flow**

**File:** `neo4j_backend/services/graphrag_retrieval.py`

**Current Function:** `hybrid_retrieve()`

**Change From:**

```python
def hybrid_retrieve(query: str, top_k: int = 10, **kwargs) -> Dict:
    # Get semantic results
    semantic_results = semantic_search(...), no filtering yet)
    semantic_results = semantic_search(query, top_k=top_k * 3)
    keyword_results = keyword_search(query, top_k=top_k * 2)
    all_results = semantic_results + keyword_results

    # Extract chunk IDs for entity seeding
    chunk_ids = [r.get('id', r.get('chunk_id')) for r in all_results[:15] if r.get('id') or r.get('chunk_id')]

    # 2. Seed entities from graph mentions (GENERIC - works for any doc type)
    seed_entities = extract_seed_entities_from_chunks(chunk_ids, max_entities=15)

    path_entity_names = []

    if len(seed_entities) >= 2:
        logger.info(f"🔗 Detected {len(seed_entities)} entities via MENTIONS - checking graph connections")

        # 3. Discover paths between entities
        paths = discover_graph_connections_between_entities(
            seed_entities,
            max_pairs=20,
            max_hops=3,
            max_paths=10
        )

        if paths:
            logger.info(f"✅ Found {len(paths)} graph paths")

            # Collect entity names from paths
            for path in paths:
                path_entity_names.extend(path['path_nodes'])
            path_entity_names = list(set(path_entity_names))  # Deduplicate

            # 4. Fetch supporting chunks for citations
            supporting_chunks = fetch_supporting_chunks_for_path_entities(
                path_entity_names,
                max_chunks=12
            )

            logger.info(f"📄 Fetched {len(supporting_chunks)} supporting chunks for citations")
            all_results.extend(supporting_chunks)

    # 5. NOW filter and rank (after graph expansion, with graph-aware boosting)
    filtered = rerank_with_graph_awareness(
        all_results,
        path_entity_names=path_entity_names,
        min_similarity=0.3
    dcoding)"""

    # 1. High-recall retrieval (get more candidates)
    semantic_results = semantic_search(query, top_k=top_k * seed_entities_from_chunks()` | ➕ Add new function |
| `neo4j_backend/services/graphrag_retrieval.py` | `discover_graph_connections_between_entities()` | ➕ Add new function |
| `neo4j_backend/services/graphrag_retrieval.py` | `fetch_supporting_chunks_for_path_entities()` | ➕ Add new function |
| `neo4j_backend/services/graphrag_retrieval.py` | `rerank_with_graph_awarenes

    # 2. Detect if graph expansion needed (GENERIC check)
    entities = extract_entities_from_results(all_results[:15])

    if len(entities) >= 2:
        logger.info(f"🔗 Detected {len(entities)} entities - checking graph connections")
        graph_results = discover_graph_connections(entities)

        if graph_results:
            logger.info(f"✅ Found {len(graph_results)} graph connections")
            all_results.extend(graph_results)

    # 3. NOW filter and rank (after graph expansion)
    filtered = rerank_results(all_results, min_similarity=0.3)

    return {'chunks': filtered[:top_k]}
```

**Key Change:** Filter AFTER graph expansion, not before.

---

## Files to Modify

| File                                           | Function                          | Change Type                         |
| ---------------------------------------------- | --------------------------------- | ----------------------------------- |
| `neo4j_backend/services/graphrag_retrieval.py` | `extract_entities_from_results()` | ➕ Add new function                 |
| `neo4j_backend/services/graphrag_retrieval.py` | `discover_graph_connections()`    | ➕ Add new function                 |
| `neo4j_backend/services/graphrag_retrieval.py` | `rerank_results()`                | ➕ Add new function                 |
| `neo4j_backend/services/graphrag_retrieval.py` | `hybrid_retrieve()`               | 🔄 Modify flow (reorder operations) |

---

## Testing Plan

### **Test 1: Relationship Query**

```python
query = "Is there any relationship between Kevin and Alexis?"

# Expected flow:
# 1. Retrieves Kevin's resume (similarity 0.25)
# 2. Retrieves Alexis's resume (similarity 0.27)
# 3. Extracts entities: {"Kevin J. Ramírez Pomales", "Alexis Torres"}
# 4. Graph query finds: (Kevin)-[:AT_ORGANIZATION]->(DXC)<-[:AT_ORGANIZATION]-(Alexis)
# 5. Returns graph result (score 0.9) + resume chunks
# 6. LLM sees: "Kevin and Alexis are connected: both AT_ORGANIZATION DXC Technology"

# Expected answer: "Yes, Kevin and Alexis both work at DXC Technology"
```

### **Test 2: Single Entity Query (No Graph)**

```python
query = "What are Alexis's skills?"

# Expected flow:
# 1. Retrieves Alexis's resume chunks
# 2. Extracts entities: {"Alexis Torres"}
# 3. Only 1 entity → skip graph expansion
# 4. Returns vector results normally

# Expected answer: List of skills from resume
```

### **Test 3: No Relationship Found**

```python
query = "Is there any relationship between Kevin and John Doe?"

# Expected flow:
# 1. Retrieves Kevin's resume (similarity 0.3)
# 2. No results for "John Doe"
# 3. Extracts entities: {"Kevin J. Ramírez Pomales"}
# 4. Only 1 entity → skip graph expansion
# 5. Returns Kevin's chunks

# Expected answer: "No information found about John Doe"
```

---

## Verification Checklist

seeding uses MENTIONS relationships (not filenames)

- [ ] Entity extraction works for any document type (resumes, financials, policies)
- [ ] Graph query uses bounded shortest paths (max 3 hops)
- [ ] Graph query returns results for Kevin-Alexis relationship
- [ ] Supporting chunks fetched for citations (source='graph_supporting_evidence')
- [ ] Supporting chunks preserved during filtering (not removed by similarity threshold)
- [ ] Vector results still filtered by similarity >= 0.3
- [ ] Chunks mentioning path entities get 15% boost
- [ ] Final results sorted by hybrid_score
- [ ] Single-entity queries skip graph expansion
- [ ] No hardcoded keywords, entity names, or relationship types
- [ ] No synthetic chunks with artificial 0.9 scorpansion
- [ ] No hardcoded keywords or entity names

---

## Expected Behavior Changes

### **Before:**

```
Query: "Is there any relationship between Kevin and Alexis?"
→ Hybrid search: 12 results (similarity 0.22-0.29)
→ Filter: 0 results kept (all < 0.3)
→ Answer: "No relationship found"
```

### **After:**

```
Query: "Is there any relationship between Kevin and Alexis?"
→ Hybrid search: 12 results (similarity 0.22-0.29)
→ Entity detection: 2 entities found
→ Graph expansion: 1 connection found (AT_ORGANIZATION → DXC)
→ Filter: 1 graph result (0.9) + 0 vector results
→ Answer: "Yes, Kevin and Alexis both work at DXC Technology"
```

---

## Non-Goals (Out of Scope)

- ❌ No hardcoded relationship types (AT_ORGANIZATION, WORKED_WITH, etc.)
- ❌ No domain-specific logic (resumes, documents, etc.)
- ❌ No keyword lists for detection
- ❌ No changes to Agent code (all changes in neo4j_backend)

---

## Success Criteria

✅ Query "Is there any relationship between Kevin and Alexis?" returns correct answer
✅ System works for ANY entities, ANY relationships
✅ No hardcoded keywords or patterns
✅ Graph results preserved through filtering
✅ Single-entity queries unaffected
✅ All existing tests pass

---

## Implementation Order

1. ✅ Create this imseed_entities_from_chunks()` function (Phase 1)
2. ⏭️ Add `discover_graph_connections_between_entities()` function (Phase 2)
3. ⏭️ Add `fetch_supporting_chunks_for_path_entities()` function (Phase 2b)
4. ⏭️ Add `rerank_with_graph_awareness()` function (Phase 3)
5. ⏭️ Modify `hybrid_retrieve()` to use new flow (Phase 4)
6. ⏭️ Test with Kevin-Alexis relationship query
7. ⏭️ Verify existing queries still work
8. ⏭️ Verify existing queries still work
9. ⏭️ Commit and push changes

--- via MENTIONS)

- Solution is **truly generic** and works across any domain (resumes, financials, policies, contracts, etc.)
- Filtering happens **after** graph expansion (not before)
- Graph facts come with **citable text chunks** (not synthetic high-scoring chunks)
- Uses **bounded shortest paths** to prevent performance issues
- Uses **undirected relationships** (no direction assumptions)
- Graph is used for **structure**; text chunks are used for **evidence/citations**

## Why This Approach Is Generic

| Aspect                   | Generic Approach                              | Avoids                                            |
| ------------------------ | --------------------------------------------- | ------------------------------------------------- |
| **Entity Detection**     | Via `Chunk → MENTIONS → Entity` relationships | Filename parsing, person-specific metadata        |
| **Entity Resolution**    | Match only `:Entity` nodes by stable IDs      | Broad `MATCH (e)` matching random nodes           |
| **Connection Discovery** | Bounded shortest paths (undirected)           | Direction assumptions, unbounded graph explosions |
| **Evidence**             | Real chunks that mention entities             | Synthetic chunks with artificial scores           |
| **Citations**            | Supporting chunks pulled from graph           | Graph structure alone (non-verifiable)            |
| **Boosting**             | Graph-aware reranking of real chunks          | Fake high-scoring "chunks"                        |

- This is a **control-flow change**, not an architectural redesign
- Graph expansion is **adaptive** (only when multiple entities detected)
- Solution is **generic** and works across any domain
- Filtering happens **after** graph expansion (not before)
- Graph facts are **preserved** (not subject to similarity thresholds)

---

**Status:** Ready for implementation
**Target Repository:** neo4j_backend (branch: assistance-agents)
**Estimated Time:** 1-2 hours
