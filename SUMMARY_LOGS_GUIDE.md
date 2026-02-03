# Summary Logs Guide: Understanding Semantic, Keyword & Graph Analysis

## Overview

Added comprehensive summary logs to help understand how the Agent analyzes queries through three different signals:
1. **Semantic Analysis** - Vector similarity matching
2. **Keyword Analysis** - Exact keyword matching
3. **Graph Analysis** - Relationship discovery

---

## Log Sections

### 1. **RETRIEVAL SUMMARY**
Shows the overall analysis of what signals were found in the results.

**What to look for:**
- Total results analyzed
- How many have semantic signals (vector matching)
- How many have keyword matches
- Whether graph connections were discovered

**Example output:**
```
🔍 [RETRIEVAL SUMMARY] Analyzing 12 results...

📊 [SEMANTIC ANALYSIS]
   Total with semantic signals: 12
   Avg similarity: 0.215, Max: 0.294
   Top semantic matches:
     - Alexis Torres - DXC Resume.pdf: similarity=0.294, header='Alexis Torres Senior Technical Consultant'
     - 20250912 Kevin Ramirez DXC Resume.pdf: similarity=0.221, header='Kevin J. Ramírez Pomales'

🔑 [KEYWORD ANALYSIS]
   Total with keyword matches: 8
   Avg keyword score: 0.450, Max: 0.560
   Top keyword matches:
     - 20250912 Kevin Ramirez DXC Resume.pdf: keyword_score=0.560, header='Kevin J. Ramírez Pomales'
     - Alexis Torres - DXC Resume.pdf: keyword_score=0.460, header='Alexis Torres Senior Technical Consultant'

🔗 [GRAPH ANALYSIS]
   Total graph connections found: 1
   Graph connections:
     - Kevin J. Ramírez Pomales <-[::AT_ORGANIZATION]-> Alexis Torres via DXC Technology
```

### 2. **SEMANTIC ANALYSIS**
Breaking down vector similarity matching.

**What this tells you:**
- How well the query matched documents semantically
- Average and maximum similarity scores
- Which files had the best semantic match
- The header/section of best matches

**Interpretation:**
- **Avg similarity > 0.3**: Good semantic matches found
- **Avg similarity < 0.15**: Weak semantic matching (may need keyword/graph help)
- **Max similarity close to avg**: Consistent matching across results
- **Max similarity >> avg**: One very strong match with weak others

### 3. **KEYWORD ANALYSIS**
Showing exact keyword matching in content.

**What this tells you:**
- How many results had explicit keyword matches
- Average and maximum keyword scores
- Which files matched keywords best
- Keyword boost effectiveness

**Interpretation:**
- **High keyword score (>0.5)**: Strong keyword presence in document
- **Low keyword score (<0.2)**: Few keywords found
- **High count**: Keyword-based search working well
- **Low count**: Query needs more semantic/graph help

### 4. **GRAPH ANALYSIS**
Relationship discovery between entities.

**What this tells you:**
- Whether graph relationships were found
- The entities connected (entity1 → entity2)
- The relationship type (e.g., AT_ORGANIZATION)
- What node connects them (via)

**Example interpretation:**
```
Kevin J. Ramírez Pomales <-[::AT_ORGANIZATION]-> Alexis Torres via DXC Technology
```
Means: Both Kevin and Alexis have `AT_ORGANIZATION` relationships with DXC Technology

**Interpretation:**
- **Found graph connections**: Indicates the query asked about relationships
- **No graph connections**: Either single-entity query or no relationships exist
- Multiple connections: Rich relationships discovered

### 5. **FILTER SUMMARY**
Shows what happened during filtering.

**What this tells you:**
- How many results were kept vs. removed
- Why each result passed filtering:
  - **By similarity**: Had similarity >= 0.3
  - **By name matching**: Contained person/entity names
  - **By graph discovery**: Was a graph connection result
- Sample of removed results and why

**Example output:**
```
📊 [FILTER SUMMARY]
   Kept 2 results:
     - By similarity (>= 0.3): 0
     - By name matching: 2
     - By graph discovery: 0
   Removed 10 results (low similarity, no match)
   Sample removed (low similarity):
     - Certifications section: similarity=0.173, header='...'
     - Skills section: similarity=0.127, header='...'
```

**What to understand:**
- If `Kept by similarity` is high: Good direct matches
- If `Kept by name matching` is high: Relationship query filtering working
- If `Kept by graph discovery` is > 0: Graph relationships found
- If `Removed` is high: Many low-quality results filtered out (good!)

---

## Real-World Example: Kevin-Alexis Relationship Query

**Query:** "Is there any relationship between Kevin and Alexis?"

**Expected Log Output:**

```
🔍 [RETRIEVAL SUMMARY] Analyzing 12 results...

📊 [SEMANTIC ANALYSIS]
   Total with semantic signals: 12
   Avg similarity: 0.215, Max: 0.294
   Top semantic matches:
     - Alexis Torres - DXC Resume.pdf: similarity=0.294
     - Kevin Ramirez DXC Resume.pdf: similarity=0.221

🔑 [KEYWORD ANALYSIS]
   Total with keyword matches: 8
   Avg keyword score: 0.450, Max: 0.560

🔗 [GRAPH ANALYSIS]
   Total graph connections found: 1
   Graph connections:
     - Kevin J. Ramírez Pomales <-[::AT_ORGANIZATION]-> Alexis Torres via DXC Technology

📊 [FILTER SUMMARY]
   Kept 2 results:
     - By similarity (>= 0.3): 0
     - By name matching: 2
     - By graph discovery: 0
   Removed 10 results (low similarity, no match)
```

**What this shows:**
1. ✅ Both people found (Kevin + Alexis)
2. ✅ Low semantic match (0.22) but both matched by name
3. ✅ Graph found the connection through DXC
4. ✅ Filter correctly kept 2 results based on name matching
5. ✅ LLM will now see both resumes + their shared organization

---

## How to Read the Logs

### Find Recent Logs
```bash
docker logs rag-agent --since "2026-01-30T12:00:00" | grep -E "RETRIEVAL|SEMANTIC|KEYWORD|GRAPH|FILTER"
```

### Full Logs with Context
```bash
docker logs rag-agent --tail 500 | grep -E "RETRIEVAL SUMMARY|SEMANTIC ANALYSIS|KEYWORD ANALYSIS|GRAPH ANALYSIS|FILTER SUMMARY" -A 10
```

### Filter to Just Graph Analysis
```bash
docker logs rag-agent --tail 500 | grep -E "GRAPH ANALYSIS" -A 5
```

---

## Interpreting Common Scenarios

### Scenario 1: Good Query (All Signals Strong)
```
📊 [SEMANTIC ANALYSIS]
   Total with semantic signals: 5
   Avg similarity: 0.75, Max: 0.92
   
🔑 [KEYWORD ANALYSIS]
   Total with keyword matches: 5
   Avg keyword score: 0.85
   
🔗 [GRAPH ANALYSIS]
   Total graph connections found: 2
```
**Meaning:** Excellent! Multiple strong signals confirm relevance.

### Scenario 2: Weak Semantic, Strong Keyword (Query Mismatch)
```
📊 [SEMANTIC ANALYSIS]
   Avg similarity: 0.12, Max: 0.25
   
🔑 [KEYWORD ANALYSIS]
   Total with keyword matches: 8
   Avg keyword score: 0.65
```
**Meaning:** Words are relevant but embedding doesn't match. May be format issue.

### Scenario 3: Relationship Query (Should See Graph Results)
```
🔗 [GRAPH ANALYSIS]
   Total graph connections found: 1
   Graph connections:
     - Entity1 <-[::RELATIONSHIP_TYPE]-> Entity2 via Connection
```
**Meaning:** Graph discovery working! Relationship found.

### Scenario 4: No Results (All Signals Weak)
```
📊 [SEMANTIC ANALYSIS]
   Total with semantic signals: 1
   Avg similarity: 0.08, Max: 0.12
   
🔑 [KEYWORD ANALYSIS]
   Total with keyword matches: 0
   
🔗 [GRAPH ANALYSIS]
   Total graph connections found: 0
```
**Meaning:** Query didn't match documents well. May need to refine search.

---

## Debugging Tips

### Query Not Finding Results?
1. Check **SEMANTIC ANALYSIS** - if Avg similarity << 0.3, query embedding doesn't match
2. Check **KEYWORD ANALYSIS** - if count = 0, no exact phrases matched
3. Check **GRAPH ANALYSIS** - if looking for relationship but count = 0, entity extraction may have failed

### Relationship Query Returning Wrong Results?
1. Check **GRAPH ANALYSIS** - are the right entities being connected?
2. Check **FILTER SUMMARY** - are graph results being kept?
3. Check entity extraction logic if `Total graph connections = 0`

### Filtering Too Aggressive?
1. Check **FILTER SUMMARY** - "Removed X results"
2. Check similarity threshold (currently 0.3)
3. Look at "Sample removed" to see what was discarded

---

## Summary

These logs provide **complete visibility** into:
- ✅ What the semantic search found
- ✅ What keyword matches exist
- ✅ What graph relationships were discovered
- ✅ Why results were kept or removed
- ✅ Whether filtering is working correctly

Use them to debug queries that aren't returning expected results!
