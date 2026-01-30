# Agent Execution Flow Diagram

**Query:** "Is there any relationship between Kevin and Alexis?"

---

## 1. Overall Sequential Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER QUERY RECEIVED                         │
│        (Session: 697cae19f039c325f2f109dd)                      │
│        "Is there any relationship between Kevin and Alexis?"    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │      TRIAGE AGENT                 │
         │  (asst_wFLqOr6s8dX3Sp1KNfZYwpkZ)  │
         └────────────────┬──────────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │ Schema Detection & Classification  │
        │ • Check available databases        │
        │ • Detect query intent              │
        │ • Route to appropriate agent       │
        └────────────────┬──────────────────┘
                         │
          ┌──────────────┴────────────────┐
          │  Intent: 'ai_search'          │
          │  Confidence: 1.0              │
          │  Targets: ['Kevin','Alexis']  │
          └──────────────┬────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────┐
    │    AI SEARCH AGENT                   │
    │ (asst_EOiaaKe3CtL5penJUc6ELHrN)      │
    └────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │ Query Analysis  │
        │ • person_names: []
        │ • is_person_query: False
        │ • intent: semantic
        │ • routing: soft
        └────────┬────────┘
                 │
                 ▼
      ┌─────────────────────────────────────┐
      │   GRAPHRAG SEARCH                   │
      │   (Neo4j Backend API Call)          │
      │   http://host.docker.internal:8000 │
      └────────┬────────────────────────────┘
               │
        ┌──────┴──────────────────────────┐
        │  1. RETRIEVE CHUNKS (Vector)    │
        │  • Query embedding              │
        │  • Semantic search: top_k=12    │
        │  • Retrieved: 17 results        │
        └──────┬──────────────────────────┘
               │
        ┌──────┴───────────────────────────┐
        │  2. HYBRID SCORING               │
        │  • Vector similarity scores      │
        │  • Keyword matching scores       │
        │  • Combine: hybrid_score         │
        └──────┬──────────────────────────┘
               │
        ┌──────┴────────────────────────────┐
        │  3. RESULTS ANALYSIS              │
        │  • Alexis chunk: sim=0.294 ✅     │
        │  • Kevin chunk: sim=0.221 ✅      │
        │  • Other chunks: sim=0.0-0.173   │
        │  • Total returned: 17 results    │
        └──────┬────────────────────────────┘
               │
               ▼
   ┌─────────────────────────────────────────┐
   │  RESULT ANALYSIS & LOGGING              │
   │                                          │
   │  📊 SEMANTIC ANALYSIS                   │
   │  ├─ Total with signals: 12              │
   │  ├─ Avg similarity: 0.091               │
   │  └─ Max similarity: 0.294 (Alexis)      │
   │                                          │
   │  🔑 KEYWORD ANALYSIS                    │
   │  ├─ Results with keywords: 6            │
   │  ├─ Avg keyword score: 0.330            │
   │  └─ Max keyword score: 0.560 (Kevin)    │
   │                                          │
   │  🔗 GRAPH ANALYSIS                      │
   │  ├─ Entities detected: 0 (limitation)   │
   │  └─ Graph connections: 0                │
   │                                          │
   │  📋 RETRIEVAL SUMMARY                   │
   │  └─ 12 results analyzed                 │
   └────────────────┬────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │  FILTER RESULTS                         │
    │  (filter_results_by_exact_match)        │
    │                                          │
    │  Input: 12 results                      │
    │  Filter mode: Generic (no person names) │
    │  Threshold: similarity >= 0.3           │
    │                                          │
    │  Detection: Relationship query          │
    │  Names extracted: ['Kevin', 'Alexis']   │
    │                                          │
    │  Output:                                │
    │  ✅ Kept 2 results:                     │
    │     • Result 1: Alexis Torres (0.294)   │
    │     • Result 2: Kevin Ramírez (0.221)   │
    │                                          │
    │  ❌ Removed 10 results (low similarity) │
    │                                          │
    │  📊 FILTER SUMMARY                      │
    │  ├─ By similarity: 0                    │
    │  ├─ By name matching: 2 ✅             │
    │  └─ By graph discovery: 0              │
    └────────────────┬──────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────┐
    │  CONTEXT ASSEMBLY                  │
    │  • Filtered results: 2 chunks      │
    │  • Context length: 1,309 chars     │
    │  • Prepare for LLM synthesis       │
    └────────────────┬───────────────────┘
                     │
                     ▼
         ┌────────────────────────────────┐
         │  ASSISTANT AGENT               │
         │ (asst_2CYLm2SZUNQYMPxYrftrgzJB)│
         │                                │
         │ Task:                          │
         │ • Synthesize answer from       │
         │   filtered chunks              │
         │ • Use LLM to generate          │
         │   response                     │
         │ • Response length: 685 chars   │
         └────────────────┬────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  REVIEW AGENT                  │
         │ (asst_hih0dA6Xc2sC5tRzyaiQyGRF)│
         │                                │
         │ Pass 1/3:                      │
         │ ✅ ACCEPT                      │
         │ Reason: Response is factually  │
         │ correct, cites sources,       │
         │ aligns with metadata          │
         └────────────────┬────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  RESPONSE SENT TO USER         │
         │                                │
         │  "Both resumes are for         │
         │  professionals associated     │
         │  with DXC (as seen from       │
         │  the file paths), but         │
         │  there is..."                 │
         └────────────────────────────────┘
```

---

## 2. Detailed Agent Flow Sequence

### **Phase 1: Triage Agent** ⏱️ 53 seconds
```
START: 17:01:18.404
END:   17:01:18.586

┌─────────────────────────────────────────┐
│  TriageAgent                            │
└─────────────────────────────────────────┘
        │
        ├─→ Initialize Schema Retriever
        │   └─ URL: http://host.docker.internal:8000
        │
        ├─→ Health Check
        │   └─ Status: 200 ✅
        │
        ├─→ List Available Databases
        │   └─ Response: Empty list ⚠️
        │
        ├─→ Retry Database Listing
        │   └─ Response: Still empty ⚠️
        │
        └─→ Fallback Classification
            ├─ Intent: 'ai_search'
            ├─ Confidence: 1.0 (100%)
            ├─ Targets: ['Kevin', 'Alexis']
            ├─ Database ID: None
            └─ Preferred Agent: None
                    │
                    ▼
        DECISION: Route to AiSearchAgent
```

---

### **Phase 2: AI Search Agent** ⏱️ 1 minute 46 seconds
```
START: 17:01:18.784
END:   17:02:04.962

┌─────────────────────────────────────────┐
│  AiSearchAgent                          │
└─────────────────────────────────────────┘
        │
        ├─→ Load Header Vocabulary
        │   ├─ File: header_vocab.json
        │   └─ Loaded: 203 phrases ✅
        │
        ├─→ Query Classification
        │   ├─ Intent: semantic
        │   ├─ Routing: soft
        │   └─ Person names: [] (empty)
        │
        ├─→ GraphRAG Schema Discovery
        │   ├─ Request: /api/graphrag/schema
        │   ├─ Node labels: 17
        │   ├─ Relationship types: 17
        │   ├─ AT_ORGANIZATION edges: 6
        │   └─ Status: 200 ✅
        │
        └─→ GRAPHRAG SEARCH
            │
            ├─→ Neo4j Query Call
            │   ├─ URL: http://host.docker.internal:8000/api/graphrag/query
            │   ├─ Method: Hybrid (vector + keyword)
            │   ├─ Query: "Is there any relationship between Kevin and Alexis?"
            │   ├─ top_k_vector: 12
            │   ├─ similarity_threshold: 0.75
            │   ├─ use_keyword_search: True
            │   ├─ keyword_match_type: any
            │   └─ enable_coworker_expansion: True
            │
            ├─→ Vector Search Results (17 total)
            │   ├─ Results 1-6: Direct chunks (hybrid_score: 0.890, sim: 0.0)
            │   │   └─ Introduction sections, contact info, skills
            │   ├─ Result 7: 📄 Alexis Torres - DXC Resume.pdf
            │   │   ├─ Similarity: 0.294 ✅ (Best match)
            │   │   ├─ Keyword score: 0.460
            │   │   ├─ Hybrid score: 0.399
            │   │   └─ Header: "Alexis Torres Senior Technical Consultant"
            │   ├─ Result 8: 📄 Kevin Ramirez DXC Resume.pdf
            │   │   ├─ Similarity: 0.221 ✅
            │   │   ├─ Keyword score: 0.560
            │   │   ├─ Hybrid score: 0.357
            │   │   └─ Header: "Kevin J. Ramírez Pomales"
            │   └─ Results 9-12: Other sections (sim: 0.127-0.173)
            │       └─ Certifications, roles, skills
            │
            ├─→ Analysis: SEMANTIC ANALYSIS
            │   ├─ Results analyzed: 12
            │   ├─ Avg similarity: 0.091
            │   ├─ Max similarity: 0.294 (Alexis)
            │   ├─ Second best: 0.221 (Kevin)
            │   └─ Third best: 0.173
            │
            ├─→ Analysis: KEYWORD ANALYSIS
            │   ├─ Results with keywords: 6 / 12
            │   ├─ Avg keyword score: 0.330
            │   ├─ Max keyword score: 0.560 (Kevin)
            │   └─ Second: 0.460 (Alexis)
            │
            ├─→ Analysis: GRAPH ANALYSIS
            │   ├─ Detected entities: 0 ❌
            │   │   └─ Limitation: header_vocab parsing
            │   └─ Graph connections: 0
            │
            ├─→ Analysis: RETRIEVAL SUMMARY
            │   └─ Processed: 12 results total
            │
            └─→ Analysis: FILTER SUMMARY (Before Filtering)
                ├─ Results by similarity threshold:
                │   └─ >= 0.3: 2 results (Alexis, Kevin)
                ├─ Results by keyword:
                │   └─ 6 matches
                └─ Results by graph:
                    └─ 0 matches
```

---

### **Phase 3: Filtering & Context Assembly** ⏱️ 2.3 seconds
```
START: 17:01:20.341
END:   17:02:04.962

FILTER FUNCTION: filter_results_by_exact_match

Input: 12 results
Mode: Generic (no person_names provided)

┌─────────────────────────────────────────┐
│  FILTER LOGIC                           │
└─────────────────────────────────────────┘
        │
        ├─→ Check mode: is_person_query=False
        │   └─ Using: Generic mode
        │
        ├─→ Extract names from query
        │   ├─ Query text: "Is there any relationship between Kevin and Alexis?"
        │   ├─ Regex pattern match
        │   └─ Names found: ['Kevin', 'Alexis']
        │
        ├─→ Relationship detection
        │   ├─ Pattern: "relationship", "between"
        │   └─ Result: Relationship query detected ✅
        │
        ├─→ Apply filtering logic
        │   │
        │   ├─ Result 1-6: Direct chunks
        │   │  └─ Similarity: 0.0 < 0.3 ❌ REMOVE
        │   │
        │   ├─ Result 7: Alexis Torres Resume
        │   │  ├─ Similarity: 0.294
        │   │  ├─ Name match: "Alexis" ✅
        │   │  └─ Status: KEEP (by name matching)
        │   │
        │   ├─ Result 8: Kevin Ramirez Resume
        │   │  ├─ Similarity: 0.221
        │   │  ├─ Name match: "Kevin" ✅
        │   │  └─ Status: KEEP (by name matching)
        │   │
        │   └─ Result 9-12: Other sections
        │      └─ Similarity: 0.127-0.173 < 0.3 ❌ REMOVE
        │
        └─→ OUTPUT FILTER SUMMARY
            ├─ Kept by similarity: 0
            ├─ Kept by name: 2 ✅
            ├─ Kept by graph: 0
            ├─ Total kept: 2
            ├─ Total removed: 10
            └─ Kept results:
                ├─ Alexis Torres (Header: "Alexis Torres Senior Technical Consultant")
                └─ Kevin Ramirez (Header: "Kevin J. Ramírez Pomales")

CONTEXT ASSEMBLY:
├─ Chunk 1: Alexis Torres - DXC Resume (Intro section)
├─ Chunk 2: Kevin Ramirez - DXC Resume (Intro section)
├─ Total context: 1,309 characters
└─ Ready for LLM synthesis
```

---

### **Phase 4: Assistant Agent** ⏱️ 13.6 seconds
```
START: 17:02:04.962
END:   17:02:18.050

┌─────────────────────────────────────────┐
│  AssistantAgent                         │
│ (asst_2CYLm2SZUNQYMPxYrftrgzJB)         │
└─────────────────────────────────────────┘
        │
        ├─→ Receive filtered context
        │   ├─ Sources: 2 chunks
        │   ├─ Length: 1,309 chars
        │   └─ Contains: Alexis & Kevin intro sections
        │
        ├─→ Call Azure OpenAI (Foundry)
        │   ├─ Model: gpt-4
        │   ├─ Temperature: Auto
        │   ├─ Prompt: Synthesize relationship query answer
        │   └─ Max tokens: 2048
        │
        └─→ Generate Response
            ├─ Duration: ~13 seconds
            ├─ Output length: 685 characters
            └─ Content: "Both resumes are for professionals
                        associated with DXC (as seen from
                        the file paths), but there is..."

                    Response includes:
                    • File path references
                    • Professional information
                    • Relationship acknowledgment
                    • Source attribution
```

---

### **Phase 5: Review Agent** ⏱️ 7.7 seconds
```
START: 17:02:18.050
END:   17:02:25.802

┌─────────────────────────────────────────┐
│  ReviewAgent                            │
│ (asst_hih0dA6Xc2sC5tRzyaiQyGRF)        │
└─────────────────────────────────────────┘
        │
        ├─→ Pass 1/3
        │   ├─ Evaluate response quality
        │   ├─ Check factual accuracy
        │   ├─ Verify source attribution
        │   └─ Decision: ✅ ACCEPT
        │       └─ Reason: Response accurately addresses
        │                  the query, is factually correct,
        │                  clearly cites file paths and
        │                  chunk indices, and aligns with
        │                  provided source metadata
        │
        └─→ Finalize Response
            ├─ Status: Approved
            ├─ Quality: High
            └─ Send to User ✅
```

---

## 3. Timeline Overview

```
17:01:18.027  ├─ USER QUERY RECEIVED
              │   "Is there any relationship between Kevin and Alexis?"
              │
17:01:18.033  ├─ TRIAGE AGENT STARTED
              │
17:01:18.404  ├─ TRIAGE AGENT → Schema Detection
              │
17:01:18.586  ├─ TRIAGE AGENT COMPLETED
              │   └─ Decision: ai_search intent
              │
17:01:18.784  ├─ AI SEARCH AGENT STARTED
              │
17:01:18.785  ├─ GraphRAG Search Initiated
              │
17:01:20.332  ├─ GraphRAG Results Received (17 chunks)
              │   ├─ Alexis Torres: similarity=0.294
              │   └─ Kevin Ramirez: similarity=0.221
              │
17:01:20.334  ├─ ANALYSIS LOGS GENERATED
              │   ├─ Semantic Analysis
              │   ├─ Keyword Analysis
              │   ├─ Graph Analysis
              │   └─ Retrieval Summary
              │
17:01:20.341  ├─ FILTERING STARTED
              │
17:01:20.346  ├─ FILTERING COMPLETED
              │   └─ Kept: 2 results (by name matching)
              │
17:02:04.962  ├─ CONTEXT ASSEMBLED (1,309 chars)
              │
17:02:04.962  ├─ ASSISTANT AGENT STARTED
              │
17:02:18.050  ├─ ASSISTANT AGENT COMPLETED
              │   └─ Response generated (685 chars)
              │
17:02:18.050  ├─ REVIEW AGENT STARTED
              │
17:02:25.802  ├─ REVIEW AGENT COMPLETED
              │   └─ Decision: ACCEPT ✅
              │
17:02:25.805  └─ RESPONSE SENT TO USER
                  Total Duration: 1 minute 7.8 seconds
```

---

## 4. Key Decision Points

### **Decision 1: Intent Classification** (Triage Agent)
```
Input: "Is there any relationship between Kevin and Alexis?"
Factors:
├─ Contains relationship keyword: YES
├─ Contains two entity names: YES ("Kevin", "Alexis")
├─ Question format: YES

Output: Intent = 'ai_search'
         Confidence = 1.0 (100%)
         Targets = ['Kevin', 'Alexis']
```

### **Decision 2: Query Routing** (Triage Agent)
```
Available routes:
├─ SQL Search ❌ (No database endpoint)
├─ AI Search ✅ (Generic, supports any query)
└─ Fallback: Use default routing

Decision: Route to AiSearchAgent
```

### **Decision 3: Search Type** (AI Search Agent)
```
Analysis:
├─ Is person query: False (Not explicitly a profile query)
├─ Intent: Semantic
├─ Routing: Soft

Search parameters:
├─ Mode: Hybrid (vector + keyword)
├─ top_k: 12
├─ similarity_threshold: 0.75
├─ keyword_boost: 0.0
└─ enable_coworker_expansion: True
```

### **Decision 4: Filtering** (AI Search Agent)
```
Initial results: 12 from GraphRAG (17 before limiting)
Semantic signals weak: Avg similarity = 0.091

Filtering applied:
├─ Threshold: similarity >= 0.3
├─ Relationship detection: YES
├─ Name extraction: ['Kevin', 'Alexis']

Result:
├─ Standard filtering: 2 results pass
├─ Name matching: 2 results (Alexis + Kevin)
└─ Graph matching: 0 results

Final kept: 2 results
```

### **Decision 5: Context Assembly** (AI Search Agent)
```
Selected chunks:
├─ Alexis Torres intro section (74 chars)
├─ Kevin Ramirez intro section (576 chars)
└─ Total: 1,309 chars

LLM receives: Both professionals' intro data
Expected: LLM synthesizes relationship info from bios
```

### **Decision 6: Response Quality** (Review Agent)
```
Evaluation criteria:
├─ Accuracy: ✅ Factually correct
├─ Attribution: ✅ Cites file paths
├─ Completeness: ✅ Addresses query
├─ Relevance: ✅ Aligns with sources

Result: ✅ ACCEPT (Pass 1/3)
```

---

## 5. Performance Metrics

| Phase | Duration | Start Time | End Time | Status |
|-------|----------|-----------|---------|--------|
| User Input | — | 17:01:18.027 | 17:01:18.027 | ✅ |
| TriageAgent | 0.559s | 17:01:18.033 | 17:01:18.586 | ✅ |
| AI Search Init | 0.001s | 17:01:18.784 | 17:01:18.785 | ✅ |
| GraphRAG Query | 1.40s | 17:01:18.785 | 17:01:20.332 | ✅ |
| Analysis Logs | 0.008s | 17:01:20.334 | 17:01:20.342 | ✅ |
| Filtering | 0.008s | 17:01:20.341 | 17:01:20.346 | ✅ |
| Context Build | 44.6s | 17:01:20.346 | 17:02:04.962 | ✅ |
| AssistantAgent | 13.1s | 17:02:04.962 | 17:02:18.050 | ✅ |
| ReviewAgent | 7.8s | 17:02:18.050 | 17:02:25.802 | ✅ |
| **TOTAL** | **67.8s** | 17:01:18.027 | 17:02:25.805 | ✅ |

**Breakdown:**
- GraphRAG processing: 1.40s (2.1%)
- LLM synthesis: 13.1s (19.3%)
- Context/embedding: 44.6s (65.7%)
- Workflow coordination: 8.4s (12.4%)

---

## 6. Data Flow Summary

```
Query Text
    ↓
[TriageAgent]
    ├─ Classify intent
    └─ Route decision
    ↓
[AiSearchAgent]
    ├─ Parse query
    ├─ Load vocabulary
    └─ Call GraphRAG
    ↓
[Neo4j Backend]
    ├─ Vector search
    ├─ Keyword search
    ├─ Hybrid scoring
    └─ Return 17 results
    ↓
[Analysis Layer]
    ├─ Semantic analysis (vector scores)
    ├─ Keyword analysis (keyword scores)
    ├─ Graph analysis (entity relationships)
    └─ Generate summary logs
    ↓
[Filtering Layer]
    ├─ Extract entity names
    ├─ Apply thresholds
    ├─ Match patterns
    └─ Return 2 results
    ↓
[AssistantAgent]
    ├─ Receive context
    ├─ Call LLM
    └─ Generate response
    ↓
[ReviewAgent]
    ├─ Evaluate quality
    ├─ Check accuracy
    └─ Approve response
    ↓
[User]
    └─ Receive answer
```

---

## 7. Error Handling & Fallbacks

```
Potential Issues & Mitigations:

1. Database Listing Failed ⚠️
   └─ Mitigation: Fall back to default routing ✅

2. Schema Discovery
   └─ Status: Success (17 node types, 17 relationships) ✅

3. Header Vocabulary Loading
   └─ Status: Success (203 phrases loaded) ✅

4. Neo4j Connection
   └─ Status: Success (200 OK response) ✅

5. Vector Search
   └─ Status: Success (17 results returned) ✅

6. Entity Extraction from Headers
   └─ Status: Limitation (0 entities detected)
   └─ Workaround: Name regex from query string ✅

7. Response Quality
   └─ Review: ACCEPT (factually correct) ✅
```

---

## 8. Key Insights

✅ **What Worked:**
- Sequential workflow orchestration
- Hybrid search (vector + keyword) effective
- Filtering preserves relevant results
- Multiple verification passes

⚠️ **Challenges:**
- Entity extraction from headers not working (limitation)
- Graph relationship discovery returned 0 (not triggered)
- Vector similarity low (0.091 avg) but acceptable with filtering
- Long context/embedding time (44.6s) - potential bottleneck

🔍 **Observations:**
- Relationship detection from query text effective
- Name-based filtering working well
- LLM synthesis from intro sections adequate
- Review agent adds quality assurance

📊 **Signals:**
- Semantic: Weak (0.091 avg)
- Keyword: Moderate (0.330 avg)
- Graph: Not triggered (0 connections)
- Combined result: Acceptable answer
