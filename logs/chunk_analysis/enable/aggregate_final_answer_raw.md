# Aggregate Final Answer - Raw Chunks (Pre-LLM)
This file captures raw chunks before LLM summarization.
(File overwrites with each new query)

---

## Execution: 2026-02-13T22:16:00.375378
**Query**: What is                             VectorCypherRetriever

---

## Passing Chunks (Append Log)

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:75
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.744827
**Selection Method:** keyword

```text
Import Libraries
This notebook imports the core libraries required for building and querying RAG pipelines with Neo4j and GraphRAG:

- neo4j.GraphDatabase: The official Python driver for connecting to and querying a Neo4j database.
- neo4j\_graphrag.llm.OpenAILLM : Integrates OpenAI language models for generating and processing natural language queries.
- neo4j\_graphrag.embeddings. OpenAIEmbeddings : Provides access to OpenAI's embedding models for generating vector representations of text.
- Neo4j\_graphrag.retrievers : Different retriever classes for semantic and hybrid search over graph data using vector similarity and Cypher queries:
- VectorRetriever
- VectorCypherRetriever
- Text2CypherRetriever
- neo4j\_graphrag.generation.GraphRAG : The main class for orchestrating RAG workflows over a Neo4j knowledge graph.
- neo4j\_graphrag.schema.get\_schema : Utility to introspect and retrieve the schema of your Neo4j database.
- dotenv.load\_dotenv : Loads environment variables (such as credentials and API keys) from an .env file for secure configuration.

These imports enable advanced semantic search, retrieval, and GenAI capabilities directly on your Neo4j knowledge graph.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:84
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.777342
**Selection Method:** keyword

```text
The Graph-Enhanced Vector Search Pattern
The basic retriever pattern typically relies on textbased embeddings, capturing only the semantic meaning of content. While this method is effective in identifying similar chunks, it leaves the LLM in the dark as to how those items interact in the real world.

The Graph-Enhanced Vector Search Pattern, also known as augmented vector search, overcomes this limitation by drawing on the graph structure (i.e., using not just what items are but also how they connect). By embedding node positions and relationships within a graph, this approach generates contextually relevant nodes, integrating both:

-  Unstructured data: Product descriptions, customer reviews, and other text content via semantic similarity
-  Structured data: Purchase patterns, category relationships, and transaction records via explicit instructions

The VectorCypherRetriever uses the full graph capabilities of Neo4j by combining vector-based similarity searches with graph traversal techniques. The retriever completes the following actions:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:85
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.783345
**Selection Method:** keyword

```text
Executes a Cypher query to traverse the graph based on these nodes.
To set up this particular query, you need to tell the graph where and how to traverse from the semantic nodes. In this example, the query is:

'What are the risk factors for companies discussing cryptocurrency in their filings?'

The following code creates a retriever to answer this query:


Let's start by looking at the parts of the graph that help to answer this query. We start by identifying the Chunk that is semantically similar to the cryptocurrency query. Then we need to traverse the graph to identify the Document the Chunk comes from, the Company that FILED the Document and collect the other RiskFactors for that Company . Once this information is retrieved, it's converted to Cypher and set as the retrieval query.


Figure 27.  VectorCypherRetriever example 1



Next, let's add this new retrieval query to the VectorCypherRetriever parameters:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:86
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.788345
**Selection Method:** keyword

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name : The name of the vector index (here, chunkEmbeddings ) used for semantic search
- Embedder : The embedding model used to generate/query vector representations
- Retrieval\_query : The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

This setup enables you to start with a semantic search (e.g., for 'cryptocurrency risk') and automatically traverse your knowledge graph to reveal which companies are involved and what other risks they face. The resulting responses are both semantically relevant and graph-aware.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:88
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.796345
**Selection Method:** keyword

```text
VectorCypher Retriever in Practice
eports (text) with the graph relationships between vulnerabilities, affected assets, and mitigation strategies to provide a holistic view of your security posture.
-  Education: Map student essays or discussion posts to learning objectives, course materials, and assessment outcomes for personalized education analytics.

Let's summarize the major tasks from  this example so you can apply it to your domain:

-  Adapt the Pattern Model Your Domain: Define the node types, relationships, and key properties relevant to your vertical (e.g., Patient, Diagnosis, Product, Supplier, Case, Asset, etc.).
- Index the Right Data: Create vector indexes on the appropriate text or document nodes for semantic retrieval.
-  Craft Domain-Specific Cypher Queries: Write Cypher queries that traverse from the retrieved nodes to related entities and/or relationships that matter in your context.
-  Integrate With VectorCypherRetriever: Use the VectorCypherRetriever with your custom query to combine semantic and structural search.

The result: You can ask complex, context-aware questions about entities in your own industry. The GraphRAG retriever will surface relevant information that connects context across
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:89
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.801344
**Selection Method:** keyword

```text
VectorCypher Retriever in Practice
structural search.

The result: You can ask complex, context-aware questions about entities in your own industry. The GraphRAG retriever will surface relevant information that connects context across structured and unstructured data to drive real-world understanding.

With this in mind, let's look at another VectorCypherRetriever example.



## VectorCypher Retrieval: A Working Example

Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:90
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.806343
**Selection Method:** keyword

```text
VectorCypher Retrieval: A Working Example
Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:91
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** keyword_pre_filter
**Captured:** 2026-02-13T22:16:00.811343
**Selection Method:** keyword

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name: The name of the vector index (here, chunkEmbeddings ) used for semantic search


- Embedder: The embedding model used to generate/query vector representations
- Retrieval\_query: The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

```
result = vector_cypher_retriever.search(query_text=query, top_k=10) for item in result.items: print(item.content[:100])
```
```

## Final Selected Chunks

### File: Developers-Guide-GraphRAG.pdf (ID: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf)

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:75

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
Import Libraries
This notebook imports the core libraries required for building and querying RAG pipelines with Neo4j and GraphRAG:

- neo4j.GraphDatabase: The official Python driver for connecting to and querying a Neo4j database.
- neo4j\_graphrag.llm.OpenAILLM : Integrates OpenAI language models for generating and processing natural language queries.
- neo4j\_graphrag.embeddings. OpenAIEmbeddings : Provides access to OpenAI's embedding models for generating vector representations of text.
- Neo4j\_graphrag.retrievers : Different retriever classes for semantic and hybrid search over graph data using vector similarity and Cypher queries:
- VectorRetriever
- VectorCypherRetriever
- Text2CypherRetriever
- neo4j\_graphrag.generation.GraphRAG : The main class for orchestrating RAG workflows over a Neo4j knowledge graph.
- neo4j\_graphrag.schema.get\_schema : Utility to introspect and retrieve the schema of your Neo4j database.
- dotenv.load\_dotenv : Loads environment variables (such as credentials and API keys) from an .env file for secure configuration.

These imports enable advanced semantic search, retrieval, and GenAI capabilities directly on your Neo4j knowledge graph.
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:84

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
The Graph-Enhanced Vector Search Pattern
The basic retriever pattern typically relies on textbased embeddings, capturing only the semantic meaning of content. While this method is effective in identifying similar chunks, it leaves the LLM in the dark as to how those items interact in the real world.

The Graph-Enhanced Vector Search Pattern, also known as augmented vector search, overcomes this limitation by drawing on the graph structure (i.e., using not just what items are but also how they connect). By embedding node positions and relationships within a graph, this approach generates contextually relevant nodes, integrating both:

-  Unstructured data: Product descriptions, customer reviews, and other text content via semantic similarity
-  Structured data: Purchase patterns, category relationships, and transaction records via explicit instructions

The VectorCypherRetriever uses the full graph capabilities of Neo4j by combining vector-based similarity searches with graph traversal techniques. The retriever completes the following actions:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:85

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
Executes a Cypher query to traverse the graph based on these nodes.
To set up this particular query, you need to tell the graph where and how to traverse from the semantic nodes. In this example, the query is:

'What are the risk factors for companies discussing cryptocurrency in their filings?'

The following code creates a retriever to answer this query:


Let's start by looking at the parts of the graph that help to answer this query. We start by identifying the Chunk that is semantically similar to the cryptocurrency query. Then we need to traverse the graph to identify the Document the Chunk comes from, the Company that FILED the Document and collect the other RiskFactors for that Company . Once this information is retrieved, it's converted to Cypher and set as the retrieval query.


Figure 27.  VectorCypherRetriever example 1



Next, let's add this new retrieval query to the VectorCypherRetriever parameters:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:86

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name : The name of the vector index (here, chunkEmbeddings ) used for semantic search
- Embedder : The embedding model used to generate/query vector representations
- Retrieval\_query : The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

This setup enables you to start with a semantic search (e.g., for 'cryptocurrency risk') and automatically traverse your knowledge graph to reveal which companies are involved and what other risks they face. The resulting responses are both semantically relevant and graph-aware.
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:88

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
VectorCypher Retriever in Practice
eports (text) with the graph relationships between vulnerabilities, affected assets, and mitigation strategies to provide a holistic view of your security posture.
-  Education: Map student essays or discussion posts to learning objectives, course materials, and assessment outcomes for personalized education analytics.

Let's summarize the major tasks from  this example so you can apply it to your domain:

-  Adapt the Pattern Model Your Domain: Define the node types, relationships, and key properties relevant to your vertical (e.g., Patient, Diagnosis, Product, Supplier, Case, Asset, etc.).
- Index the Right Data: Create vector indexes on the appropriate text or document nodes for semantic retrieval.
-  Craft Domain-Specific Cypher Queries: Write Cypher queries that traverse from the retrieved nodes to related entities and/or relationships that matter in your context.
-  Integrate With VectorCypherRetriever: Use the VectorCypherRetriever with your custom query to combine semantic and structural search.

The result: You can ask complex, context-aware questions about entities in your own industry. The GraphRAG retriever will surface relevant information that connects context across
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:89

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
VectorCypher Retriever in Practice
structural search.

The result: You can ask complex, context-aware questions about entities in your own industry. The GraphRAG retriever will surface relevant information that connects context across structured and unstructured data to drive real-world understanding.

With this in mind, let's look at another VectorCypherRetriever example.



## VectorCypher Retrieval: A Working Example

Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:90

**Analyzed At:** 2026-02-13T22:16:00.871873
**Status:** Keyword selected

```text
VectorCypher Retrieval: A Working Example
Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:91

**Analyzed At:** 2026-02-13T22:16:00.872871
**Status:** Keyword selected

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name: The name of the vector index (here, chunkEmbeddings ) used for semantic search


- Embedder: The embedding model used to generate/query vector representations
- Retrieval\_query: The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

```
result = vector_cypher_retriever.search(query_text=query, top_k=10) for item in result.items: print(item.content[:100])
```
```

**Summary:**
- Selected chunks: 8
- Total chars: 7008

