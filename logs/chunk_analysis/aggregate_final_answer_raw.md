# Aggregate Final Answer - Raw Chunks (Pre-LLM)
This file captures raw chunks before LLM summarization.
(File overwrites with each new query)

---

## Execution: 2026-02-08T02:22:13.857850
**Query**: What is VectorCypher Retrieva

---

## Passing Chunks (Append Log)

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:0
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:15.475411

```text
THE DEVELOPER'S GUIDE TO GraphRAG
Alison Cossette Zach Blumenfeld Damaso Sanoja
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:1
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:16.411537

```text
Table of Contents
| PART I: The Problem With Current RAG...............................................................................             |   4 |
|---------------------------------------------------------------------------------------------------------------------------------|-----|
| PART II: What Makes It GraphRAG-Structure, Logic, and Meaning........................                                           |   5 |
| What Is RAG? .................................................................................................................. |   5 |
| What Is GraphRAG?......................................................................................................         |   6 |
| 1. Context-Aware Responses.....................................................................                                 |   6 |
| 2. Traceability and Explainability ............................................................                                 |   6 |
| 3, Access to Structured and Unstructured Data ..............................                                                    |   7 |
| How GraphRAG
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:5
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:17.708873

```text
Table of Contents (continued)
| PART IV: Implementing GraphRAG Retrieval Patterns ..................................................                            |   17 |
|---------------------------------------------------------------------------------------------------------------------------------|------|
| Import Libraries............................................................................................................... |   17 |
| Load Environment Variables and Initialize Neo4j Driver................................                                          |   18 |
| Initialize the LLM and Embedder.............................................................................                    |   18 |
| The Basic Retriever Pattern.......................................................................................              |   18 |
| The Graph-Enhanced Vector Search Pattern.....................................................                                   |   20 |
| VectorCypher Retriever in Practice........................................................................                      |   21 |
| VectorCypher Retrieval: AWorking
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:6
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:18.815234

```text
Table of Contents (continued)
|   20 |
| VectorCypher Retriever in Practice........................................................................                      |   21 |
| VectorCypher Retrieval: AWorking Example....................................................                                    |   22 |
| Text2CypherRetriever..................................................................................................          |   23 |
| Community Summary Pattern...................................................................................                    |   25 |
| Concluding Thoughts and Next Steps..................................................................                            |   25 |
| Appendix: Technical Resources in Workflow Order........................................................                         |   27 |




## PART I: The Problem With Current RAG

Why chunk-based RAG hits a ceiling - and why developers need more context to answer well

You've built a retrieval-augmented generation (RAG) system. You embedded the docs, connected the vector store, wrapped a prompt around the output, and deployed it. For a minute, it felt like you cracked
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:7
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:19.690053

```text
Table of Contents (continued)
've built a retrieval-augmented generation (RAG) system. You embedded the docs, connected the vector store, wrapped a prompt around the output, and deployed it. For a minute, it felt like you cracked the code. The model was grounded in your own data, giving answers that sounded smarter than base GPT.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:10
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:20.786587

```text
RAG retrieves semantically similar text, but it doesn't know how the pieces fit together.
It has no map of your domain. No memory of what matters. It's like hiring a new developer and giving them a stack of index cards with code snippets from your repo. They can parrot back functions, maybe even modify them, but they don't understand the architecture. They don't know the 'why,' only the 'what.'

That's the ceiling of traditional RAG. And that's what this book is here to fix.

## Here's the core issue: RAG retrieves based on similarity, not understanding.

You give it a query, it vectorizes that query, and fetches the top-k similar chunks. That's fine if the answer you need lives entirely within isolated chunks. But most real-world questions don't work that way.

Let's say a user asks about a contract clause, but


the meaning depends on a sales addendum from three weeks earlier. Or maybe they ask a support question that only makes sense in the context of their infrastructure and license tier. The information is there, but it's scattered across multiple documents, formats, and timelines. Chunk-based retrieval can't bridge that gap.

Traditional RAG doesn't have shared context across documents. That's because it doesn't track relationships. It doesn't know which concepts
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:11
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:21.864591

```text
RAG retrieves semantically similar text, but it doesn't know how the pieces fit together.
nd timelines. Chunk-based retrieval can't bridge that gap.

Traditional RAG doesn't have shared context across documents. That's because it doesn't track relationships. It doesn't know which concepts are upstream, downstream, dependent, or mutually exclusive. It doesn't distinguish between definitions, instructions, timelines, policies, or decision logic.

## The bottom line: Traditional RAG treats all chunks as equal, flat, unstructured blobs of text.

Even more problematic is that the system has no mental model for your business. It cannot understand what a 'customer' is in your world. Or how a support ticket relates to a contract. Or what a system diagram implies about downstream integrations. The mental model that represents the structure behind your content is absent in RAG.

Without it, RAG can't reason. It can only retrieve, and that isn't enough.

You already know what your RAG system should be able to do. It's the kind of reasoning your team does every day without thinking. Consider this: If a customer reaches out to your support team, the employee will listen to the customer's concern, look up their account and tech stack, check previous service requests, etc. When
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:12
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:23.446276

```text
RAG retrieves semantically similar text, but it doesn't know how the pieces fit together.
ng. Consider this: If a customer reaches out to your support team, the employee will listen to the customer's concern, look up their account and tech stack, check previous service requests, etc. When answering the customer's question, the employee brings context. They may answer differently if the person is a new customer vs. a long-term customer.

You want your RAG application to do what humans do naturally: use context to inform its answer. As examples, you might want the RAG system to:

-  Answer a support question and understand the user's tech stack, contract level, and product version.
-  Explain a contract term - and know what the sales path looked like, who signed off, and which systems were impacted.

- Interpret a customer review and place it in context with purchase history, usage data, and net promoter score (NPS).

These shouldn't feel like advanced use cases they're basic context. They're what you, as a human developer, bring into every decision without even realizing it. And that's the problem: Your RAG system has none of that. Sure, it has some document metadata available, but no user metadata, no business logic, no connected data - just isolated chunks in a vector
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:13
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:24.977812

```text
RAG retrieves semantically similar text, but it doesn't know how the pieces fit together.
And that's the problem: Your RAG system has none of that. Sure, it has some document metadata available, but no user metadata, no business logic, no connected data - just isolated chunks in a vector store. But RAG can't use what it can't see. So until you give it structure - until you teach it relationships, timelines, ownership, and dependencies - it will keep retrieving the right words for the wrong reasons.

This isn't a whitepaper. It's a build-it-yourself playbook. We're going to walk you through:

-  Ingesting documents and turning them into a knowledge graph
-  Structuring real-world context from messy PDFs, CSVs, and APIs
- Building retrievers that combine vector search and graph traversal
-  Using text-to-query generation to run dynamic Cypher queries (a query language for graphs) and pull precise information and calculations from your data

And we're going to do it with code. No fluff. Just the stack, the logic, and the patterns that actually work. If you've built RAG, and you know it's not enough, then this is the guide to take you further.

## PART II: What Makes It GraphRAG - Structure, Logic, and Meaning

To understand GraphRAG, let's explore its foundational
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:14
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:26.801314

```text
RAG retrieves semantically similar text, but it doesn't know how the pieces fit together.
, and you know it's not enough, then this is the guide to take you further.

## PART II: What Makes It GraphRAG - Structure, Logic, and Meaning

To understand GraphRAG, let's explore its foundational components - RAG and knowledge graphs - and why they work so well together.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:15
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:27.913308

```text
Here's the core issue: RAG retrieves based on similarity, not understanding.
You give it a query, it vectorizes that query, and fetches the top-k similar chunks. That's fine if the answer you need lives entirely within isolated chunks. But most real-world questions don't work that way.

Let's say a user asks about a contract clause, but


the meaning depends on a sales addendum from three weeks earlier. Or maybe they ask a support question that only makes sense in the context of their infrastructure and license tier. The information is there, but it's scattered across multiple documents, formats, and timelines. Chunk-based retrieval can't bridge that gap.

Traditional RAG doesn't have shared context across documents. That's because it doesn't track relationships. It doesn't know which concepts are upstream, downstream, dependent, or mutually exclusive. It doesn't distinguish between definitions, instructions, timelines, policies, or decision logic.

## The bottom line: Traditional RAG treats all chunks as equal, flat, unstructured blobs of text.

Even more problematic is that the system has no mental model for your business. It cannot understand what a 'customer' is in your world. Or how a support ticket relates to a contract. Or what a system diagram
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:16
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:29.449486

```text
Here's the core issue: RAG retrieves based on similarity, not understanding.
problematic is that the system has no mental model for your business. It cannot understand what a 'customer' is in your world. Or how a support ticket relates to a contract. Or what a system diagram implies about downstream integrations. The mental model that represents the structure behind your content is absent in RAG.

Without it, RAG can't reason. It can only retrieve, and that isn't enough.

You already know what your RAG system should be able to do. It's the kind of reasoning your team does every day without thinking. Consider this: If a customer reaches out to your support team, the employee will listen to the customer's concern, look up their account and tech stack, check previous service requests, etc. When answering the customer's question, the employee brings context. They may answer differently if the person is a new customer vs. a long-term customer.

You want your RAG application to do what humans do naturally: use context to inform its answer. As examples, you might want the RAG system to:

-  Answer a support question and understand the user's tech stack, contract level, and product version.
-  Explain a contract term - and know what the sales path looked like,
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:17
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:31.112983

```text
Here's the core issue: RAG retrieves based on similarity, not understanding.
want the RAG system to:

-  Answer a support question and understand the user's tech stack, contract level, and product version.
-  Explain a contract term - and know what the sales path looked like, who signed off, and which systems were impacted.

- Interpret a customer review and place it in context with purchase history, usage data, and net promoter score (NPS).

These shouldn't feel like advanced use cases they're basic context. They're what you, as a human developer, bring into every decision without even realizing it. And that's the problem: Your RAG system has none of that. Sure, it has some document metadata available, but no user metadata, no business logic, no connected data - just isolated chunks in a vector store. But RAG can't use what it can't see. So until you give it structure - until you teach it relationships, timelines, ownership, and dependencies - it will keep retrieving the right words for the wrong reasons.

This isn't a whitepaper. It's a build-it-yourself playbook. We're going to walk you through:

-  Ingesting documents and turning them into a knowledge graph
-  Structuring real-world context from messy PDFs, CSVs, and APIs
- Building retrievers that
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:18
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:32.391370

```text
Here's the core issue: RAG retrieves based on similarity, not understanding.
playbook. We're going to walk you through:

-  Ingesting documents and turning them into a knowledge graph
-  Structuring real-world context from messy PDFs, CSVs, and APIs
- Building retrievers that combine vector search and graph traversal
-  Using text-to-query generation to run dynamic Cypher queries (a query language for graphs) and pull precise information and calculations from your data

And we're going to do it with code. No fluff. Just the stack, the logic, and the patterns that actually work. If you've built RAG, and you know it's not enough, then this is the guide to take you further.

## PART II: What Makes It GraphRAG - Structure, Logic, and Meaning

To understand GraphRAG, let's explore its foundational components - RAG and knowledge graphs - and why they work so well together.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:19
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:33.600981

```text
The bottom line: Traditional RAG treats all chunks as equal, flat, unstructured blobs of text.
Even more problematic is that the system has no mental model for your business. It cannot understand what a 'customer' is in your world. Or how a support ticket relates to a contract. Or what a system diagram implies about downstream integrations. The mental model that represents the structure behind your content is absent in RAG.

Without it, RAG can't reason. It can only retrieve, and that isn't enough.

You already know what your RAG system should be able to do. It's the kind of reasoning your team does every day without thinking. Consider this: If a customer reaches out to your support team, the employee will listen to the customer's concern, look up their account and tech stack, check previous service requests, etc. When answering the customer's question, the employee brings context. They may answer differently if the person is a new customer vs. a long-term customer.

You want your RAG application to do what humans do naturally: use context to inform its answer. As examples, you might want the RAG system to:

-  Answer a support question and understand the user's tech stack, contract level, and product version.
-  Explain a contract term - and know what the sales path looked
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:20
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:34.830448

```text
The bottom line: Traditional RAG treats all chunks as equal, flat, unstructured blobs of text.
might want the RAG system to:

-  Answer a support question and understand the user's tech stack, contract level, and product version.
-  Explain a contract term - and know what the sales path looked like, who signed off, and which systems were impacted.

- Interpret a customer review and place it in context with purchase history, usage data, and net promoter score (NPS).

These shouldn't feel like advanced use cases they're basic context. They're what you, as a human developer, bring into every decision without even realizing it. And that's the problem: Your RAG system has none of that. Sure, it has some document metadata available, but no user metadata, no business logic, no connected data - just isolated chunks in a vector store. But RAG can't use what it can't see. So until you give it structure - until you teach it relationships, timelines, ownership, and dependencies - it will keep retrieving the right words for the wrong reasons.

This isn't a whitepaper. It's a build-it-yourself playbook. We're going to walk you through:

-  Ingesting documents and turning them into a knowledge graph
-  Structuring real-world context from messy PDFs, CSVs, and APIs
- Building retrievers
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:21
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:35.822818

```text
The bottom line: Traditional RAG treats all chunks as equal, flat, unstructured blobs of text.
self playbook. We're going to walk you through:

-  Ingesting documents and turning them into a knowledge graph
-  Structuring real-world context from messy PDFs, CSVs, and APIs
- Building retrievers that combine vector search and graph traversal
-  Using text-to-query generation to run dynamic Cypher queries (a query language for graphs) and pull precise information and calculations from your data

And we're going to do it with code. No fluff. Just the stack, the logic, and the patterns that actually work. If you've built RAG, and you know it's not enough, then this is the guide to take you further.

## PART II: What Makes It GraphRAG - Structure, Logic, and Meaning

To understand GraphRAG, let's explore its foundational components - RAG and knowledge graphs - and why they work so well together.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:23
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:36.857593

```text
What Is RAG?
Let's start with the well-known problems of large language models (LLMs), which power chatbots


such as ChatGPT, Gemini, and Claude. When a user's prompt goes directly to the LLM, it generates a response based on its training data. Due to the probabilistic nature of response generation, LLMs often produce responses that lack accuracy and nuance and don't draw on knowledge specific to your business. In addition, the LLM in question may have limited explainability, which limits its adoption in enterprise settings.

RAG addresses these challenges by intercepting a user's prompt, querying external data, usually a vector store, and passing relevant documents back to the LLM. Adding retrieval to the LLM enables the application to answer questions with knowledge from a specific dataset. This simple technique suddenly makes it possible to build applications for a variety of use cases. As examples:

-  Knowledge assistants can tap into companyspecific information for accurate, contextual responses.
-  Recommendation systems can incorporate real-time data for more personalized suggestions.
-  Search APIs can deliver more nuanced and context-aware results.

RAG consists of three key
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:24
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:37.747598

```text
What Is RAG?
al responses.
-  Recommendation systems can incorporate real-time data for more personalized suggestions.
-  Search APIs can deliver more nuanced and context-aware results.

RAG consists of three key components:

-  An LLM that serves as the generator
-  A knowledge base or database that stores the information to be retrieved
-  A retrieval mechanism to find relevant information from the knowledge base, based on the input query


Figure 1. Querying a knowledge graph with an LLM
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:25
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:39.185332

```text
The quality of a RAG response depends heavily on the database type the information is retrieved from.
If you use a vector store (as in traditional RAG), the process goes like this: The user query is turned into a vector, which is then used to retrieve semantically similar text chunks from a vector database. While retrieval based on semantic similarity can work across multiple documents, it often falls short when questions require understanding implicit context or relationships that span those documents. Traditional RAG treats each chunk in isolation, as it lacks a holistic view of the domain.

Retrieval based on semantic similarity can only get you so far. And this is where GraphRAG comes in. GraphRAG gives the LLM a mental model of your domain so that it can answer questions by drawing on the correct context.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:26
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:40.252999

```text
What Is GraphRAG?
In GraphRAG, the knowledge base used for retrieval is a knowledge graph. A knowledge graph organizes facts as connected entities and relationships, which helps the system understand how pieces of information relate to each other.

The knowledge graph becomes a mental map of your domain, providing the LLM with information about dependencies, sequences, hierarchies, and meaning. This makes GraphRAG especially effective at answering complex, multi-step questions that require reasoning across multiple sources.

Imagine that a customer calls to request support regarding a recent purchase. Customer Service uses an internal chatbot to troubleshoot the request. A traditional system built on vector-only RAG would retrieve a product name from the customer support ticket:

|   Service Ticket | Service Ticket Text                       | Embedding            |
|------------------|-------------------------------------------|----------------------|
|           234381 | My new JavaCo coffee maker isn't working. | [.234, .789, .123……] |

But that's all the RAG system would surface.

A GraphRAG system, on the other hand, would show not only this service ticket text but also the



customer's
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:27
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:41.161384

```text
What Is GraphRAG?
er isn't working. | [.234, .789, .123……] |

But that's all the RAG system would surface.

A GraphRAG system, on the other hand, would show not only this service ticket text but also the



customer's purchase history, known issues with that product version, related documentation, and prior support conversations.

Figure 2. Order issue flow


A knowledge graph holds all related information together across both structured and unstructured data. A RAG system built on a knowledge graph  or GraphRAG - excels at generating context-aware responses.

The main reasons to implement a GraphRAG solution include:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:28
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:42.406144

```text
1.  Context-Aware Responses
Unlike traditional RAG, which retrieves isolated chunks of text based on similarity, GraphRAG retrieves facts in context. Since the knowledge graph explicitly encodes relationships, GraphRAG returns relevant information, as well as related information. This structured retrieval ensures that application outputs are comprehensive, reducing hallucinations and leading to more accurate, reliable outputs and improving real-world applicability.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:29
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:43.595681

```text
2.  Traceability and Explainability
LLMs and even standard RAG approaches operate as black boxes, making it difficult to know why and how a certain answer was generated. GraphRAG increases transparency by structuring retrieval paths through the knowledge graph. The knowledge graph will show the sources and relationships that contributed to a response. This makes it easier to audit results, build trust, and meet compliance needs.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:33
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:44.649083

```text
Implementing GraphRAG retrieval patterns
Figure 3. Implementing GraphRAG retrieval patterns flow


The rest of this book walks you through these two critical steps.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:34
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:45.833027

```text
Prepare a Knowledge Graph for GraphRAG
Effective retrieval in GraphRAG starts with a wellstructured knowledge graph. The data needs to be structured to model the business domain as it relates to the documents. That means having a clear data model that defines both the content you're working with and how it is connected.

There are two aspects to consider when you're modeling a knowledge graph for AI workflows:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:37
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:47.108408

```text
Ground With Unstructured and Structured Data
If you've worked with RAG systems, you're already familiar with vector databases and unstructured content - PDFs, contracts, reports. But the most important  context for your data rarely lives in a single format. In fact, most of the time you'll want to use more than just unstructured data. Structured data like CRM exports, product catalogs, and relational databases often contains crucial grounding information for the answers your users need.

To build systems that retrieve the right answer at the right time, you need to connect two worlds: unstructured and structured. That's where knowledge graphs come in. By linking unstructured chunks to structured business entities and relationships, you create a semantic network that makes retrieval smarter, safer, and more transparent. So, where do you start? With your documents or your structured schema?

Technically, you can begin from either side. But in practice, most teams start with unstructured data because that's where the buried context usually lives. Think financial disclosures, legal contracts, emails, and support tickets. These contain implicit business logic, risk factors, and decision-making signals that don't show up in
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:40
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:48.601691

```text
Ground With Unstructured and Structured Data
Guide             |
| Boundary's Annotation Modeling Language (BAML) | Declarative language for extracting structured data from unstructured sources, demonstrated with Neo4j           | BAML to Neo4j Tutorial by Jason Koo |
| pdfplumber                                     | Parses tables and text from PDF files, ideal for extracting structured data from documents                       | GitHub Repository                   |
| LangChain                                      | Framework for developing applications powered by language models, with support for Neo4j integration             | Neo4j Integration                   |


For this exercise, you'll start with unstructured financial documents. Using an LLM-powered pipeline to extract entities like Company and Risk Factor, you'll look for relationships such as FACES\_RISK to build a knowledge graph in Neo4j. This process mirrors what many teams face: extracting meaning from dense reports, contracts, or disclosures.

You'll then use Neo4j's Data Importer to load structured datasets - the kind of CSVs or database connectors most companies already have - further enriching the graph with known entities and relationships.

Finally,
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:41
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:49.730953

```text
Ground With Unstructured and Structured Data
Neo4j's Data Importer to load structured datasets - the kind of CSVs or database connectors most companies already have - further enriching the graph with known entities and relationships.

Finally, you'll test retrieval strategies, from vector search to graph-enhanced queries, to dynamic Cypher generation with Text2Cypher. The same process can be applied to your own PDFs, internal databases, and business domain to build a semantic layer over enterprise knowledge, making it accessible to GenAI systems with precision, transparency, and context.

## PART III: Constructing the Graph
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:45
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:50.862634

```text
Key Features of Neo4j GraphRAG Package
-  Knowledge Graph Construction Pipeline: Automates the extraction of entities and relationships from unstructured text and structures them into a Neo4j graph.
-  Vector Indexing and Retrieval: Facilitates the creation of vector indices for efficient semantic search within the graph.
-  Integration with LLMs: Seamlessly integrates with LLMs for tasks like entity extraction and relation identification.
-  Document Chunking and Storage: The package uses the SimpleKGPipeline class to automate chunking and storage. This class handles the parsing of documents, the chunking of text, and storage of chunks as nodes in Neo4j.
- neo4j: Official Python driver for interacting with a Neo4j database.
- GraphDatabase: Connects to Neo4j to interact with the graph database.
- SimpleKGPipeline : Automates chunking, entity recognition, and storage in Neo4j.
- OpenAILLM : Integrates GPT-4 for text-based processing and knowledge extraction.
- OpenAIEmbeddings : Handles vector embeddings to enable semantic search in Neo4j.
- ERExtractionTemplate: Supplies prompt templates for entity-relation extraction.




The LLM does the thinking by extracting meaningful concepts from text. The embedder turns the
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:49
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:52.204920

```text
Define Node Labels and Relationship Types
```
entities = [ {'label': 'Executive', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'Product', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'FinancialMetric', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'RiskFactor', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'StockType', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'Transaction', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'TimePeriod', 'properties': [{'name': 'name', 'type': 'STRING'}]}, {'label': 'Company', 'properties': [{'name': 'name', 'type': 'STRING'}]} ] relations = [ {'label': 'HAS_METRIC', 'source': 'Company', 'target': 'FinancialMetric'}, {'label': 'FACES_RISK', 'source': 'Company', 'target': 'RiskFactor'}, {'label': 'ISSUED_STOCK', 'source': 'Company', 'target': 'StockType'}, {'label': 'MENTIONS', 'source': 'Company', 'target': 'Product'} ]
```


Defining your nodes and relationships in two lists is a key moment in the knowledge graph construction process. This is when you determine the data model. These lists control what the SimpleKGBuilder will look for in the text and how it will organize that
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:50
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:53.610490

```text
Define Node Labels and Relationship Types
key moment in the knowledge graph construction process. This is when you determine the data model. These lists control what the SimpleKGBuilder will look for in the text and how it will organize that information in your graph. To understand how you might want to construct these lists, let's take a look at some general ideas.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:51
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:54.668800

```text
Entities = Nouns
What are the real-world concepts you're trying to capture?

Company, Executive, RiskFactor, Product whatever matters to your domain.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:52
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:56.007469

```text
Relationships = Verbs or Connectors
How do those concepts relate?

Perhaps a Company  FACES\_RISK  RiskFactor, or Company  ISSUED\_STOCK  StockType.

If you aren't sure which entities and relationships to include in your first project, ask yourself: What information would help my chunk provide a better answer? Alternatively, what information connects various chunks? Ultimately, you want to think through the application's use case and start with the entities and relationships that will move the needle the most on your project. This step isn't just configuration; it's your chance to define the mental model of your data.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:53
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:57.462236

```text
Initialize and Run the Pipeline
The SimpleKGPipeline sets up a structured pipeline for extracting and storing knowledge from unstructured text into a graph database. It starts with the driver , which is the Neo4j connection used to write data into the graph. The llm parameter specifies the language model that will interpret and extract meaningful entities and relationships from the input text. The embedder is the embedding

model used to vectorize text, which supports similarity-based retrieval alongside structured querying.

The entities and relations define the schema: what kinds of objects (like Customers, Contracts, Products) and relationships (like HAS\_CONTRACT , CONTAINS , REFERENCES ) the pipeline should look for. Finally, enforce\_schema=True ensures that only the entity and relationship types that have been explicitly defined in those lists are allowed into the graph. This prevents schema drift and keeps the resulting knowledge graph clean and reliable.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:56
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:58.718548

```text
Create the Vector Index
A vector index is a type of database index that enables fast similarity search over high-dimensional vectors, such as embeddings from models like OpenAI's. Unlike traditional indexes that look for exact matches, vector indexes retrieve items most similar to a query vector using metrics like cosine similarity or Euclidean distance.

In the context of Neo4j and RAG, here's what you need to know:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:57
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:22:59.984252

```text
This capability is essential for semantic search, question answering, and other AI-powered applications where meaning and context matter more than exact keywords.
By using a vector index, Neo4j enables scalable, realtime retrieval of relevant knowledge from large and complex graphs.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:72
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:01.209536

```text
FILED Relationship
Note that the relationship between Company and Document is the linchpin that connects the structured and the unstructured data in this GraphRAG application.

Relationship Type:

FILED

Table:

Company\_Filings.csv

Node ID Mapping

From:

Node:

Company

ID: name

ID column:

companyName

To:

Node:

Document

ID: path

ID column:

path\_Windows or path\_Mac\_ix

Figure 25. FILED relationship


As you see in the diagram above, each entity and relationship will have a green check mark when it has been properly mapped. Now you're ready to run the


import. Click the blue Run import button in the upper right corner of the screen.

Figure 26.  Run import button


Now that your unstructured and structured data is loaded, you can use the Explore and Query functions to refine your graph structure and data to accurately represent your business domain. Use Explore to visualize and navigate your graph with Neo4j Bloom and Query to investigate the graph.

For a detailed walkthrough of graph data modeling, see The Developer's Guide: How to Build a Knowledge Graph.

## PART IV: Implementing GraphRAG Retrieval Patterns

GraphRAG retrieval patterns are practical mechanisms that define how the
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:73
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:02.627530

```text
FILED Relationship
ta modeling, see The Developer's Guide: How to Build a Knowledge Graph.

## PART IV: Implementing GraphRAG Retrieval Patterns

GraphRAG retrieval patterns are practical mechanisms that define how the LLM in your GraphRAG solution accesses the context and connections in your knowledge graph.

Let's examine some of the most common GraphRAG patterns and how to use them.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:74
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:03.709398

```text
PART IV: Implementing GraphRAG Retrieval Patterns
GraphRAG retrieval patterns are practical mechanisms that define how the LLM in your GraphRAG solution accesses the context and connections in your knowledge graph.

Let's examine some of the most common GraphRAG patterns and how to use them.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:75
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:04.986592

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

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:77
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:06.168072

```text
Initialize the LLM and Embedder
Just as you selected a specific LLM and embedding model when processing your PDFs, you should do the same when generating embeddings for your text data. It's important to keep track of the language model and embedding tools that you use during this process.

For the retrievers to work correctly, the embedding model used during retrieval must match the one used to generate the dataset's embeddings. This ensures accurate and meaningful search results.

llm = OPENAILLM (model\_name='gpt-4o', api\_key=OPENAI\_API\_KEY) embedder = OPENAIEmbeddings(api\_key=OPENAI\_API\_KEY)
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:78
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:07.397178

```text
The Basic Retriever Pattern
The basic retriever uses vector embeddings to find nodes that are semantically similar based on content. This retriever is useful only for handling specific information requests about topics contained in just one or a few chunks. It's a starting point for more complex graph-based retrievals, and it's easy to implement if you're familiar with RAG but new to GraphRAG.


There are two components in the process:

-  Chunks as nodes: The pattern uses the already chunked data to create a graph, where each chunk becomes a node in the graph.
- Retrieval: When a query is performed, the basic retriever pattern searches through these chunk nodes to find the most relevant information.

Let's look at how you would implement this pattern using the SEC dataset.

You can now execute vector similarity searches to retrieve a company's current challenges based on certain text in their filing. The retriever compares a query vector generated from the search prompt (i.e., the numeric representation of the question) against the indexed text embeddings of the chunks. Vector similarity searches work well for simple queries with a narrow focus, such as: 'What are the risks around cryptocurrency?'


Be sure
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:79
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:08.535890

```text
The Basic Retriever Pattern
uestion) against the indexed text embeddings of the chunks. Vector similarity searches work well for simple queries with a narrow focus, such as: 'What are the risks around cryptocurrency?'


Be sure to review your retrieval results before generating any text output. This step helps you confirm that your retriever is functioning as intended and returning relevant data from your knowledge graph. For example, in the query above, a sample of the retrieved content is displayed for inspection:

```
result_table=pd.DataFrame([(item.metadata['score'], item. content [10:80], item.metadata['id']) for item in result.items], columns=['Score', 'Content', 'ID']
```



|    Score | Content                                                                  | ID                |
|----------|--------------------------------------------------------------------------|-------------------|
| 0.913177 | cryptocurrency assets could be treated as a general unsecured claim ag.. | 6064a2f775a8:1724 |
| 0.908264 | agencyofferings could subject us to additional regulations, licensing r… | 6064a2f775a8:1723 |
| 0.903259 | cyberextortion, distributed denial- of-service attacks, ransomware, spe… |
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:82
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:09.688387

```text
The Basic Retriever Pattern
gh a third-party custodian are susceptible to various risks, such as inappropriate access, theft, or destruction. Inadequate insurance coverage by custodians and their potential inability to maintain effective controls can expose customers to losses. In the event of a custodian's bankruptcy, the treatment of custodial holdings in proceedings remains uncertain, which could delay or prevent the return of assets.
3. Third-Party Partner Risks: Dependence on third-party custodians and financial institutions means exposure to operational disruptions, inability to safeguard holdings, and financial defaults by these partners, which could harm business operations and customer trust.

These risks underscore the need for robust regulatory compliance, secure custodial arrangements, and the management of thirdparty relationships to mitigate potential negative impacts on businesses offering cryptocurrency products.



While the vector search provided useful information about cryptocurrency risks, it did not answer deeper, more actionable questions, such as:

-  Which specific companies are exposed to these risks?
-  What other risks may be occurring concurrently?
-  Which asset managers are
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:83
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:10.632154

```text
The Basic Retriever Pattern
, it did not answer deeper, more actionable questions, such as:

-  Which specific companies are exposed to these risks?
-  What other risks may be occurring concurrently?
-  Which asset managers are associated with the affected companies?  (e.g., multi-hop relationships from risk to company to asset manager)

In other words, the approach demonstrated here retrieves relevant text fragments. However, it doesn't use the graph's structure to connect the risks to companies or asset managers, nor does it show related or concurrent risks. There's no traversal or multi-hop reasoning, so you miss out on the rich, contextual insights that a knowledge graph can provide.

To answer these more complex, relationship-driven questions, you need to combine vector search with graph-powered Cypher queries that can traverse and analyze connections between entities. This is where graph-enhanced retrieval patterns come in.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:84
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:12.113680

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
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:13.232730

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
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:14.562150

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name : The name of the vector index (here, chunkEmbeddings ) used for semantic search
- Embedder : The embedding model used to generate/query vector representations
- Retrieval\_query : The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

This setup enables you to start with a semantic search (e.g., for 'cryptocurrency risk') and automatically traverse your knowledge graph to reveal which companies are involved and what other risks they face. The resulting responses are both semantically relevant and graph-aware.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:87
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:15.685753

```text
VectorCypher Retriever in Practice
The power of the Graph-Enhanced Vector Search Pattern lies in its flexibility. While the example above focuses on linking companies to risk factors in financial filings, the approach can be applied to any domain or vertical by customizing the graph schema and Cypher queries.

How might this look for other industries?

- Healthcare: Retrieve patient records, diagnoses, and treatment plans by combining semantic search of clinical notes with graph traversal across relationships like doctorpatient, medication-prescribed, or symptomdiagnosis.
-  Ecommerce: Connect customer reviews or product descriptions (unstructured text) to

purchase behavior, category hierarchies, or supplier relationships (a structured graph), enabling recommendations and/or supply chain insights.

- Law: Link case law or legal opinions to statutes, precedents, and involved parties, surfacing not just relevant text but also the legal context and network of citations.
-  Cybersecurity: Combine threat intelligence reports (text) with the graph relationships between vulnerabilities, affected assets, and mitigation strategies to provide a holistic view of your security posture.
-  Education: Map student essays or
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:88
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:16.633680

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
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:18.025929

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
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:19.117508

```text
VectorCypher Retrieval: A Working Example
Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:93
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:21.524082

```text
WELLS FARGO &amp; COMPANY/MN
This is where GraphRAG really shines. You may be wondering how to construct the retrieval query that traverses the graph. In this example, you can see that the retrieval\_query is a string of Cypher code, the language of graph querying. Now let's look at one last retriever pattern found in the Neo4j library: the Text2CypherRetriever .
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:94
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:22.609189

```text
Text2CypherRetriever
You can use Text2CypherRetriever to seamlessly generate Cypher queries from natural language questions. Instead of manually crafting each Cypher statement, the retriever uses an LLM to translate your plain-English queries into Cypher based on its understanding of your Neo4j schema.

The process begins with a natural language question, such as:

'What are the names of companies owned by BlackRock Inc.?'

The retriever then uses the schema, described as a string outlining the main node types and relationships in your graph (for example, companies, risk factors, and asset managers), to guide the LLM in generating an appropriate Cypher query. While you could pass a hard-coded schema to the retriever, it's best practice to access the schema as it currently exists in your instance. Here's a sample of the full schema:

```
result = get_schema (driver) Node properties: Document {id: STRING, path: STRING, createdAt: STRING} Chunk {id: STRING, index: INTEGER, text: STRING, embedding: LIST} Company {id: STRING, name: STRING, chunk_ index: INTEGER, ticker: STRING} Product {id: STRING, name: STRING, chunk_ index: INTEGER} . . . Relationship properties: OWNS {position_status: STRING, Value:
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:100
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:23.780735

```text
Concluding Thoughts and Next Steps
Integrating a knowledge graph with RAG gives GenAI systems structured context and relationships, improving the relevance and quality of generated results.

This guide has equipped you with the foundational skills needed to implement GraphRAG. You learned how to use Neo4j's cloud-based graph database service, Neo4j Aura, to prepare a knowledge graph for GraphRAG, Data Importer, and the GraphRAG Python library to create a knowledge graph from unstructured data. You also learned how to implement foundational GraphRAG retrieval patterns, including the basic retriever, graph-enhanced vector search, and Text2Cypher.


Like other AI technologies, GraphRAG is rapidly evolving. A few trends to watch:

-  More advanced, dynamic Cypher queries and sophisticated retrieval patterns that use graph algorithms and machine learning techniques are pushing the boundaries of what's possible in information retrieval and generation.
- Deeper integration with other AI technologies, such as knowledge graph embeddings and graph neural networks, promises to enhance the semantic understanding and reasoning capabilities of GraphRAG systems.
-  Integrating GraphRAG with agentic systems and other multi-tool,
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:101
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:25.015986

```text
Concluding Thoughts and Next Steps
embeddings and graph neural networks, promises to enhance the semantic understanding and reasoning capabilities of GraphRAG systems.
-  Integrating GraphRAG with agentic systems and other multi-tool, multi-step RAG chains can result in more autonomous and intelligent systems capable of handling complex, multifaceted tasks with greater efficiency and accuracy.
-  Incorporating semantic layers in GraphRAG systems can provide even more nuanced understanding and context awareness in information retrieval and generation tasks.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:102
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:26.243022

```text
Explore GenAI With Neo4j
Neo4j uncovers hidden relationships and patterns across billions of data connections deeply, easily, and quickly, making graph databases an ideal choice for building your first GraphRAG application.

Learn More



Build on what you learned in this guide:

-  The Neo4j for GenAI use case page offers guides, tutorials, and best practices about GraphRAG implementation.
-  The GraphRAG site contains explanations of GraphRAG principles and step-bystep guides for various implementation scenarios.
-  Neo4j GraphAcademy offers free, handson online courses.
```

### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:104
**File ID:** 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf
**File Name:** Developers-Guide-GraphRAG.pdf
**Phase:** iterative_boolean_eval
**Captured:** 2026-02-08T02:23:27.364797

```text
Technical Resources in Workflow Order
Data Importer Tool                            | Visual UI for mapping CSVs and relational data to graph nodes and relationships.                  |
| 5. Data Ingestion (Unstructured) | Neo4j GraphRAG Python Library                       | Convert PDFs and text to a knowledge graph using LLM-powered entity + relationship extraction.    |
| 6. Data Ingestion (Unstructured) | KGBuilder Tutorial-SEC Filings Example              | Walkthrough for turning dense financial disclosu- res into structured graph nodes and edges.      |
| 7. Embeddings + Vector Indexing  | Neo4j Vector Indexing Docs                          | Build and manage vector embeddings inside Neo4j for hybrid retrieval.                             |
| 8. Retrieval: Basic + Vector     | Neo4j GraphRAGBasicRetriever Pattern                | First step: combine chunked content and embed- ding for basic semantic retrieval.                 |
| 9. Retrieval: Graph-Enhanced     | Graph-EnhancedVector Search with Neo4               | Augment vector search with traversal logic to im- prove contextual accuracy.                      |
| 10. Test2Cypher Automation       | Text2Cypher Documentation & Examples                |
```

## Final Selected Chunks

### File: Developers-Guide-GraphRAG.pdf (ID: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf)

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:89

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

```text
VectorCypher Retrieval: A Working Example
Which Asset Managers are most affected by reseller concerns?

Let's again start with the Chunks semantically similar to 'reseller concerns,' and then traverse through the Document to the Company through OWNS to identify the AssetManagers relevant to the query. We'll also include the property shares from the relationship OWNS and order by largest holdings.


Figure 28. VectorCypherRetriever example 2


Next, add this new retrieval query to the VectorCypherRetriever parameters:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:88

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

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:85

```text
Executes a Cypher query to traverse the graph based on these nodes.
To set up this particular query, you need to tell the graph where and how to traverse from the semantic nodes. In this example, the query is:

'What are the risk factors for companies discussing cryptocurrency in their filings?'

The following code creates a retriever to answer this query:


Let's start by looking at the parts of the graph that help to answer this query. We start by identifying the Chunk that is semantically similar to the cryptocurrency query. Then we need to traverse the graph to identify the Document the Chunk comes from, the Company that FILED the Document and collect the other RiskFactors for that Company . Once this information is retrieved, it's converted to Cypher and set as the retrieval query.


Figure 27.  VectorCypherRetriever example 1



Next, let's add this new retrieval query to the VectorCypherRetriever parameters:
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:6

```text
Table of Contents (continued)
|   20 |
| VectorCypher Retriever in Practice........................................................................                      |   21 |
| VectorCypher Retrieval: AWorking Example....................................................                                    |   22 |
| Text2CypherRetriever..................................................................................................          |   23 |
| Community Summary Pattern...................................................................................                    |   25 |
| Concluding Thoughts and Next Steps..................................................................                            |   25 |
| Appendix: Technical Resources in Workflow Order........................................................                         |   27 |




## PART I: The Problem With Current RAG

Why chunk-based RAG hits a ceiling - and why developers need more context to answer well

You've built a retrieval-augmented generation (RAG) system. You embedded the docs, connected the vector store, wrapped a prompt around the output, and deployed it. For a minute, it felt like you cracked
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:5

```text
Table of Contents (continued)
| PART IV: Implementing GraphRAG Retrieval Patterns ..................................................                            |   17 |
|---------------------------------------------------------------------------------------------------------------------------------|------|
| Import Libraries............................................................................................................... |   17 |
| Load Environment Variables and Initialize Neo4j Driver................................                                          |   18 |
| Initialize the LLM and Embedder.............................................................................                    |   18 |
| The Basic Retriever Pattern.......................................................................................              |   18 |
| The Graph-Enhanced Vector Search Pattern.....................................................                                   |   20 |
| VectorCypher Retriever in Practice........................................................................                      |   21 |
| VectorCypher Retrieval: AWorking
```

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:75

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

#### Chunk: 8dcd8cd1-62c4-4607-b0bc-ffce165cbf0b:C:\Alexis\DXC\AI\RAG\Developers-Guide-GraphRAG.pdf:chunk:86

```text
VectorCypherRetriever parameters:
- Driver : The Neo4j database connection
- Index\_name : The name of the vector index (here, chunkEmbeddings ) used for semantic search
- Embedder : The embedding model used to generate/query vector representations
- Retrieval\_query : The Cypher query (defined above) that tells Neo4j how to traverse the graph from the semantically matched nodes

This setup enables you to start with a semantic search (e.g., for 'cryptocurrency risk') and automatically traverse your knowledge graph to reveal which companies are involved and what other risks they face. The resulting responses are both semantically relevant and graph-aware.
```

**Summary:**
- Selected chunks: 8
- Total chars: 7827

