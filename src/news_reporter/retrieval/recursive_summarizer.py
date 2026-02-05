"""Phase 4: Recursive Summarization - Summarize expanded file chunks using LLM.

MIT RLM (Recursive Inspection Model) Implementation:
- Generate executable Python code for chunk relevance evaluation
- Apply the code to filter chunks relevant to the user query
- Selectively summarize matched content per file
- Return file-level summaries with metadata for citations

The core innovation: Instead of using rules, the LLM generates actual Python code
(evaluate_chunk_relevance function) that the system executes to determine which
chunks are relevant to the user's query. This follows the MIT RLM paper approach
for generating small, executable inspection programs.

Uses Azure OpenAI API for LLM-based code generation and summarization.
"""

import logging
import asyncio
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _build_completion_params(model_deployment: str, **kwargs) -> dict:
    """
    Build completion parameters compatible with the model.
    o1/o3 models don't support temperature parameter.
    
    Args:
        model_deployment: Model deployment name
        **kwargs: Other parameters like max_completion_tokens, messages, etc.
    
    Returns:
        Dict with appropriate parameters for the model
    """
    # Check if this is an o1/o3 model
    is_o_series = any(model_deployment.startswith(prefix) for prefix in ['o1', 'o3'])
    
    # Remove temperature for o-series models
    if is_o_series and 'temperature' in kwargs:
        kwargs.pop('temperature')
        logger.debug(f"Removed 'temperature' parameter for {model_deployment} (o-series model)")
    
    return kwargs


@dataclass
class FileSummary:
    """Summary of a file's relevant chunks."""
    file_id: str
    file_name: str
    summary_text: str
    source_chunk_ids: List[str]  # Chunk IDs that contributed to summary
    chunk_count: int
    summarized_chunk_count: int
    expansion_ratio: float  # (expanded_chunks / entry_chunks)


async def recursive_summarize_files(
    expanded_files: Dict[str, Dict],
    query: str,
    llm_client: Optional[Any] = None,
    model_deployment: Optional[str] = None
) -> List[FileSummary]:
    """
    Apply LLM-based recursive summarization to expanded file chunks.

    Phase 4: Recursive Summarization
    - For each file, analyze expanded chunks
    - Generate summarization logic (e.g., regex rules or filtering criteria)
    - Selectively summarize matched content
    - Return file summaries with citations

    Args:
        expanded_files: Output from Phase 3 {file_id: {chunks: [...], file_name: str, ...}}
        query: User query for context
        llm_client: Optional Azure OpenAI client (created if not provided)
        model_deployment: Azure OpenAI deployment name (reads from AZURE_OPENAI_CHAT_DEPLOYMENT if not provided)

    Returns:
        List of FileSummary objects with file-level summaries
    """
    # Initialize Azure OpenAI client if not provided
    if llm_client is None:
        try:
            from openai import AsyncAzureOpenAI
            
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            if not (azure_endpoint and api_key):
                logger.warning("⚠️  Phase 4: Azure OpenAI credentials not configured; skipping recursive summarization")
                return []
            
            llm_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        except ImportError:
            logger.warning("⚠️  Phase 4: Azure OpenAI SDK not available; skipping recursive summarization")
            return []
        except Exception as e:
            logger.warning(f"⚠️  Phase 4: Failed to initialize Azure OpenAI client: {e}")
            return []
    
    # Get deployment name from environment or parameter
    if model_deployment is None:
        model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini")
    
    logger.info(f"🔄 Phase 4: Starting recursive summarization for {len(expanded_files)} files (deployment: {model_deployment})")

    summaries = []
    inspection_code = {}  # Store LLM-generated inspection logic per file

    for file_id, file_data in expanded_files.items():
        try:
            file_name = file_data.get("file_name", "unknown")
            chunks = file_data.get("chunks", [])
            entry_chunk_count = file_data.get("entry_chunk_count", 0)
            total_chunks = len(chunks)

            if not chunks:
                logger.warning(f"⚠️  Phase 4: File {file_name} has no chunks; skipping")
                continue

            logger.info(
                f"📍 Phase 4.1: Analyzing file '{file_name}' "
                f"({entry_chunk_count} entry → {total_chunks} total chunks, deployment: {model_deployment})"
            )

            # Prepare chunk text for LLM analysis
            chunk_texts = []
            chunk_map = {}  # Map text segments back to chunk IDs

            for idx, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
                    text = chunk.get("text", "").strip()
                    if text:
                        chunk_texts.append(text)
                        chunk_map[text] = chunk_id

            if not chunk_texts:
                logger.warning(f"⚠️  Phase 4: File {file_name} has no extractable text")
                continue

            # Step 1: Generate Python code for chunk relevance evaluation (MIT RLM)
            logger.info(f"  → Step 1: Generating inspection code (Python filter function) for query: '{query[:50]}...'")
            generated_code = await _generate_inspection_logic(
                query=query,
                file_name=file_name,
                sample_chunks=chunk_texts[:3],  # Use first 3 chunks as sample
                llm_client=llm_client,
                model_deployment=model_deployment
            )
            
            # Store inspection code for logging and audit trail
            inspection_code[file_id] = generated_code

            # Step 2: Apply inspection logic to identify relevant chunks
            logger.info(f"  → Step 2: Identifying relevant chunks using inspection code")
            relevant_chunks = await _apply_inspection_logic(
                chunks=chunk_texts,
                inspection_logic=generated_code,
                llm_client=llm_client,
                model_deployment=model_deployment
            )

            if not relevant_chunks:
                logger.warning(f"⚠️  Phase 4: No relevant chunks identified in {file_name}")
                # Fallback: use all chunks if none pass relevance
                relevant_chunks = chunk_texts[:min(5, len(chunk_texts))]

            # Step 3: Summarize relevant chunks
            logger.info(
                f"  → Step 3: Summarizing {len(relevant_chunks)} relevant chunks "
                f"(from {total_chunks} total)"
            )
            summary_text = await _summarize_chunks(
                chunks=relevant_chunks,
                query=query,
                file_name=file_name,
                llm_client=llm_client,
                model_deployment=model_deployment
            )

            # Map back to chunk IDs for citations
            source_chunk_ids = [chunk_map.get(chunk, "unknown") for chunk in relevant_chunks]

            file_summary = FileSummary(
                file_id=file_id,
                file_name=file_name,
                summary_text=summary_text,
                source_chunk_ids=source_chunk_ids,
                chunk_count=total_chunks,
                summarized_chunk_count=len(relevant_chunks),
                expansion_ratio=total_chunks / max(1, entry_chunk_count)
            )

            summaries.append(file_summary)

            logger.info(
                f"  ✅ File '{file_name}': "
                f"Summary generated from {len(relevant_chunks)} chunks, "
                f"expansion ratio: {file_summary.expansion_ratio:.2f}x"
            )

        except Exception as e:
            logger.warning(f"⚠️  Phase 4: Failed to summarize {file_name}: {e}", exc_info=True)
            continue

    logger.info(
        f"✅ Phase 4: Recursive summarization complete - "
        f"{len(summaries)} file summaries generated"
    )

    # Log inspection code for debugging and analysis
    if inspection_code:
        try:
            await log_inspection_code_to_markdown(
                inspection_rules=inspection_code,
                query=query,
                rlm_enabled=True,
                output_dir="/app/logs/chunk_analysis"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code: {e}")

    return summaries


async def _generate_inspection_logic(
    query: str,
    file_name: str,
    sample_chunks: List[str],
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate LLM-based inspection logic (executable Python code) for relevance filtering.

    MIT RLM approach: Generate small, executable Python code that can evaluate chunks.
    The code is a filter function that identifies chunks relevant to the user query.

    Args:
        query: User query
        file_name: Name of file being analyzed
        sample_chunks: First few chunks for context
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Python code as string that can evaluate chunk relevance
    """
    sample_text = "\n---\n".join(sample_chunks[:2])

    prompt = f"""You are implementing the MIT Recursive Inspection Model (RLM) for document analysis.

Given a user query and document content, generate a Python filter function that evaluates 
chunk relevance. The function must be executable Python code.

Document: {file_name}
User Query: {query}

Sample content from the document:
{sample_text}

Generate a Python function called `evaluate_chunk_relevance(chunk_text: str) -> bool` that:
1. Takes a chunk text as input
2. Returns True if the chunk contains information relevant to the user query
3. Implements 2-3 specific criteria based on the query and document context

The function should:
- Use simple string operations and regex patterns
- Be deterministic and efficient
- Focus on key terms, entities, and relationships from the query
- Handle edge cases (empty strings, case insensitivity)

Requirements:
- Function signature: def evaluate_chunk_relevance(chunk_text: str) -> bool:
- Must be valid, executable Python code
- Can import: re, json, collections (built-in only)
- Return only the function code, no explanations or comments

Example format (for a different query about company annual revenue):

def evaluate_chunk_relevance(chunk_text: str) -> bool:
    import re
    text_lower = chunk_text.lower()
    
    # Criterion 1: Contains revenue-related keywords
    revenue_keywords = ['revenue', 'sales', 'income', 'earnings', 'profit']
    has_revenue = any(keyword in text_lower for keyword in revenue_keywords)
    
    # Criterion 2: Contains time periods (years) indicating financial reports
    has_year = bool(re.search(r'\\b(19|20)\\d{2}\\b', text_lower))
    
    # Criterion 3: Contains company name or financial metrics
    has_metrics = re.search(r'\\b(million|billion|k|percentage|%|rate)\\b', text_lower) is not None
    
    return (has_revenue and has_year) or (has_revenue and has_metrics)

Generate executable Python code now:"""

    try:
        params = _build_completion_params(
            model_deployment,
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a Python expert implementing the MIT RLM model. Generate executable Python code for chunk evaluation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=500
        )
        response = await llm_client.chat.completions.create(**params)
        inspection_code = response.choices[0].message.content.strip()
        logger.debug(f"Generated inspection code:\n{inspection_code}")
        return inspection_code
    except Exception as e:
        logger.warning(f"⚠️  Failed to generate inspection code: {e}")
        # Fallback: return simple query-based filter function
        return f"""def evaluate_chunk_relevance(chunk_text: str) -> bool:
    \"\"\"Fallback relevance filter based on query terms.\"\"\"
    text_lower = chunk_text.lower()
    query_terms = {repr(query.lower().split())}
    return sum(1 for term in query_terms if term in text_lower) >= 2"""


async def _apply_inspection_logic(
    chunks: List[str],
    inspection_logic: str,
    llm_client: Any,
    model_deployment: str,
    max_chunks: int = 10
) -> List[str]:
    """
    Apply inspection logic (executable Python code) to select relevant chunks.

    MIT RLM approach: Execute the generated Python code to filter chunks.
    The inspection_logic should be a Python function: evaluate_chunk_relevance(chunk_text: str) -> bool

    Args:
        chunks: List of chunk texts
        inspection_logic: Python code containing evaluate_chunk_relevance() function
        llm_client: Azure OpenAI client (kept for compatibility)
        model_deployment: Azure deployment name (kept for compatibility)
        max_chunks: Max chunks to select

    Returns:
        Filtered list of relevant chunks
    """
    if not chunks:
        return []

    try:
        # Execute the generated Python code to create the filter function
        namespace = {}
        exec(inspection_logic, namespace)
        
        # Get the evaluate_chunk_relevance function
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        
        if evaluate_func is None:
            logger.warning("⚠️  Generated code does not contain evaluate_chunk_relevance function; using LLM fallback")
            return await _apply_inspection_logic_llm_fallback(
                chunks, inspection_logic, llm_client, model_deployment, max_chunks
            )
        
        # Apply the filter function to each chunk
        relevant_chunks = []
        for chunk in chunks[:max_chunks * 2]:  # Evaluate up to 2x max_chunks
            try:
                if evaluate_func(chunk):
                    relevant_chunks.append(chunk)
                    if len(relevant_chunks) >= max_chunks:
                        break
            except Exception as e:
                logger.debug(f"Error evaluating chunk with filter function: {e}")
                continue
        
        # If no chunks matched, return top few as fallback
        if not relevant_chunks:
            logger.warning(f"⚠️  No chunks matched filter criteria; using first few chunks as fallback")
            relevant_chunks = chunks[:min(3, len(chunks))]
        
        logger.debug(f"Selected {len(relevant_chunks)} relevant chunks from {len(chunks)} using Python filter")
        return relevant_chunks
    
    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in generated inspection code: {e}; using LLM fallback")
        return await _apply_inspection_logic_llm_fallback(
            chunks, inspection_logic, llm_client, model_deployment, max_chunks
        )
    except Exception as e:
        logger.warning(f"⚠️  Error executing inspection code: {e}; using LLM fallback")
        return await _apply_inspection_logic_llm_fallback(
            chunks, inspection_logic, llm_client, model_deployment, max_chunks
        )


async def _apply_inspection_logic_llm_fallback(
    chunks: List[str],
    inspection_logic: str,
    llm_client: Any,
    model_deployment: str,
    max_chunks: int = 10
) -> List[str]:
    """
    LLM-based fallback for inspection logic when code execution fails.
    
    Uses the LLM to apply the inspection logic to chunks.
    """
    if not chunks:
        return []

    # Prepare chunk list with indices for tracking
    chunks_with_idx = [(i, chunk) for i, chunk in enumerate(chunks[:max_chunks * 2])]

    prompt = f"""Analyze the following chunks and identify which ones contain information 
relevant according to these inspection criteria:

INSPECTION CRITERIA/CODE:
{inspection_logic}

CHUNKS TO ANALYZE:
{chr(10).join(f'[Chunk {i}]: {chunk[:200]}...' if len(chunk) > 200 else f'[Chunk {i}]: {chunk}' for i, chunk in chunks_with_idx)}

Return a JSON list of chunk indices that are relevant. Format: {{"relevant_indices": [0, 2, 5]}}
Include only chunks that clearly match the criteria. If fewer than 3 chunks match, include the most relevant ones anyway."""

    try:
        params = _build_completion_params(
            model_deployment,
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=200
        )
        response = await llm_client.chat.completions.create(**params)
        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        result = json.loads(response_text)
        relevant_indices = result.get("relevant_indices", [])

        # Filter chunks based on selected indices
        relevant_chunks = [chunks[i] for i in relevant_indices if i < len(chunks)]

        logger.debug(f"Selected {len(relevant_chunks)} relevant chunks from {len(chunks)} using LLM fallback")
        return relevant_chunks

    except Exception as e:
        logger.warning(f"⚠️  LLM fallback also failed: {e}; returning first few chunks")
        # Final fallback: return first few chunks
        return chunks[:min(3, len(chunks))]


async def _summarize_chunks(
    chunks: List[str],
    query: str,
    file_name: str,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Summarize selected chunks into a cohesive file-level summary.

    Args:
        chunks: List of relevant chunk texts
        query: User query for context
        file_name: Name of file being summarized
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Summary text
    """
    chunks_text = "\n---\n".join(chunks)

    prompt = f"""Summarize the following chunks from "{file_name}" to answer: {query}

RELEVANT CHUNKS:
{chunks_text}

Provide a concise summary (3-5 sentences) that:
1. Directly addresses the user's query
2. Includes specific details from the chunks
3. Maintains the context and relationships mentioned
4. Is suitable for inclusion in a final answer

Return only the summary text, without preamble."""

    try:
        params = _build_completion_params(
            model_deployment,
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a summarization expert. Provide clear, concise summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_completion_tokens=500
        )
        response = await llm_client.chat.completions.create(**params)
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.warning(f"⚠️  Failed to summarize chunks: {e}")
        # Fallback: return concatenated chunks
        return " ".join(chunks[:min(2, len(chunks))])


async def log_file_summaries_to_markdown(
    file_summaries: List[FileSummary],
    query: str,
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log Phase 4 file summaries to markdown file.

    Args:
        file_summaries: List of FileSummary objects
        query: User query
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        # Determine output file
        file_suffix = "enabled" if rlm_enabled else "disabled"
        output_path = Path(output_dir) / f"summaries_rlm_{file_suffix}.md"

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: Recursive Summarization (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Summaries:** {len(file_summaries)}",
            "\n---\n",
        ]

        # Add each file summary
        for idx, summary in enumerate(file_summaries, 1):
            lines.extend([
                f"## {idx}. {summary.file_name}",
                f"\n**File ID:** {summary.file_id}",
                f"**Chunks:** {summary.summarized_chunk_count}/{summary.chunk_count} summarized",
                f"**Expansion Ratio:** {summary.expansion_ratio:.2f}x",
                f"\n### Summary",
                f"\n{summary.summary_text}",
                f"\n### Source Chunks",
                f"\n{', '.join(summary.source_chunk_ids)}",
                "\n---\n",
            ])

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 summaries logged to {output_path}")
        print(f"✅ Phase 4 summaries logged: {len(file_summaries)} files")

    except Exception as e:
        logger.error(f"❌ Failed to log Phase 4 summaries: {e}", exc_info=True)
        raise


async def log_inspection_code_to_markdown(
    inspection_rules: Dict[str, str],
    query: str,
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log LLM-generated Python inspection code to markdown file.

    This stores the executable Python code generated by the LLM per the MIT RLM model.
    Each file gets a Python function (evaluate_chunk_relevance) that determines if 
    a chunk is relevant to the user query.

    Args:
        inspection_rules: Dict mapping file_id to generated inspection code (Python functions)
        query: User query that drove the analysis
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        # Determine output file
        file_suffix = "enabled" if rlm_enabled else "disabled"
        output_path = Path(output_dir) / f"inspection_code_rlm_{file_suffix}.md"

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Inspection Programs:** {len(inspection_rules)}",
            f"\n**Implementation:** MIT Recursive Inspection Model (RLM)",
            "\n---\n",
            "## Overview\n",
            "This file stores the **executable Python code** generated by the LLM per MIT RLM.",
            "Each code block contains a Python function `evaluate_chunk_relevance(chunk_text: str) -> bool`",
            "that determines if a chunk contains information relevant to the user's query.\n",
            "### Purpose\n",
            "- Evaluate which chunks contain relevant information",
            "- Extract specific signals or matches from the content",
            "- Decide which subsets require deeper inspection/summarization\n",
            "### Usage\n",
            "These functions are executed by the recursive summarizer to filter chunks before LLM-based summarization.\n",
            "---\n",
        ]

        # Add each file's inspection code
        for idx, (file_id, inspection_code) in enumerate(inspection_rules.items(), 1):
            lines.extend([
                f"## {idx}. Inspection Code (File ID: {file_id})",
                f"\n**Purpose:** Filter chunks relevant to query: \"{query}\"\n",
                "```python",
                inspection_code,
                "```\n",
                "---\n",
            ])

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code logged to {output_path}")
        logger.info(f"   Stored {len(inspection_rules)} inspection programs following MIT RLM model")

    except Exception as e:
        logger.error(f"❌ Failed to log inspection code: {e}", exc_info=True)
        raise
