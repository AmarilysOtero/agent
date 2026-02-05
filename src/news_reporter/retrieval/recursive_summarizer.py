"""Phase 4: Recursive Summarization - Summarize expanded file chunks using LLM.

This module implements the MIT RLM recursive inspection model:
- Apply LLM-generated inspection logic to expanded chunks
- Selectively summarize matched content per file
- Return file-level summaries with metadata for citations

Uses Azure OpenAI API for LLM-based summarization.
"""

import logging
import asyncio
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
            from azure.openai import AsyncAzureOpenAI
            
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
                f"({entry_chunk_count} entry → {total_chunks} total chunks)"
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

            # Step 1: Generate inspection logic using LLM
            logger.info(f"  → Step 1: Generating inspection logic for query: '{query[:50]}...'")
            inspection_logic = await _generate_inspection_logic(
                query=query,
                file_name=file_name,
                sample_chunks=chunk_texts[:3],  # Use first 3 chunks as sample
                llm_client=llm_client,
                model_deployment=model_deployment
            )

            # Step 2: Apply inspection logic to identify relevant chunks
            logger.info(f"  → Step 2: Identifying relevant chunks using inspection logic")
            relevant_chunks = await _apply_inspection_logic(
                chunks=chunk_texts,
                inspection_logic=inspection_logic,
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

    return summaries


async def _generate_inspection_logic(
    query: str,
    file_name: str,
    sample_chunks: List[str],
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate LLM-based inspection logic (rules/patterns) for relevance filtering.

    MIT RLM approach: Let LLM generate small executable logic based on query.
    For now, we'll use this as a ruleset description that informs chunk selection.

    Args:
        query: User query
        file_name: Name of file being analyzed
        sample_chunks: First few chunks for context
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Inspection logic description/rules as string
    """
    sample_text = "\n---\n".join(sample_chunks[:2])

    prompt = f"""You are analyzing a document to find relevant information for a user query.

Document: {file_name}
User Query: {query}

Sample content from the document:
{sample_text}

Generate a concise set of rules or patterns (3-5 specific criteria) that would identify 
chunks containing information relevant to the user's query. Focus on:
1. Key terms or concepts the user is asking about
2. Patterns in how relevant information is typically presented
3. Relationships or connections mentioned in the query

Return only the rules, formatted as a numbered list. Be specific and actionable."""

    try:
        response = await llm_client.chat.completions.create(
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a document analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        inspection_logic = response.choices[0].message.content.strip()
        logger.debug(f"Generated inspection logic:\n{inspection_logic}")
        return inspection_logic
    except Exception as e:
        logger.warning(f"⚠️  Failed to generate inspection logic: {e}")
        # Fallback: return query-based heuristics
        return f"Return chunks containing these terms: {query}"


async def _apply_inspection_logic(
    chunks: List[str],
    inspection_logic: str,
    llm_client: Any,
    model_deployment: str,
    max_chunks: int = 10
) -> List[str]:
    """
    Apply inspection logic to select relevant chunks.

    Args:
        chunks: List of chunk texts
        inspection_logic: Rules/patterns for relevance
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name
        max_chunks: Max chunks to select

    Returns:
        Filtered list of relevant chunks
    """
    if not chunks:
        return []

    # Prepare chunk list with indices for tracking
    chunks_with_idx = [(i, chunk) for i, chunk in enumerate(chunks[:max_chunks * 2])]

    prompt = f"""Analyze the following chunks and identify which ones contain information 
relevant according to these rules:

RELEVANCE RULES:
{inspection_logic}

CHUNKS TO ANALYZE:
{chr(10).join(f'[Chunk {i}]: {chunk[:200]}...' if len(chunk) > 200 else f'[Chunk {i}]: {chunk}' for i, chunk in chunks_with_idx)}

Return a JSON list of chunk indices that are relevant. Format: {{"relevant_indices": [0, 2, 5]}}
Include only chunks that clearly match the rules. If fewer than 3 chunks match, include the most relevant ones anyway."""

    try:
        response = await llm_client.chat.completions.create(
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        result = json.loads(response_text)
        relevant_indices = result.get("relevant_indices", [])

        # Filter chunks based on selected indices
        relevant_chunks = [chunks[i] for i in relevant_indices if i < len(chunks)]

        logger.debug(f"Selected {len(relevant_chunks)} relevant chunks from {len(chunks)}")
        return relevant_chunks

    except Exception as e:
        logger.warning(f"⚠️  Failed to apply inspection logic: {e}")
        # Fallback: return first few chunks
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
        response = await llm_client.chat.completions.create(
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are a summarization expert. Provide clear, concise summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
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
