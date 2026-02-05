"""Phase 5: Cross-File Merge + Final Answer + Citations

Complete the RLM flow by:
- Merging file summaries into a global understanding
- Generating final answer based on merged content
- Enforcing citation policy (strict or best_effort)
- Respecting safety caps (RLM_MAX_FILES, RLM_MAX_CHUNKS)

Citation Policy:
- strict: Every claim must have a direct citation to a chunk ID
- best_effort: Include citations where possible, summarized claims otherwise

Uses Azure OpenAI API for cross-file merging and final answer generation.
"""

import logging
import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CitationPolicy(str, Enum):
    """Citation policy for final answer."""
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


@dataclass
class Citation:
    """Citation reference for an answer claim."""
    chunk_id: str
    file_id: str
    file_name: str
    quote: Optional[str] = None  # Extracted quote from the chunk


@dataclass
class Answer:
    """Final answer with citations."""
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    file_count: int = 0
    chunk_count: int = 0
    expansion_ratio: float = 0.0


async def generate_final_answer(
    file_summaries: List[Any],  # List[FileSummary]
    query: str,
    llm_client: Optional[Any] = None,
    model_deployment: Optional[str] = None,
    citation_policy: str = "best_effort",
    max_files: int = 10,
    max_chunks: int = 50
) -> Answer:
    """
    Generate final answer by merging file summaries with citations.

    Phase 5: Cross-File Merge + Final Answer + Citations
    - Merge file-level summaries into global understanding
    - Generate final answer addressing the query
    - Enforce citation policy
    - Respect safety caps

    Args:
        file_summaries: List of FileSummary objects from Phase 4
        query: User query for context
        llm_client: Optional Azure OpenAI client
        model_deployment: Azure deployment name
        citation_policy: "strict" or "best_effort"
        max_files: Maximum files to reference
        max_chunks: Maximum chunks to cite

    Returns:
        Answer object with answer_text and citations
    """
    # Initialize Azure OpenAI client if not provided
    if llm_client is None:
        try:
            from openai import AsyncAzureOpenAI
            
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            if not (azure_endpoint and api_key):
                logger.warning("⚠️  Phase 5: Azure OpenAI credentials not configured")
                # Return answer without LLM generation
                return _create_fallback_answer(file_summaries, max_files, max_chunks)
            
            llm_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
        except ImportError:
            logger.warning("⚠️  Phase 5: Azure OpenAI SDK not available")
            return _create_fallback_answer(file_summaries, max_files, max_chunks)
        except Exception as e:
            logger.warning(f"⚠️  Phase 5: Failed to initialize Azure OpenAI client: {e}")
            return _create_fallback_answer(file_summaries, max_files, max_chunks)
    
    if model_deployment is None:
        model_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini")
    
    logger.info(f"🔄 Phase 5: Generating final answer from {len(file_summaries)} file summaries (deployment: {model_deployment})")

    try:
        # Step 1: Enforce safety caps
        logger.info(f"  → Step 1: Enforcing safety caps (max_files: {max_files}, max_chunks: {max_chunks})")
        capped_summaries = _enforce_safety_caps(file_summaries, max_files, max_chunks)
        
        if not capped_summaries:
            logger.warning("⚠️  Phase 5: No summaries left after applying caps")
            return _create_empty_answer()
        
        logger.info(f"  ✓ Using {len(capped_summaries)} files after caps (from {len(file_summaries)})")

        # Step 2: Merge file summaries into global understanding
        logger.info(f"  → Step 2: Merging {len(capped_summaries)} file summaries into global understanding")
        merged_context = await _merge_file_summaries(
            capped_summaries,
            query,
            llm_client,
            model_deployment
        )

        # Step 3: Generate final answer with merged context
        logger.info(f"  → Step 3: Generating final answer")
        answer_text = await _generate_answer_text(
            merged_context=merged_context,
            query=query,
            llm_client=llm_client,
            model_deployment=model_deployment,
            citation_policy=citation_policy
        )

        # Step 4: Extract and enforce citations
        logger.info(f"  → Step 4: Extracting citations (policy: {citation_policy})")
        citations = _extract_citations(
            answer_text=answer_text,
            file_summaries=capped_summaries,
            citation_policy=citation_policy
        )

        # Build final answer
        answer = Answer(
            answer_text=answer_text,
            citations=citations,
            file_count=len(capped_summaries),
            chunk_count=sum(s.summarized_chunk_count for s in capped_summaries),
            expansion_ratio=sum(s.expansion_ratio for s in capped_summaries) / max(1, len(capped_summaries))
        )

        logger.info(
            f"  ✅ Final answer generated with {len(citations)} citations "
            f"from {answer.file_count} files"
        )

        return answer

    except Exception as e:
        logger.error(f"❌ Phase 5: Failed to generate final answer: {e}", exc_info=True)
        return _create_fallback_answer(file_summaries, max_files, max_chunks)


async def _merge_file_summaries(
    file_summaries: List[Any],
    query: str,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Merge file-level summaries into a cohesive global understanding.

    Args:
        file_summaries: List of FileSummary objects
        query: User query
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Merged context string
    """
    if not file_summaries:
        return ""

    if len(file_summaries) == 1:
        # Only one file, return its summary
        return file_summaries[0].summary_text

    # Build merged context from multiple files
    summaries_text = "\n\n---\n\n".join([
        f"**{summary.file_name}** (chunks: {summary.summarized_chunk_count}/{summary.chunk_count})\n\n{summary.summary_text}"
        for summary in file_summaries
    ])

    prompt = f"""You are synthesizing information from multiple documents to answer a user query.

User Query: {query}

DOCUMENT SUMMARIES:
{summaries_text}

Create a merged, cohesive summary that:
1. Synthesizes information across all documents
2. Identifies key relationships and overlaps
3. Prioritizes information relevant to the user's query
4. Maintains factual accuracy from source summaries
5. Notes important context or caveats

Return only the merged summary, no preamble. Focus on answering the query comprehensively."""

    try:
        # Use gpt-4o-mini if o3-mini is configured (o3 returns empty responses)
        effective_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment
        
        response = await llm_client.chat.completions.create(
            model=effective_deployment,
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing information from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_completion_tokens=1000
        )
        merged_context = response.choices[0].message.content.strip()
        return merged_context
    except Exception as e:
        logger.warning(f"⚠️  Failed to merge summaries: {e}; using concatenated summaries")
        return summaries_text


async def _generate_answer_text(
    merged_context: str,
    query: str,
    llm_client: Any,
    model_deployment: str,
    citation_policy: str = "best_effort"
) -> str:
    """
    Generate final answer from merged context.

    Args:
        merged_context: Merged summary from all files
        query: User query
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name
        citation_policy: "strict" or "best_effort"

    Returns:
        Final answer text
    """
    citation_instruction = (
        "Use the following format for citations: [source: file_name, chunks: chunk_id_1, chunk_id_2]"
        if citation_policy == "strict"
        else "When possible, include citations in format: [source: file_name, chunks: chunk_ids]"
    )

    prompt = f"""Answer the following user query based on the provided merged context.

User Query: {query}

MERGED CONTEXT FROM DOCUMENTS:
{merged_context}

Requirements:
- Provide a direct, comprehensive answer to the query
- Be specific and cite information from the context
- {citation_instruction}
- If information is insufficient, note what is missing
- When citing, format as: [source: document_name, chunks: chunk_ids]

Return only the answer, no preamble. Answer should be 2-5 sentences."""

    try:
        # Use gpt-4o-mini if o3-mini is configured (o3 returns empty responses)
        effective_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment
        
        response = await llm_client.chat.completions.create(
            model=effective_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on provided documents. Include citations in your answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=800
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.warning(f"⚠️  Failed to generate answer text: {e}")
        return merged_context[:500]  # Fallback: return start of merged context


def _extract_citations(
    answer_text: str,
    file_summaries: List[Any],
    citation_policy: str = "best_effort"
) -> List[Citation]:
    """
    Extract citations from answer text based on citation policy.

    Args:
        answer_text: Generated answer text (may contain citation markers)
        file_summaries: List of FileSummary objects for lookup
        citation_policy: "strict" or "best_effort"

    Returns:
        List of Citation objects
    """
    citations = []
    
    # Create file lookup by name
    file_lookup = {summary.file_name: summary for summary in file_summaries}
    file_lookup_by_id = {summary.file_id: summary for summary in file_summaries}
    
    # Parse citation markers from answer text: [source: file_name, chunks: chunk_id_1, chunk_id_2]
    import re
    citation_pattern = r'\[source:\s*([^,]+),\s*chunks:\s*([^\]]+)\]'
    
    found_citations = re.findall(citation_pattern, answer_text)
    
    for file_name_raw, chunks_raw in found_citations:
        file_name = file_name_raw.strip()
        chunk_ids = [cid.strip() for cid in chunks_raw.split(',')]
        
        # Look up file
        file_summary = file_lookup.get(file_name)
        if file_summary:
            for chunk_id in chunk_ids:
                if chunk_id in file_summary.source_chunk_ids:
                    citations.append(Citation(
                        chunk_id=chunk_id,
                        file_id=file_summary.file_id,
                        file_name=file_summary.file_name,
                        quote=None  # Could extract from original chunks, but not available here
                    ))
    
    # If no citations found but best_effort, create citations from ALL relevant source chunks
    # (source_chunk_ids are already filtered to relevant chunks from Phase 4)
    if not citations and citation_policy == "best_effort":
        for summary in file_summaries:
            for chunk_id in summary.source_chunk_ids:  # Use ALL relevant chunks, not just first 3
                citations.append(Citation(
                    chunk_id=chunk_id,
                    file_id=summary.file_id,
                    file_name=summary.file_name,
                    quote=None
                ))
    
    # Enforce strict policy: if no citations found, citation is required
    if citation_policy == "strict" and not citations:
        logger.warning(f"⚠️  Strict citation policy: no citations found in answer text")
        # Still return empty list; caller should handle this
    
    logger.debug(f"Extracted {len(citations)} citations from {len(file_summaries)} files (policy: {citation_policy})")
    return citations


def _enforce_safety_caps(
    file_summaries: List[Any],
    max_files: int = 10,
    max_chunks: int = 50
) -> List[Any]:
    """
    Enforce safety caps on file and chunk counts.

    Prioritizes:
    1. Files with higher expansion ratios (more relevant)
    2. Files with more source chunks

    Args:
        file_summaries: List of FileSummary objects
        max_files: Maximum files to include
        max_chunks: Maximum total chunks to reference

    Returns:
        Filtered list of FileSummary objects
    """
    if not file_summaries:
        return []
    
    # Sort by expansion ratio (descending) then chunk count (descending)
    sorted_summaries = sorted(
        file_summaries,
        key=lambda s: (s.expansion_ratio, s.summarized_chunk_count),
        reverse=True
    )
    
    # Enforce max_files cap
    capped_by_files = sorted_summaries[:max_files]
    
    # Enforce max_chunks cap by removing least relevant chunks
    total_chunks = sum(s.summarized_chunk_count for s in capped_by_files)
    
    if total_chunks > max_chunks:
        logger.info(f"  ⚠️  Total chunks {total_chunks} exceeds max {max_chunks}; culling less relevant files")
        
        # Remove files until under cap, starting from lowest expansion ratio
        remaining = list(capped_by_files)
        remaining.sort(key=lambda s: s.expansion_ratio)
        
        chunks_so_far = sum(s.summarized_chunk_count for s in remaining)
        while chunks_so_far > max_chunks and remaining:
            removed = remaining.pop(0)
            chunks_so_far -= removed.summarized_chunk_count
            logger.info(f"    Removed {removed.file_name} ({removed.summarized_chunk_count} chunks)")
        
        capped_by_files = remaining
    
    logger.info(f"  Safety caps applied: {len(capped_by_files)} files, {sum(s.summarized_chunk_count for s in capped_by_files)} total chunks")
    return capped_by_files


def _create_fallback_answer(
    file_summaries: List[Any],
    max_files: int = 10,
    max_chunks: int = 50
) -> Answer:
    """
    Create fallback answer when LLM generation fails.

    Args:
        file_summaries: List of FileSummary objects
        max_files: Maximum files to include
        max_chunks: Maximum chunks to reference

    Returns:
        Answer object with concatenated summaries
    """
    # Apply safety caps
    capped_summaries = _enforce_safety_caps(file_summaries, max_files, max_chunks)
    
    if not capped_summaries:
        return _create_empty_answer()
    
    # Concatenate summaries
    answer_text = "\n\n".join([
        f"**{summary.file_name}:**\n{summary.summary_text}"
        for summary in capped_summaries
    ])
    
    # Create citations from source chunks
    citations = []
    for summary in capped_summaries:
        for chunk_id in summary.source_chunk_ids[:3]:  # Limit to first 3 per file
            citations.append(Citation(
                chunk_id=chunk_id,
                file_id=summary.file_id,
                file_name=summary.file_name
            ))
    
    return Answer(
        answer_text=answer_text,
        citations=citations,
        file_count=len(capped_summaries),
        chunk_count=sum(s.summarized_chunk_count for s in capped_summaries),
        expansion_ratio=sum(s.expansion_ratio for s in capped_summaries) / max(1, len(capped_summaries))
    )


def _create_empty_answer() -> Answer:
    """Create empty answer response."""
    return Answer(
        answer_text="No relevant information found to answer the query.",
        citations=[],
        file_count=0,
        chunk_count=0,
        expansion_ratio=0.0
    )


async def log_final_answer_to_markdown(
    answer: Answer,
    query: str,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log Phase 5 final answer with citations to markdown file.

    Args:
        answer: Answer object with citations
        query: User query
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        # Determine output file
        output_path = Path(output_dir) / "final_answer_phase5.md"
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 5: Final Answer with Citations",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Files Used:** {answer.file_count}",
            f"**Total Chunks Cited:** {answer.chunk_count}",
            f"**Average Expansion Ratio:** {answer.expansion_ratio:.2f}x",
            "\n---\n",
            "## Answer\n",
            f"{answer.answer_text}",
            "\n---\n",
            "## Citations\n",
        ]

        # Add citations by file
        if answer.citations:
            citations_by_file = {}
            for citation in answer.citations:
                if citation.file_name not in citations_by_file:
                    citations_by_file[citation.file_name] = []
                citations_by_file[citation.file_name].append(citation.chunk_id)
            
            for file_name, chunk_ids in citations_by_file.items():
                lines.extend([
                    f"### {file_name}",
                    f"- Chunks: {', '.join(chunk_ids)}",
                    "\n",
                ])
        else:
            lines.append("No citations extracted from answer.")

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 5 final answer logged to {output_path}")
        print(f"✅ Phase 5 complete: {len(answer.citations)} citations from {answer.file_count} files")

    except Exception as e:
        logger.error(f"❌ Failed to log final answer: {e}", exc_info=True)
