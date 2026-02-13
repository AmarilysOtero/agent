"""Chunk logging utility for Phase 3 - writes chunk information to markdown files."""

import logging
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _chunk_logs_base_dir() -> Path:
    """Base directory for chunk logs (project-local on Windows/non-Docker)."""
    if os.name == "nt" or not Path("/.dockerenv").exists():
        # __file__ = .../src/news_reporter/retrieval/chunk_logger.py -> repo root = parent x4
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        return repo_root / "logs" / "chunk_analysis"
    return Path("/app/logs/chunk_analysis")


# Output directory for chunk logs (project-local on Windows so enable/disable exist under repo)
CHUNK_LOGS_DIR = _chunk_logs_base_dir()
CHUNK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_LOGS_ENABLE_DIR = CHUNK_LOGS_DIR / "enable"
CHUNK_LOGS_DISABLE_DIR = CHUNK_LOGS_DIR / "disable"
CHUNK_LOGS_ENABLE_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_LOGS_DISABLE_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_RLM_DISABLED_FILE = CHUNK_LOGS_DISABLE_DIR / "chunks_rlm_disabled.md"
CHUNKS_RLM_ENABLED_FILE = CHUNK_LOGS_ENABLE_DIR / "chunks_rlm_enabled.md"
AGGREGATE_RAW_FILE_NAME = "aggregate_final_answer_raw.md"


def _resolve_chunk_logs_dir(output_dir: Optional[str] = None, rlm_enabled: bool = False) -> Path:
    """Resolve chunk logs directory with RLM enable/disable subfolder."""
    base_dir = Path(output_dir) if output_dir else _chunk_logs_base_dir()
    subfolder = "enable" if rlm_enabled else "disable"
    target_dir = base_dir / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def init_aggregate_raw_log(query: Optional[str] = None, output_dir: Optional[str] = None, rlm_enabled: bool = True) -> Path:
    """Initialize the aggregate raw log (overwrite per query)."""
    target_dir = _resolve_chunk_logs_dir(output_dir, rlm_enabled=rlm_enabled)
    output_path = target_dir / AGGREGATE_RAW_FILE_NAME

    content = [
        "# Aggregate Final Answer - Raw Chunks (Pre-LLM)\n",
        "This file captures raw chunks before LLM summarization.\n",
        "(File overwrites with each new query)\n\n",
        "---\n\n",
        f"## Execution: {datetime.now().isoformat()}\n",
    ]
    if query:
        content.append(f"**Query**: {query}\n\n")
    content.append("---\n\n")
    content.append("## Passing Chunks (Append Log)\n\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(content))

    logger.info(f"📝 Initialized aggregate raw log: {output_path}")
    return output_path


def ensure_rlm_enable_log_files(query: Optional[str] = None) -> None:
    """Ensure logs/chunk_analysis/enable exists with initial .md files (for RLM-enabled runs)."""
    init_aggregate_raw_log(query=query, rlm_enabled=True)
    if not CHUNKS_RLM_ENABLED_FILE.exists():
        CHUNKS_RLM_ENABLED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_RLM_ENABLED_FILE, "w", encoding="utf-8") as f:
            f.write("# Chunks (RLM Enabled)\n\nChunk log (append).\n\n")
        logger.info(f"📝 Initialized chunks log: {CHUNKS_RLM_ENABLED_FILE}")


def append_aggregate_raw_chunk(
    chunk_id: str,
    file_id: str,
    file_name: str,
    chunk_text: str,
    phase: str,
    output_dir: Optional[str] = None,
    rlm_enabled: bool = True
) -> None:
    """Append a passing chunk to the aggregate raw log."""
    output_path = _resolve_chunk_logs_dir(output_dir, rlm_enabled=rlm_enabled) / AGGREGATE_RAW_FILE_NAME


    # Always include selection method
    if phase == "keyword_pre_filter":
        selection_method = "keyword"
    elif phase == "iterative_boolean_eval":
        selection_method = "recursive"
    else:
        selection_method = "unknown"

    entry = [
        f"### Chunk: {chunk_id}\n",
        f"**File ID:** {file_id}\n",
        f"**File Name:** {file_name}\n",
        f"**Phase:** {phase}\n",
        f"**Captured:** {datetime.now().isoformat()}\n",
        f"**Selection Method:** {selection_method}\n",
        "\n",
        "```text\n",
        f"{chunk_text}\n",
        "```\n\n",
    ]

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("".join(entry))


def log_aggregate_raw_final_set(
    file_id: str,
    file_name: str,
    selected_chunks: List[Dict],
    output_dir: Optional[str] = None,
    rlm_enabled: bool = True
) -> None:
    """Append the final selected chunks before summarization."""
    output_path = _resolve_chunk_logs_dir(output_dir, rlm_enabled=rlm_enabled) / AGGREGATE_RAW_FILE_NAME

    header = [
        "## Final Selected Chunks\n\n",
        f"### File: {file_name} (ID: {file_id})\n\n",
    ]

    total_chars = sum(len(chunk.get("text", "")) for chunk in selected_chunks)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("".join(header))
        for chunk in selected_chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            chunk_text = chunk.get("text", "")
            timestamp = datetime.now().isoformat()
            f.write(f"#### Chunk: {chunk_id}\n\n")
            f.write(f"**Analyzed At:** {timestamp}\n\n")
            f.write("```text\n")
            f.write(f"{chunk_text}\n")
            f.write("```\n\n")

        f.write("**Summary:**\n")
        f.write(f"- Selected chunks: {len(selected_chunks)}\n")
        f.write(f"- Total chars: {total_chars}\n\n")


async def log_chunks_to_markdown(
    chunks: List[Dict],
    rlm_enabled: bool = False,
    query: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Log chunk information to markdown files.
    
    Maintains two reusable markdown files:
    - chunks_rlm_disabled.md: Chunks retrieved with RLM disabled (standard retrieval)
    - chunks_rlm_enabled.md: Chunks retrieved/expanded with RLM enabled (Phase 3 expansion)
    
    Args:
        chunks: List of chunk dictionaries with chunk_id, text, file, etc.
        rlm_enabled: Whether RLM (High-Recall Learner) was enabled
        query: Original query string (optional)
        metadata: Additional metadata like file_id, expansion_info, etc. (optional)
    """
    
    target_file = CHUNKS_RLM_ENABLED_FILE if rlm_enabled else CHUNKS_RLM_DISABLED_FILE
    mode_label = "RLM ENABLED" if rlm_enabled else "RLM DISABLED"
    
    try:
        # Check if file exists to determine if we should add a section
        file_exists = target_file.exists()
        
        # Build markdown content
        content = []
        
        # Add header on every write (overwrite mode)
        content.append(f"# Chunk Analysis - {mode_label}\n")
        content.append("This file contains chunk information from the latest query execution.\n")
        content.append("(File overwrites with each new query)\n\n")
        content.append("---\n\n")
        
        # Add timestamp and query info
        content.append(f"## Execution: {datetime.now().isoformat()}\n")
        
        if query:
            content.append(f"**Query**: {query}\n\n")
        
        if metadata:
            content.append("**Metadata**:\n")
            for key, value in metadata.items():
                content.append(f"- {key}: {value}\n")
            content.append("\n")
        
        # Add chunk summary
        content.append(f"**Total Chunks**: {len(chunks)}\n\n")
        
        # Add chunks table
        content.append("### Chunks\n\n")
        content.append("| # | Timestamp | Chunk ID | File | Text Preview | Size |\n")
        content.append("|---|-----------|----------|------|--------------|------|\n")
        
        for idx, chunk in enumerate(chunks, 1):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            chunk_id = chunk.get("chunk_id", chunk.get("id", "N/A"))
            file_name = chunk.get("file", "N/A")
            text = chunk.get("text", "")
            text_preview = (text[:100] + "...") if len(text) > 100 else text
            text_preview = text_preview.replace("\n", " ").replace("|", "\\|")
            size = len(text)
            
            content.append(f"| {idx} | {timestamp} | `{chunk_id}` | {file_name} | {text_preview} | {size} bytes |\n")
        
        content.append("\n---\n\n")
        
        # Write to file (OVERWRITE mode - 'w' instead of 'a')
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("".join(content))
        
        logger.info(
            f"📝 Logged {len(chunks)} chunks to {target_file.name} (RLM: {rlm_enabled})"
        )
        print(f"✅ Chunk logging: {len(chunks)} chunks logged to {target_file.name}")
        
    except Exception as e:
        logger.error(
            f"❌ Failed to log chunks to markdown: {str(e)}",
            exc_info=True
        )


async def log_phase3_expansion(
    entry_chunks: List[Dict],
    expanded_chunks: Dict[str, Dict],
    query: Optional[str] = None,
    rlm_enabled: bool = True
) -> None:
    """
    Log Phase 3 file expansion details to markdown.
    
    Args:
        entry_chunks: Original entry chunks from retrieval
        expanded_chunks: Result from expand_to_full_files() - {file_id: {chunks: [...], ...}}
        query: Original query string
        rlm_enabled: Whether RLM is enabled (defaults to True since Phase 3 only runs with RLM)
    """
    
    try:
        # Flatten expanded chunks for logging
        all_expanded_chunks = []
        for file_id, file_data in expanded_chunks.items():
            chunks = file_data.get("chunks", [])
            all_expanded_chunks.extend(chunks)
        
        # Build metadata
        metadata = {
            "Entry Chunks": len(entry_chunks),
            "Expanded Chunks": len(all_expanded_chunks),
            "Source Files": len(expanded_chunks),
            "Expansion Ratio": f"{len(all_expanded_chunks)} / {len(entry_chunks)}" if entry_chunks else "N/A"
        }
        
        if expanded_chunks:
            file_info = []
            for file_id, file_data in expanded_chunks.items():
                file_name = file_data.get("file_name", file_id)
                total = file_data.get("total_chunks", 0)
                entry = file_data.get("entry_chunk_count", 0)
                file_info.append(f"{file_name} ({entry}/{total})")
            metadata["Files Expanded"] = "; ".join(file_info)
        
        # Log expanded chunks
        await log_chunks_to_markdown(
            chunks=all_expanded_chunks,
            rlm_enabled=rlm_enabled,
            query=query,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(
            f"❌ Failed to log Phase 3 expansion: {str(e)}",
            exc_info=True
        )


def get_chunk_log_file_path(rlm_enabled: bool) -> Path:
    """Get the path to the chunk log file."""
    return CHUNKS_RLM_ENABLED_FILE if rlm_enabled else CHUNKS_RLM_DISABLED_FILE


def clear_chunk_logs(rlm_enabled: Optional[bool] = None) -> None:
    """
    Clear chunk log files.
    
    Args:
        rlm_enabled: Which log to clear. None = clear both
    """
    files_to_clear = []
    
    if rlm_enabled is None:
        files_to_clear = [CHUNKS_RLM_DISABLED_FILE, CHUNKS_RLM_ENABLED_FILE]
    elif rlm_enabled:
        files_to_clear = [CHUNKS_RLM_ENABLED_FILE]
    else:
        files_to_clear = [CHUNKS_RLM_DISABLED_FILE]
    
    for file_path in files_to_clear:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️  Cleared chunk log: {file_path.name}")
