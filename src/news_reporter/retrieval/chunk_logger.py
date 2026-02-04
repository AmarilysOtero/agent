"""Chunk logging utility for Phase 3 - writes chunk information to markdown files."""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Output directory for chunk logs
CHUNK_LOGS_DIR = Path("/app/logs/chunk_analysis")
CHUNK_LOGS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_RLM_DISABLED_FILE = CHUNK_LOGS_DIR / "chunks_rlm_disabled.md"
CHUNKS_RLM_ENABLED_FILE = CHUNK_LOGS_DIR / "chunks_rlm_enabled.md"


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
        
        if not file_exists:
            # Add header only on first write
            content.append(f"# Chunk Analysis - {mode_label}\n")
            content.append("This file contains chunk information logged from Phase 3 retrieval operations.\n")
            content.append("File is reused and updated with each query execution.\n\n")
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
        content.append("| # | Chunk ID | File | Text Preview | Size |\n")
        content.append("|---|----------|------|--------------|------|\n")
        
        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("chunk_id", chunk.get("id", "N/A"))
            file_name = chunk.get("file", "N/A")
            text = chunk.get("text", "")
            text_preview = (text[:100] + "...") if len(text) > 100 else text
            text_preview = text_preview.replace("\n", " ").replace("|", "\\|")
            size = len(text)
            
            content.append(f"| {idx} | `{chunk_id}` | {file_name} | {text_preview} | {size} bytes |\n")
        
        content.append("\n---\n\n")
        
        # Write to file
        with open(target_file, "a", encoding="utf-8") as f:
            f.write("".join(content))
        
        logger.info(
            f"📝 Logged {len(chunks)} chunks to {target_file.name} (RLM: {rlm_enabled})"
        )
        
    except Exception as e:
        logger.error(
            f"❌ Failed to log chunks to markdown: {str(e)}",
            exc_info=True
        )


async def log_phase3_expansion(
    entry_chunks: List[Dict],
    expanded_chunks: Dict[str, Dict],
    query: Optional[str] = None
) -> None:
    """
    Log Phase 3 file expansion details to markdown.
    
    Args:
        entry_chunks: Original entry chunks from retrieval
        expanded_chunks: Result from expand_to_full_files() - {file_id: {chunks: [...], ...}}
        query: Original query string
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
            rlm_enabled=True,
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
