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

# MIT RLM Configuration (Phase 2 feature flag)
USE_MIT_RLM_RECURSION = os.getenv("USE_MIT_RLM_RECURSION", "false").lower() == "true"
MAX_RLM_ITERATIONS = int(os.getenv("MAX_RLM_ITERATIONS", "5"))

logger.info(f"🔧 MIT RLM Recursion: {'ENABLED' if USE_MIT_RLM_RECURSION else 'DISABLED'}")


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
    inspection_code_with_text = {}  # Store inspection code with chunk text details

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

            # MIT RLM: Generate and apply inspection code per chunk
            file_inspection_codes = {}  # Store generated code for each chunk
            file_inspection_payloads = {}  # Store code + text metadata per chunk
            relevant_chunks = []
            
            for idx, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id", f"unknown-{idx}")
                    chunk_text = chunk.get("text", "").strip()
                    
                    if not chunk_text:
                        continue
                    
                    logger.debug(f"  → Generating inspection code for chunk {idx} ({chunk_id[:30]}...)")
                    
                    # Step 1: Generate Python code specific to this chunk (MIT RLM per-chunk approach)
                    generated_code = await _generate_inspection_logic(
                        query=query,
                        file_name=file_name,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                        llm_client=llm_client,
                        model_deployment=model_deployment
                    )
                    
                    file_inspection_codes[chunk_id] = generated_code
                    file_inspection_payloads[chunk_id] = {
                        "code": generated_code,
                        "chunk_text": chunk_text,
                        "first_read_text": chunk_text[:500]
                    }
                    
                    # Step 2: Apply the generated code to evaluate this specific chunk
                    logger.debug(f"  → Evaluating chunk {idx} with generated code")
                    is_relevant = await _evaluate_chunk_with_code(
                        chunk_text=chunk_text,
                        inspection_code=generated_code,
                        chunk_id=chunk_id
                    )
                    
                    if is_relevant:
                        relevant_chunks.append(chunk_text)
                        logger.debug(f"    ✓ Chunk {idx} is relevant")
                    else:
                        logger.debug(f"    ✗ Chunk {idx} is not relevant")
            
            # Store all chunk-level inspection codes for this file
            inspection_code[file_id] = file_inspection_codes
            inspection_code_with_text[file_id] = file_inspection_payloads

            if not relevant_chunks:
                logger.warning(f"⚠️  Phase 4: No relevant chunks identified in {file_name}")
                # Fallback: use first few chunks if none pass relevance
                relevant_chunks = [chunk.get("text", "").strip() for chunk in chunks[:min(3, len(chunks))] if chunk.get("text", "").strip()]

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

            # Get chunk IDs from file_inspection_codes for citations
            source_chunk_ids = list(file_inspection_codes.keys())[:len(relevant_chunks)]

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
    summary_by_file_id = {summary.file_id: summary.summary_text for summary in summaries}

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

    if inspection_code_with_text:
        try:
            await log_inspection_code_with_text_to_markdown(
                inspection_rules=inspection_code_with_text,
                query=query,
                summary_by_file_id=summary_by_file_id,
                rlm_enabled=True,
                output_dir="/app/logs/chunk_analysis"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to log inspection code with text: {e}")

    return summaries


async def _generate_inspection_logic(
    query: str,
    file_name: str,
    chunk_id: str,
    chunk_text: str,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate LLM-based inspection logic (executable Python code) for a specific chunk.

    MIT RLM approach: Generate small, executable Python code for each chunk
    to determine if it's relevant to the user query.

    Args:
        query: User query
        file_name: Name of file being analyzed
        chunk_id: ID of the chunk being evaluated
        chunk_text: The actual chunk content
        llm_client: Azure OpenAI client
        model_deployment: Azure deployment name

    Returns:
        Python code as string that can evaluate chunk relevance
    """
    # Use gpt-4o-mini for code generation (o3-mini returns empty responses)
    code_generation_deployment = "gpt-4o-mini"
    if model_deployment.startswith(('gpt-', 'gpt4')):
        code_generation_deployment = model_deployment

    prompt = f"""You are implementing the MIT Recursive Inspection Model (RLM) for document analysis.

TASK: Generate a Python function to evaluate if this specific chunk is relevant to the user's query.

Document: {file_name}
Chunk ID: {chunk_id}
User Query: {query}

CHUNK CONTENT:
{chunk_text[:500]}

Generate a Python function called `evaluate_chunk_relevance(chunk_text: str) -> bool` that:
1. Returns True if the chunk is relevant to the user query
2. Returns False if the chunk is NOT relevant
3. Uses simple string operations (case-insensitive matching)
4. Implements 2-3 specific criteria based on this chunk's content

Requirements:
- Function signature: def evaluate_chunk_relevance(chunk_text: str) -> bool:
- Valid, executable Python code only
- Return ONLY the function code with no explanations
- No imports needed

NOW generate the function:"""

    try:
        params = _build_completion_params(
            code_generation_deployment,
            model=code_generation_deployment,
            messages=[
                {"role": "system", "content": "You are a Python expert. Generate clean, executable Python code with no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=300
        )
        
        response = await llm_client.chat.completions.create(**params)
        inspection_code = response.choices[0].message.content.strip()
        
        # Handle markdown-wrapped code (```python ... ```)
        if inspection_code.startswith("```"):
            logger.debug("    Extracting code from markdown wrapper")
            lines = inspection_code.split("\n")
            inspection_code = "\n".join(lines[1:-1]) if len(lines) > 2 else inspection_code
        
        # Check if response is empty or doesn't contain the function signature
        if not inspection_code or "def evaluate_chunk_relevance" not in inspection_code:
            logger.warning(f"⚠️  LLM returned empty or incomplete code for chunk {chunk_id}")
            # Fallback: return simple query-based filter function
            fallback_code = f"""def evaluate_chunk_relevance(chunk_text: str) -> bool:
    \"\"\"Fallback relevance filter based on query terms.\"\"\"
    text_lower = chunk_text.lower()
    query_terms = {repr(query.lower().split())}
    return sum(1 for term in query_terms if term in text_lower) >= 2"""
            return fallback_code
        
        logger.debug(f"    Generated {len(inspection_code)} char code for chunk {chunk_id}")
        return inspection_code
    except Exception as e:
        logger.warning(f"⚠️  Failed to generate code for chunk {chunk_id}: {e}")
        # Fallback: return simple query-based filter function
        fallback_code = f"""def evaluate_chunk_relevance(chunk_text: str) -> bool:
    \"\"\"Fallback relevance filter based on query terms.\"\"\"
    text_lower = chunk_text.lower()
    query_terms = {repr(query.lower().split())}
    return sum(1 for term in query_terms if term in text_lower) >= 2"""
        return fallback_code


async def _generate_recursive_inspection_program(
    query: str,
    file_name: str,
    active_chunks: List[Dict],
    iteration: int,
    previous_data: Dict,
    llm_client: Any,
    model_deployment: str
) -> str:
    """
    Generate MIT RLM-compliant inspection program for one iteration.

    Returns Python code that evaluates all chunks and returns structured output:
    {
        "selected_chunk_ids": [...],
        "extracted_data": {...},
        "confidence": 0.0-1.0,
        "stop": True/False
    }
    """
    chunk_summaries = "\n".join([
        f"  - Chunk {i} (ID: {c.get('chunk_id', '')[:20]}...): {c.get('text', '')[:100]}..."
        for i, c in enumerate(active_chunks[:10])
    ])

    prompt = f"""You are implementing the MIT Recursive Language Model (RLM) for document analysis.

ITERATION {iteration + 1}/5
Document: {file_name}
User Query: {query}

ACTIVE CHUNKS ({len(active_chunks)} total):
{chunk_summaries}

PREVIOUS ITERATIONS DATA:
{previous_data if previous_data else "None (first iteration)"}

Generate a Python function with this EXACT signature:

def inspect_iteration(chunks):
    \"\""
    Evaluate chunks for this iteration and return structured output.

    Args:
        chunks: List of dicts with keys: chunk_id, text

    Returns:
        {{
            "selected_chunk_ids": [...],  # IDs of relevant chunks to keep
            "extracted_data": {{}},        # Any data extracted this iteration
            "confidence": 0.0-1.0,        # Confidence in completeness
            "stop": True/False            # Whether we have enough information
        }}
    \"\""
    # Your implementation here
    pass

Requirements:
1. Evaluate ALL chunks in the chunks list
2. Return structured dict matching the schema above
3. Use simple string operations (no imports)
4. Be specific to this iteration and previous findings
5. Narrow focus each iteration (select fewer chunks)
6. selected_chunk_ids MUST be a subset of provided chunk IDs (validate before returning)
7. Return at least 2 selected_chunk_ids unless stop=True
8. confidence MUST be a float in range [0.0, 1.0]
9. Return ONLY function code with no markdown or explanations

Generate ONLY the function code with no explanations:"""

    try:
        code_generation_deployment = "gpt-4o-mini"
        if model_deployment.startswith(('gpt-', 'gpt4')):
            code_generation_deployment = model_deployment

        params = _build_completion_params(
            code_generation_deployment,
            model=code_generation_deployment,
            messages=[
                {"role": "system", "content": "You are a Python expert implementing MIT RLM. Generate clean, executable code with no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=600
        )

        response = await llm_client.chat.completions.create(**params)
        program_code = response.choices[0].message.content.strip()

        if program_code.startswith("```"):
            lines = program_code.split("\n")
            program_code = "\n".join(lines[1:-1]) if len(lines) > 2 else program_code

        if not program_code or "def inspect_iteration" not in program_code:
            logger.warning(f"⚠️  LLM returned incomplete program for iteration {iteration}")
            return _get_fallback_inspection_program(query, iteration)

        logger.debug(f"    Generated {len(program_code)} char program for iteration {iteration}")
        return program_code

    except Exception as e:
        logger.warning(f"⚠️  Failed to generate program for iteration {iteration}: {e}")
        return _get_fallback_inspection_program(query, iteration)


def _get_fallback_inspection_program(query: str, iteration: int) -> str:
    """Fallback inspection program when LLM fails."""
    query_terms = query.lower().split()

    return f"""def inspect_iteration(chunks):
    \"\""Fallback program based on query term matching.\"\""
    query_terms = {repr(query_terms)}
    selected_ids = []
    extracted_data = {{}}

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        if matches >= max(2, len(query_terms) // 2):
            selected_ids.append(chunk['chunk_id'])

    confidence = min(1.0, len(selected_ids) / max(1, len(chunks)))
    stop = iteration >= 3 or confidence > 0.8

    return {{
        "selected_chunk_ids": selected_ids,
        "extracted_data": {{"fallback": True}},
        "confidence": confidence,
        "stop": stop
    }}"""


async def _evaluate_chunk_with_code(
    chunk_text: str,
    inspection_code: str,
    chunk_id: str
) -> bool:
    """
    Execute generated inspection code against a specific chunk.

    MIT RLM: Execute the Python function to evaluate if chunk is relevant.

    Args:
        chunk_text: The chunk to evaluate
        inspection_code: Python code containing evaluate_chunk_relevance function
        chunk_id: ID of chunk for logging

    Returns:
        True if chunk is relevant, False otherwise
    """
    try:
        # Execute the generated Python code
        namespace = {}
        exec(inspection_code, namespace)
        
        # Get the evaluate_chunk_relevance function
        evaluate_func = namespace.get("evaluate_chunk_relevance")
        
        if evaluate_func is None:
            logger.warning(f"⚠️  Code for {chunk_id} doesn't contain evaluate_chunk_relevance; returning False")
            return False
        
        # Execute the function against the chunk
        result = evaluate_func(chunk_text)
        return bool(result)
        
    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in code for {chunk_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error executing code for {chunk_id}: {e}")
        return False


async def _execute_inspection_program(
    chunks: List[Dict],
    program: str,
    iteration: int
) -> Dict:
    """
    Execute MIT RLM inspection program and return structured output.

    The program is expected to define:
        def inspect_iteration(chunks) -> Dict
    """
    try:
        safe_globals = {
            "__builtins__": {
                "len": len,
                "list": list,
                "dict": dict,
                "set": set,
                "sum": sum,
                "min": min,
                "max": max,
                "int": int,
                "float": float,
                "str": str,
                "range": range,
                "enumerate": enumerate,
                "sorted": sorted,
                "any": any,
                "all": all
            }
        }
        namespace = {}
        exec(program, safe_globals, namespace)

        inspect_func = namespace.get("inspect_iteration")

        if inspect_func is None:
            logger.warning(f"⚠️  Program for iteration {iteration} missing inspect_iteration function")
            return _get_fallback_result(chunks, iteration)

        chunk_list = [
            {"chunk_id": chunk.get("chunk_id", f"unknown-{i}"), "text": chunk.get("text", "")}
            for i, chunk in enumerate(chunks)
        ]
        valid_chunk_ids = set(chunk.get("chunk_id") for chunk in chunk_list)

        result = inspect_func(chunk_list)

        if not isinstance(result, dict):
            logger.warning(f"⚠️  Program returned non-dict: {type(result)}")
            return _get_fallback_result(chunks, iteration)

        MIN_KEEP = 2
        selected_ids = result.get("selected_chunk_ids", [])
        should_stop = result.get("stop", False)

        if isinstance(selected_ids, list):
            selected_ids = [cid for cid in selected_ids if cid in valid_chunk_ids]
        else:
            selected_ids = []

        if not should_stop and len(selected_ids) < MIN_KEEP:
            candidate_ids_ordered = [chunk.get("chunk_id") for chunk in chunk_list if chunk.get("chunk_id") in valid_chunk_ids]
            selected_ids = candidate_ids_ordered[:MIN_KEEP]
            logger.debug(f"    📌 Enforcing MIN_KEEP={MIN_KEEP}, selected first {len(selected_ids)} chunks by original order")

        extracted = result.get("extracted_data", {})
        MAX_EXTRACTED_SIZE = 50000
        if isinstance(extracted, dict):
            extracted_size = len(str(extracted))
            if extracted_size > MAX_EXTRACTED_SIZE:
                logger.warning(
                    f"    ⚠️  Iteration {iteration}: extracted_data too large ({extracted_size} bytes), truncating"
                )
                truncated = {}
                for k, v in extracted.items():
                    if isinstance(v, str):
                        truncated[k] = v[:500]
                    elif isinstance(v, (int, float, bool)):
                        truncated[k] = v
                extracted = truncated

        validated_result = {
            "selected_chunk_ids": selected_ids,
            "extracted_data": extracted,
            "confidence": max(0.0, min(1.0, result.get("confidence", 0.5))),
            "stop": bool(should_stop)
        }

        return validated_result

    except SyntaxError as e:
        logger.warning(f"⚠️  Syntax error in program for iteration {iteration}: {e}")
        return _get_fallback_result(chunks, iteration)
    except Exception as e:
        logger.warning(f"⚠️  Error executing program for iteration {iteration}: {e}")
        return _get_fallback_result(chunks, iteration)


def _get_fallback_result(chunks: List[Dict], iteration: int) -> Dict:
    """Generate safe fallback result when program execution fails."""
    chunk_ids = [chunk.get("chunk_id", f"unknown-{i}") for i, chunk in enumerate(chunks)]

    return {
        "selected_chunk_ids": chunk_ids[:min(10, len(chunk_ids))],
        "extracted_data": {"fallback": True, "iteration": iteration},
        "confidence": 0.3,
        "stop": iteration >= 3
    }


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

    # Use gpt-4o-mini if o3-mini is configured
    analysis_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment

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
            analysis_deployment,
            model=analysis_deployment,
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
    # Use gpt-4o-mini if o3-mini is configured
    summary_deployment = "gpt-4o-mini" if model_deployment.startswith('o3') else model_deployment
    
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
            summary_deployment,
            model=summary_deployment,
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
    inspection_rules: Dict[str, Dict[str, str]],
    query: str,
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log LLM-generated Python inspection code to markdown file.

    This stores the executable Python code generated by the LLM per the MIT RLM model.
    Each chunk gets its own Python function (evaluate_chunk_relevance) that determines 
    if that specific chunk is relevant to the user query.

    Args:
        inspection_rules: Dict mapping file_id to dict of chunk_id -> code (MIT RLM per-chunk approach)
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

        # Calculate total number of inspection programs (all chunks)
        total_programs = sum(len(chunk_codes) for chunk_codes in inspection_rules.values())

        # Build markdown content
        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Inspection Programs:** {total_programs}",
            f"\n**Implementation:** MIT Recursive Inspection Model (RLM) - Per-Chunk Code Generation",
            "\n---\n",
            "## Overview\n",
            "This file stores the **executable Python code** generated by the LLM per MIT RLM.",
            "Each chunk gets its own Python function `evaluate_chunk_relevance(chunk_text: str) -> bool`",
            "that determines if that specific chunk contains information relevant to the user's query.\n",
            "### Purpose\n",
            "- Generate chunk-specific relevance evaluation code",
            "- Each chunk receives tailored inspection logic",
            "- More precise relevance filtering per MIT RLM approach\n",
            "### Usage\n",
            "These functions are executed by the recursive summarizer to evaluate each chunk individually.\n",
            "---\n",
        ]

        # Add each file's inspection codes (per chunk)
        file_counter = 1
        for file_id, chunk_codes in inspection_rules.items():
            lines.extend([
                f"## {file_counter}. File (ID: {file_id})",
                "\n",
            ])
            
            chunk_counter = 1
            for chunk_id, code in chunk_codes.items():
                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Query:** {query}\n",
                    "```python",
                    code,
                    "```\n",
                ])
                chunk_counter += 1
            
            lines.append("---\n")
            file_counter += 1

        content = "\n".join(lines)

        # Write to file (overwrite mode - one query per file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code logged to {output_path}")
        logger.info(f"   Stored {total_programs} per-chunk inspection programs following MIT RLM model")


    except Exception as e:
        logger.error(f"❌ Failed to log inspection code: {e}", exc_info=True)
        raise


async def log_inspection_code_with_text_to_markdown(
    inspection_rules: Dict[str, Dict[str, Dict[str, str]]],
    query: str,
    summary_by_file_id: Dict[str, str],
    rlm_enabled: bool = True,
    output_dir: str = "/app/logs/chunk_analysis"
) -> None:
    """
    Log inspection code with chunk text, first read, and recursive text.

    Args:
        inspection_rules: Dict mapping file_id to dict of chunk_id -> payload
            payload keys: code, chunk_text, first_read_text
        query: User query that drove the analysis
        summary_by_file_id: Dict mapping file_id to recursive summary text
        rlm_enabled: Whether RLM is enabled
        output_dir: Output directory for logs
    """
    from pathlib import Path
    from datetime import datetime

    try:
        file_suffix = "enable" if rlm_enabled else "disable"
        output_path = Path(output_dir) / f"inspection_code_chunk_rlm_{file_suffix}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_programs = sum(len(chunk_payloads) for chunk_payloads in inspection_rules.values())

        timestamp = datetime.now().isoformat()
        lines = [
            f"# Phase 4: LLM-Generated Inspection Logic (RLM {'Enabled' if rlm_enabled else 'Disabled'})",
            f"\n**Execution Time:** {timestamp}",
            f"\n**Query:** {query}",
            f"\n**Total Inspection Programs:** {total_programs}",
            f"\n**Implementation:** MIT Recursive Inspection Model (RLM) - Per-Chunk Code Generation",
            "\n---\n",
            "## Overview\n",
            "This file stores the **executable Python code** generated by the LLM per MIT RLM.",
            "Each chunk gets its own Python function `evaluate_chunk_relevance(chunk_text: str) -> bool`",
            "that determines if that specific chunk contains information relevant to the user's query.\n",
            "### Purpose\n",
            "- Generate chunk-specific relevance evaluation code",
            "- Preserve the exact chunk text and first read passed to the model",
            "- Record the recursive summary text used for Phase 4\n",
            "### Usage\n",
            "These functions are executed by the recursive summarizer to evaluate each chunk individually.\n",
            "---\n",
        ]

        file_counter = 1
        for file_id, chunk_payloads in inspection_rules.items():
            recursive_text = summary_by_file_id.get(file_id, "")
            lines.extend([
                f"## {file_counter}. File (ID: {file_id})",
                "\n",
            ])

            chunk_counter = 1
            for chunk_id, payload in chunk_payloads.items():
                code = payload.get("code", "")
                chunk_text = payload.get("chunk_text", "")
                first_read_text = payload.get("first_read_text", "")

                lines.extend([
                    f"### {file_counter}.{chunk_counter} Chunk: {chunk_id}",
                    f"\n**Query:** {query}\n",
                    "```python",
                    code,
                    "```\n",
                    "#### Chunk Text",
                    "```text",
                    chunk_text,
                    "```\n",
                    "#### First Read",
                    "```text",
                    first_read_text,
                    "```\n",
                    "#### Recursive Text",
                    "```text",
                    recursive_text,
                    "```\n",
                ])
                chunk_counter += 1

            lines.append("---\n")
            file_counter += 1

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Phase 4 inspection code with text logged to {output_path}")
        logger.info(f"   Stored {total_programs} per-chunk inspection programs with text")

    except Exception as e:
        logger.error(f"❌ Failed to log inspection code with text: {e}", exc_info=True)
        raise
