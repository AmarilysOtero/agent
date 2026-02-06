"""Unit tests for MIT RLM recursive summarizer helpers."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.news_reporter.retrieval.recursive_summarizer import (
    _generate_recursive_inspection_program,
    _execute_inspection_program,
    _get_fallback_inspection_program,
    _get_fallback_result,
)


class TestRLMRecursion:
    """Unit tests for recursive inspection program generation and execution."""

    @pytest.mark.asyncio
    async def test_generate_recursive_program_valid_output(self):
        """Program generation returns valid Python code."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "def inspect_iteration(chunks):\n"
            "    return {\n"
            "        'selected_chunk_ids': [c['chunk_id'] for c in chunks[:3]],\n"
            "        'extracted_data': {'test': 'data'},\n"
            "        'confidence': 0.7,\n"
            "        'stop': False\n"
            "    }"
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        chunks = [{"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(5)]

        program = await _generate_recursive_inspection_program(
            query="test query",
            file_name="test.pdf",
            active_chunks=chunks,
            iteration=0,
            previous_data={},
            llm_client=mock_client,
            model_deployment="gpt-4o-mini",
        )

        assert "def inspect_iteration" in program
        assert len(program) > 50

    @pytest.mark.asyncio
    async def test_execute_program_structured_output(self):
        """Program execution returns properly structured output."""
        valid_program = (
            "def inspect_iteration(chunks):\n"
            "    return {\n"
            "        'selected_chunk_ids': [chunks[0]['chunk_id'], chunks[1]['chunk_id']],\n"
            "        'extracted_data': {'entities': ['Entity1', 'Entity2']},\n"
            "        'confidence': 0.75,\n"
            "        'stop': False\n"
            "    }"
        )

        chunks = [
            {"chunk_id": "chunk_1", "text": "Text 1"},
            {"chunk_id": "chunk_2", "text": "Text 2"},
            {"chunk_id": "chunk_3", "text": "Text 3"},
        ]

        result = await _execute_inspection_program(
            chunks=chunks,
            program=valid_program,
            iteration=0,
        )

        assert isinstance(result, dict)
        assert "selected_chunk_ids" in result
        assert "extracted_data" in result
        assert "confidence" in result
        assert "stop" in result
        assert len(result["selected_chunk_ids"]) == 2
        assert result["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_execute_program_invalid_output_uses_fallback(self):
        """Invalid program output triggers fallback."""
        invalid_program = (
            "def inspect_iteration(chunks):\n"
            "    return 'invalid string output'"
        )

        chunks = [{"chunk_id": "chunk_1", "text": "Text 1"}]

        result = await _execute_inspection_program(
            chunks=chunks,
            program=invalid_program,
            iteration=0,
        )

        assert result["extracted_data"].get("fallback") is True
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_execute_program_syntax_error_uses_fallback(self):
        """Syntax errors in program trigger fallback."""
        broken_program = (
            "def inspect_iteration(chunks)\n"
            "    return {}"
        )

        chunks = [{"chunk_id": "chunk_1", "text": "Text 1"}]

        result = await _execute_inspection_program(
            chunks=chunks,
            program=broken_program,
            iteration=1,
        )

        assert result["extracted_data"].get("fallback") is True

    def test_fallback_program_is_valid_python(self):
        """Fallback program compiles without errors."""
        fallback = _get_fallback_inspection_program("test query", 0)

        compile(fallback, "<string>", "exec")

        assert "def inspect_iteration" in fallback
        assert "selected_chunk_ids" in fallback

    def test_get_fallback_result_structure(self):
        """Fallback result has correct structure."""
        chunks = [{"chunk_id": f"c{i}", "text": f"Text {i}"} for i in range(5)]

        result = _get_fallback_result(chunks, iteration=2)

        assert isinstance(result["selected_chunk_ids"], list)
        assert isinstance(result["extracted_data"], dict)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["stop"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
