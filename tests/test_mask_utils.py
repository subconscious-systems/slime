"""Tests for mask_utils, specifically tool_result masking."""

import pytest
from slime.utils.mask_utils import find_tool_result_value_spans


class TestFindToolResultValueSpans:
    def test_no_tool_result(self):
        content = '{"tooluse": {"tool_name": "search", "parameters": {"q": "hello"}}}'
        assert find_tool_result_value_spans(content) == []

    def test_simple_tool_result(self):
        content = '{"tooluse": {"tool_name": "search", "parameters": {"q": "hello"}, "tool_result": {"answer": "world"}}}'
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 1
        assert content[spans[0][0] : spans[0][1]] == '{"answer": "world"}'

    def test_nested_braces_in_tool_result(self):
        content = '{"tooluse": {"tool_name": "search", "tool_result": {"data": {"nested": {"deep": 1}}}}}'
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 1
        assert content[spans[0][0] : spans[0][1]] == '{"data": {"nested": {"deep": 1}}}'

    def test_multiple_tool_results(self):
        content = (
            '[{"tooluse": {"tool_name": "a", "tool_result": {"r": 1}}}, '
            '{"tooluse": {"tool_name": "b", "tool_result": {"r": 2}}}]'
        )
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 2
        assert content[spans[0][0] : spans[0][1]] == '{"r": 1}'
        assert content[spans[1][0] : spans[1][1]] == '{"r": 2}'

    def test_braces_inside_string_values(self):
        content = '{"tooluse": {"tool_name": "x", "tool_result": {"msg": "use {braces} here"}}}'
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 1
        assert content[spans[0][0] : spans[0][1]] == '{"msg": "use {braces} here"}'

    def test_escaped_quotes_in_string(self):
        content = r'{"tooluse": {"tool_result": {"msg": "he said \"hi\""}}}'
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 1
        extracted = content[spans[0][0] : spans[0][1]]
        assert extracted.startswith("{")
        assert extracted.endswith("}")

    def test_whitespace_after_colon(self):
        content = '{"tool_result" :  {"val": 1}}'
        spans = find_tool_result_value_spans(content)
        assert len(spans) == 1
        assert content[spans[0][0] : spans[0][1]] == '{"val": 1}'

    def test_empty_string(self):
        assert find_tool_result_value_spans("") == []

    def test_tool_result_not_object(self):
        # Non-object value after tool_result: should be skipped
        content = '{"tool_result": "just a string"}'
        assert find_tool_result_value_spans(content) == []


class TestMaskToolResultInContent:
    """Test the full masking pipeline with a tokenizer.

    These tests require the transformers library with Qwen tokenizer access.
    They are skipped if the tokenizer is not available.
    """

    @pytest.fixture
    def generator(self):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
            from slime.utils.mask_utils import MultiTurnLossMaskGenerator

            return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen", mask_tool_results=True)
        except Exception:
            pytest.skip("Qwen tokenizer not available")

    def test_tool_result_tokens_masked(self, generator):
        content = '{"tooluse": {"tool_name": "search", "parameters": {"q": "hello"}, "tool_result": {"answer": "world"}}}'
        tokens = generator.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        n_tokens = len(tokens["input_ids"])
        mask = [1] * n_tokens

        result = generator.mask_tool_result_in_content(content, mask)

        # Some tokens should be masked (0)
        assert 0 in result
        # The masked region should correspond to {"answer": "world"}
        spans = find_tool_result_value_spans(content)
        offsets = tokens["offset_mapping"]
        for i, (ts, te) in enumerate(offsets):
            in_span = any(ts < se and te > ss for ss, se in spans)
            assert result[i] == (0 if in_span else 1), f"Token {i} ({ts}-{te}) mismatch"

    def test_no_tool_result_unchanged(self, generator):
        content = '{"tooluse": {"tool_name": "search", "parameters": {"q": "hello"}}}'
        tokens = generator.tokenizer(content, add_special_tokens=False)
        n_tokens = len(tokens["input_ids"])
        mask = [1] * n_tokens

        result = generator.mask_tool_result_in_content(content, mask)
        assert result == mask
