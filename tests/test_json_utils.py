"""Tests for JSON compaction utilities."""

import pytest
from slime.utils.json_utils import compact_json_string, compact_json_response


class TestCompactJsonString:
    """Tests for compact_json_string function."""

    def test_empty_string(self):
        assert compact_json_string("") == ""
        assert compact_json_string(None) is None

    def test_already_compact(self):
        compact = '{"key": "value"}'
        assert compact_json_string(compact) == compact

    def test_simple_pretty_json(self):
        pretty = '''{
  "key": "value"
}'''
        expected = '{"key": "value"}'
        assert compact_json_string(pretty) == expected

    def test_nested_json(self):
        pretty = '''{
  "outer": {
    "inner": "value"
  }
}'''
        expected = '{"outer": {"inner": "value"}}'
        assert compact_json_string(pretty) == expected

    def test_array_json(self):
        pretty = '''[
  1,
  2,
  3
]'''
        expected = '[1, 2, 3]'
        assert compact_json_string(pretty) == expected

    def test_preserve_string_whitespace(self):
        # Whitespace inside strings should be preserved
        pretty = '{"key": "value with   spaces"}'
        expected = '{"key": "value with   spaces"}'
        assert compact_json_string(pretty) == expected

    def test_preserve_newlines_in_string(self):
        # Newlines inside strings should be preserved
        pretty = '{"key": "line1\\nline2"}'
        expected = '{"key": "line1\\nline2"}'
        assert compact_json_string(pretty) == expected

    def test_escape_sequences(self):
        # Escape sequences should be handled properly
        pretty = '{"key": "value with \\"quotes\\""}'
        expected = '{"key": "value with \\"quotes\\""}'
        assert compact_json_string(pretty) == expected

    def test_incomplete_json_missing_close(self):
        # Incomplete JSON - missing closing brace
        pretty = '''{
  "key": "value"'''
        expected = '{"key": "value"'
        assert compact_json_string(pretty) == expected

    def test_incomplete_json_partial_string(self):
        # Incomplete JSON - string not closed
        pretty = '{"key": "incom'
        expected = '{"key": "incom'
        assert compact_json_string(pretty) == expected

    def test_complex_nested_structure(self):
        pretty = '''{
  "reasoning": [
    {
      "thought": "First step",
      "subtasks": [
        {
          "thought": "Subtask 1"
        }
      ],
      "conclusion": "Done"
    }
  ],
  "answer": "42"
}'''
        expected = '{"reasoning": [{"thought": "First step", "subtasks": [{"thought": "Subtask 1"}], "conclusion": "Done"}], "answer": "42"}'
        assert compact_json_string(pretty) == expected

    def test_tabs_and_mixed_whitespace(self):
        pretty = '{\t\n  "key":\t"value"\n}'
        expected = '{"key": "value"}'
        assert compact_json_string(pretty) == expected

    def test_multiple_key_values(self):
        pretty = '''{
  "a": 1,
  "b": 2,
  "c": 3
}'''
        expected = '{"a": 1, "b": 2, "c": 3}'
        assert compact_json_string(pretty) == expected


class TestCompactJsonResponse:
    """Tests for compact_json_response function."""

    def test_empty_response(self):
        result, orig_len, new_len = compact_json_response("")
        assert result == ""
        assert orig_len == 0
        assert new_len == 0

    def test_json_only_response(self):
        pretty = '{\n  "key": "value"\n}'
        result, orig_len, new_len = compact_json_response(pretty)
        assert result == '{"key": "value"}'
        assert orig_len > new_len

    def test_response_with_prefix(self):
        response = 'Here is the answer:\n{\n  "key": "value"\n}'
        result, orig_len, new_len = compact_json_response(response)
        assert result == 'Here is the answer:\n{"key": "value"}'
        assert orig_len > new_len

    def test_response_with_marker(self):
        response = '<think>reasoning</think>\n{\n  "key": "value"\n}'
        result, _, _ = compact_json_response(response, json_start_marker="</think>")
        # The newline after marker is part of the JSON section and gets compacted
        assert result == '<think>reasoning</think>{"key": "value"}'

    def test_no_json_found(self):
        response = "This is just plain text without JSON"
        result, orig_len, new_len = compact_json_response(response)
        assert result == response
        assert orig_len == new_len

    def test_array_json_response(self):
        response = 'Result:\n[\n  1,\n  2\n]'
        result, _, _ = compact_json_response(response)
        assert result == 'Result:\n[1, 2]'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
