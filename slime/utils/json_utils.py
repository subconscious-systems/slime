"""Utilities for JSON string processing, including compaction of pretty-printed JSON."""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def compact_json_string(pretty_json: str) -> str:
    """Convert pretty-printed JSON to single-line JSON.
    
    Handles incomplete/partial JSON strings gracefully by processing character-by-character.
    Removes newlines and indentation while preserving:
    - Whitespace inside string literals
    - Escape sequences
    - Single space after colons and commas (for readability)
    
    Args:
        pretty_json: A JSON string that may contain newlines and indentation.
                    Can be incomplete/partial JSON.
    
    Returns:
        Single-line JSON string with spaces after colons and commas.
    
    Examples:
        >>> compact_json_string('{"key": "value"}')
        '{"key": "value"}'
        >>> compact_json_string('{\\n  "key": "value"\\n}')
        '{"key": "value"}'
        >>> compact_json_string('{"key": "value with  spaces"}')
        '{"key": "value with  spaces"}'
        >>> compact_json_string('{"incomplete": "json')  # Partial JSON
        '{"incomplete": "json'
    """
    if not pretty_json:
        return pretty_json
    
    result = []
    i = 0
    n = len(pretty_json)
    in_string = False
    need_space = False  # Track if we need to add a space before the next non-whitespace char
    
    while i < n:
        char = pretty_json[i]
        
        if in_string:
            # Inside a string literal - preserve everything
            result.append(char)
            if char == '\\' and i + 1 < n:
                # Escape sequence - include next char as-is
                i += 1
                result.append(pretty_json[i])
            elif char == '"':
                # End of string
                in_string = False
        else:
            # Outside string literal
            if char == '"':
                # Start of string - add pending space if needed
                if need_space:
                    result.append(' ')
                    need_space = False
                in_string = True
                result.append(char)
            elif char in ' \t\n\r':
                # Skip whitespace outside strings
                # Space will be added via need_space flag when we see next char
                pass
            elif char in ':,':
                # Colon and comma - output them and mark that we need space after
                result.append(char)
                need_space = True
            elif char in '}]':
                # Closing brackets - no space needed before them
                need_space = False
                result.append(char)
            else:
                # Other characters (digits, true, false, null, opening brackets)
                if need_space:
                    result.append(' ')
                    need_space = False
                result.append(char)
        
        i += 1
    
    return ''.join(result)


def compact_json_response(response: str, json_start_marker: str = None) -> Tuple[str, int, int]:
    """Compact JSON content within a response string.
    
    If json_start_marker is provided, only compact content after that marker.
    Otherwise, attempts to find JSON content (starting with '{' or '[').
    
    Args:
        response: The full response string that may contain JSON.
        json_start_marker: Optional marker indicating where JSON starts.
    
    Returns:
        Tuple of (compacted_response, original_length, new_length)
    """
    if not response:
        return response, 0, 0
    
    original_length = len(response)
    
    if json_start_marker and json_start_marker in response:
        # Split at marker and compact only the JSON part
        marker_idx = response.index(json_start_marker)
        prefix = response[:marker_idx + len(json_start_marker)]
        json_part = response[marker_idx + len(json_start_marker):]
        compacted_json = compact_json_string(json_part)
        result = prefix + compacted_json
    else:
        # Try to find JSON start
        json_start = -1
        for i, char in enumerate(response):
            if char in '{[':
                json_start = i
                break
        
        if json_start >= 0:
            prefix = response[:json_start]
            json_part = response[json_start:]
            compacted_json = compact_json_string(json_part)
            result = prefix + compacted_json
        else:
            # No JSON found, return as-is
            result = response
    
    return result, original_length, len(result)


def recompute_tokens_after_compaction(
    tokenizer,
    original_tokens: list[int],
    original_response: str,
    compacted_response: str,
    response_length: int,
) -> Tuple[list[int], int, list[float] | None]:
    """Recompute tokens after JSON compaction.
    
    Args:
        tokenizer: The tokenizer to use for re-tokenization.
        original_tokens: Original token ids (prompt + response).
        original_response: Original response string.
        compacted_response: Compacted response string.
        response_length: Original response length in tokens.
    
    Returns:
        Tuple of (new_tokens, new_response_length, None)
        Note: rollout_log_probs cannot be reused after re-tokenization.
    """
    # Extract prompt tokens (everything except response)
    prompt_length = len(original_tokens) - response_length
    prompt_tokens = original_tokens[:prompt_length]
    
    # Tokenize the compacted response
    new_response_tokens = tokenizer.encode(compacted_response, add_special_tokens=False)
    
    # Combine prompt + new response tokens
    new_tokens = prompt_tokens + new_response_tokens
    new_response_length = len(new_response_tokens)
    
    return new_tokens, new_response_length, None


def _get_sample_reward_value(sample) -> float | None:
    """Extract the numeric reward value from a sample.
    
    Args:
        sample: A Sample object with reward attribute.
    
    Returns:
        The numeric reward value, or None if not available.
    """
    if sample.reward is None:
        return None
    
    if isinstance(sample.reward, (int, float)):
        return float(sample.reward)
    
    if isinstance(sample.reward, dict):
        # Try common keys for reward dictionaries
        for key in ['reward', 'score', 'value']:
            if key in sample.reward:
                val = sample.reward[key]
                if isinstance(val, (int, float)):
                    return float(val)
        # If no known key, try the first numeric value
        for val in sample.reward.values():
            if isinstance(val, (int, float)):
                return float(val)
    
    return None


def process_sample_json_compaction(
    sample,
    tokenizer,
    json_start_marker: str = None,
    only_positive_reward: bool = False,
    store_original_for_old_log_prob: bool = False,
):
    """Process a single sample to compact its JSON response.
    
    Modifies the sample in-place:
    - Updates sample.response with compacted JSON
    - Updates sample.tokens with re-tokenized content  
    - Updates sample.response_length
    - Sets sample.rollout_log_probs to None (cannot be reused after re-tokenization)
    - If store_original_for_old_log_prob=True, stores original tokens in sample.original_tokens
      and original response_length in sample.original_response_length for old_log_prob computation
    
    Args:
        sample: A Sample object with response, tokens, and response_length.
        tokenizer: The tokenizer for re-tokenization.
        json_start_marker: Optional marker indicating where JSON starts.
        only_positive_reward: If True, only compact samples with positive reward.
        store_original_for_old_log_prob: If True, store original tokens for old_log_prob computation.
    
    Returns:
        The modified sample (also modified in-place).
    """
    if not sample.response:
        return sample
    
    # Check reward condition if enabled
    if only_positive_reward:
        reward_value = _get_sample_reward_value(sample)
        if reward_value is None or reward_value <= 0:
            # Skip compaction for non-positive reward samples
            return sample
    
    # Compact the JSON
    compacted_response, orig_len, new_len = compact_json_response(
        sample.response, json_start_marker
    )
    
    # Skip if no change
    if compacted_response == sample.response:
        return sample
    
    # Store original tokens for old_log_prob computation if requested
    if store_original_for_old_log_prob:
        sample.original_tokens = sample.tokens.copy()
        sample.original_response_length = sample.response_length
        sample.original_response = sample.response
    
    # Recompute tokens
    new_tokens, new_response_length, _ = recompute_tokens_after_compaction(
        tokenizer=tokenizer,
        original_tokens=sample.tokens,
        original_response=sample.response,
        compacted_response=compacted_response,
        response_length=sample.response_length,
    )
    
    # Update sample
    sample.response = compacted_response
    sample.tokens = new_tokens
    sample.response_length = new_response_length
    # rollout_log_probs are no longer valid after re-tokenization
    sample.rollout_log_probs = None
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"JSON compaction: response {orig_len} -> {new_len} chars, "
            f"tokens {len(sample.tokens)} -> {len(new_tokens)}"
        )
    
    return sample


def process_samples_json_compaction(
    samples: list,
    tokenizer,
    json_start_marker: str = None,
    only_positive_reward: bool = False,
    store_original_for_old_log_prob: bool = False,
) -> list:
    """Process multiple samples to compact their JSON responses.
    
    Args:
        samples: List of Sample objects (can be flat list or nested list of groups).
        tokenizer: The tokenizer for re-tokenization.
        json_start_marker: Optional marker indicating where JSON starts.
        only_positive_reward: If True, only compact samples with positive reward.
        store_original_for_old_log_prob: If True, store original tokens for old_log_prob computation.
    
    Returns:
        The same samples list (modified in-place).
    """
    # Handle nested structure (groups of samples)
    if samples and isinstance(samples[0], list):
        for group in samples:
            for sample in group:
                if isinstance(sample, list):
                    for s in sample:
                        process_sample_json_compaction(
                            s, tokenizer, json_start_marker, only_positive_reward, store_original_for_old_log_prob
                        )
                else:
                    process_sample_json_compaction(
                        sample, tokenizer, json_start_marker, only_positive_reward, store_original_for_old_log_prob
                    )
    else:
        for sample in samples:
            process_sample_json_compaction(
                sample, tokenizer, json_start_marker, only_positive_reward, store_original_for_old_log_prob
            )
    
    return samples
