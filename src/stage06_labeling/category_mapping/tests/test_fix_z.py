"""Tests for fix Z functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.stage06_labeling.category_mapping.fix_z_topics import (
    convert_llm_categories_to_weights,
    identify_z_topics,
)


def test_identify_z_topics():
    """Test identification of Z topics."""
    # Sample category mappings
    category_probs = {
        "0": {"categories": {"A_commitment_hea": 1.0}},
        "1": {"categories": {"Z_noise_oog": 1.0}},  # Z only
        "2": {"categories": {"C_explicit": 0.7, "B_mutual_intimacy": 0.3}},
        "3": {"categories": {"Z_noise_oog": 1.0}},  # Z only
        "4": {"categories": {"Z_noise_oog": 0.5, "A_commitment_hea": 0.5}},  # Not Z only
    }
    
    z_topics = identify_z_topics(category_probs)
    
    # Should identify topics 1 and 3 (Z only)
    assert 1 in z_topics
    assert 3 in z_topics
    assert 0 not in z_topics
    assert 2 not in z_topics
    assert 4 not in z_topics  # Has multiple categories


def test_identify_z_topics_old_format():
    """Test identification with old format (direct dict)."""
    # Old format: direct category dict
    category_probs = {
        "0": {"A_commitment_hea": 1.0},
        "1": {"Z_noise_oog": 1.0},  # Z only
        "2": {"C_explicit": 0.7, "B_mutual_intimacy": 0.3},
    }
    
    z_topics = identify_z_topics(category_probs)
    
    assert 1 in z_topics
    assert 0 not in z_topics
    assert 2 not in z_topics


def test_convert_llm_categories_to_weights_single():
    """Test conversion with single category."""
    weights = convert_llm_categories_to_weights(["A_commitment_hea"])
    
    assert weights == {"A_commitment_hea": 1.0}
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_convert_llm_categories_to_weights_two():
    """Test conversion with two categories."""
    weights = convert_llm_categories_to_weights(
        ["C_explicit"],
        ["B_mutual_intimacy"]
    )
    
    assert "C_explicit" in weights
    assert "B_mutual_intimacy" in weights
    assert weights["C_explicit"] == 0.67
    assert weights["B_mutual_intimacy"] == 0.33
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_convert_llm_categories_to_weights_three():
    """Test conversion with three categories."""
    weights = convert_llm_categories_to_weights(
        ["A_commitment_hea", "N_separation_reunion"],
        ["G_rituals_gifts"]
    )
    
    assert len(weights) == 3
    assert weights["A_commitment_hea"] == 0.5
    assert weights["N_separation_reunion"] == 0.3
    assert weights["G_rituals_gifts"] == 0.2
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_convert_llm_categories_to_weights_empty():
    """Test conversion with empty categories."""
    weights = convert_llm_categories_to_weights([])
    
    assert weights == {"Z_noise_oog": 1.0}


def test_convert_llm_categories_to_weights_no_secondary():
    """Test conversion without secondary categories."""
    weights = convert_llm_categories_to_weights(["C_explicit"])
    
    assert weights == {"C_explicit": 1.0}


@patch('src.stage06_labeling.category_mapping.fix_z_topics.OpenAI')
def test_fix_topic_with_llm_mock(mock_openai):
    """Test fix_topic_with_llm with mocked API."""
    from src.stage06_labeling.category_mapping.fix_z_topics import fix_topic_with_llm
    
    # Mock API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "topic_id": 37,
        "label": "Wet Oral Foreplay",
        "primary_categories": ["C_explicit"],
        "secondary_categories": ["B_mutual_intimacy"],
        "is_noise": False,
        "rationale": "The focus is on erotic oral play and intimate kissing."
    })
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 150
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Call function
    result = fix_topic_with_llm(
        topic_id=37,
        label="Moist Mouth Play",
        keywords=["tongue", "mouth", "wet", "kissing"],
        client=mock_client,
        model_name="mistralai/mistral-nemo",
        temperature=0.3,
    )
    
    # Verify result
    assert result["topic_id"] == 37
    assert result["label"] == "Wet Oral Foreplay"
    assert result["primary_categories"] == ["C_explicit"]
    assert result["is_noise"] is False
    assert "api_time" in result
    assert "api_tokens" in result


@patch('src.stage06_labeling.category_mapping.fix_z_topics.OpenAI')
def test_fix_topic_with_llm_json_in_markdown(mock_openai):
    """Test parsing JSON from markdown code blocks."""
    from src.stage06_labeling.category_mapping.fix_z_topics import fix_topic_with_llm
    
    # Mock API response with markdown
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """Here's the classification:

```json
{
  "topic_id": 8,
  "label": "Misunderstood Feelings",
  "primary_categories": ["Q_miscommunication"],
  "secondary_categories": ["F_negative_affect"],
  "is_noise": false,
  "rationale": "The focus is on misunderstandings."
}
```"""
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 200
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    result = fix_topic_with_llm(
        topic_id=8,
        label="Wrong Thoughts",
        keywords=["wrong", "thought", "doubt"],
        client=mock_client,
        model_name="mistralai/mistral-nemo",
    )
    
    assert result["topic_id"] == 8
    assert result["label"] == "Misunderstood Feelings"
    assert result["primary_categories"] == ["Q_miscommunication"]
    assert result["is_noise"] is False

