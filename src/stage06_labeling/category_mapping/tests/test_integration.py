"""Integration tests for category mapping pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.stage06_labeling.category_mapping.map_topics_to_categories import (
    infer_categories,
    load_labels,
    save_json,
)


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return {
        "0": {"label": "True Promise", "keywords": ["means", "idea", "promise", "work"]},
        "1": {"label": "Tongue & Nipple Play", "keywords": ["hips", "tongue", "breasts", "nipples"]},
        "2": {"label": "Ice Hockey Players", "keywords": ["game", "hockey", "puck", "players"]},
        "3": {"label": "Moist Mouth Play", "keywords": ["tongue", "mouth", "wet", "kissing"]},
        "4": {"label": "Long-Awaited Reunion", "keywords": ["way", "years", "reunion", "relationship"]},
    }


@pytest.fixture
def temp_labels_file(sample_labels):
    """Create a temporary labels JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_labels, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_end_to_end_category_mapping(temp_labels_file, temp_output_dir):
    """Test end-to-end category mapping workflow."""
    # Load labels
    labels = load_labels(temp_labels_file)
    assert len(labels) == 5
    
    # Map to categories
    topic_to_cat = {}
    for tid, rec in labels.items():
        label = rec.get("label", "")
        keywords = rec.get("keywords", [])
        cats = infer_categories(label, keywords=keywords)
        topic_to_cat[tid] = {
            "categories": cats,
            "label": label,
            "keywords": keywords
        }
    
    # Verify all topics have categories
    assert len(topic_to_cat) == 5
    for tid, data in topic_to_cat.items():
        assert "categories" in data
        assert len(data["categories"]) > 0
        # Verify weights sum to 1.0
        total = sum(data["categories"].values())
        assert abs(total - 1.0) < 1e-6, f"Topic {tid} weights sum to {total}"
    
    # Save to JSON
    output_file = temp_output_dir / "topic_to_category_probs.json"
    save_json(topic_to_cat, output_file)
    
    # Verify file was created
    assert output_file.exists()
    
    # Load and verify
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    assert len(loaded) == 5
    assert loaded["0"]["categories"] == topic_to_cat["0"]["categories"]


def test_z_topic_identification(temp_labels_file, temp_output_dir):
    """Test that Z topics are correctly identified."""
    # Load labels
    labels = load_labels(temp_labels_file)
    
    # Map to categories
    topic_to_cat = {}
    for tid, rec in labels.items():
        label = rec.get("label", "")
        keywords = rec.get("keywords", [])
        cats = infer_categories(label, keywords=keywords)
        topic_to_cat[tid] = {
            "categories": cats,
            "label": label,
            "keywords": keywords
        }
    
    # Identify Z topics
    z_topics = []
    for tid, data in topic_to_cat.items():
        cats = data["categories"]
        if len(cats) == 1 and "Z_noise_oog" in cats:
            z_topics.append(tid)
    
    # "Ice Hockey Players" should be Z
    assert "2" in z_topics, "Ice Hockey Players should be classified as Z_noise_oog"
    
    # Other topics should not be Z
    assert "0" not in z_topics, "True Promise should not be Z"
    assert "1" not in z_topics, "Tongue & Nipple Play should not be Z"
    assert "3" not in z_topics, "Moist Mouth Play should not be Z"
    assert "4" not in z_topics, "Long-Awaited Reunion should not be Z"


def test_category_mapping_output_format(temp_labels_file, temp_output_dir):
    """Test that output format matches expected structure."""
    # Load and map
    labels = load_labels(temp_labels_file)
    topic_to_cat = {}
    for tid, rec in labels.items():
        label = rec.get("label", "")
        keywords = rec.get("keywords", [])
        cats = infer_categories(label, keywords=keywords)
        topic_to_cat[tid] = {
            "categories": cats,
            "label": label,
            "keywords": keywords
        }
    
    # Save
    output_file = temp_output_dir / "topic_to_category_probs.json"
    save_json(topic_to_cat, output_file)
    
    # Verify structure
    with open(output_file, 'r') as f:
        loaded = json.load(f)
    
    for tid, data in loaded.items():
        # Should have required fields
        assert "categories" in data
        assert "label" in data
        assert "keywords" in data
        
        # Categories should be a dict
        assert isinstance(data["categories"], dict)
        
        # All category values should be numbers
        for cat, weight in data["categories"].items():
            assert isinstance(weight, (int, float))
            assert 0.0 <= weight <= 1.0


def test_explicit_content_detection(sample_labels):
    """Test that explicit content is correctly detected."""
    # Test "Moist Mouth Play" with keywords
    label = sample_labels["3"]["label"]
    keywords = sample_labels["3"]["keywords"]
    cats = infer_categories(label, keywords=keywords)
    
    # Should map to C_explicit
    assert "C_explicit" in cats, f"Expected C_explicit, got {cats}"
    assert cats.get("C_explicit", 0.0) > 0.0
    assert "Z_noise_oog" not in cats or cats.get("Z_noise_oog", 0.0) == 0.0


def test_commitment_and_reunion_detection(sample_labels):
    """Test that commitment and reunion topics are correctly detected."""
    label = sample_labels["4"]["label"]
    keywords = sample_labels["4"]["keywords"]
    cats = infer_categories(label, keywords=keywords)
    
    # Should map to both A and N
    assert "A_commitment_hea" in cats
    assert "N_separation_reunion" in cats

