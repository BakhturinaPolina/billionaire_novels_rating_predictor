"""Test suite for category mapping inference function."""

import pytest

# Try package import first, then local file as fallback
try:
    from src.stage06_labeling.category_mapping.map_topics_to_categories import (
        infer_categories,
    )
except ImportError:  # local dev / notebook
    from map_topics_to_categories import infer_categories


def _cats(label, keywords=None):
    """Helper to get categories for a label."""
    return infer_categories(label, keywords or [])


def _weight(cats, cat):
    """Helper to get weight for a category."""
    return float(cats.get(cat, 0.0))


def _has_only(cats, *expected):
    """Helper to check if categories contain only the expected ones."""
    nonzero = {c for c, w in cats.items() if w > 1e-8}
    return nonzero == set(expected)


def test_weights_sum_to_one():
    """Test that category weights always sum to 1.0."""
    cats = _cats("True Promise")
    total = sum(cats.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"


def test_commitment_and_reunion():
    """Test that reunion topics map to both A and N categories."""
    cats = _cats("Long-Awaited Reunion")
    assert "A_commitment_hea" in cats
    assert "N_separation_reunion" in cats
    assert abs(sum(cats.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("label", [
    "Ice Hockey Players",
    "Guacamole Dilemma",
    "Animal Encounter",
    "Purring Rescue Kittens",
])
def test_noise_topics_are_Z(label):
    """Test that noise topics (sports, animals, food) map to Z_noise_oog."""
    cats = _cats(label)
    assert _has_only(cats, "Z_noise_oog"), f"Expected only Z_noise_oog, got {cats}"


def test_humor_backoff():
    """Test that humor topics map to I_humor_lightness."""
    cats = _cats("Playful Banter")
    assert _has_only(cats, "I_humor_lightness"), f"Expected only I_humor_lightness, got {cats}"


def test_scene_anchor_backoff():
    """Test that scene anchor topics map to S_scene_anchor."""
    cats = _cats("Elevator Encounter")
    assert _has_only(cats, "S_scene_anchor"), f"Expected only S_scene_anchor, got {cats}"


def test_explicit_sex_detected_as_C():
    """Test that explicit sex topics are detected as C_explicit."""
    # These are currently mis-mapped as Z in your code
    labels = [
        ("Moist Mouth Play", ["tongue", "mouth", "wet", "kissing"]),
        ("Hesitant Intercourse", ["sex", "hesitant", "bedroom"]),
    ]
    
    for label, kw in labels:
        cats = _cats(label, kw)
        assert _weight(cats, "C_explicit") > 0.0, \
            f"Expected C_explicit > 0 for '{label}', got {cats}"
        assert _weight(cats, "Z_noise_oog") == 0.0, \
            f"Expected Z_noise_oog == 0 for '{label}', got {cats}"


def test_apology_vs_miscommunication():
    """Test that apology topics map to T_repair_apology, not Q_miscommunication."""
    cats = _cats("Shameless Apologies")
    # This topic should be repair-focused, not miscommunication-focused
    assert _weight(cats, "T_repair_apology") > 0.0, \
        f"Expected T_repair_apology > 0, got {cats}"
    assert _weight(cats, "Q_miscommunication") == 0.0, \
        f"Expected Q_miscommunication == 0, got {cats}"


def test_protectiveness_vs_jealousy():
    """Test that protectiveness and jealousy map to different R categories."""
    protect = _cats("Protective Bodyguard")
    jealous = _cats("Jealous Fit")
    
    assert _weight(protect, "R1_protectiveness") > 0.0, \
        f"Expected R1_protectiveness > 0 for 'Protective Bodyguard', got {protect}"
    assert _weight(protect, "R2_jealousy") == 0.0, \
        f"Expected R2_jealousy == 0 for 'Protective Bodyguard', got {protect}"
    
    assert _weight(jealous, "R2_jealousy") > 0.0, \
        f"Expected R2_jealousy > 0 for 'Jealous Fit', got {jealous}"
    # jealousy can co-occur with negative affect, but it should not be pure noise
    assert _weight(jealous, "Z_noise_oog") == 0.0, \
        f"Expected Z_noise_oog == 0 for 'Jealous Fit', got {jealous}"


def test_explicit_with_keywords():
    """Test that explicit content is detected even when keywords are needed."""
    # Test that keywords help disambiguate
    cats = _cats("Moist Mouth Play", ["tongue", "mouth", "wet", "kissing", "oral"])
    assert _weight(cats, "C_explicit") > 0.0, \
        f"Expected C_explicit > 0 with keywords, got {cats}"


def test_tender_intimacy():
    """Test that tender intimacy maps to B_mutual_intimacy."""
    cats = _cats("Tender Tongue Play", ["tongue", "kiss", "tender", "soft"])
    assert _weight(cats, "B_mutual_intimacy") > 0.0, \
        f"Expected B_mutual_intimacy > 0, got {cats}"


def test_wedding_planning():
    """Test that wedding topics map to A_commitment_hea."""
    cats = _cats("Wedding Planning", ["wedding", "ceremony", "planner"])
    assert _weight(cats, "A_commitment_hea") > 0.0, \
        f"Expected A_commitment_hea > 0, got {cats}"


def test_wrong_thoughts_miscommunication():
    """Test that 'Wrong Thoughts' maps to Q_miscommunication."""
    cats = _cats("Wrong Thoughts", ["wrong", "thought", "doubt", "misunderstand"])
    assert _weight(cats, "Q_miscommunication") > 0.0, \
        f"Expected Q_miscommunication > 0, got {cats}"

