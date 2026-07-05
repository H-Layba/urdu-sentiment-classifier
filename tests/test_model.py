from transformers import pipeline

MODEL_ID = "H-Layba/urdu-sentiment-classifier"

def get_classifier():
    return pipeline("text-classification", model=MODEL_ID)

def test_positive_sentence():
    """Clearly positive Urdu sentence should return positive."""
    clf = get_classifier()
    result = clf("یہ فلم بہت اچھی تھی")[0]
    assert result["label"] == "positive", f"Expected positive, got {result['label']}"
    assert result["score"] > 0.8, f"Confidence too low: {result['score']}"

def test_negative_sentence():
    """Clearly negative Urdu sentence should return negative."""
    clf = get_classifier()
    result = clf("آج کا دن بہت برا تھا")[0]
    assert result["label"] == "negative", f"Expected negative, got {result['label']}"
    assert result["score"] > 0.8, f"Confidence too low: {result['score']}"

def test_positive_food_review():
    """Positive food review should return positive."""
    clf = get_classifier()
    result = clf("کھانا لاجواب تھا")[0]
    assert result["label"] == "positive", f"Expected positive, got {result['label']}"

def test_negative_service_review():
    """Negative service review should return negative."""
    clf = get_classifier()
    result = clf("خدمت بہت خراب تھی")[0]
    assert result["label"] == "negative", f"Expected negative, got {result['label']}"

def test_high_confidence_on_clear_cases():
    """Model should be confident (>85%) on clearly polar sentences."""
    clf = get_classifier()
    results = [
        clf("یہ بہترین تجربہ تھا")[0],   # This was the best experience
        clf("بہت مایوسی ہوئی")[0],         # Very disappointed
    ]
    for r in results:
        assert r["score"] > 0.85, f"Confidence too low: {r['score']} for label {r['label']}"
