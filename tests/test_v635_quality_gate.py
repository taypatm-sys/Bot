from app.mockup_generator import MockupQualityCheck


def _quality(**updates):
    values = {
        "overall_score": 92,
        "product_type_match": True,
        "print_side_match": True,
        "garment_color_score": 91,
        "garment_fit_score": 88,
        "garment_silhouette_score": 88,
        "print_fidelity_score": 93,
        "print_scale_position_score": 90,
        "reference_composition_score": 88,
        "old_reference_artwork_removed": True,
        "unrequested_scene_changes": False,
        "reason": "all locked fields match",
        "correction_instruction": "",
    }
    values.update(updates)
    return MockupQualityCheck(**values)


def test_quality_gate_accepts_precise_result():
    assert _quality().acceptable(minimum_score=82, has_separate_print=True)


def test_quality_gate_rejects_wrong_print_even_with_high_overall_score():
    check = _quality(print_fidelity_score=60, overall_score=95)
    assert not check.acceptable(minimum_score=82, has_separate_print=True)


def test_quality_gate_rejects_wrong_side():
    check = _quality(print_side_match=False)
    assert not check.acceptable(minimum_score=82, has_separate_print=False)
