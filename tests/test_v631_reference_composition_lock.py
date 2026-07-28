from app.mockup_generator import MockupSpec, PhotoDirection, build_model_photo_prompt


def _cap_spec() -> MockupSpec:
    return MockupSpec(
        side="front",
        garment_type="cap",
        shirt_color="black washed",
        fabric_finish="washed cotton",
        fit="soft unstructured baseball cap",
        print_width_percent=42,
        print_height_percent=18,
        print_top_offset_percent=35,
        print_left_offset_percent=29,
        print_center_x_percent=50,
        construction_details="six panel cap with curved brim",
        target_gender="unisex",
        target_age_group="adult-universal",
        moods=["street"],
        print_theme="text",
        geometry_mode="source-guided",
    )


def _direction() -> PhotoDirection:
    return PhotoDirection(
        label="reference cap",
        gender="women",
        pose_kind="close-up",
        person="fictional adult",
        setting="ordinary indoor room",
        pose="head lowered",
        camera="close phone photo",
        framing="close-up",
        light="soft indoor light",
        seed=123,
    )


def test_manual_cap_reference_locks_hidden_face_and_crop():
    prompt = build_model_photo_prompt(
        _cap_spec(),
        _direction(),
        "ref-lock",
        has_style_reference=True,
        style_reference_tags={
            "face_visibility": "hidden",
            "head_direction": "down",
            "shot_character": "candid-phone",
            "face_occlusion": "cap brim covers the entire face",
            "composition_notes": "cap dominates frame; head tilted down",
        },
    )
    assert "FACE VISIBILITY IS LOCKED" in prompt
    assert "If a cap brim" in prompt
    assert "The face may remain completely hidden" in prompt
    assert "Do not turn the result into a face-forward portrait" in prompt
    assert "match the reference crop and asymmetric placement exactly" in prompt.lower()
    assert "clean flawless skin" not in prompt
    assert "crisp 35 mm portrait lens" not in prompt
    assert "premium streetwear lookbook quality" not in prompt
    assert "must keep the entire head" not in prompt.lower()
    assert "CAP REFERENCE COMPOSITION LOCK - HIGHEST PRIORITY" in prompt
    assert "Only the cap and the immediate contact area around it may change" in prompt
    assert "Remove every trace of the reference cap's original text" in prompt
    assert "Do not redesign the person, pose or background" in prompt


def test_no_reference_keeps_general_product_photo_direction():
    prompt = build_model_photo_prompt(
        _cap_spec(),
        _direction(),
        "normal",
        has_style_reference=False,
    )
    assert "EVERYDAY CAP PHOTO REALISM" in prompt
    assert "CAP COMPOSITION WITHOUT STYLE REFERENCE" in prompt
    assert "Keep the cap and print comfortably inside the 4:5 frame" in prompt
