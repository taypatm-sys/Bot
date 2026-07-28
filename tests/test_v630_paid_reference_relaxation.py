from app.reference_catalog import _is_soft_generative_rejection


def test_existing_print_is_soft_for_paid_generation() -> None:
    assert _is_soft_generative_rejection(
        "На футболке уже есть крупный принт, который занимает более 10% площади"
    )


def test_chain_is_soft_for_paid_generation() -> None:
    assert _is_soft_generative_rejection(
        "Наличие цепочки, перекрывающей область печати"
    )


def test_shoulder_touch_is_soft_for_paid_generation() -> None:
    assert _is_soft_generative_rejection(
        "Руки модели касаются плеч и верхней части футболки"
    )


def test_wrong_side_stays_hard() -> None:
    assert not _is_soft_generative_rejection(
        "Неправильная сторона: видна спина, а нужен перед"
    )
