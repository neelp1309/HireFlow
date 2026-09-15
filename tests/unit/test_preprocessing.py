from hireflow.preprocessing import RESUME_HEADINGS, extract_bullets, normalize_pdf_text, split_sections


def test_normalize_removes_synthetic_achievement_placeholder() -> None:
    text = "ACHIEVEMENTS\n{generate_achievements(experience_years)}\nREFERENCES"
    cleaned = normalize_pdf_text(text)
    assert "generate_achievements" not in cleaned


def test_del_character_becomes_bullet_and_items_extract() -> None:
    cleaned = normalize_pdf_text("TECHNICAL SKILLS\x7f Python\x7f Financial Reporting")
    sections = split_sections(cleaned, RESUME_HEADINGS)
    assert extract_bullets(sections["TECHNICAL SKILLS"]) == ["Python", "Financial Reporting"]
