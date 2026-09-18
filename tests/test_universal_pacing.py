import pytest
from app.services.script_engine import ScriptEngine
from app.core.config import SUPPORTED_LANGUAGES

def test_universal_language_pacing_matrix():
    """
    Verifies that all supported languages have empirical speech rates
    and that Urdu is realistically calibrated to ~132 WPM to prevent 14+ min overshoots.
    """
    assert hasattr(ScriptEngine, "LANGUAGE_WPM")
    for lang_code in SUPPORTED_LANGUAGES.keys():
        assert lang_code in ScriptEngine.LANGUAGE_WPM, f"Missing WPM for {lang_code}"
    
    # Urdu empirical calibration (Edge-TTS ur-PK-AsadNeural is ~132 WPM)
    assert 128 <= ScriptEngine.LANGUAGE_WPM["ur"] <= 136, f"Urdu WPM not calibrated: {ScriptEngine.LANGUAGE_WPM['ur']}"
    
    # Hindi empirical calibration (~138 WPM)
    assert 132 <= ScriptEngine.LANGUAGE_WPM["hi"] <= 142, f"Hindi WPM not calibrated: {ScriptEngine.LANGUAGE_WPM['hi']}"
    
    # English empirical calibration (150 WPM)
    assert ScriptEngine.LANGUAGE_WPM["en"] == 150
    
    # Spanish empirical calibration (165 WPM)
    assert 160 <= ScriptEngine.LANGUAGE_WPM["es"] <= 170

    # 10 minute Urdu at normal speed: 10 * 132 = 1320 words
    ur_10_norm = ScriptEngine.calculate_target_words(10, voice_speed="normal", target_lang="ur")
    assert 1280 <= ur_10_norm <= 1360, f"Expected ~1320 words for 10m Urdu normal, got {ur_10_norm}"

    # 10 minute Urdu at fast speed (1.15x): 10 * 132 * 1.15 = 1518 words
    ur_10_fast = ScriptEngine.calculate_target_words(10, voice_speed="fast", target_lang="ur")
    assert 1480 <= ur_10_fast <= 1560, f"Expected ~1518 words for 10m Urdu fast, got {ur_10_fast}"


def test_universal_timeline_partitioner():
    """
    Verifies that partition_timeline accurately divides any movie runtime (e.g. 84 mins)
    into 5 cinematic acts covering from 0% to at least 95% of total source duration.
    """
    source_sec = 84 * 60  # 5040 seconds (84 minutes)
    milestones = ScriptEngine.partition_timeline(source_duration_sec=source_sec, target_duration_mins=10)
    
    assert len(milestones) == 5, f"Expected 5 acts, got {len(milestones)}"
    assert milestones[0]["act"] == "Act 1"
    assert milestones[0]["start_sec"] == 0
    
    # Check Epilogue covers up to source video ending
    epilogue = milestones[-1]
    assert epilogue["act"] == "Epilogue"
    assert epilogue["end_sec"] >= source_sec * 0.98, f"Timeline did not reach movie end: {epilogue['end_sec']} vs {source_sec}"
    assert "timestamp_range" in epilogue
    assert "%" in epilogue["budget_pct"]


def test_non_destructive_clamp_preserves_climax_and_ending():
    """
    Verifies that clamp_script_word_budget NEVER severs the final climax/ending scenes,
    even when total words significantly exceed the target word budget.
    """
    scenes = []
    # Create 10 scenes, total ~2500 words
    for i in range(1, 10):
        scenes.append(
            f"[SCENE: {i*5:02d}:00 - {i*5+2:02d}:00]\n[VOICEOVER]\n"
            + f"Scene {i} events unfold in great detail with suspense and tension building progressively. " * 18
        )
    # The final scene is the Climax & Resolution at the end of the movie (81:00 - 84:00)
    scenes.append(
        "[SCENE: 81:00 - 84:00]\n[VOICEOVER]\n"
        "In the shocking final climax, the killer is finally unmasked. The dark mystery comes to an end, "
        "and justice is served. In conclusion, this movie delivers a masterclass in psychological dread. Subscribe for more!"
    )
    long_script = "\n\n".join(scenes)
    assert len(long_script.split()) > 2000

    # Clamp to 10 minutes Urdu (budget ~1518 words)
    clamped = ScriptEngine.clamp_script_word_budget(
        long_script,
        target_duration_mins=10,
        target_lang="ur",
        voice_speed="fast"
    )
    
    # 1. Budget constraint met
    clamped_words = len(clamped.split())
    assert clamped_words <= 1650, f"Script was not clamped, words: {clamped_words}"
    
    # 2. Opening Scene 1 is preserved
    assert "Scene 1" in clamped
    
    # 3. CRITICAL: Final Climax Scene (81:00 - 84:00) is 100% PRESERVED
    assert "81:00 - 84:00" in clamped, "FAIL: Climax was severed by clamp_script_word_budget!"
    assert "shocking final climax" in clamped, "FAIL: Ending narration was dropped!"


def test_validate_script_integrity():
    """
    Verifies the automated quality gatekeeper:
    - Rejects scripts that stop early (missing climax/ending).
    - Accepts scripts that cover >= 85% of source duration.
    """
    source_sec = 84 * 60  # 84 minutes = 5040 seconds

    # Case A: Truncated script (stopped at 58:10 -> 3490 seconds = 69% coverage)
    truncated_script = (
        "[SCENE: 01:00 - 03:00]\n[VOICEOVER]\nHook and opening.\n\n"
        "[SCENE: 25:00 - 27:00]\n[VOICEOVER]\nMiddle conflict.\n\n"
        "[SCENE: 56:40 - 58:10]\n[VOICEOVER]\nCharacters are investigating the cemetery."
    )
    is_valid, report = ScriptEngine.validate_script_integrity(
        script_text=truncated_script,
        source_duration_sec=source_sec,
        target_duration_mins=10,
        target_lang="ur"
    )
    assert not is_valid, "Truncated script should fail validation!"
    assert "coverage" in report.lower()

    # Case B: Complete script covering 83:30 (covers 99.4%) with full word budget
    complete_script = (
        "[SCENE: 01:00 - 03:00]\n[VOICEOVER]\n" + "Hook and opening setup with dramatic narration and suspense. " * 30 + "\n\n"
        "[SCENE: 45:00 - 47:00]\n[VOICEOVER]\n" + "Midpoint turning point with shocking revelation and conflict. " * 30 + "\n\n"
        "[SCENE: 81:00 - 83:30]\n[VOICEOVER]\n" + "Final climax and conclusion where justice is served and truth prevails. " * 30
    )
    is_valid, report = ScriptEngine.validate_script_integrity(
        script_text=complete_script,
        source_duration_sec=source_sec,
        target_duration_mins=10,
        target_lang="ur"
    )
    assert is_valid, f"Complete script should pass validation, got errors: {report}"
