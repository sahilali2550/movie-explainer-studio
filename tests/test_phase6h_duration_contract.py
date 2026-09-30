import os
import pytest
import asyncio
from typing import Optional, Dict, Any, List
from app.services.script_engine import ScriptEngine, SceneBlock

SAMPLE_STORYBOARD_WITH_TAGS = """[SCENE: 00:00 - 00:20]
[DIALOGUE_REF: "Give me the money for the car wash."]
[VOICEOVER]
A wealthy car owner stands aggressively beside his luxury vehicle. He accuses an impoverished driver of causing a minor scratch on the polished bumper. The driver desperately pleads his innocence.

[SCENE: 00:20 - 00:45]
[DIALOGUE_REF: "That is all I have."]
[VOICEOVER]
The driver empties his pockets in distress before the watching crowd. He hands over his last remaining small bills to satisfy the angry owner. He has nothing left for his family."""

# ---------------------------------------------------------------------------
# Phase 6H Legacy Preserved Tests
# ---------------------------------------------------------------------------

def test_spoken_word_extraction_and_markup_exclusion():
    """Verify get_spoken_word_count extracts only spoken text and completely excludes structural markup (Req M)."""
    raw_word_count = len(SAMPLE_STORYBOARD_WITH_TAGS.split())
    spoken_word_count = ScriptEngine.get_spoken_word_count(SAMPLE_STORYBOARD_WITH_TAGS)

    assert raw_word_count > spoken_word_count
    clean_narr, _, _ = ScriptEngine.parse_storyboard(SAMPLE_STORYBOARD_WITH_TAGS)
    assert "SCENE" not in clean_narr
    assert "DIALOGUE_REF" not in clean_narr
    assert "VOICEOVER" not in clean_narr
    assert spoken_word_count == len(clean_narr.split())
    assert spoken_word_count == 61

def test_target_word_calculation_and_calibration():
    """Verify target words calculation supports anti-copyright drift calibration and backward compatibility."""
    w_uncal_5m = ScriptEngine.calculate_target_words(5, voice_speed="fast", target_lang="en", anti_copyright_drift=False)
    assert w_uncal_5m == 862

    w_cal_5m = ScriptEngine.calculate_target_words(5, voice_speed="fast", target_lang="en", anti_copyright_drift=True)
    assert w_cal_5m == 880
    assert w_cal_5m > w_uncal_5m

    w_3m = ScriptEngine.calculate_target_words(3, voice_speed="fast", target_lang="en", anti_copyright_drift=True)
    assert w_3m == 528

def test_lower_bound_duration_validation_rejection():
    """Verify validate_script_length rejects materially under-length scripts as pre-TTS planning heuristic."""
    under_length_narration = " ".join(["Word"] * 598)
    under_length_script = f"""[SCENE: 00:00 - 11:30]
[DIALOGUE_REF: "A key line"]
[VOICEOVER]
{under_length_narration}"""

    is_valid, msg, details = ScriptEngine.validate_script_length(
        script_text=under_length_script,
        source_duration_sec=764.0,
        target_duration_mins=5,
        target_lang="en",
        voice_speed="fast",
        return_details=True
    )

    assert not is_valid
    assert details["error"] == "under_budget"
    assert details["spoken_words"] == 598
    assert details["min_allowed"] >= 689

def test_valid_near_target_script_acceptance():
    """Verify validate_script_length accepts scripts meeting the duration conformance threshold."""
    near_target_narration = " ".join(["Story"] * 750)
    near_target_script = f"""[SCENE: 00:00 - 10:00]
[DIALOGUE_REF: "Final revelation"]
[VOICEOVER]
{near_target_narration}"""

    is_valid, msg = ScriptEngine.validate_script_length(
        script_text=near_target_script,
        source_duration_sec=700.0,
        target_duration_mins=5,
        target_lang="en",
        voice_speed="fast"
    )
    assert is_valid
    assert "passed" in msg.lower()

def test_raw_vs_spoken_word_mismatch_in_clamping():
    """Verify clamp_script_word_budget does not clamp scripts whose spoken words are within budget."""
    scenes = []
    for i in range(10):
        scenes.append(f"""[SCENE: {i:02d}:00 - {i+1:02d}:00]
[DIALOGUE_REF: "Detailed dialogue reference sentence number {i} accurately transcribed from the source video"]
[VOICEOVER]
{" ".join([f"Narration{i}"] * 80)}.""")
    full_script = "\n\n".join(scenes)

    raw_words = len(full_script.split())
    spoken_words = ScriptEngine.get_spoken_word_count(full_script)
    assert spoken_words == 800
    assert raw_words > 950

    clamped = ScriptEngine.clamp_script_word_budget(
        full_script,
        target_duration_mins=5,
        target_lang="en",
        voice_speed="fast"
    )

    clamped_spoken = ScriptEngine.get_spoken_word_count(clamped)
    assert clamped_spoken == 800
    assert "[SCENE: 00:00 - 01:00]" in clamped
    assert '[DIALOGUE_REF: "Detailed dialogue reference sentence number 0 accurately transcribed from the source video"]' in clamped
    assert "[VOICEOVER]" in clamped

def test_expansion_prompt_preserves_storyboard_structure():
    """Verify expansion instructions require preserving [SCENE], [DIALOGUE_REF], and [VOICEOVER]."""
    prompt = ScriptEngine.build_expansion_prompt(
        current_script=SAMPLE_STORYBOARD_WITH_TAGS,
        actual_words=52,
        target_words=500,
        target_lang="en",
        duration_mins=3
    )

    assert "[SCENE" in prompt
    assert "[DIALOGUE_REF" in prompt
    assert "[VOICEOVER]" in prompt
    assert "PRESERVE" in prompt

# ---------------------------------------------------------------------------
# Phase 6H.2: TTS-Duration Feedback Loop Tests (Req A - M)
# ---------------------------------------------------------------------------

def test_duration_feedback_actual_tts_duration_authoritative(tmp_path):
    """Req A: Actual TTS duration is authoritative over word count."""
    async def _run():
        script = SAMPLE_STORYBOARD_WITH_TAGS

        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock audio")
            return True

        def mock_measure(path):
            return 422.64

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            target_lang="en",
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=0,
            synthesize_fn=mock_synth,
            measure_fn=mock_measure
        )

    res = asyncio.run(_run())
    assert res["initial_tts_duration"] == 422.64
    assert pytest.approx(res["initial_projected_final_duration"], 0.01) == 422.64 / 1.02
    assert res["duration_converged"] is False
    assert res["duration_error_seconds"] < -80.0

def test_duration_feedback_low_word_count_long_tts_is_long(tmp_path):
    """Req B: A script with low word count but long TTS duration is treated as LONG."""
    script = SAMPLE_STORYBOARD_WITH_TAGS
    assert ScriptEngine.get_spoken_word_count(script) < 100

    recorded_directions = []

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        def mock_llm(p):
            if "Compress" in p:
                recorded_directions.append("compress")
            elif "Expand" in p:
                recorded_directions.append("expand")
            return script

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            target_lang="en",
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=1,
            llm_correction_fn=mock_llm,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 450.0
        )

    res = asyncio.run(_run())
    assert "compress" in recorded_directions

def test_duration_feedback_high_word_count_short_tts_is_short(tmp_path):
    """Req C: A script with high word count but short TTS duration is treated as SHORT."""
    script = f"""[SCENE: 00:00 - 05:00]
[DIALOGUE_REF: "Fast dialogue"]
[VOICEOVER]
{" ".join(['Fast'] * 1000)}."""

    recorded_directions = []

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        def mock_llm(p):
            if "Expand" in p:
                recorded_directions.append("expand")
            elif "Compress" in p:
                recorded_directions.append("compress")
            return script

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            target_lang="en",
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=1,
            llm_correction_fn=mock_llm,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 150.0
        )

    res = asyncio.run(_run())
    assert "expand" in recorded_directions

def test_duration_feedback_projected_final_math(tmp_path):
    """Req D: Duration decision uses actual_tts_duration / 1.02."""
    script = SAMPLE_STORYBOARD_WITH_TAGS
    measured_tts = 306.0  # 306.0 / 1.02 = 300.0s!

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=5.0,
            max_corrections=0,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: measured_tts
        )

    res = asyncio.run(_run())
    assert pytest.approx(res["final_projected_final_duration"], 0.01) == 300.0
    assert pytest.approx(res["duration_error_seconds"], 0.01) == 0.0
    assert res["duration_converged"] is True

def test_duration_feedback_short_script_triggers_expansion_and_converges(tmp_path):
    """Req E: Short script triggers expansion and converges when corrected."""
    script_v1 = SAMPLE_STORYBOARD_WITH_TAGS
    expanded_script = f"""[SCENE: 00:00 - 00:20]
[DIALOGUE_REF: "Give me the money for the car wash."]
[VOICEOVER]
A wealthy car owner stands aggressively beside his luxury vehicle. {" ".join(['Detailed dramatic expansion.'] * 50)}

[SCENE: 00:20 - 00:45]
[DIALOGUE_REF: "That is all I have."]
[VOICEOVER]
The driver empties his pockets in distress before the watching crowd. {" ".join(['Detailed dramatic climax.'] * 50)}"""

    durations = [180.0, 305.0]
    call_idx = [0]

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        def mock_measure(path):
            dur = durations[min(call_idx[0], len(durations) - 1)]
            call_idx[0] += 1
            return dur

        def mock_llm(prompt):
            assert "Expand" in prompt
            return expanded_script

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script_v1,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=2,
            llm_correction_fn=mock_llm,
            synthesize_fn=mock_synth,
            measure_fn=mock_measure
        )

    res = asyncio.run(_run())
    assert res["correction_attempts"] == 1
    assert res["duration_converged"] is True
    assert pytest.approx(res["final_projected_final_duration"], 1.0) == 305.0 / 1.02
    assert "converged" in res["reason"]

def test_duration_feedback_long_script_triggers_compression_and_converges(tmp_path):
    """Req F: Long script triggers compression and converges."""
    script_v1 = f"""[SCENE: 00:00 - 05:00]
[DIALOGUE_REF: "Important quote"]
[VOICEOVER]
{" ".join(['Sentence number one in the long narration.', 'Sentence number two in the long narration.', 'Sentence number three in the long narration.', 'Sentence number four in the long narration.'] * 30)}"""

    durations = [440.0, 304.0]
    call_idx = [0]

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        def mock_measure(path):
            dur = durations[min(call_idx[0], len(durations) - 1)]
            call_idx[0] += 1
            return dur

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script_v1,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=1,
            llm_correction_fn=lambda p: None,
            synthesize_fn=mock_synth,
            measure_fn=mock_measure
        )

    res = asyncio.run(_run())
    assert res["correction_attempts"] == 1
    assert res["duration_converged"] is True
    assert "[SCENE: 00:00 - 05:00]" in res["script"]
    assert '[DIALOGUE_REF: "Important quote"]' in res["script"]

def test_duration_feedback_correct_duration_skips_correction(tmp_path):
    """Req G: Correct initial duration skips correction (0 attempts)."""
    script = SAMPLE_STORYBOARD_WITH_TAGS

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=30.0,
            max_corrections=2,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 300.0 * 1.02
        )

    res = asyncio.run(_run())
    assert res["correction_attempts"] == 0
    assert res["duration_converged"] is True
    assert res["initial_tts_duration"] == res["final_tts_duration"]

def test_duration_feedback_maximum_corrections_enforced(tmp_path):
    """Req H: Maximum correction attempts (2) are strictly enforced."""
    script = SAMPLE_STORYBOARD_WITH_TAGS
    attempts_seen = [0]

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        def mock_llm(p):
            attempts_seen[0] += 1
            return script

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=10.0,
            max_corrections=2,
            llm_correction_fn=mock_llm,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 500.0
        )

    res = asyncio.run(_run())
    assert res["correction_attempts"] == 2
    assert attempts_seen[0] == 2
    assert res["duration_converged"] is False

def test_duration_feedback_failed_convergence_safety(tmp_path):
    """Req I: Failed convergence does not silently report success."""
    script = SAMPLE_STORYBOARD_WITH_TAGS

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script,
            duration_mins=5,
            temp_dir=str(tmp_path),
            tolerance_sec=15.0,
            max_corrections=1,
            llm_correction_fn=lambda p: script,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 500.0
        )

    res = asyncio.run(_run())
    assert res["duration_converged"] is False
    assert "Failed to converge" in res["reason"]
    for k in [
        "requested_duration_seconds", "initial_tts_duration", "final_tts_duration",
        "initial_projected_final_duration", "final_projected_final_duration",
        "duration_error_seconds", "duration_converged", "correction_attempts",
        "initial_spoken_words", "final_spoken_words"
    ]:
        assert k in res

def test_duration_feedback_chronology_validation_active(tmp_path):
    """Req J: Existing chronology validation remains active during feedback loop."""
    timeline = [
        {"start": 10.0, "end": 15.0, "text": "First quote"},
        {"start": 20.0, "end": 25.0, "text": "Second quote"}
    ]
    script_valid = """[SCENE: 00:00 - 00:30]
[DIALOGUE_REF: "First quote"]
[VOICEOVER]
Scene one narration.

[SCENE: 00:30 - 01:00]
[DIALOGUE_REF: "Second quote"]
[VOICEOVER]
Scene two narration."""

    script_inverted = """[SCENE: 00:00 - 00:30]
[DIALOGUE_REF: "Second quote"]
[VOICEOVER]
Inverted scene narration.

[SCENE: 00:30 - 01:00]
[DIALOGUE_REF: "First quote"]
[VOICEOVER]
Inverted scene narration."""

    async def _run():
        async def mock_synth(text, voice, out_p, rate, pitch):
            with open(out_p, "w") as f: f.write("mock")
            return True

        return await ScriptEngine.converge_script_duration_with_tts_feedback(
            script_text=script_valid,
            duration_mins=5,
            temp_dir=str(tmp_path),
            dialogue_timeline=timeline,
            max_corrections=1,
            llm_correction_fn=lambda p: script_inverted,
            synthesize_fn=mock_synth,
            measure_fn=lambda p: 200.0
        )

    res = asyncio.run(_run())
    assert '[DIALOGUE_REF: "First quote"]' in res["script"]
    pos1 = res["script"].index("First quote")
    pos2 = res["script"].index("Second quote")
    assert pos1 < pos2

def test_duration_feedback_dialogue_provenance_active():
    """Req K: Existing dialogue provenance remains active and verifiable."""
    timeline = [
        {"start": 10.0, "end": 15.0, "text": "Exact matching dialogue quote"}
    ]
    script = """[SCENE: 00:00 - 00:30]
[DIALOGUE_REF: "Exact matching dialogue quote"]
[VOICEOVER]
Scene narration."""

    blocks = ScriptEngine.parse_storyboard_blocks(script, dialogue_timeline=timeline)
    assert len(blocks) == 1
    assert blocks[0].dialogue_ref == "Exact matching dialogue quote"
    assert blocks[0].movie_start == 10.0

def test_duration_feedback_timeline_behavior_unchanged():
    """Req L: Existing audio/video timeline behavior remains unchanged."""
    blocks = [
        SceneBlock(movie_start=10.0, movie_end=20.0, narration_text="First scene", word_count=2),
        SceneBlock(movie_start=25.0, movie_end=35.0, narration_text="Second scene", word_count=2)
    ]
    total_tts = 100.0
    timed = ScriptEngine.assign_narration_timing(blocks, total_tts)

    assert len(timed) == 2
    assert timed[0].narration_start == 0.0
    assert timed[0].narration_end == timed[1].narration_start
    assert timed[1].narration_end == total_tts
    assert sum(b.speech_dur for b in timed) == pytest.approx(total_tts, 0.001)
