import pytest
from typing import Optional, Dict, Any, List
from app.schemas.explainer import ScriptGenerateRequest
from app.services.script_engine import ScriptEngine, SceneBlock


def create_sample_cues(count: int = 40, total_duration_sec: float = 3600.0) -> List[Dict[str, Any]]:
    """Helper to create realistic timed dialogue cues distributed across a movie."""
    cues = []
    interval = total_duration_sec / max(1, count)
    for i in range(count):
        start = i * interval + 2.0
        end = start + 8.0
        cues.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": f"Spoken dialogue event {i+1} at timestamp {int(start)} seconds"
        })
    return cues


# ---------------------------------------------------------------------------
# 1. API Schema Duration Tests
# ---------------------------------------------------------------------------

def test_schema_accepts_long_form_duration():
    """Verify ScriptGenerateRequest accepts 20, 30, and 60 minute durations without le=15 error."""
    req_20 = ScriptGenerateRequest(duration_mins=20)
    assert req_20.duration_mins == 20

    req_30 = ScriptGenerateRequest(duration_mins=30)
    assert req_30.duration_mins == 30

    req_60 = ScriptGenerateRequest(duration_mins=60)
    assert req_60.duration_mins == 60

    # Minimum bound ge=1 must still be enforced
    with pytest.raises(Exception):
        ScriptGenerateRequest(duration_mins=0)


# ---------------------------------------------------------------------------
# 2. Chunk Planning & Word Budget Conservation Tests
# ---------------------------------------------------------------------------

def test_plan_long_form_chunks_20m():
    """Verify 20-minute explainer is partitioned into bounded narrative chunks with chronological boundaries."""
    cues = create_sample_cues(count=30, total_duration_sec=7200.0)
    evidence_packets = ScriptEngine.build_evidence_packets(
        total_movie_dur=7200.0,
        target_output_dur_mins=20,
        dialogue_timeline=cues
    )
    story_plan = ScriptEngine.create_grounded_story_plan(
        evidence_packets=evidence_packets,
        target_duration_mins=20,
        genre="movie_recap",
        target_lang="en",
        voice_speed="fast"
    )

    chunks = ScriptEngine.plan_long_form_chunks(
        target_duration_mins=20,
        voice_speed="fast",
        target_lang="en",
        source_duration_sec=7200.0,
        dialogue_timeline=cues,
        evidence_packets=evidence_packets,
        story_plan=story_plan,
        max_chunk_words=1200
    )

    assert len(chunks) >= 5
    total_target_words = ScriptEngine.calculate_target_words(20, voice_speed="fast", target_lang="en", anti_copyright_drift=True)
    allocated_words = sum(c["target_words"] for c in chunks)
    assert abs(allocated_words - total_target_words) <= len(chunks) * 5

    # Check chronological order of chunk milestones
    for i in range(len(chunks) - 1):
        assert chunks[i]["chunk_index"] == i
        assert chunks[i]["start_sec"] <= chunks[i+1]["start_sec"]
        assert chunks[i]["end_sec"] <= chunks[i+1]["end_sec"]
        assert chunks[i]["target_words"] > 0
        assert chunks[i]["target_words"] <= 1250


def test_plan_long_form_chunks_subdivision_on_large_acts():
    """Verify that an act whose word target exceeds max_chunk_words is subdivided deterministically."""
    cues = create_sample_cues(count=50, total_duration_sec=7200.0)
    evidence_packets = ScriptEngine.build_evidence_packets(
        total_movie_dur=7200.0,
        target_output_dur_mins=30,
        dialogue_timeline=cues
    )
    story_plan = ScriptEngine.create_grounded_story_plan(
        evidence_packets=evidence_packets,
        target_duration_mins=30,
        genre="movie_recap",
        target_lang="en",
        voice_speed="fast"
    )

    # In 30m, Act 2A and 2B have 25% of ~5280 words = ~1320 words.
    # With max_chunk_words=1000, both should be subdivided into Part 1 and Part 2.
    chunks = ScriptEngine.plan_long_form_chunks(
        target_duration_mins=30,
        voice_speed="fast",
        target_lang="en",
        source_duration_sec=7200.0,
        dialogue_timeline=cues,
        evidence_packets=evidence_packets,
        story_plan=story_plan,
        max_chunk_words=1000
    )

    assert len(chunks) > 5
    for c in chunks:
        assert c["target_words"] <= 1050


# ---------------------------------------------------------------------------
# 3. Chunk Prompt Construction & Continuity Context
# ---------------------------------------------------------------------------

def test_build_chunk_prompt_continuity():
    """Verify build_chunk_prompt includes minimal continuity context from previous chunk without leaking full script."""
    chunks = ScriptEngine.plan_long_form_chunks(
        target_duration_mins=20,
        voice_speed="fast",
        target_lang="en",
        source_duration_sec=3600.0
    )

    chunk_1 = chunks[0]
    prompt_1 = ScriptEngine.build_chunk_prompt(
        chunk=chunk_1,
        title="Inception Recap",
        description="A thief who steals corporate secrets.",
        plot_summary="Dom Cobb enters dreams.",
        genre="movie_recap",
        persona="hollywood_trailer",
        mood="suspense",
        spoiler_mode="full_recap",
        target_lang="en",
        voice_speed="fast",
        previous_chunk_summary=None
    )

    assert "Act 1" in prompt_1
    assert "PREVIOUS_SCENE_END_CONTEXT" not in prompt_1
    assert "DO NOT conclude the movie" in prompt_1

    # Chunk 2 with previous context
    prev_summary = {
        "last_scene_timestamp": "08:15 - 08:45",
        "last_dialogue_ref": "Are you ready to take a leap of faith?",
        "last_narration_snippet": "Cobb watches the spinning top hesitate on the table before the train roars through the city streets."
    }

    chunk_2 = chunks[1]
    prompt_2 = ScriptEngine.build_chunk_prompt(
        chunk=chunk_2,
        title="Inception Recap",
        description="A thief who steals corporate secrets.",
        plot_summary="Dom Cobb enters dreams.",
        genre="movie_recap",
        persona="hollywood_trailer",
        mood="suspense",
        spoiler_mode="full_recap",
        target_lang="en",
        voice_speed="fast",
        previous_chunk_summary=prev_summary
    )

    assert "PREVIOUS_SCENE_END_CONTEXT" in prompt_2
    assert "08:15 - 08:45" in prompt_2
    assert "Are you ready to take a leap of faith?" in prompt_2
    assert "DO NOT re-introduce characters" in prompt_2


# ---------------------------------------------------------------------------
# 4. Chunk Generation & Bounded Retry Tests
# ---------------------------------------------------------------------------

def test_generate_chunk_with_retry_success():
    """Verify generate_chunk_with_retry returns cleanly on first valid attempt."""
    chunk = {
        "chunk_index": 0,
        "total_chunks": 5,
        "act": "Act 1",
        "start_sec": 0.0,
        "end_sec": 720.0,
        "target_words": 150
    }

    valid_response = """[SCENE: 01:00 - 01:30]
[DIALOGUE_REF: "Don't move or I'll shoot."]
[VOICEOVER]
The rain beats heavily against the cracked asphalt as John corners his former partner in the shadows of the alleyway. The betrayal has shattered what was left of their fragile brotherhood, and now the reckoning has finally arrived. Neither man is willing to back down from the fatal confrontation that will seal their destiny forever."""

    def mock_gen(p: str) -> Optional[str]:
        return valid_response

    text, is_valid, report, attempts = ScriptEngine.generate_chunk_with_retry(
        generate_fn=mock_gen,
        chunk_prompt="prompt",
        chunk=chunk,
        target_lang="en"
    )

    assert is_valid is True
    assert attempts == 1
    assert "01:00 - 01:30" in text


def test_generate_chunk_with_retry_recovers_on_second_attempt():
    """Verify generate_chunk_with_retry retries once on malformed first output and succeeds on second attempt."""
    chunk = {
        "chunk_index": 0,
        "total_chunks": 5,
        "act": "Act 1",
        "start_sec": 0.0,
        "end_sec": 720.0,
        "target_words": 120
    }

    call_count = 0
    def mock_flaky_gen(p: str) -> Optional[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Too short"  # Fails length & storyboard validation
        return """[SCENE: 02:00 - 02:30]
[DIALOGUE_REF: "You will regret this."]
[VOICEOVER]
The tension in the courtroom explodes as the final piece of evidence is presented before the stunned jury. Every eye turns toward the defendant whose calm demeanor completely vanishes in an instant of sheer panic."""

    text, is_valid, report, attempts = ScriptEngine.generate_chunk_with_retry(
        generate_fn=mock_flaky_gen,
        chunk_prompt="prompt",
        chunk=chunk,
        target_lang="en",
        max_retries=1
    )

    assert is_valid is True
    assert attempts == 2
    assert call_count == 2
    assert "02:00 - 02:30" in text


def test_generate_chunk_with_retry_aborts_on_repeated_failure():
    """Verify generate_chunk_with_retry stops after max_retries without looping infinitely."""
    chunk = {
        "chunk_index": 0,
        "total_chunks": 5,
        "act": "Act 1",
        "start_sec": 0.0,
        "end_sec": 720.0,
        "target_words": 120
    }

    def mock_failing_gen(p: str) -> Optional[str]:
        return "Always bad output"

    text, is_valid, report, attempts = ScriptEngine.generate_chunk_with_retry(
        generate_fn=mock_failing_gen,
        chunk_prompt="prompt",
        chunk=chunk,
        target_lang="en",
        max_retries=1
    )

    assert is_valid is False
    assert attempts == 2
    assert "Chunk generation failed" in report


# ---------------------------------------------------------------------------
# 5. Merging & Boundary Deduplication Tests
# ---------------------------------------------------------------------------

def test_merge_storyboard_chunks_deduplication():
    """Verify merge_storyboard_chunks removes duplicate boundary scenes and consecutive repeated narration."""
    chunk_1_text = """[SCENE: 00:00 - 00:30]
[DIALOGUE_REF: "Hello world"]
[VOICEOVER]
The morning sun rises over the quiet city as Marcus prepares for his morning patrol across the district.

[SCENE: 05:00 - 05:30]
[DIALOGUE_REF: "Look at the radar"]
[VOICEOVER]
Marcus detects an unidentified signal approaching the perimeter. It moves with impossible speed and precision."""

    # Chunk 2 accidentally repeats the last scene of chunk 1 at its beginning
    chunk_2_text = """[SCENE: 05:00 - 05:30]
[DIALOGUE_REF: "Look at the radar"]
[VOICEOVER]
Marcus detects an unidentified signal approaching the perimeter. It moves with impossible speed and precision.

[SCENE: 12:00 - 12:30]
[DIALOGUE_REF: "We need backup immediately"]
[VOICEOVER]
The command center goes completely dark as the alien vessel breaches the atmosphere above the capital."""

    chunks_output = [
        {"chunk_index": 0, "act": "Act 1", "text": chunk_1_text},
        {"chunk_index": 1, "act": "Act 2A", "text": chunk_2_text}
    ]

    merged_text, is_valid, report, details = ScriptEngine.merge_storyboard_chunks(
        chunks_output=chunks_output,
        target_duration_mins=20,
        target_lang="en",
        voice_speed="fast",
        source_duration_sec=3600.0
    )

    # The repeated scene [SCENE: 05:00 - 05:30] must appear exactly ONCE
    occurrences = merged_text.count("05:00 - 05:30")
    assert occurrences == 1
    assert "00:00 - 00:30" in merged_text
    assert "12:00 - 12:30" in merged_text


# ---------------------------------------------------------------------------
# 6. End-to-End Long-Form Generation (20m and 30m)
# ---------------------------------------------------------------------------

def test_long_form_20m_end_to_end_structural_validation():
    """Verify 20-minute long-form explainer workflow generates multiple chunks, merges, and validates."""
    cues = create_sample_cues(count=30, total_duration_sec=7200.0)

    def mock_provider(prompt: str) -> Optional[str]:
        base_filler = (
            "Detective Ryan carefully examines the forensic evidence scattered across the metropolitan crime scene. "
            "Every subtle trace reveals a calculated conspiracy reaching deep into the corporate syndicate. "
            "The ticking clock forces the investigation into high gear as hidden enemies watch from the shadows. "
            "With unwavering determination, Ryan navigates dangerous underground networks to uncover the truth. "
        )
        body = " ".join([base_filler] * 8)  # ~380 words

        if "Act 1" in prompt:
            return f"""[SCENE: 00:00 - 02:00]
[DIALOGUE_REF: "Spoken dialogue event 1 at timestamp 2 seconds"]
[VOICEOVER]
In the sprawling heart of the futuristic metropolis, Detective Ryan investigates an unprecedented digital robbery. {body}"""
        elif "Act 2A" in prompt:
            return f"""[SCENE: 24:00 - 26:00]
[DIALOGUE_REF: "Spoken dialogue event 10 at timestamp 2402 seconds"]
[VOICEOVER]
Ryan traces the encrypted transmission to an abandoned industrial shipyard on the outskirts of the harbor. {body}"""
        elif "Act 2B" in prompt:
            return f"""[SCENE: 48:00 - 50:00]
[DIALOGUE_REF: "Spoken dialogue event 20 at timestamp 4802 seconds"]
[VOICEOVER]
A devastating revelation shocks the department as Cipher's true identity is finally decrypted on the mainframe. {body}"""
        elif "Act 3" in prompt:
            return f"""[SCENE: 75:00 - 78:00]
[DIALOGUE_REF: "Spoken dialogue event 25 at timestamp 6002 seconds"]
[VOICEOVER]
Ryan storms the executive penthouse in a breathtaking high-stakes showdown against the corrupt commissioner. {body}"""
        else:  # Epilogue
            return f"""[SCENE: 116:00 - 118:00]
[DIALOGUE_REF: "Spoken dialogue event 30 at timestamp 7202 seconds"]
[VOICEOVER]
As dawn breaks over the restored city, Ryan walks away from the precinct with the quiet satisfaction of justice served. {body} Make sure to subscribe to our channel for more thrilling recaps!"""

    res = ScriptEngine.generate_script(
        title="The Cipher Protocol",
        description="A detective takes down a corrupt mastermind.",
        subs_text=" ".join(c["text"] for c in cues),
        target_lang="en",
        duration_mins=20,
        genre="movie_recap",
        source_video_duration_sec=7200.0,
        dialogue_timeline=cues,
        generate_fn=mock_provider
    )

    assert res.get("success") is True
    assert "00:00 - 02:00" in res["script"]
    assert "116:00 - 118:00" in res["script"]
    assert res.get("actual_words", 0) > 1800


def test_long_form_30m_end_to_end_structural_validation():
    """Verify 30-minute long-form explainer workflow generates multiple chunks, merges, and validates."""
    cues = create_sample_cues(count=40, total_duration_sec=7200.0)

    def mock_provider(prompt: str) -> Optional[str]:
        base_filler = (
            "The deep-sea operative navigates through the darkened pressure chambers of the ocean laboratory. "
            "Hostile submarine units circle the perimeter while encrypted telemetry flows to foreign servers. "
            "Every structural bulkhead groans under the relentless pressure as time runs out for the evacuation team. "
            "With ice-cold focus, the agent prepares the countermeasures that will decide the fate of the entire mission. "
        )
        body = " ".join([base_filler] * 8)  # ~400 words

        if "Part 1" in prompt and "Act 2A" in prompt:
            return f"""[SCENE: 18:00 - 20:00]
[DIALOGUE_REF: "Spoken dialogue event 10 at timestamp 1802 seconds"]
[VOICEOVER]
The facility suffers a sudden catastrophic containment failure as shadowy corporate mercenaries sever external lines. {body}"""
        elif "Part 2" in prompt and "Act 2A" in prompt:
            return f"""[SCENE: 30:00 - 32:00]
[DIALOGUE_REF: "Spoken dialogue event 15 at timestamp 2702 seconds"]
[VOICEOVER]
Water levels rise rapidly as the operative fights through flooded maintenance corridors toward the central core. {body}"""
        elif "Part 1" in prompt and "Act 2B" in prompt:
            return f"""[SCENE: 42:00 - 44:00]
[DIALOGUE_REF: "Spoken dialogue event 20 at timestamp 3602 seconds"]
[VOICEOVER]
Desperate survivors discover that the director himself planned the sabotage to destroy evidence of quantum tampering. {body}"""
        elif "Part 2" in prompt and "Act 2B" in prompt:
            return f"""[SCENE: 55:00 - 57:00]
[DIALOGUE_REF: "Spoken dialogue event 25 at timestamp 4502 seconds"]
[VOICEOVER]
The operative confronts the rogue research team in the central observation bay amidst sparking electrical panels. {body}"""
        elif "Act 1" in prompt:
            return f"""[SCENE: 00:00 - 02:00]
[DIALOGUE_REF: "Spoken dialogue event 1 at timestamp 2 seconds"]
[VOICEOVER]
An elite deep-cover operative infiltrates an undersea facility where illegal quantum experiments threaten global stability. {body}"""
        elif "Act 3" in prompt:
            return f"""[SCENE: 78:00 - 80:00]
[DIALOGUE_REF: "Spoken dialogue event 30 at timestamp 5402 seconds"]
[VOICEOVER]
The operative triggers the emergency ballast vents and neutralizes the rogue director before escaping in the deep-sea pod. {body}"""
        else:  # Epilogue
            return f"""[SCENE: 116:00 - 118:00]
[DIALOGUE_REF: "Spoken dialogue event 40 at timestamp 7202 seconds"]
[VOICEOVER]
Surfacing under the clear northern sky, the agent transmits the recovered data to international authorities. {body} Subscribe for more thrilling recaps!"""

    res = ScriptEngine.generate_script(
        title="Abyssal Protocol",
        description="Undersea action thriller.",
        subs_text=" ".join(c["text"] for c in cues),
        target_lang="en",
        duration_mins=30,
        genre="movie_recap",
        source_video_duration_sec=7200.0,
        dialogue_timeline=cues,
        generate_fn=mock_provider
    )

    assert res.get("success") is True
    assert "00:00 - 02:00" in res["script"]
    assert "116:00 - 118:00" in res["script"]
    assert res.get("actual_words", 0) > 2800


def test_short_form_duration_regression_unchanged():
    """Verify that durations <= 15 minutes continue to use single-turn generation without chunking."""
    assert ScriptEngine.LONG_FORM_DURATION_THRESHOLD_MINS == 15

    # Target words calculation for short-form
    w_3m = ScriptEngine.calculate_target_words(3, voice_speed="fast", target_lang="en")
    w_15m = ScriptEngine.calculate_target_words(15, voice_speed="fast", target_lang="en")
    assert w_3m < w_15m

