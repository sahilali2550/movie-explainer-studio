import os
import math
import pytest
from typing import Dict, List, Optional, Any, Tuple

from app.services.script_engine import (
    SceneBlock,
    ScriptEngine,
    VisualSemanticProvider,
    DictVisualSemanticProvider,
    OpenAICompatibleMultimodalVisualProvider,
    VisualEmbeddingCache,
    _validate_visual_vector,
)


class TestPhase7AVisualSemanticSelection:
    """
    Phase 7A: Automated Test Suite for Visual Semantic Scene Selection.
    Verifies:
      A. Visual provider abstraction
      B. Image/text embedding dimension validation
      C. Invalid/NaN/empty provider result handling
      D. Candidate generation is bounded
      E. Strong exact dialogue anchor remains protected
      F. Weak timestamp candidate can be visually reranked
      G. Visual provider unavailable -> existing anchor path preserved
      H. Visual score cannot dominate beyond configured cap
      I. Silent-event candidate can be selected in a deterministic mocked visual environment
      J. Cache isolation by provider/model
      K. Existing chronology remains valid
      L. Existing dialogue provenance remains valid
      M. Existing audio-locked timing remains unchanged
    """

    # --- Test A: Visual provider abstraction ---
    def test_a_provider_abstraction(self):
        class IncompleteProvider(VisualSemanticProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

        provider = DictVisualSemanticProvider(
            text_embeddings={"a man walks": [1.0, 0.0]},
            image_embeddings={"frame_01.jpg": [1.0, 0.0]},
            provider_name="test_prov",
            model_name="test_model"
        )
        assert provider.provider_id == "test_prov"
        assert provider.model_id == "test_model"
        assert provider.namespace == "test_prov:test_model"
        assert provider.encode_text("a man walks") == [1.0, 0.0]
        assert provider.encode_image("frame_01.jpg") == [1.0, 0.0]
        assert provider.similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    # --- Test B: Image/text embedding dimension validation ---
    def test_b_dimension_validation(self):
        assert _validate_visual_vector([1.0, 2.0, 3.0], expected_dim=3) == [1.0, 2.0, 3.0]
        assert _validate_visual_vector([1.0, 2.0], expected_dim=3) is None
        assert _validate_visual_vector([1.0, 2.0, 3.0, 4.0], expected_dim=3) is None

        prov = DictVisualSemanticProvider(dim=4)
        assert prov.validate_vector([1.0, 2.0, 3.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]
        assert prov.validate_vector([1.0, 2.0]) is None

    # --- Test C: Invalid/NaN/empty provider result handling ---
    def test_c_invalid_nan_empty_handling(self):
        assert _validate_visual_vector([]) is None
        assert _validate_visual_vector(None) is None
        assert _validate_visual_vector([1.0, float("nan"), 3.0]) is None
        assert _validate_visual_vector([1.0, float("inf"), 3.0]) is None
        assert _validate_visual_vector(["not", "numbers"]) is None

        prov = DictVisualSemanticProvider()
        # Unknown query/frame returns 0.0 safely
        score = prov.score_frame("non_existent_frame.jpg", "unknown text query")
        assert score == 0.0

    # --- Test D: Candidate generation is bounded ---
    def test_d_candidate_generation_is_bounded(self):
        block = SceneBlock(
            movie_start=100.0,
            movie_end=115.0,
            narration_text="The detective discovers the hidden safe behind the painting.",
            word_count=10
        )
        timeline = [
            {"start": float(t), "end": float(t + 4.0), "text": f"Dialogue cue at {t}"}
            for t in range(0, 300, 10)
        ]
        candidates = ScriptEngine.generate_visual_candidates(
            block=block,
            dialogue_timeline=timeline,
            video_duration=300.0,
            search_window_radius=30.0,
            max_candidates=5
        )
        assert len(candidates) <= 5
        assert len(candidates) >= 1
        # All candidates must be within bounded window [100 - 30, 100 + 30]
        for c in candidates:
            assert 70.0 <= c["start"] <= 130.0

    # --- Test E: Strong exact dialogue anchor remains protected ---
    def test_e_strong_exact_dialogue_anchor_remains_protected(self):
        block = SceneBlock(
            movie_start=50.0,
            movie_end=65.0,
            narration_text="He demands payment for the damaged bumper.",
            word_count=8,
            dialogue_ref="Give me the money for the car right now."
        )
        timeline = [
            {"start": 50.0, "end": 55.0, "text": "Give me the money for the car right now."},
            {"start": 60.0, "end": 65.0, "text": "Something completely unrelated."}
        ]
        # Visual provider gives higher score to t=60.0 than t=50.0
        provider = DictVisualSemanticProvider(
            direct_scores={
                ("frame_50.jpg", "He demands payment for the damaged bumper."): 0.40,
                ("frame_60.jpg", "He demands payment for the damaged bumper."): 0.95,
            }
        )
        # Even with higher visual score at t=60.0, strong dialogue ref protects t=50.0
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            dialogue_timeline=timeline,
            visual_provider=provider,
            mock_frame_resolver=lambda ts: f"frame_{int(ts)}.jpg"
        )
        assert len(reranked) == 1
        assert reranked[0].movie_start == 50.0
        assert reranked[0].visual_selection_reason == "dialogue_ref_protected"

    # --- Test F: Weak timestamp candidate can be visually reranked ---
    def test_f_weak_timestamp_candidate_can_be_visually_reranked(self):
        # Scene block with no dialogue ref (weak / generic conversational match)
        block = SceneBlock(
            movie_start=120.0,
            movie_end=130.0,
            narration_text="She examines the mysterious antique clock in the hallway.",
            word_count=10,
            dialogue_ref=None
        )
        timeline = [
            {"start": 120.0, "end": 125.0, "text": "Yeah okay."},  # weak generic cue
            {"start": 128.0, "end": 133.0, "text": "I see it."}     # weak generic cue
        ]
        provider = DictVisualSemanticProvider(
            direct_scores={
                ("frame_120.jpg", "She examines the mysterious antique clock in the hallway."): 0.20,
                ("frame_128.jpg", "She examines the mysterious antique clock in the hallway."): 0.88,
            }
        )
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            dialogue_timeline=timeline,
            visual_provider=provider,
            mock_frame_resolver=lambda ts: f"frame_{int(ts)}.jpg"
        )
        assert len(reranked) == 1
        assert reranked[0].movie_start == 128.0
        assert reranked[0].visual_confidence == "VISUAL_CONFIDENT"
        assert reranked[0].visual_score == 0.88

    # --- Test G: Visual provider unavailable -> existing anchor path preserved ---
    def test_g_visual_provider_unavailable_preserves_anchor(self):
        block = SceneBlock(
            movie_start=85.0,
            movie_end=95.0,
            narration_text="The car drives away into the evening sunset.",
            word_count=9
        )
        # 1. None provider
        reranked_none = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            visual_provider=None
        )
        assert len(reranked_none) == 1
        assert reranked_none[0].movie_start == 85.0
        assert reranked_none[0].visual_confidence == "VISUAL_UNAVAILABLE"

        # 2. Failing provider
        class BrokenProvider(VisualSemanticProvider):
            provider_id = "broken"
            model_id = "broken"
            def encode_text(self, text: str):
                raise RuntimeError("Network failure")
            def encode_image(self, image_path: str):
                raise RuntimeError("Disk failure")
            def similarity(self, a, b):
                return 0.0
            def score_frame(self, image_path: str, query: str):
                raise RuntimeError("Vision service timeout")

        reranked_broken = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            visual_provider=BrokenProvider()
        )
        assert len(reranked_broken) == 1
        assert reranked_broken[0].movie_start == 85.0
        assert reranked_broken[0].visual_confidence == "VISUAL_UNAVAILABLE"

    # --- Test H: Visual score cannot dominate beyond configured cap ---
    def test_h_visual_score_cannot_dominate_beyond_cap(self):
        score_low = ScriptEngine.compute_visual_bonus(0.20, max_bonus=1.0)
        assert score_low == 0.0  # weak threshold < 0.65 yields 0.0 bonus

        score_high = ScriptEngine.compute_visual_bonus(0.90, max_bonus=1.0)
        assert 0.0 < score_high <= 1.0

        score_extreme = ScriptEngine.compute_visual_bonus(500.0, max_bonus=1.0)
        assert score_extreme == 1.0  # capped strictly at max_bonus

    # --- Test I: Silent-event candidate can be selected in a deterministic mocked visual environment ---
    def test_i_silent_event_candidate_selected_visually(self):
        block = SceneBlock(
            movie_start=150.0,
            movie_end=160.0,
            narration_text="The mother silently watches her son leave through the hospital doors.",
            word_count=11,
            dialogue_ref=None
        )
        # Empty dialogue timeline at this interval (silent scene)
        timeline = [
            {"start": 100.0, "end": 105.0, "text": "Goodbye."},
            {"start": 210.0, "end": 215.0, "text": "Welcome back."}
        ]
        # Candidate generation identifies silent gaps at t=150.0 and t=170.0
        provider = DictVisualSemanticProvider(
            direct_scores={
                ("frame_150.jpg", "The mother silently watches her son leave through the hospital doors."): 0.15,
                ("frame_170.jpg", "The mother silently watches her son leave through the hospital doors."): 0.92,
            }
        )
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            dialogue_timeline=timeline,
            visual_provider=provider,
            mock_frame_resolver=lambda ts: f"frame_{int(ts)}.jpg"
        )
        assert len(reranked) == 1
        assert reranked[0].movie_start == 170.0
        assert reranked[0].visual_confidence == "VISUAL_CONFIDENT"
        assert reranked[0].visual_score == 0.92

    # --- Test J: Cache isolation by provider/model ---
    def test_j_cache_isolation_by_provider_and_model(self):
        cache = VisualEmbeddingCache(max_size=100)
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]

        cache.put("frame_01.jpg", v1, namespace="providerA:model1:2")
        cache.put("frame_01.jpg", v2, namespace="providerB:model2:2")

        res_a = cache.get("frame_01.jpg", namespace="providerA:model1:2")
        res_b = cache.get("frame_01.jpg", namespace="providerB:model2:2")
        res_c = cache.get("frame_01.jpg", namespace="providerA:model2:2")

        assert res_a == v1
        assert res_b == v2
        assert res_c is None

    # --- Test K: Existing chronology remains valid ---
    def test_k_existing_chronology_remains_valid(self):
        block1 = SceneBlock(movie_start=50.0, movie_end=60.0, narration_text="Scene one.", word_count=2)
        block2 = SceneBlock(movie_start=100.0, movie_end=110.0, narration_text="Scene two.", word_count=2)
        block3 = SceneBlock(movie_start=150.0, movie_end=160.0, narration_text="Scene three.", word_count=2)

        # Provider tempts block2 to jump backward to t=40.0 (before block1)
        provider = DictVisualSemanticProvider(
            direct_scores={
                ("frame_40.jpg", "Scene two."): 0.99,   # illegal backward jump
                ("frame_105.jpg", "Scene two."): 0.85,  # valid forward candidate
            }
        )
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block1, block2, block3],
            dialogue_timeline=[],
            visual_provider=provider,
            mock_frame_resolver=lambda ts: f"frame_{int(ts)}.jpg"
        )
        # Must strictly preserve non-decreasing order: block1 <= block2 <= block3
        assert reranked[0].movie_start <= reranked[1].movie_start <= reranked[2].movie_start
        assert reranked[1].movie_start >= reranked[0].movie_start

    # --- Test L: Existing dialogue provenance remains valid ---
    def test_l_dialogue_provenance_preserved(self):
        block = SceneBlock(
            movie_start=100.0,
            movie_end=110.0,
            narration_text="Scene text.",
            word_count=2,
            dialogue_ref="Verified ASR quote from movie",
            evidence_ref="EP-003",
            story_step=3
        )
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            visual_provider=None
        )
        assert reranked[0].dialogue_ref == "Verified ASR quote from movie"
        assert reranked[0].evidence_ref == "EP-003"
        assert reranked[0].story_step == 3

    # --- Test M: Existing audio-locked timing remains unchanged ---
    def test_m_audio_locked_timing_remains_unchanged(self):
        block = SceneBlock(
            movie_start=100.0,
            movie_end=110.0,
            narration_text="Scene with TTS duration lock.",
            word_count=5,
            speech_dur=12.45,
            actual_duration=12.45,
            narration_start=34.50,
            narration_end=46.95,
            is_authoritative=True
        )
        provider = DictVisualSemanticProvider(
            direct_scores={("frame_108.jpg", "Scene with TTS duration lock."): 0.90}
        )
        reranked = ScriptEngine.rerank_scene_blocks_visually(
            blocks=[block],
            visual_provider=provider,
            mock_frame_resolver=lambda ts: f"frame_{int(ts)}.jpg"
        )
        # Source footage start can be adjusted
        # But narration timing coordinates must remain EXACTLY unchanged
        assert reranked[0].speech_dur == 12.45
        assert reranked[0].actual_duration == 12.45
        assert reranked[0].narration_start == 34.50
        assert reranked[0].narration_end == 46.95
        assert reranked[0].is_authoritative is True
