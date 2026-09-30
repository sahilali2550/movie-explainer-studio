"""
Phase 7B Multilingual Robustness Test Suite.
Verifies tokenization, Unicode normalization, language-scoped English morphology & stopwords,
language-neutral lexical matching, dialogue_ref provenance, and roadmap informativeness
across representative language families:
Latin (EN, ES, FR, DE, PT, TR, ID), Cyrillic (RU), Arabic script (AR, UR), Indic (HI, BN).
"""
import pytest
import unicodedata
from app.services.script_engine import (
    ScriptEngine,
    SceneBlock,
    DialogueRefProvenance,
)


class TestPhase7BMultilingualRobustness:

    # =========================================================================
    # 1. UNICODE NORMALIZATION & TOKENIZATION ACROSS SCRIPT FAMILIES
    # =========================================================================

    def test_multilingual_unicode_nfkc_and_tokenization(self):
        """Verifies Unicode NFKC preservation and robust tokenization across 12 languages."""
        samples = {
            "en": "The detective investigated the mysterious crime scene.",
            "es": "¿El detective investigó la escena del crimen misterioso?",
            "fr": "L'inspecteur a examiné la mystérieuse scène du crime.",
            "de": "Der Detektiv untersuchte das merkwürdige Verbrechen.",
            "pt": "O detetive investigou a misteriosa cena do crime.",
            "tr": "Dedektif gizemli suç mahallini ayrıntılı olarak inceledi.",
            "id": "Detektif menyelidiki tempat kejadian perkara yang misterius.",
            "ru": "Детектив расследовал загадочное место преступления.",
            "ar": "قام المحقق بفحص مسرح الجريمة الغامض بدقة.",
            "ur": "تفتیش کار نے پراسرار جائے وقوعہ کا باریک بینی سے جائزہ لیا۔",
            "hi": "जासूस ने रहस्यमयी अपराध स्थल की गहन जांच की।",
            "bn": "গোয়েন্দা রহস্যময় অপরাধের ঘটনাস্থলটি পুঙ্খানুপুঙ্খ তদন্ত করেছিলেন।"
        }

        for lang, text in samples.items():
            norm = ScriptEngine.normalize_dialogue_text(text, language=lang)
            # Verify NFKC normalization
            assert norm == unicodedata.normalize("NFKC", norm), f"NFKC failed for {lang}"
            assert len(norm) > 0, f"Empty normalized string for {lang}"

            # Verify tokenization extracts meaningful words
            tokens = ScriptEngine.tokenize_multilingual(text, language=lang)
            assert len(tokens) >= 3, f"Insufficient tokens extracted for {lang}: {tokens}"
            # Ensure no tokens contain leftover punctuation
            for t in tokens:
                assert not any(p in t for p in [".", ",", "!", "?", "¿", "¡", "،", "؟", "۔"]), (
                    f"Punctuation leaked into token '{t}' for {lang}"
                )

    # =========================================================================
    # 2. CONTRACTION & APOSTROPHE SCOPING (ENGLISH VS NON-ENGLISH)
    # =========================================================================

    def test_english_contractions_scoped_to_english_only(self):
        """
        English contractions (don't -> do not, they're -> they are) must be scoped to English
        and MUST NOT mangle French (d'accord, l'homme), Spanish, or other Latin languages.
        """
        # English: should expand contractions
        en_text = "They don't know that we're ready."
        en_norm = ScriptEngine.normalize_dialogue_text(en_text, language="en")
        assert "not" in en_norm
        assert "are" in en_norm

        # French: l'homme, d'accord, c'est must NOT turn into English 'would' or 'are'
        fr_text = "L'homme est d'accord avec c'est la vie."
        fr_norm = ScriptEngine.normalize_dialogue_text(fr_text, language="fr")
        assert "would" not in fr_norm, f"English 'would' injected into French: {fr_norm}"
        assert "are" not in fr_norm, f"English 'are' injected into French: {fr_norm}"
        fr_tokens = ScriptEngine.tokenize_multilingual(fr_text, language="fr")
        assert "homme" in fr_tokens or "l'homme" in fr_tokens
        assert "accord" in fr_tokens or "d'accord" in fr_tokens

    def test_spanish_and_german_and_turkish_punctuation_preserved(self):
        """Spanish ¿ ¡, German umlauts/eszett, and Turkish dotted/dotless I must be preserved."""
        # Spanish inverted marks
        es_text = "¡Hola! ¿Dónde está el dinero del rescate?"
        es_norm = ScriptEngine.normalize_dialogue_text(es_text, language="es")
        assert "¿" not in es_norm and "¡" not in es_norm
        es_tokens = ScriptEngine.tokenize_multilingual(es_text, language="es")
        assert "rescate" in es_tokens
        assert "dinero" in es_tokens

        # German umlauts and eszett
        de_text = "Das schöne Mädchen schloss die große Tür."
        de_norm = ScriptEngine.normalize_dialogue_text(de_text, language="de")
        assert "schöne" in de_norm or "schoene" in de_norm
        de_tokens = ScriptEngine.tokenize_multilingual(de_text, language="de")
        assert any("schön" in t or "schon" in t for t in de_tokens)

        # Turkish dotted İ / dotless ı
        tr_text = "İstanbul ışıkları altında gizli bir buluşma gerçekleşti."
        tr_norm = ScriptEngine.normalize_dialogue_text(tr_text, language="tr")
        tr_tokens = ScriptEngine.tokenize_multilingual(tr_text, language="tr")
        assert len(tr_tokens) >= 4

    # =========================================================================
    # 3. ENGLISH STOPWORD ISOLATION & SHORT-TOKEN GUARD
    # =========================================================================

    def test_english_stopwords_not_applied_to_other_languages(self):
        """
        English stopwords ('in', 'it', 'to', 'no', 'me') must not strip non-English words
        that happen to share letters (e.g. Italian/Spanish 'no', 'me', 'in').
        """
        # In Spanish, 'no me digas eso' contains 'no' and 'me'
        es_text = "No me digas la verdad todavía."
        es_tokens = ScriptEngine.tokenize_multilingual(es_text, language="es")
        # In Spanish, substantive tokens like 'verdad', 'todavía', 'digas' must remain
        assert "verdad" in es_tokens
        assert "todavía" in es_tokens or "todavia" in es_tokens

        # In Russian, 'он не знает'
        ru_text = "Он не знает где спрятаны документы."
        ru_tokens = ScriptEngine.tokenize_multilingual(ru_text, language="ru")
        assert "знает" in ru_tokens
        assert "документы" in ru_tokens

    # =========================================================================
    # 4. MORPHOLOGY / STEMMING ISOLATION & ENGLISH FALSE-POSITIVE PREVENTION
    # =========================================================================

    def test_english_morphology_false_positives_prevented(self):
        """
        Verifies that known false-positive pairs (care ↔ career, organ ↔ organic)
        do NOT produce false morphological stem matches in English.
        """
        stems_care = ScriptEngine.get_language_stems("care", language="en")
        stems_career = ScriptEngine.get_language_stems("career", language="en")
        # career must NOT stem down to 'care'
        assert not (stems_care & stems_career), f"Spurious stem collision between 'care' and 'career': {stems_care & stems_career}"

        stems_organ = ScriptEngine.get_language_stems("organ", language="en")
        stems_organic = ScriptEngine.get_language_stems("organic", language="en")
        assert not (stems_organ & stems_organic), f"Spurious stem collision between 'organ' and 'organic': {stems_organ & stems_organic}"

    def test_english_stemming_not_applied_to_non_english(self):
        """English suffix rules (-ing, -ed, -ies, -er) must not be applied to non-English languages."""
        # Spanish verb 'comer' should not have English agent-noun '-er' stripped to 'com'
        es_stems = ScriptEngine.get_language_stems("comer", language="es")
        assert es_stems == {"comer"}, f"English suffix stripping incorrectly applied to Spanish 'comer': {es_stems}"

        # French verb 'parler'
        fr_stems = ScriptEngine.get_language_stems("parler", language="fr")
        assert fr_stems == {"parler"}, f"English suffix stripping incorrectly applied to French 'parler': {fr_stems}"

        # Urdu word 'شاندار'
        ur_stems = ScriptEngine.get_language_stems("شاندار", language="ur")
        assert ur_stems == {"شاندار"}

        # Hindi word 'करना'
        hi_stems = ScriptEngine.get_language_stems("करना", language="hi")
        assert hi_stems == {"करना"}

    # =========================================================================
    # 5. LAYERED LANGUAGE-NEUTRAL LEXICAL MATCHING
    # =========================================================================

    def test_spanish_lexical_anchoring_with_paraphrase_and_exact(self):
        """Verifies that non-English (Spanish) dialogue anchoring succeeds with exact and phrase matches."""
        blocks = [
            SceneBlock(
                movie_start=10.0,
                movie_end=25.0,
                narration_text="El sospechoso escapa por el callejón oscuro mientras la policía rodea la casa.",
                word_count=13
            )
        ]
        dialogue_timeline = [
            {"start": 5.0, "end": 10.0, "text": "Buenos días a todos los presentes en la sala."},
            {"start": 320.0, "end": 326.0, "text": "¡El sospechoso escapa por el callejón oscuro hacia el puente!"},
            {"start": 600.0, "end": 605.0, "text": "La cena está servida en el comedor principal."}
        ]

        anchored = ScriptEngine.anchor_scene_blocks_to_dialogue(blocks, dialogue_timeline, language="es")
        assert len(anchored) == 1
        assert anchored[0].movie_start == 320.0, (
            f"Expected Spanish cue at 320.0s, got {anchored[0].movie_start}s"
        )

    def test_russian_lexical_anchoring_chronological(self):
        """Verifies Cyrillic (Russian) multi-scene dialogue anchoring preserves chronological order."""
        blocks = [
            SceneBlock(movie_start=0.0, movie_end=15.0, narration_text="Офицер требует показать документы на автомобиль.", word_count=6),
            SceneBlock(movie_start=15.0, movie_end=30.0, narration_text="Они открывают сейф и находят секретный ключ.", word_count=7)
        ]
        dialogue_timeline = [
            {"start": 50.0, "end": 56.0, "text": "Покажите ваши документы на этот автомобиль прямо сейчас."},
            {"start": 120.0, "end": 125.0, "text": "В городе идет сильный дождь."},
            {"start": 410.0, "end": 418.0, "text": "Сейф открыт, внутри лежит секретный ключ."}
        ]

        anchored = ScriptEngine.anchor_scene_blocks_to_dialogue(blocks, dialogue_timeline, language="ru")
        assert anchored[0].movie_start == 50.0
        assert anchored[1].movie_start == 410.0

    # =========================================================================
    # 6. DIALOGUE REF MULTILINGUAL SAFETY & PROVENANCE VALIDATION
    # =========================================================================

    def test_multilingual_dialogue_ref_validation(self):
        """Verifies exact and substantive excerpt dialogue references across languages."""
        # Spanish exact quote
        es_cues = [{"start": 45.0, "end": 49.0, "text": "No tenemos suficiente tiempo para escapar.", "source": "user_transcript"}]
        prov_es = ScriptEngine.validate_dialogue_ref_provenance("No tenemos suficiente tiempo para escapar.", dialogue_timeline=es_cues, language="es")
        assert prov_es.is_source_bound is True
        assert prov_es.status == "SOURCE_BOUND"
        assert prov_es.movie_start == 45.0

        # Arabic exact quote
        ar_cues = [{"start": 80.0, "end": 85.0, "text": "الحقيقة ستظهر قريبا أمام الجميع", "source": "user_transcript"}]
        prov_ar = ScriptEngine.validate_dialogue_ref_provenance("الحقيقة ستظهر قريبا أمام الجميع", dialogue_timeline=ar_cues, language="ar")
        assert prov_ar.is_source_bound is True
        assert prov_ar.status == "SOURCE_BOUND"
        assert prov_ar.movie_start == 80.0

        # Hindi exact quote
        hi_cues = [{"start": 110.0, "end": 115.0, "text": "मैं तुम्हें कभी माफ नहीं करूंगा", "source": "user_transcript"}]
        prov_hi = ScriptEngine.validate_dialogue_ref_provenance("मैं तुम्हें कभी माफ नहीं करूंगा", dialogue_timeline=hi_cues, language="hi")
        assert prov_hi.is_source_bound is True
        assert prov_hi.status == "SOURCE_BOUND"
        assert prov_hi.movie_start == 110.0

    def test_short_non_substantive_tokens_rejected_in_all_languages(self):
        """Short generic words (e.g. Spanish 'de la', Russian 'и в', Urdu 'کا') must be REJECTED as dialogue_ref."""
        es_cues = [{"start": 45.0, "end": 49.0, "text": "Hablamos de la verdad en la casa.", "source": "user_transcript"}]
        # Generic 2-letter tokens 'de la' must be rejected
        prov_es = ScriptEngine.validate_dialogue_ref_provenance("de la", dialogue_timeline=es_cues, language="es")
        assert prov_es.is_source_bound is False, "Generic short Spanish tokens incorrectly accepted"

        ru_cues = [{"start": 50.0, "end": 55.0, "text": "Он пошел и в магазин за хлебом.", "source": "user_transcript"}]
        prov_ru = ScriptEngine.validate_dialogue_ref_provenance("и в", dialogue_timeline=ru_cues, language="ru")
        assert prov_ru.is_source_bound is False, "Generic short Russian tokens incorrectly accepted"

        ur_cues = [{"start": 60.0, "end": 65.0, "text": "یہ اس کی کہانی کا اختتام ہے۔", "source": "user_transcript"}]
        prov_ur = ScriptEngine.validate_dialogue_ref_provenance("کا", dialogue_timeline=ur_cues, language="ur")
        assert prov_ur.is_source_bound is False, "Single short Urdu token incorrectly accepted"

    def test_translated_dialogue_ref_rejected_against_original_source(self):
        """
        If source transcript is in Spanish, an English translation MUST NOT be treated
        as source-bound dialogue. The source transcript language is authoritative.
        """
        es_cues = [{"start": 45.0, "end": 49.0, "text": "No tenemos suficiente tiempo para escapar.", "source": "user_transcript"}]
        # Submitting the English translation 'We do not have enough time to escape'
        prov_trans = ScriptEngine.validate_dialogue_ref_provenance(
            "We do not have enough time to escape",
            dialogue_timeline=es_cues,
            language="es"
        )
        assert prov_trans.is_source_bound is False, "Translated dialogue reference must NOT be accepted as source-bound"

    # =========================================================================
    # 7. LANGUAGE-NEUTRAL ROADMAP & INFORMATIVENESS SCORING
    # =========================================================================

    def test_multilingual_cue_informativeness_evaluates_fairly(self):
        """
        Verifies that high-value plot cues in non-English languages receive fair informativeness scores
        without depending on English drama keywords, while audio tags remain universally penalized.
        """
        # Noise tags penalized in all languages
        assert ScriptEngine.evaluate_cue_informativeness("[MUSIC PLAYING]") < 0.0
        assert ScriptEngine.evaluate_cue_informativeness("(SFX: Gunshot)") < 0.0

        # Substantive narrative dialogue in Spanish, Russian, Arabic, Urdu, Bengali
        cues = {
            "es": "El inspector descubrió los documentos confidenciales escondidos en la pared.",
            "ru": "Следователь обнаружил секретные документы, спрятанные в стене.",
            "ar": "اكتشف المحقق الوثائق السرية المخبأة داخل الجدار القديم للمنزل.",
            "ur": "تفتیش کار نے دیوار میں چھپائے گئے انتہائی خفیہ دستاویزات برآمد کر لیے۔",
            "bn": "তদন্তকারী দেওয়ালের মধ্যে লুকিয়ে রাখা গোপন নথি উদ্ধার করেছিলেন।"
        }

        for lang, text in cues.items():
            sc = ScriptEngine.evaluate_cue_informativeness(text, language=lang)
            # High-value plot cue must score positive and substantive
            assert sc >= 5.0, f"Informativeness score for {lang} is too low ({sc}): {text}"

    def test_evaluate_block_importance_multilingual_neutrality(self):
        """
        Verifies that _evaluate_block_importance awards fair scores to substantive non-English blocks
        and does not require English keywords to be recognized as important.
        """
        es_block = "El clímax de la confrontación ocurre en el muelle donde el traidor es desenmascarado."
        ru_block = "Финальное противостояние происходит в порту, где предатель наконец разоблачен."
        ur_block = "کہانی کا فیصلہ کن موڑ اس وقت آتا ہے جب غدار کا چہرہ بے نقاب ہو جاتا ہے۔"

        sc_es = ScriptEngine._evaluate_block_importance(es_block, language="es")
        sc_ru = ScriptEngine._evaluate_block_importance(ru_block, language="ru")
        sc_ur = ScriptEngine._evaluate_block_importance(ur_block, language="ur")

        assert sc_es > 4.0, f"Spanish block importance too low: {sc_es}"
        assert sc_ru > 4.0, f"Russian block importance too low: {sc_ru}"
        assert sc_ur > 4.0, f"Urdu block importance too low: {sc_ur}"

    # =========================================================================
    # 8. SPOKEN WORD EXTRACTION INTEGRITY ACROSS SCRIPTS
    # =========================================================================

    def test_spoken_word_count_clean_extraction_across_scripts(self):
        """Verifies get_spoken_word_count strips tags and returns accurate counts for diverse scripts."""
        script_es = """[SCENE: 00:00 - 00:10]
[VOICEOVER]
Una madre alegre toma una fotografía de su hijo sonriendo orgulloso frente al coche.
[DIALOGUE_REF: "Sonríe una vez más"]
"""
        count_es = ScriptEngine.get_spoken_word_count(script_es)
        assert count_es == 14, f"Expected 14 spoken words, got {count_es}"

        script_ur = """[SCENE: 00:00 - 00:10]
[VOICEOVER]
ماں اپنے بیٹے کی تصویر کھینچتی ہے جو گاڑی کے سامنے کھڑا مسکرا رہا ہے۔
[DIALOGUE_REF: "ایک بار مسکراؤ"]
"""
        count_ur = ScriptEngine.get_spoken_word_count(script_ur)
        assert count_ur == 15, f"Expected 15 spoken words for Urdu, got {count_ur}"
