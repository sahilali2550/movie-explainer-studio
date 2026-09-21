import os
import re
import json
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from deep_translator import GoogleTranslator, MyMemoryTranslator
from app.core.config import SUPPORTED_LANGUAGES, STORY_PERSONAS


@dataclass
class SceneBlock:
    movie_start: float          # seconds into SOURCE movie (e.g. 185.0)
    movie_end: float            # seconds into SOURCE movie (e.g. 290.0)
    narration_text: str         # clean spoken narration for this scene
    word_count: int             # word count in narration
    speech_dur: float = 0.0     # actual allocated speech duration (seconds)
    narration_start: float = 0.0 # start timestamp in explainer voiceover
    narration_end: float = 0.0   # end timestamp in explainer voiceover


class ScriptEngine:
    """
    Multilingual Storyboard & Narrative Generator for Movie Explainers.
    Supports 15+ languages, 4 storytelling personas, spoiler toggles, and virality scoring.
    """

    @staticmethod
    def auto_detect_creative_context(
        title: str,
        description: str = "",
        transcript_sample: str = ""
    ) -> Dict[str, str]:
        """
        Analyzes movie title, description, and transcript text to auto-detect
        the ideal genre, storytelling tone, background music mood, and spoiler mode.
        Guarantees 1-click mistake-proof alignment for users.
        """
        combined = f"{title} {description} {transcript_sample}".lower()

        horror_kw = ["ghost", "demon", "haunted", "witch", "curse", "creepy", "monsters", "chucky", "jigsaw", "killer", "conjuring", "mansion", "attic", "blood", "خوفناک", "بھوت", "चुड़ैल", "भूत"]
        action_kw = ["mafia", "gangster", "heist", "police", "gun", "assassin", "agent", "spy", "thriller", "chase", "combat", "fight", "revenge", "explosion", "cartel", "hitman", "rounds", "گینگسٹر", "مافیا", "गैंगस्टर", "गोलियां"]
        romance_kw = ["love", "romance", "heartbreak", "crying", "dying", "illness", "tears", "husband", "wife", "marriage", "divorce", "lover", "tragic", "hospital", "corridor", "letter", "محبت", "عشق", "آنسو", "प्यार", "दर्द", "आंसू"]
        scifi_kw = ["space", "alien", "robot", "galaxy", "future", "simulation", "matrix", "planet", "time travel", "cyborg"]
        doc_kw = ["documentary", "historical documentary", "history", "ancient", "empire", "wwii", "world war", "civilization", "archaeological"]
        bio_kw = ["biography", "biopic", "life story", "autobiography"]
        crime_kw = ["true crime", "forensic", "serial killer", "unsolved mystery", "cold case"]
        tech_kw = ["case study", "silicon valley", "startup", "tech giant", "billion dollar"]

        # Action/Spy/Thriller in title or description takes priority for movies
        if any(kw in combined for kw in action_kw):
            return {
                "genre": "movie_recap",
                "persona": "hollywood_trailer",
                "mood": "tense",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in horror_kw):
            return {
                "genre": "movie_recap",
                "persona": "documentary",
                "mood": "suspense",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in doc_kw):
            return {
                "genre": "documentary",
                "persona": "documentary",
                "mood": "suspense",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in bio_kw):
            return {
                "genre": "biography",
                "persona": "documentary",
                "mood": "emotional",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in crime_kw):
            return {
                "genre": "true_crime",
                "persona": "documentary",
                "mood": "tense",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in tech_kw):
            return {
                "genre": "tech_science",
                "persona": "viral_fast",
                "mood": "upbeat",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in romance_kw):
            return {
                "genre": "movie_recap",
                "persona": "hollywood_trailer",
                "mood": "emotional",
                "spoiler_mode": "full_recap"
            }
        elif any(kw in combined for kw in scifi_kw):
            return {
                "genre": "movie_recap",
                "persona": "hollywood_trailer",
                "mood": "suspense",
                "spoiler_mode": "full_recap"
            }
        else:
            return {
                "genre": "movie_recap",
                "persona": "hollywood_trailer",
                "mood": "suspense",
                "spoiler_mode": "full_recap"
            }

    @staticmethod
    def detect_episode_info(title: str, description: str = "") -> Dict[str, Any]:
        """
        Detects if content is an episodic series / drama (e.g. Episode 01, Ep 14, قسط 5).
        Returns dict with is_episodic bool, current episode number, and next episode number.
        """
        combined = f"{title} {description}".lower()
        patterns = [
            r'(?:episode|ep|ep\.)\s*0*(\d+)',
            r'(?:قسط|قسمت|حلقہ)\s*(?:نمبر)?\s*0*(\d+)',
            r'\be0*(\d+)\b'
        ]
        for pat in patterns:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                try:
                    num = int(m.group(1))
                    return {
                        "is_episodic": True,
                        "episode_num": num,
                        "next_episode_num": num + 1
                    }
                except Exception:
                    pass
        return {
            "is_episodic": False,
            "episode_num": None,
            "next_episode_num": None
        }

    @staticmethod
    def strip_code_and_developer_artifacts(raw_text: str) -> str:
        """
        Strips markdown code blocks, python test functions (e.g. def test_word_count),
        assertions, reasoning tags (<think>...</think>), and developer comments
        inadvertently generated by reasoning/coding LLMs.
        Preserves spoken narration, [SCENE: MM:SS - MM:SS] brackets, and [SFX: ...] cues.
        """
        if not raw_text:
            return ""

        text = raw_text

        # 1. Strip <think>...</think> blocks
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)

        # 2. Strip markdown code fences (e.g. ```python ... ``` or ``` ... ```)
        text = re.sub(r'```[a-zA-Z0-9_-]*[\s\S]*?```', '', text)

        # 3. Strip raw python functions (def test_... or def check_... up to unindented text or EOF)
        text = re.sub(r'(?m)^def\s+[a-zA-Z0-9_]+\s*\(.*?\).*?:\s*\n(?:[ \t]+.*\n)*', '', text)

        # 4. Strip assertion lines, dev notes, and ponytail test harness tags
        text = re.sub(r'(?m)^[ \t]*assert\s+.*$', '', text)
        text = re.sub(r'(?m)^[ \t]*#\s*(?:ponytail|test|todo|ci|unit\s*test).*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^[ \t]*->\s*(?:skipped|note|test|full\s*testing).*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^[ \t]*return\s+(?:True|False|count).*$', '', text)

        # 5. Clean up redundant empty lines
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    @staticmethod
    def fetch_wikipedia_plot(movie_title: str) -> Optional[str]:
        """
        Fetches the movie plot summary from Wikipedia using the official MediaWiki API.
        Supports movie titles with or without year.
        """
        if not movie_title or not movie_title.strip():
            return None

        import urllib.parse
        clean_title = re.sub(r'[^\w\s-]', '', movie_title).strip()
        search_candidates = [
            f"{clean_title} (film)",
            clean_title,
            f"{clean_title} movie"
        ]

        for term in search_candidates:
            try:
                encoded = urllib.parse.quote(term)
                api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=0&explaintext=0&titles={encoded}&format=json"
                req = urllib.request.Request(api_url, headers={"User-Agent": "AutoExplainer/2.0 (movie-explainer-saas)"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        if pid == "-1":
                            continue
                        extract = pdata.get("extract", "")
                        if not extract:
                            continue

                        # Search for Plot section in HTML/plain extract
                        plot_match = re.search(r'<h3>(?:<span[^>]*>)?\s*Plot\s*(?:</span>)?</h3>\s*(.*?)(?=<h[23]|\Z)', extract, flags=re.DOTALL | re.IGNORECASE)
                        if not plot_match:
                            plot_match = re.search(r'<h2>(?:<span[^>]*>)?\s*Plot\s*(?:</span>)?</h2>\s*(.*?)(?=<h[23]|\Z)', extract, flags=re.DOTALL | re.IGNORECASE)
                        if not plot_match:
                            plot_match = re.search(r'===\s*Plot\s*===\s*(.*?)(?====|\Z)', extract, flags=re.DOTALL | re.IGNORECASE)

                        raw_plot = plot_match.group(1) if plot_match else extract
                        clean_plot = re.sub(r'<[^>]+>', ' ', raw_plot)
                        clean_plot = re.sub(r'\s+', ' ', clean_plot).strip()
                        if len(clean_plot) > 50:
                            return clean_plot[:4500]
            except Exception as e:
                continue

        return None

    @staticmethod
    def clamp_script_word_budget(
        script_text: str,
        target_duration_mins: int = 3,
        target_lang: str = "en",
        voice_speed: str = "fast"
    ) -> str:
        """
        Non-Destructive Narrative Auto-Budgeting.
        Clamps storyboard script length to strict duration-budget word ceilings
        WITHOUT severing the climax or final scenes.
        Pillars (Scene 1 Hook & Final Climax/Ending) are permanently protected.
        """
        if not script_text or not script_text.strip():
            return script_text

        target_words = ScriptEngine.calculate_target_words(target_duration_mins, voice_speed, target_lang)
        # Allow buffer up to +8%
        max_words = max(180, int(round(target_words * 1.08)))

        words = script_text.split()
        if len(words) <= max_words:
            return script_text

        # Split into scene blocks
        blocks = [b.strip() for b in re.split(r'\n\s*\n', script_text.strip()) if b.strip()]
        if len(blocks) > 2:
            # Pillar 1: First scene (Hook)
            pillar_first = blocks[0]
            # Pillar 2: Last scene (Climax / Ending)
            pillar_last = blocks[-1]

            p1_words = len(pillar_first.split())
            p2_words = len(pillar_last.split())

            middle_blocks = blocks[1:-1]
            remaining_budget = max(50, max_words - (p1_words + p2_words))

            # Total middle words
            mid_words = sum(len(b.split()) for b in middle_blocks)

            if mid_words <= remaining_budget:
                retained_middle = middle_blocks
            else:
                # Proportional compression of middle blocks
                compression_ratio = remaining_budget / max(1, mid_words)
                retained_middle = []
                for b in middle_blocks:
                    b_sentences = re.split(r'(?<=[.!?۔؟\n])\s+', b.strip())
                    if len(b_sentences) > 1:
                        target_s_count = max(1, int(round(len(b_sentences) * compression_ratio)))
                        compressed_b = " ".join(b_sentences[:target_s_count]).strip()
                        if not any(compressed_b.endswith(p) for p in [".", "۔", "!", "?", "؟"]):
                            compressed_b += "۔" if any(ord(c) > 1500 for c in compressed_b) else "."
                        retained_middle.append(compressed_b)
                    else:
                        retained_middle.append(b)

                # If still over budget, drop least critical intermediate blocks from the middle
                cur_total = p1_words + p2_words + sum(len(b.split()) for b in retained_middle)
                while cur_total > max_words and len(retained_middle) > 1:
                    drop_idx = len(retained_middle) // 2
                    retained_middle.pop(drop_idx)
                    cur_total = p1_words + p2_words + sum(len(b.split()) for b in retained_middle)

            final_blocks = [pillar_first] + retained_middle + [pillar_last]
            result = "\n\n".join(final_blocks).strip()
            if not any(result.endswith(p) for p in [".", "۔", "!", "?", "؟"]):
                result += "۔" if any(ord(c) > 1500 for c in result) else "."
            return result

        # Single block or only 2 blocks: clamp at sentence boundary while preserving first and last sentence
        sentences = re.split(r'(?<=[.!?۔؟\n])\s+', script_text.strip())
        if len(sentences) > 2:
            first_s = sentences[0]
            last_s = sentences[-1]
            rem_words = max_words - (len(first_s.split()) + len(last_s.split()))
            retained_mid = []
            cur_w = 0
            for s in sentences[1:-1]:
                s_w = len(s.split())
                if cur_w + s_w <= rem_words:
                    retained_mid.append(s)
                    cur_w += s_w
                else:
                    break
            res = " ".join([first_s] + retained_mid + [last_s]).strip()
            if not any(res.endswith(p) for p in [".", "۔", "!", "?", "؟"]):
                res += "۔" if any(ord(c) > 1500 for c in res) else "."
            return res

        return script_text

    @staticmethod
    def calculate_hook_score(script_text: str, lang: str = "en") -> Dict[str, Any]:
        """
        Evaluates the opening 5 seconds (first 25-40 words) for emotional hook,
        curiosity gap, high-stakes words, and viewer retention potential.
        """
        if not script_text:
            return {"score": 0, "rating": "Empty", "analysis": "No script provided."}

        first_passage = " ".join(script_text.strip().split()[:45]).lower()
        score = 65  # Base score

        # Universal curiosity and high-stakes markers
        trigger_keywords = [
            # English
            "secret", "killed", "mystery", "shock", "trap", "never", "nobody", "twisted", "died", "lie",
            # Urdu / Hindi
            "راز", "قتل", "دھوکہ", "خوفناک", "ہوش", "حیران", "خطرناک", "سازش", "انجام",
            "सच", "मौत", "धोखा", "रहस्य", "चौंकाने", "खतरनाक", "साजिश",
            # Spanish
            "secreto", "muerte", "peligro", "giro", "nadie", "aterrador", "trampa",
            # Indonesian
            "rahasia", "terjebak", "mati", "mengerikan", "misteri", "tak terduga",
            # Arabic
            "سر", "موت", "كارثة", "صدمة", "غامض", "مرعب", "خيانة"
        ]

        matches = [kw for kw in trigger_keywords if kw in first_passage]
        score += min(25, len(matches) * 7)

        # Sentence brevity bonus (punchy sentences retain attention)
        sentences = [s.strip() for s in re.split(r'[.!?۔\n]+', first_passage) if len(s.strip()) > 3]
        if len(sentences) >= 2:
            score += 8

        final_score = min(98, max(50, score))
        rating = "Viral Platinum 🔥" if final_score >= 88 else ("High Retention ⚡" if final_score >= 75 else "Moderate Pacing 📈")

        return {
            "score": final_score,
            "rating": rating,
            "analysis": f"Detected {len(matches)} curiosity triggers in the opening hook. High retention expected."
        }

    @staticmethod
    def parse_storyboard(raw_script: str) -> Tuple[str, List[Tuple[float, float]], List[str]]:
        """
        Extracts clean voiceover narration text, scene timestamp cuts (e.g. 01:23 - 02:45),
        and per-scene subtitle lines from the generated or edited storyboard.
        """
        if not raw_script:
            return "", [], []

        def time_to_sec(t: str) -> float:
            parts = t.strip().split(':')
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:
                pass
            return 0.0

        # 1. Parse timestamps
        ranges = []
        range_matches = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—to]+\s*(\d{1,2}:\d{2}(?::\d{2})?)', raw_script)
        for rm in range_matches:
            s_sec = time_to_sec(rm[0])
            e_sec = time_to_sec(rm[1])
            if e_sec > s_sec:
                ranges.append((s_sec, e_sec))

        # 2. Extract clean voiceover blocks
        vo_blocks = re.findall(r'\[VOICEOVER\]\s*(.*?)(?=\[SCENE:|===PART|📝|\Z)', raw_script, flags=re.DOTALL | re.IGNORECASE)
        clean_blocks = []
        scene_subs = []

        if vo_blocks:
            for b in vo_blocks:
                lines = [l.strip() for l in b.splitlines() if l.strip() and not l.strip().startswith(('#', '[', '📌', '🏷️', '📝', '---', '==='))]
                cleaned_b = ' '.join(lines)
                cleaned_b = re.sub(r'\[.*?\]', '', cleaned_b)
                cleaned_b = re.sub(r'\s+', ' ', cleaned_b).strip()
                if len(cleaned_b) > 5:
                    clean_blocks.append(cleaned_b)
                    scene_subs.append(cleaned_b[:80])
            final_text = ' '.join(clean_blocks)
        else:
            # Strip tags and markdown
            clean_lines = []
            for line in raw_script.splitlines():
                l = line.strip()
                if not l or l.startswith(('#', '[SCENE', '[TIME', '[BANNER', '===PART', 'Option:')):
                    continue
                # Strip leading numbering (e.g. '1. ') while keeping narration text
                l = re.sub(r'^\d+[\.\)]\s*', '', l)
                l = re.sub(r'\[.*?\]', '', l)
                l = re.sub(r'\*+', '', l)
                if len(l) > 8:
                    clean_lines.append(l)
            final_text = ' '.join(clean_lines)

        final_text = re.sub(r'\s+', ' ', final_text).strip()
        if not scene_subs and final_text:
            sentences = [s.strip() for s in re.split(r'[.!?۔।\n]+', final_text) if len(s.strip()) > 5]
            scene_subs = [s[:80] for s in sentences]

        return final_text, ranges, scene_subs

    @staticmethod
    def compress_transcript_to_roadmap(subs_text: str, total_movie_dur: float = 3600.0, max_points: int = 42) -> str:
        """
        Compresses full raw transcripts (SRT, VTT, or timestamped text) into an evenly spaced
        chronological event & dialogue roadmap spanning from opening (2%) to climax (92%).
        Prevents LLM truncation while ensuring full story arc visibility without token blowup.
        """
        if not subs_text or not subs_text.strip():
            return ""

        from app.services.video_engine import VideoEngine

        cues: List[Dict[str, Any]] = []
        try:
            # 1. Try parsing through VideoEngine's multi-format parser
            res = VideoEngine.parse_raw_transcript_text(subs_text)
            cues = res.get("dialogue_timeline", [])
        except Exception:
            cues = []

        # 2. Fallback regex line-by-line parsing if empty
        if not cues:
            ts_regex = re.compile(r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.*)')
            for line in subs_text.splitlines():
                l = line.strip()
                if not l:
                    continue
                m = ts_regex.match(l)
                if m:
                    ts_str = m.group(1)
                    txt = m.group(2).strip()
                    parts = ts_str.split(':')
                    sec = 0.0
                    if len(parts) == 2:
                        sec = int(parts[0]) * 60 + float(parts[1])
                    elif len(parts) == 3:
                        sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    if txt:
                        cues.append({"start": sec, "end": sec + 5.0, "text": txt})

        if not cues:
            return subs_text[:3500]

        cues.sort(key=lambda x: x["start"])

        # Determine movie time bounds
        max_cue_time = cues[-1]["start"]
        effective_movie_dur = max(float(total_movie_dur), max_cue_time + 10.0)

        # Cover from 2% of movie to 92% of movie (avoiding studio logos and end credits)
        start_bound = max(0.0, 0.02 * effective_movie_dur)
        end_bound = min(effective_movie_dur, 0.92 * effective_movie_dur)
        if end_bound <= start_bound:
            start_bound = cues[0]["start"]
            end_bound = max_cue_time

        span = max(1.0, end_bound - start_bound)
        num_buckets = min(max_points, len(cues))
        step = span / max(1, num_buckets)

        selected_cues = []
        for i in range(num_buckets):
            b_start = start_bound + i * step
            b_end = b_start + step
            bucket_cues = [c for c in cues if b_start <= c["start"] < b_end]
            if bucket_cues:
                best_cue = max(bucket_cues, key=lambda x: len(x.get("text", "")))
                selected_cues.append(best_cue)
            else:
                target_mid = b_start + step / 2.0
                closest_cue = min(cues, key=lambda x: abs(x["start"] - target_mid))
                if closest_cue not in selected_cues:
                    selected_cues.append(closest_cue)

        selected_cues.sort(key=lambda x: x["start"])

        # Deduplicate while preserving order
        unique_cues = []
        seen_starts = set()
        for c in selected_cues:
            k = round(c["start"], 1)
            if k not in seen_starts:
                seen_starts.add(k)
                unique_cues.append(c)

        lines = []
        for c in unique_cues:
            sec = c["start"]
            m = int(sec // 60)
            s = int(sec % 60)
            text_snippet = c["text"].replace("\n", " ").strip()
            text_snippet = re.sub(r'<[^>]+>', '', text_snippet)
            if len(text_snippet) > 85:
                text_snippet = text_snippet[:82] + "..."
            lines.append(f"[{m:02d}:{s:02d}] {text_snippet}")

        return "\n".join(lines)

    @staticmethod
    def parse_storyboard_blocks(
        raw_script: str,
        dialogue_timeline: Optional[List[Dict[str, Any]]] = None
    ) -> List[SceneBlock]:
        """
        Parses structured SceneBlock objects pairing each scene timestamp cut [SCENE: MM:SS - MM:SS]
        with its clean narration text and word count.

        T-03 Extension: If dialogue_timeline is supplied (list of {start, end, text} cues from
        the source movie transcript), anchor_scenes_to_dialogue() is called automatically after
        parsing to replace AI-invented timestamps with real movie timestamps.
        Omit dialogue_timeline for backward-compatible behavior.
        """
        if not raw_script:
            return []

        def time_to_sec(t: str) -> float:
            parts = t.strip().split(':')
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except Exception:
                pass
            return 0.0

        def clean_narration_chunk(txt: str) -> str:
            vo_m = re.search(r'\[VOICEOVER\]\s*(.*?)(?=\[SCENE:|===PART|📝|\Z)', txt, flags=re.DOTALL | re.IGNORECASE)
            target = vo_m.group(1) if vo_m else txt

            lines = []
            for l in target.splitlines():
                line = l.strip()
                if not line or line.startswith(('#', '[', '📌', '🏷️', '📝', '---', '===', 'Option:')):
                    continue
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'\[.*?\]', '', line)
                line = re.sub(r'\*+', '', line)
                line = re.sub(r'^(?:voiceover|narration)\s*:\s*', '', line, flags=re.IGNORECASE)
                if len(line) > 3:
                    lines.append(line)
            cleaned = ' '.join(lines)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned

        ts_header_pattern = re.compile(
            r'(?:\[SCENE:\s*|\bSCENE\s*\d*:\s*|\[)?(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—to]+\s*(\d{1,2}:\d{2}(?::\d{2})?)\]?',
            re.IGNORECASE
        )

        matches = list(ts_header_pattern.finditer(raw_script))
        blocks: List[SceneBlock] = []

        if matches:
            for idx, m in enumerate(matches):
                s_sec = time_to_sec(m.group(1))
                e_sec = time_to_sec(m.group(2))
                if e_sec <= s_sec:
                    e_sec = s_sec + 5.0

                start_pos = m.end()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_script)
                chunk = raw_script[start_pos:end_pos]
                cleaned = clean_narration_chunk(chunk)
                if not cleaned and idx == 0:
                    cleaned = clean_narration_chunk(raw_script[:m.start()])

                w_count = len(cleaned.split()) if cleaned else 0
                if cleaned and w_count > 0:
                    blocks.append(SceneBlock(
                        movie_start=s_sec,
                        movie_end=e_sec,
                        narration_text=cleaned,
                        word_count=w_count
                    ))

        if not blocks:
            final_text, ranges, _ = ScriptEngine.parse_storyboard(raw_script)
            if final_text:
                sentences = [s.strip() for s in re.split(r'[.!?۔।\n]+', final_text) if len(s.strip()) > 5]
                if not ranges:
                    ranges = [(0.0, 60.0)]

                step = len(sentences) / max(1, len(ranges))
                for r_idx, (r_s, r_e) in enumerate(ranges):
                    start_s_idx = int(r_idx * step)
                    end_s_idx = int((r_idx + 1) * step) if r_idx < len(ranges) - 1 else len(sentences)
                    scene_text = " ".join(sentences[start_s_idx:end_s_idx])
                    if not scene_text and sentences:
                        scene_text = sentences[min(r_idx, len(sentences) - 1)]
                    wc = len(scene_text.split())
                    blocks.append(SceneBlock(
                        movie_start=r_s,
                        movie_end=r_e,
                        narration_text=scene_text,
                        word_count=max(1, wc)
                    ))

        # T-03: Auto-anchor to real dialogue timestamps when timeline is provided
        if dialogue_timeline:
            blocks = ScriptEngine.anchor_scenes_to_dialogue(blocks, dialogue_timeline)

        return blocks

    @staticmethod
    def anchor_scenes_to_dialogue(
        blocks: List["SceneBlock"],
        dialogue_timeline: List[Dict[str, Any]],
        min_scene_dur: float = 5.0,
        max_scene_dur: float = 90.0
    ) -> List["SceneBlock"]:
        """
        Dialogue-Anchor Algorithm (T-01).

        Replaces AI-invented [SCENE: MM:SS] timestamps in each SceneBlock with
        REAL timestamps from the source movie's dialogue_timeline, ensuring that
        the video clip cut corresponds to the actual movie moment being narrated.

        Strategy (keyword overlap — no external deps):
          1. Tokenise each block's narration_text into a set of meaningful words.
          2. For every dialogue_timeline cue, count the number of shared tokens with
             the block's narration.
          3. The cue with the highest overlap score becomes the anchor.
          4. Chronological integrity: each block's anchor must be >= previous block's
             anchor (stable sort).
          5. Graceful fallback: if no useful match is found (score == 0), the block
             keeps its original AI timestamp — never silently dropped.

        Args:
            blocks:            SceneBlocks parsed from the AI-generated script.
            dialogue_timeline: [{start, end, text}, ...] from parse_raw_transcript_text().
            min_scene_dur:     Minimum scene window in seconds (default 5.0s).
            max_scene_dur:     Maximum scene window in seconds (default 90.0s).

        Returns:
            Same list of SceneBlocks with movie_start / movie_end updated to real
            timestamps where a confident match was found.
        """
        if not blocks:
            return []
        if not dialogue_timeline:
            return blocks  # graceful fallback: keep AI timestamps

        # --- Stop-word filter (language-agnostic common words to ignore) ---
        STOP_WORDS = {
            "a", "an", "the", "is", "in", "it", "of", "to", "and", "or",
            "on", "at", "by", "as", "be", "we", "he", "she", "his", "her",
            "was", "are", "this", "that", "with", "for", "from", "not",
            "but", "so", "if", "its", "into", "up", "out", "now", "then",
            "were", "have", "has", "had", "would", "could", "will", "do",
        }

        def tokenize(text: str) -> set:
            """Unicode alpha tokens for English, Urdu, Hindi, Arabic, etc., dropping stop-words and pure digits."""
            tokens = re.findall(r"\w{2,}", text.lower(), flags=re.UNICODE)
            return {t for t in tokens if t not in STOP_WORDS and not t.isdigit()}

        def score_match(block_tokens: set, cue_text: str) -> int:
            """Token overlap count between block narration and cue text."""
            cue_tokens = tokenize(cue_text)
            if not cue_tokens:
                return 0
            return len(block_tokens & cue_tokens)

        # Pre-tokenise all blocks once
        block_token_sets = [tokenize(b.narration_text) for b in blocks]

        # Find best-matching cue for each block (greedy, forward-only)
        anchors: List[float] = []  # matched movie_start for each block
        prev_anchor = 0.0

        for b_idx, block in enumerate(blocks):
            b_tokens = block_token_sets[b_idx]

            if not b_tokens:
                # No meaningful tokens → keep original timestamp
                anchors.append(block.movie_start)
                continue

            best_score = 0
            best_cue_start = None

            for cue in dialogue_timeline:
                cue_start = float(cue.get("start", 0.0))
                # Only consider cues AFTER previous anchor (chronological lock)
                if cue_start < prev_anchor:
                    continue
                sc = score_match(b_tokens, cue.get("text", ""))
                if sc > best_score:
                    best_score = sc
                    best_cue_start = cue_start

            if best_score > 0 and best_cue_start is not None:
                anchor_start = max(prev_anchor, best_cue_start)
                anchors.append(anchor_start)
                prev_anchor = anchor_start
            else:
                # No match → keep original AI timestamp but respect chronological order
                fallback = max(prev_anchor, block.movie_start)
                anchors.append(fallback)
                prev_anchor = fallback

        # Apply anchors back to SceneBlocks
        for b_idx, block in enumerate(blocks):
            new_start = anchors[b_idx]
            # Compute window: preserve original span but clamp to [min, max]
            original_span = max(min_scene_dur, block.movie_end - block.movie_start)
            clamped_span = min(original_span, max_scene_dur)
            new_end = round(new_start + clamped_span, 2)
            block.movie_start = round(new_start, 2)
            block.movie_end = new_end

        return blocks

    @staticmethod
    def assign_narration_timing(
        blocks: List[SceneBlock],
        total_speech_dur: float,
        cues: Optional[List[Dict[str, Any]]] = None
    ) -> List[SceneBlock]:
        """
        Locks each SceneBlock to its exact spoken duration and start/end time.
        Level 1 (Authoritative): Matches against Edge-TTS sentence boundary cues (_cues.json).
        Level 2 (Proportional Fallback): Allocates duration based on word-count weighting.
        Enforces mathematical invariant: sum(block.speech_dur) == total_speech_dur.
        """
        if not blocks:
            return []

        total_speech_dur = max(1.0, float(total_speech_dur))

        if len(blocks) == 1:
            blocks[0].speech_dur = round(total_speech_dur, 3)
            blocks[0].narration_start = 0.0
            blocks[0].narration_end = round(total_speech_dur, 3)
            return blocks

        used_cues = False
        if cues and len(cues) > 0:
            try:
                def norm(t: str) -> str:
                    return re.sub(r'[^\w\s]', '', t.lower()).strip()

                cue_idx = 0
                block_boundaries: List[Tuple[float, float]] = []

                for b_idx, b in enumerate(blocks):
                    b_words = norm(b.narration_text).split()
                    start_t = cues[min(cue_idx, len(cues) - 1)]["start"] if b_idx > 0 else 0.0

                    matched_words = 0
                    target_words = max(1, len(b_words))
                    last_end = start_t

                    while cue_idx < len(cues):
                        c = cues[cue_idx]
                        c_words = norm(c.get("text", "")).split()
                        matched_words += len(c_words)
                        last_end = c.get("end", last_end)
                        cue_idx += 1

                        if b_idx == len(blocks) - 1:
                            if cue_idx < len(cues):
                                continue
                        if matched_words >= target_words * 0.85:
                            break

                    block_boundaries.append((start_t, last_end))

                if len(block_boundaries) == len(blocks):
                    curr_t = 0.0
                    for i, b in enumerate(blocks):
                        s_t, e_t = block_boundaries[i]
                        b.narration_start = round(curr_t, 3)
                        remaining_blocks = len(blocks) - 1 - i
                        max_e_t = max(curr_t + 1.0, total_speech_dur - remaining_blocks * 1.0)
                        e_t = max(curr_t + 1.0, min(e_t, max_e_t))
                        if i == len(blocks) - 1:
                            e_t = max(e_t, total_speech_dur)
                        b.narration_end = round(e_t, 3)
                        b.speech_dur = round(b.narration_end - b.narration_start, 3)
                        curr_t = b.narration_end

                    blocks[-1].narration_end = round(total_speech_dur, 3)
                    blocks[-1].speech_dur = round(blocks[-1].narration_end - blocks[-1].narration_start, 3)
                    used_cues = True
            except Exception as ce:
                print(f"[Assign Narration Timing Notice] Cue match fallback: {ce}")
                used_cues = False

        if not used_cues:
            total_words = sum(max(1, b.word_count) for b in blocks)
            curr_t = 0.0
            for i, b in enumerate(blocks):
                w = max(1, b.word_count)
                dur = (w / total_words) * total_speech_dur
                b.speech_dur = round(dur, 3)
                b.narration_start = round(curr_t, 3)
                curr_t += dur
                b.narration_end = round(curr_t, 3)

            diff = total_speech_dur - blocks[-1].narration_end
            blocks[-1].narration_end = round(total_speech_dur, 3)
            blocks[-1].speech_dur = round(blocks[-1].narration_end - blocks[-1].narration_start, 3)

        return blocks

    @staticmethod
    def extract_sfx_cues(raw_script: str, total_duration: float = 60.0) -> List[Dict[str, Any]]:
        """
        Extracts emotional [SFX: ...] tags from the script and maps them to chronological timestamps.
        Supported tags: [SFX: HEARTBEAT], [SFX: SUB_BOOM], [SFX: WHOOSH], [SFX: CLOCK_TICK], [SFX: CASH_CHIME].
        """
        if not raw_script:
            return []

        alias_map = {
            "heartbeat": "heartbeat", "heart_beat": "heartbeat", "pulse": "heartbeat",
            "sub_boom": "sub_boom", "boom": "sub_boom", "impact": "sub_boom", "explosion": "sub_boom",
            "whoosh": "whoosh", "transition": "whoosh", "swoosh": "whoosh",
            "clock_tick": "clock_tick", "tick": "clock_tick", "countdown": "clock_tick",
            "cash_chime": "cash_chime", "chime": "cash_chime", "ding": "cash_chime",
            "riser": "sub_boom", "dramatic_riser": "sub_boom", "suspense": "heartbeat"
        }

        volume_map = {
            "heartbeat": 0.45,
            "sub_boom": 0.55,
            "whoosh": 0.40,
            "clock_tick": 0.35,
            "cash_chime": 0.40
        }

        cues = []
        clean_len = max(1, len(raw_script))
        dur = max(5.0, total_duration)

        matches = list(re.finditer(r'\[SFX:\s*([a-zA-Z0-9_\s-]+)\]', raw_script, re.IGNORECASE))
        for m in matches:
            tag_raw = m.group(1).strip().lower().replace(" ", "_")
            canonical = alias_map.get(tag_raw, "whoosh")
            vol = volume_map.get(canonical, 0.40)

            # Position relative to entire script text
            char_pos = m.start()
            t_sec = (char_pos / clean_len) * dur
            t_sec = max(0.5, min(dur - 2.0, t_sec))
            cues.append({
                "time": round(t_sec, 2),
                "sfx": canonical,
                "volume": vol
            })

        cues.sort(key=lambda x: x["time"])
        return cues

    LANGUAGE_WPM = {
        "ur": 132,  # Urdu (Edge-TTS Asad/Uzma speaks ~130-135 WPM at 1.0x)
        "hi": 138,  # Hindi (Edge-TTS Madhur/Swara speaks ~135-140 WPM at 1.0x)
        "es": 165,  # Spanish (Edge-TTS Alvaro/Elvira speaks ~160-170 WPM)
        "pt": 160,  # Portuguese (Edge-TTS Antonio/Francisca ~155-165 WPM)
        "id": 155,  # Indonesian (Edge-TTS Ardi/Gadis ~150-160 WPM)
        "vi": 160,  # Vietnamese (Edge-TTS NamMinh/HoaiMy ~155-165 WPM)
        "en": 150,  # English (Edge-TTS Christopher/Guy/Aria ~145-155 WPM)
        "fr": 150,  # French (Edge-TTS Henri/Denise ~145-155 WPM)
        "it": 155,  # Italian (Edge-TTS Diego/Elsa ~150-160 WPM)
        "tr": 150,  # Turkish (Edge-TTS Ahmet/Emel ~145-155 WPM)
        "de": 135,  # German (Edge-TTS Conrad/Katja ~130-140 WPM)
        "ar": 130,  # Arabic (Edge-TTS Shakir/Hamed ~125-135 WPM)
        "ru": 130,  # Russian (Edge-TTS Dmitry/Svetlana ~125-135 WPM)
        "th": 155,  # Thai (Edge-TTS Niwat/Premwadee ~150-160 WPM)
        "ja": 280,  # Japanese characters/min (Edge-TTS Keita/Nanami)
        "ko": 200,  # Korean blocks/min (Edge-TTS InJoon/SunHi)
    }

    SPEED_MULTIPLIER = {
        "normal": 1.0,
        "fast": 1.15,
        "ultra_fast": 1.25,
    }

    @staticmethod
    def calculate_target_words(duration_mins: int, voice_speed: str = "fast", target_lang: str = "en") -> int:
        """Calculates spoken target word count dynamically based on duration, language pace, and speed."""
        base_wpm = ScriptEngine.LANGUAGE_WPM.get(target_lang, 150)
        mult = ScriptEngine.SPEED_MULTIPLIER.get(voice_speed, 1.15)
        return max(130, int(round(duration_mins * base_wpm * mult)))

    @staticmethod
    def calculate_dynamic_pacing(source_duration_sec: float) -> Dict[str, Any]:
        """
        Feature #2: Dynamic Pacing & Scene Segmentation Formula.
        Calculates optimal explainer duration and scene block count based on source length.
        """
        sec = float(source_duration_sec) if source_duration_sec and source_duration_sec > 0 else 3600.0
        
        if sec < 900:  # Short Video (< 15 mins)
            target_duration_mins = 3
            target_scenes = 5
            category = "short"
        elif sec <= 2700:  # Medium / Drama (15 to 45 mins)
            target_duration_mins = 5
            target_scenes = 7
            category = "medium"
        else:  # Full Movie (1 to 3+ hours)
            target_duration_mins = 8
            target_scenes = 11
            category = "feature_film"

        return {
            "source_duration_sec": sec,
            "target_duration_mins": target_duration_mins,
            "target_scenes": target_scenes,
            "category": category,
            "rule_description": f"{category.upper()}: {target_duration_mins} mins explainer across ~{target_scenes} scenes"
        }

    @staticmethod
    def partition_timeline(source_duration_sec: float, target_duration_mins: int = 10) -> List[Dict[str, Any]]:
        """
        Universal 5-Act Timeline Milestone Partitioner.
        Proportionately divides source video duration (whether 40m or 180m) into 5 chronological acts
        to ensure full 0% to 100% movie coverage with clear timestamp brackets and word budgets.
        """
        total_sec = float(source_duration_sec) if source_duration_sec and source_duration_sec > 60 else float(target_duration_mins * 60 * 6)

        def sec_to_ts(s: float) -> str:
            m = int(s // 60)
            sec = int(s % 60)
            return f"{m:02d}:{sec:02d}"

        acts_def = [
            ("Act 1", "Opening Hook & Inciting Incident", 0.0, 0.20, "20%"),
            ("Act 2A", "Rising Stakes & Early Conflict", 0.20, 0.45, "25%"),
            ("Act 2B", "Midpoint Twist & Deep Crisis", 0.45, 0.70, "25%"),
            ("Act 3", "The Climax & Killer Reveal / Final Confrontation", 0.70, 0.90, "20%"),
            ("Epilogue", "Resolution, Character Fate & Short Review", 0.90, 1.00, "10%"),
        ]

        milestones = []
        for act_name, label, start_ratio, end_ratio, budget_pct in acts_def:
            start_s = round(total_sec * start_ratio)
            end_s = total_sec if end_ratio == 1.00 else round(total_sec * end_ratio)
            milestones.append({
                "act": act_name,
                "label": label,
                "start_sec": start_s,
                "end_sec": end_s,
                "timestamp_range": f"{sec_to_ts(start_s)} - {sec_to_ts(end_s)}",
                "budget_pct": budget_pct
            })
        return milestones

    @staticmethod
    def validate_script_integrity(
        script_text: str,
        source_duration_sec: float = 0,
        target_duration_mins: int = 10,
        target_lang: str = "en",
        voice_speed: str = "fast"
    ) -> Tuple[bool, str]:
        """
        Universal Automated Quality Gatekeeper.
        Verifies:
        1. Timeline Coverage: Last scene timestamp covers >= 85% of source video timeline.
        2. Word Budget Drift: Actual words are within acceptable range (not severely under-budget).
        3. Structural Pillars: Valid scene format and narrative closure.
        """
        if not script_text or not script_text.strip():
            return False, "Script is empty."

        # 1. Timeline Coverage Check
        ts_matches = re.findall(r'(?:\[(?:SCENE:\s*)?|\b)(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]?', script_text)
        max_end_sec = 0.0
        if ts_matches:
            for m in ts_matches:
                end_m, end_s = int(m[2]), int(m[3])
                end_sec = end_m * 60 + end_s
                if end_sec > max_end_sec:
                    max_end_sec = end_sec

        if source_duration_sec and source_duration_sec > 120:
            if max_end_sec == 0:
                return False, "Missing scene timestamps across narrative."
            coverage_pct = max_end_sec / source_duration_sec
            if coverage_pct < 0.85:
                return False, f"Timeline coverage insufficient (covers {coverage_pct*100:.1f}%, minimum required 85%). Missing ending/climax."

        # 2. Word Budget Check (Measured on clean spoken narration)
        clean_narr, _, _ = ScriptEngine.parse_storyboard(script_text)
        spoken_words = len(clean_narr.split()) if clean_narr else len(script_text.split())
        target_words = ScriptEngine.calculate_target_words(target_duration_mins, voice_speed, target_lang)
        min_allowed = int(target_words * 0.55) if target_duration_mins >= 5 else int(target_words * 0.50)
        if spoken_words < min_allowed:
            return False, f"Script is severely under-budget ({spoken_words} spoken words, minimum expected {min_allowed} for {target_duration_mins}m video)."

        return True, "Script passed all universal integrity gates."

    @staticmethod
    def build_prompt_for_genre(
        genre: str,
        title: str,
        description: str,
        subs_text: str,
        target_lang: str = "en",
        duration_mins: int = 5,
        plot_summary: str = "",
        persona: str = "hollywood_trailer",
        mood: str = "suspense",
        spoiler_mode: str = "full_recap",
        voice_speed: str = "fast",
        source_video_duration_sec: float = 0,
        story_beats: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Builds tailored, multi-act prompt blueprint for universal video genres with strict length quotas."""
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
        lang_name = lang_info["name"]
        target_words = ScriptEngine.calculate_target_words(duration_mins, voice_speed, target_lang)

        act1_words = int(round(target_words * 0.25))
        act2_words = int(round(target_words * 0.50))
        act3_words = int(round(target_words * 0.25))

        source_pacing_note = ""
        if source_video_duration_sec and source_video_duration_sec > 60:
            s_mins = round(source_video_duration_sec / 60, 1)
            ratio = round(s_mins / max(1, duration_mins), 1)
            source_pacing_note = f"\nSource Video Total Duration: {s_mins} minutes (Compression Ratio: {ratio}x).\nCover the entire narrative from opening events to the final climax proportionally."

        genre_configs = {
            "biography": {
                "role": f"World-Class Biographical Storyteller & Documentarian in {lang_name} ({lang_info['native']})",
                "label": "Biography & Real Story",
                "structure": f"""- Opening Hook (First 5 seconds): The dramatic defining moment, paradox, or highest achievement of this icon.
- Act 1 (~{act1_words} words): Early Life, Humble Beginnings, formative struggles, and their initial spark.
- Act 2 (~{act2_words} words): The Breakthrough, relentless grit, pivotal turning points, failures, and astronomical rise.
- Act 3 (~{act3_words} words): Enduring Legacy, life-defining lessons, and their eternal mark on history.
- Call to Action: Punchy closing inspiring viewers to like and subscribe for more legendary true stories!"""
            },
            "documentary": {
                "role": f"Investigative Crime & Historical Documentary Chronicler in {lang_name} ({lang_info['native']})",
                "label": "Investigative / True Crime / Historical Documentary",
                "structure": f"""- Opening Hook (First 5 seconds): The terrifying anomaly, unsolved crime, or historical turning point.
- Act 1 (~{act1_words} words): The Inciting Mystery, initial shock, discovery of evidence, and early investigations.
- Act 2 (~{act2_words} words): Deep Forensic Breakdown, conflicting testimonies, cover-ups, and escalating theories.
- Act 3 (~{act3_words} words): The Breakthrough, hard facts, lingering mysteries, and societal aftermath.
- Call to Action: Thought-provoking outro urging viewers to comment their theories and subscribe!"""
            },
            "true_crime": {
                "role": f"Investigative True Crime & Forensic Documentarian in {lang_name} ({lang_info['native']})",
                "label": "True Crime & Forensic Investigation",
                "structure": f"""- Opening Hook (First 5 seconds): The chilling crime scene, shocking disappearance, or forensic contradiction.
- Act 1 (~{act1_words} words): The Incident & First 48 Hours, timeline of disappearance/crime, and initial suspects.
- Act 2 (~{act2_words} words): Cold Case Reopened & Forensic Clues, digital forensics, DNA revelations, and interrogations.
- Act 3 (~{act3_words} words): The Trial, Final Confession / Verdict, and unsolved questions.
- Call to Action: Gripping outro inviting viewers to discuss theories in comments!"""
            },
            "tech_science": {
                "role": f"Visionary Tech & Science Chronicler in {lang_name} ({lang_info['native']})",
                "label": "Science, Technology & Future Innovation",
                "structure": f"""- Opening Hook (First 5 seconds): The mind-bending breakthrough, paradigm shift, or existential threat.
- Act 1 (~{act1_words} words): The Problem / Origin Story, why previous paradigms failed, and the breakthrough idea.
- Act 2 (~{act2_words} words): How It Works & Engineering Genius, the breakthroughs, obstacles, and revolutionary tech.
- Act 3 (~{act3_words} words): Future Impact & Society, ethical questions, what happens next.
- Call to Action: Exciting closing inviting viewers to share what tech they want explained next!"""
            },
            "video_essay": {
                "role": f"Master Cultural Critic & Visual Essayist in {lang_name} ({lang_info['native']})",
                "label": "Cultural Video Essay & Philosophy",
                "structure": f"""- Opening Hook (First 5 seconds): The central paradox or cultural critique that challenges common beliefs.
- Act 1 (~{act1_words} words): The Thesis & Cultural Context, breaking down the illusion.
- Act 2 (~{act2_words} words): The Evidence, cinematic/historical parallels, psychology, and hidden motifs.
- Act 3 (~{act3_words} words): The Synthesis, why this matters today, and philosophical takeaway.
- Call to Action: Thoughtful closing question to spark high-engagement comments!"""
            },
            "movie_recap": {
                "role": f"Master Hollywood Cinema Storyteller & Explainer in {lang_name} ({lang_info['native']})",
                "label": "Cinematic Movie Story Recap",
                "structure": f"""- Opening Hook (First 5 seconds): The high-stakes inciting incident or intense teaser moment.
- Act 1 (~{act1_words} words): Protagonist introduction, world setup, the inciting event, and initial stakes.
- Act 2 (~{act2_words} words): The Escalation, rising tension, plot twists, betrayals, and deep crisis.
- Act 3 (~{act3_words} words): The Climax, ultimate confrontation, killer/mastermind reveal, and full resolution.
- Short Review & Outro: A 15-20 second review & moral takeaway followed by like & subscribe call-to-action!"""
            }
        }

        cfg = genre_configs.get(genre, genre_configs["movie_recap"])

        persona_map = {
            "hollywood_trailer": "Epic, dramatic, cinematic, high-stakes with breathless pacing.",
            "viral_fast": "Hyper-fast, punchy, high-energy, modern TikTok/Reels retention style.",
            "sarcastic_roaster": "Witty, humorous, sarcastic commentary pointing out absurd plot choices.",
            "documentary": "Serious, objective, chilling, investigative tone like true crime docuseries."
        }
        persona_guide = persona_map.get(persona, persona_map["hollywood_trailer"])

        spoiler_rule = "Deliver the complete ending, final plot twists, and character fates clearly without withholding information." if spoiler_mode == "full_recap" else "Build maximum suspense up to the final cliffhanger without revealing the ultimate ending!"

        beats_section = ""
        if story_beats:
            beats_lines = ["\nCHRONOLOGICAL STORY BEATS DETECTED BY DETECTIVE AGENT:"]
            for b in story_beats:
                beat_num = b.get("beat", "")
                ts = b.get("time_range") or b.get("timestamp", "")
                btitle = b.get("title", "")
                action = b.get("action", "")
                line = f"- Beat {beat_num} [{ts}]: {btitle}"
                if action:
                    line += f" - {action}"
                beats_lines.append(line)
            beats_section = "\n".join(beats_lines) + "\n"

        # Compress transcript into full-movie roadmap if available (prevents truncation)
        if subs_text and subs_text.strip():
            roadmap = ScriptEngine.compress_transcript_to_roadmap(
                subs_text=subs_text,
                total_movie_dur=float(source_video_duration_sec or (duration_mins * 60.0 * 6.0)),
                max_points=42
            )
            transcript_section = f"Full-Movie Dialogue & Event Roadmap (Beginning to Climax):\n{roadmap}" if roadmap else f"Source Transcript Highlights:\n{subs_text[:4000]}"
        else:
            transcript_section = "Source Transcript: None provided. Structure narrative from beginning to climax based on plot guide."

        milestones = ScriptEngine.partition_timeline(source_video_duration_sec, duration_mins)
        milestone_lines = ["\nMANDATORY 5-ACT TIMELINE PROGRESSION (FULL MOVIE COVERAGE REQUIRED):"]
        milestone_lines.append("You MUST structure your narrative across these chronological acts and ensure your timestamps reach the final climax and ending:")
        for m in milestones:
            milestone_lines.append(f"- {m['act']} [{m['timestamp_range']}]: {m['label']} (Target Word Budget: ~{m['budget_pct']})")
        milestone_lines.append(f"CRITICAL MILESTONE LOCK: The final scene MUST reach the Epilogue timeframe ({milestones[-1]['timestamp_range']}) to deliver the full climax and resolution!")
        milestone_section = "\n".join(milestone_lines) + "\n"

        ep_info = ScriptEngine.detect_episode_info(title, description)
        episodic_instructions = ""
        if ep_info.get("is_episodic"):
            c_num = ep_info["episode_num"]
            n_num = ep_info["next_episode_num"]
            episodic_instructions = f"""
8. MANDATORY EPISODIC CLIFFHANGER & NEXT EPISODE CTA:
   - This video covers Episode {c_num}!
   - Establish character stakes and interpersonal conflicts clearly in Act 1.
   - Act 3 MUST end with an intense cliffhanger on the major unanswered dramatic question of this episode!
   - Your closing Call-To-Action (CTA) in {lang_name} MUST explicitly instruct viewers to watch Episode {n_num} on the channel:
     "To find out what happens next, watch Episode {n_num} right now on our channel! Don't forget to like and subscribe!"
     (In Urdu: "کہانی کا اگلا سنسنی خیز موڑ جاننے کے لیے Episode {n_num} ابھی ہمارے چینل پر دیکھیں! ویڈیو کو لائک کریں اور چینل کو سبسکرائب کریں!")
"""

        prompt = f"""You are a {cfg['role']}.
Title: {title}
Genre: {cfg['label']}
Plot Guide / Extra Notes: {plot_summary}
Description / Background: {description[:800]}
{transcript_section}
{beats_section}
{milestone_section}
OBJECTIVES:
1. Write a captivating, immersive THIRD-PERSON {genre} narrative in natural, colloquial {lang_name}.
   - STORY ARCHITECTURE & CHARACTER MOTIVATION: Do NOT merely produce a chronological list of isolated dialogue quotes. In Act 1, introduce the protagonist and main figures, explain the premise/logline and dramatic conflict (who are they, what is their feud or goal?), and weave spoken dialogues naturally into narrative sentences (e.g. 'As the party spiraled into chaos, Dawood coldly warned Arbaaz: ...').
2. Tone & Master Storyteller Voice: {persona_guide} (Overall Mood: {mood.upper()}).
   - CONVERSATIONAL STORYTELLING: Deliver the story in an engaging, conversational tone — as if you are telling an intense, captivating story directly to a friend. Maintain suspense, dramatic momentum, and curiosity throughout.
3. MANDATORY LENGTH REQUIREMENT: You MUST write at least {target_words} spoken words to match a full {duration_mins}-minute video (at {voice_speed} pace). DO NOT summarize briefly or skip scenes. Elaborate on dialogues, character emotions, and scene details.
4. Narrative Act Quotas:
- Hook & Act 1 Target: ~{act1_words} words
- Act 2 Target: ~{act2_words} words
- Act 3 Target: ~{act3_words} words
5. Ending Rule & Short Review: {spoiler_rule}
   - In the final 15-20 seconds of Act 3, provide a punchy conclusion and short review / moral takeaway summarizing the core theme or fate of the characters before the call-to-action!
6. DIALOGUE & CHARACTER QUOTING: You MUST naturally quote and reference key dialogues and character exchanges throughout the story (e.g. hero shouts: 'Get out of the way!' while firing; heroine reveals the truth: 'He was never on our side'; villain threatens...). Narrate critical turning-point actions (explosions, gunfire, car chases, confrontations) at their exact timestamps from the roadmap. This creates an authentic, emotionally charged story and guarantees perfect synchronization with the movie footage.
7. STRICT ANTI-CODE RULE: Do NOT include code blocks, python scripts, unit tests, or markdown backticks under any circumstance. Output pure spoken storytelling narration.
{episodic_instructions}
{source_pacing_note}

NARRATIVE STRUCTURE:
{cfg['structure']}

FORMATTING & AI DIRECTOR REQUIREMENTS:
1. SCENE TIMESTAMPS & DIALOGUE ANCHORING: For each narrative scene block, include a timestamp bracket mapping directly to the source movie/video timeline, e.g. [SCENE: 03:15 - 03:20] or [03:15 - 03:20].
   - MANDATORY: Anchor your timestamps to the chronological Full-Movie Roadmap and 5-Act Milestones above!
   - Match your timestamps directly to the actual dialogues/events in the Roadmap. When narrating what a character says, or a major action (gunfire, chase, explosion, confrontation), use the real timestamp from the dialogue roadmap where that event happens.
   - Do NOT invent arbitrary or fictional timestamps. The video studio cuts the exact video footage at these timestamps to sync with your voiceover!
   - Every scene MUST begin with [SCENE: MM:SS - MM:SS] followed immediately by [VOICEOVER].
   - The scene timestamps MUST progress chronologically across the entire film from Act 1 (opening) through Act 2 (middle) to Act 3 (climax) and Epilogue (ending).
   - NEVER stay in the first 10 or 50 minutes of the movie. Visuals are auto-sliced from these exact timestamps!
2. EMOTIONAL SFX CUES: At key emotional moments, insert sound cues inside brackets:
   - [SFX: HEARTBEAT] for tension, suspicion, or creeping danger.
   - [SFX: SUB_BOOM] for sudden shocking reveals, jumpscares, or plot twists.
   - [SFX: WHOOSH] for rapid chapter or scene transitions.
   - [SFX: CLOCK_TICK] for racing against time or countdowns.
3. Keep the narration natural and engaging; the voice engine will speak the story while our studio automatically cuts the corresponding scenes and triggers the sound effects!
"""
        return prompt

    @staticmethod
    def generate_script(
        title: str,
        description: str,
        subs_text: str,
        target_lang: str = "en",
        persona: str = "hollywood_trailer",
        mood: str = "suspense",
        format_mode: str = "reels_parts",
        num_parts: int = 3,
        duration_mins: int = 3,
        spoiler_mode: str = "full_recap",
        plot_summary: str = "",
        gemini_api_key: Optional[str] = None,
        voice_speed: str = "fast",
        genre: str = "movie_recap",
        source_video_duration_sec: float = 0,
        story_beats: Optional[List[Dict[str, Any]]] = None,
        openai_api_key: Optional[str] = None,
        ai_provider: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generates a viral cinematic storytelling recap or universal explainer in any of the 15+ supported languages.
        Guarantees length precision with automatic multi-act verification and expansion.
        """
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        target_words = ScriptEngine.calculate_target_words(duration_mins, voice_speed, target_lang)

        # Pass 1: If story_beats not provided but subs_text available, extract story beats via 9Router
        if story_beats is None and subs_text:
            try:
                from app.services.nine_router_client import is_ninerouter_available, extract_story_beats_with_9router
                if is_ninerouter_available(timeout_sec=5.0):
                    story_beats = extract_story_beats_with_9router(
                        title=title,
                        dialogues_text=subs_text,
                        genre=genre,
                        duration_sec=source_video_duration_sec
                    )
            except Exception:
                pass

        prompt = ScriptEngine.build_prompt_for_genre(
            genre=genre,
            title=title,
            description=description,
            subs_text=subs_text,
            target_lang=target_lang,
            duration_mins=duration_mins,
            plot_summary=plot_summary,
            persona=persona,
            mood=mood,
            spoiler_mode=spoiler_mode,
            voice_speed=voice_speed,
            source_video_duration_sec=source_video_duration_sec,
            story_beats=story_beats
        )

        if format_mode == "reels_parts" and num_parts > 1:
            prompt += f"""\nDivide the story into {num_parts} distinct parts (Part 1 to Part {num_parts}).
Each part must begin with a powerful hook.
Separate each part strictly with '===PART===' on its own line."""

        # 0. Attempt generation via OpenAI ChatGPT (GPT-4o) if requested or available
        try:
            from app.services.openai_client import is_openai_available, call_chatgpt_llm, load_openai_settings
            cfg_oa = load_openai_settings()
            oa_key = (openai_api_key or cfg_oa.get("api_key", "")).strip()
            use_openai = bool(oa_key and (ai_provider in ("openai", "auto") or is_openai_available()))
            if use_openai:
                lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
                system_instruction = f"You are an elite, world-class viral YouTube movie & drama narrator and storyboard director in natural colloquial {lang_info['name']}."
                gpt_model = cfg_oa.get("model", "gpt-4o")
                gpt_text = call_chatgpt_llm(
                    prompt=prompt,
                    system_prompt=system_instruction,
                    model=gpt_model,
                    max_tokens=4000,
                    api_key=oa_key
                )
                if gpt_text and len(gpt_text.strip()) > 80:
                    gpt_text = ScriptEngine.strip_code_and_developer_artifacts(gpt_text)
                    clean_test_oa, _, _ = ScriptEngine.parse_storyboard(gpt_text)
                    actual_words_oa = len(clean_test_oa.split())

                    # Auto-Expansion if under budget
                    if duration_mins >= 3 and actual_words_oa < int(target_words * 0.80):
                        exp_prompt = f"""The following {lang_info['name']} script is only {actual_words_oa} words, but the video duration requires AT LEAST {target_words} words:

--- CURRENT SCRIPT ---
{gpt_text}
--- END ---

TASK: Elaborate, expand and enrich the story across Act 1, Act 2, and Act 3 with detailed character dialogues, dramatic internal monologues, intense scene descriptions, and escalating emotional stakes to reach {target_words} words. Return the complete, expanded narrative with milestone timestamp brackets like [01:15 - 02:30]."""
                        try:
                            expanded_oa = call_chatgpt_llm(prompt=exp_prompt, system_prompt=system_instruction, model=gpt_model, max_tokens=4000, api_key=oa_key)
                            if expanded_oa and len(expanded_oa.split()) > actual_words_oa:
                                gpt_text = ScriptEngine.strip_code_and_developer_artifacts(expanded_oa.strip())
                        except Exception:
                            pass

                    gpt_text = ScriptEngine.clamp_script_word_budget(
                        gpt_text.strip(),
                        target_duration_mins=duration_mins,
                        target_lang=target_lang,
                        voice_speed=voice_speed
                    )
                    is_valid, val_report = ScriptEngine.validate_script_integrity(
                        script_text=gpt_text,
                        source_duration_sec=source_video_duration_sec,
                        target_duration_mins=duration_mins,
                        target_lang=target_lang,
                        voice_speed=voice_speed
                    )
                    clean_narr_oa, _, _ = ScriptEngine.parse_storyboard(gpt_text)
                    spoken_word_count_oa = len(clean_narr_oa.split()) if clean_narr_oa else len(gpt_text.split())
                    hook_metrics = ScriptEngine.calculate_hook_score(gpt_text, target_lang)
                    return {
                        "success": True,
                        "model": f"openai:{gpt_model}",
                        "script": gpt_text.strip(),
                        "language": target_lang,
                        "genre": genre,
                        "target_words": target_words,
                        "actual_words": spoken_word_count_oa,
                        "raw_words": len(gpt_text.split()),
                        "hook_score": hook_metrics,
                        "story_beats": story_beats or [],
                        "is_valid": is_valid,
                        "integrity_report": val_report
                    }
        except Exception as e:
            print(f"[OpenAI ChatGPT Integration Notice] Fallback triggered: {e}")

        # 1. Attempt generation via 9Router (Primary Local / Cloud LLM)
        try:
            from app.services.nine_router_client import is_ninerouter_available, call_ninerouter_llm
            if is_ninerouter_available(timeout_sec=8.0):
                lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
                system_instruction = f"You are an elite, viral YouTube video narrator and storyboard writer in natural colloquial {lang_info['name']}."
                nine_text = call_ninerouter_llm(prompt=prompt, system_prompt=system_instruction, timeout_sec=90, max_tokens=4000)
                if nine_text and len(nine_text.strip()) > 80:
                    nine_text = ScriptEngine.strip_code_and_developer_artifacts(nine_text)
                    clean_test, _, _ = ScriptEngine.parse_storyboard(nine_text)
                    actual_words = len(clean_test.split())
                    
                    # Auto-Expansion Loop: If returned script is under 80% of target for long videos (>=3 mins)
                    if duration_mins >= 3 and actual_words < int(target_words * 0.80):
                        exp_prompt = f"""The following {lang_info['name']} script is only {actual_words} words, but the video duration requires AT LEAST {target_words} words:

--- CURRENT SCRIPT ---
{nine_text}
--- END ---

TASK: Elaborate, expand and enrich the story across Act 1, Act 2, and Act 3 with detailed character dialogues, dramatic internal monologues, intense scene descriptions, and escalating emotional stakes to reach {target_words} words. Return the complete, expanded narrative with milestone timestamp brackets like [01:15 - 02:30]."""
                        try:
                            expanded_text = call_ninerouter_llm(prompt=exp_prompt, system_prompt=system_instruction, timeout_sec=90, max_tokens=4000)
                            if expanded_text and len(expanded_text.split()) > actual_words:
                                nine_text = ScriptEngine.strip_code_and_developer_artifacts(expanded_text.strip())
                        except Exception:
                            pass

                    nine_text = ScriptEngine.clamp_script_word_budget(
                        nine_text.strip(),
                        target_duration_mins=duration_mins,
                        target_lang=target_lang,
                        voice_speed=voice_speed
                    )
                    is_valid, val_report = ScriptEngine.validate_script_integrity(
                        script_text=nine_text,
                        source_duration_sec=source_video_duration_sec,
                        target_duration_mins=duration_mins,
                        target_lang=target_lang,
                        voice_speed=voice_speed
                    )
                    clean_narr, _, _ = ScriptEngine.parse_storyboard(nine_text)
                    spoken_word_count = len(clean_narr.split()) if clean_narr else len(nine_text.split())
                    hook_metrics = ScriptEngine.calculate_hook_score(nine_text, target_lang)
                    return {
                        "success": True,
                        "model": "9router:new-combo",
                        "script": nine_text.strip(),
                        "language": target_lang,
                        "genre": genre,
                        "target_words": target_words,
                        "actual_words": spoken_word_count,
                        "raw_words": len(nine_text.split()),
                        "hook_score": hook_metrics,
                        "story_beats": story_beats or [],
                        "is_valid": is_valid,
                        "integrity_report": val_report
                    }
        except Exception as e:
            print(f"[9Router Integration Warning] {e}")

        # 2. Attempt generation via Gemini API if key is available
        if api_key:
            for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=35) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if text:
                            text = ScriptEngine.strip_code_and_developer_artifacts(text)
                            clean_test_g, _, _ = ScriptEngine.parse_storyboard(text)
                            actual_words_g = len(clean_test_g.split())

                            # Auto-Expansion Loop for Gemini: If returned script is under 80% of target for long videos (>=3 mins)
                            if duration_mins >= 3 and actual_words_g < int(target_words * 0.80):
                                exp_res = ScriptEngine.expand_script(
                                    current_script=text,
                                    target_lang=target_lang,
                                    duration_mins=duration_mins,
                                    voice_speed=voice_speed,
                                    genre=genre,
                                    gemini_api_key=api_key
                                )
                                if exp_res.get("success") and exp_res.get("script") and len(exp_res["script"].split()) > actual_words_g:
                                    text = exp_res["script"]

                            text = ScriptEngine.clamp_script_word_budget(
                                text.strip(),
                                target_duration_mins=duration_mins,
                                target_lang=target_lang,
                                voice_speed=voice_speed
                            )
                            is_valid, val_report = ScriptEngine.validate_script_integrity(
                                script_text=text,
                                source_duration_sec=source_video_duration_sec,
                                target_duration_mins=duration_mins,
                                target_lang=target_lang,
                                voice_speed=voice_speed
                            )
                            clean_narr_g, _, _ = ScriptEngine.parse_storyboard(text)
                            spoken_word_count_g = len(clean_narr_g.split()) if clean_narr_g else len(text.split())
                            hook_metrics = ScriptEngine.calculate_hook_score(text, target_lang)
                            return {
                                "success": True,
                                "model": model,
                                "script": text,
                                "language": target_lang,
                                "genre": genre,
                                "target_words": target_words,
                                "actual_words": spoken_word_count_g,
                                "raw_words": len(text.split()),
                                "hook_score": hook_metrics,
                                "story_beats": story_beats or [],
                                "is_valid": is_valid,
                                "integrity_report": val_report
                            }
                except Exception:
                    continue

        # 3. Fallback multi-language template generator
        fallback_text = ScriptEngine._generate_fallback_script(title, target_lang, mood, persona)
        hook_metrics = ScriptEngine.calculate_hook_score(fallback_text, target_lang)
        return {
            "success": True,
            "model": "local_fallback_engine",
            "script": fallback_text,
            "language": target_lang,
            "genre": genre,
            "target_words": target_words,
            "hook_score": hook_metrics,
            "story_beats": story_beats or []
        }

    @staticmethod
    def _generate_fallback_script(title: str, lang: str, mood: str, persona: str) -> str:
        """Resilient fallback storytelling templates for when LLM API is unreachable."""
        templates = {
            "en": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
Nobody could have predicted the horrific secret hidden behind {title}.
When our protagonist stepped into this dark mystery, every exit was already sealed.
Slowly, the unsettling truth began to emerge, revealing a conspiracy that will leave you breathless.
Make sure to follow and like for the shocking conclusion!""",

            "ur": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
یہ کہانی شروع ہوتی ہے ایک ایسے پراسرار موڑ سے جہاں ہر لمحہ جان لیوا ثابت ہو سکتا ہے۔
{title} کی اس داستان میں جب مرکزی کردار ایک تاریک راز کے پیچھے نکلا، تو اسے اندازہ نہیں تھا کہ اصل دشمن کون ہے۔
آگے کیا ہوتا ہے؟ جاننے کے لیے ابھی لائک اور فالو کریں!""",

            "hi": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
यह कहानी शुरू होती है एक ऐसे ख़ौफ़नाक मोड़ से जहाँ हर कदम पर मौत खड़ी थी।
{title} की इस कहानी में एक ऐसा भयानक सच सामने आता है जो आपके रोंगटे खड़े कर देगा।
आगे की पूरी कहानी जानने के लिए अभी फ़ॉलो और लाइक करें!""",

            "es": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
Todo comienza con un giro aterrador que nadie vio venir en {title}.
Cuando el protagonista descubre la verdad, ya era demasiado tarde para escapar de la trampa.
¡Síguenos ahora para no perderte el impactante desenlace!""",

            "id": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
Cerita bermula ketika sebuah rahasia kelam dalam {title} terungkap ke permukaan.
Ketika sang tokoh utama mencoba mencari kebenaran, bahaya besar sudah menantinya di setiap sudut.
Follow dan like sekarang untuk kelanjutan kisah menegangkan ini!""",

            "ar": f"""[SCENE: 00:00 - 00:45]
[VOICEOVER]
تبدأ هذه القصة مع لغز غامض ومرعب في {title} لا يمكن لأحد توقعه.
عندما اقترب البطل من كشف الحقيقة، أدرك أن الخطر يحيط به من كل جانب.
تابعنا الآن لمشاهدة النهاية الصادمة!"""
        }
        return templates.get(lang, templates["en"])

    @staticmethod
    def translate_script_to_languages(primary_script: str, languages: List[str]) -> Dict[str, str]:
        """Translates a primary script into multiple target languages using chunked translation."""
        results = {}
        if not primary_script:
            return results

        from deep_translator import GoogleTranslator
        for lang in languages:
            l_code = lang.strip().lower()
            if not l_code:
                continue
            if l_code == "en":
                results[l_code] = primary_script
                continue
            try:
                if len(primary_script) > 2200:
                    chunks = [primary_script[i:i+2000] for i in range(0, len(primary_script), 2000)]
                    translated_chunks = [GoogleTranslator(source="auto", target=l_code).translate(c) for c in chunks]
                    results[l_code] = " ".join(translated_chunks)
                else:
                    results[l_code] = GoogleTranslator(source="auto", target=l_code).translate(primary_script)
            except Exception as e:
                print(f"[Translate Scripts Error {l_code}] {e}")
                results[l_code] = primary_script
        return results

    @staticmethod
    def expand_script(
        current_script: str,
        target_lang: str = "en",
        duration_mins: int = 10,
        voice_speed: str = "fast",
        genre: str = "movie_recap",
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Expands an existing script to reach the full target word count."""
        target_words = ScriptEngine.calculate_target_words(duration_mins, voice_speed, target_lang)
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
        clean_text, _, _ = ScriptEngine.parse_storyboard(current_script)
        current_words = len(clean_text.split())

        prompt = f"""You are an elite YouTube storytelling storyboard writer in natural colloquial {lang_info['name']}.
The creator requires a full {duration_mins}-minute explainer which needs at least {target_words} spoken words, but the current script only has {current_words} words.

CURRENT SCRIPT:
{current_script}

EXPANSION INSTRUCTIONS:
1. Preserve the core story arc, character names, tone, and final climax/twist.
2. Greatly expand and elaborate on Act 1 and Act 2 by detailing scene interactions, character dialogues, inner turmoil, and escalating dramatic confrontations.
3. Bring the total script length to AT LEAST {target_words} spoken words.
4. Maintain milestone scene timestamp brackets like [01:15 - 02:30].
Return the complete, expanded storyboard narrative."""

        # 1. 9Router expansion
        try:
            from app.services.nine_router_client import is_ninerouter_available, call_ninerouter_llm
            if is_ninerouter_available(timeout_sec=8.0):
                system_instruction = f"You are an elite YouTube video scriptwriter in {lang_info['name']}."
                exp_text = call_ninerouter_llm(prompt=prompt, system_prompt=system_instruction, timeout_sec=90, max_tokens=4000)
                if exp_text and len(exp_text.strip()) > 100:
                    exp_text = ScriptEngine.strip_code_and_developer_artifacts(exp_text)
                    hook_metrics = ScriptEngine.calculate_hook_score(exp_text, target_lang)
                    return {
                        "success": True,
                        "script": exp_text.strip(),
                        "target_words": target_words,
                        "actual_words": len(exp_text.split()),
                        "hook_score": hook_metrics
                    }
        except Exception as e:
            print(f"[expand_script 9Router error] {e}")

        # 2. Gemini expansion fallback
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            for model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if text:
                            text = ScriptEngine.strip_code_and_developer_artifacts(text)
                            hook_metrics = ScriptEngine.calculate_hook_score(text, target_lang)
                            return {
                                "success": True,
                                "script": text.strip(),
                                "target_words": target_words,
                                "actual_words": len(text.split()),
                                "hook_score": hook_metrics
                            }
                except Exception as ge:
                    print(f"[expand_script Gemini error] {ge}")
                    continue

        return {"success": False, "error": "LLM expansion unavailable", "script": current_script}

