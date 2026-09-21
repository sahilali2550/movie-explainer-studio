import os
import re
import math
import subprocess
from typing import List, Dict, Any, Optional
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter
from app.core.config import get_ffmpeg_binary, THUMBNAILS_DIR, FONTS_DIR, SUPPORTED_LANGUAGES

class ThumbnailEngine:
    """
    Hybrid Viral Thumbnail Generator.
    Extracts high-contrast movie milestone frames, applies cinematic color grading,
    renders 1920x1080 Full HD canvas, and handles multi-line auto-wrapped typography
    with RTL Arabic/Urdu and Latin font support.
    """

    @staticmethod
    def extract_candidate_frames(
        video_path: str,
        output_prefix: str,
        count: int = 6,
        target_timestamps: Optional[List[float]] = None
    ) -> List[str]:
        """
        Extracts candidate keyframes evenly across the video duration or from target scene timestamps.
        """
        ffmpeg_bin = get_ffmpeg_binary()
        from app.core.config import get_ffprobe_binary

        dur = 120.0
        try:
            cmd = [get_ffprobe_binary(), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                dur = float(res.stdout.strip())
        except Exception:
            pass

        min_safe_t = max(10.0, dur * 0.16) if dur > 60.0 else 2.0
        max_safe_t = min(dur - 10.0, dur * 0.82) if dur > 60.0 else max(2.0, dur - 2.0)

        safe_ratios = [0.22, 0.32, 0.48, 0.60, 0.74, 0.82]
        fallback_ts = [round(dur * p, 2) for p in safe_ratios]

        if target_timestamps:
            timestamps = [t for t in target_timestamps if min_safe_t <= t <= max_safe_t]
            if len(timestamps) < count:
                for ft in fallback_ts:
                    if ft not in timestamps and min_safe_t <= ft <= max_safe_t:
                        timestamps.append(ft)
        else:
            timestamps = fallback_ts

        extracted_frames = []
        for idx, t in enumerate(timestamps[:count]):
            frame_path = f"{output_prefix}_cand_{idx}.jpg"
            cmd = [
                ffmpeg_bin, "-y",
                "-ss", str(round(t, 2)),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                frame_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 1000:
                extracted_frames.append(frame_path)

        return extracted_frames

    @staticmethod
    def calculate_sharpness(img_path: str) -> float:
        """
        Estimates image sharpness using edge detection filter standard deviation.
        """
        try:
            with Image.open(img_path) as im:
                im_gray = im.convert("L").resize((320, 180))
                edges = im_gray.filter(ImageFilter.FIND_EDGES)
                stat = edges.histogram()
                mean = sum(i * n for i, n in enumerate(stat)) / sum(stat)
                variance = sum((i - mean) ** 2 * n for i, n in enumerate(stat)) / sum(stat)
                return math.sqrt(variance)
        except Exception:
            return 0.0

    @staticmethod
    def apply_cinematic_grading(img: Image.Image) -> Image.Image:
        """
        Enhances image for YouTube feed contrast pop on 1920x1080 canvas.
        """
        enhanced = ImageEnhance.Contrast(img).enhance(1.22)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.18)

        w, h = enhanced.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bottom_h = int(h * 0.40)
        for y in range(bottom_h):
            alpha = int(185 * (y / bottom_h))
            draw.line([(0, h - bottom_h + y), (w, h - bottom_h + y)], fill=(0, 0, 0, alpha))

        top_h = int(h * 0.25)
        for y in range(top_h):
            alpha = int(120 * (1 - (y / top_h)))
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

        enhanced_rgba = enhanced.convert("RGBA")
        result = Image.alpha_composite(enhanced_rgba, overlay)
        return result.convert("RGB")

    @staticmethod
    def apply_face_zoom_style(img: Image.Image, zoom_factor: float = 1.25) -> Image.Image:
        """
        Layout 1: Climax Face Zoom (1920x1080 Full HD).
        """
        img = img.convert("RGB")
        w, h = img.size
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        left = max(0, (w - crop_w) // 2)
        top = max(0, (h - crop_h) // 2)
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        resized = cropped.resize((1920, 1080), Image.Resampling.LANCZOS)

        enhanced = ImageEnhance.Contrast(resized).enhance(1.25)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.20)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.25)
        return ThumbnailEngine.apply_cinematic_grading(enhanced)

    @staticmethod
    def apply_split_screen_style(img1: Image.Image, img2: Image.Image) -> Image.Image:
        """
        Layout 2: Split-Screen Confrontation (1920x1080 Full HD).
        """
        def crop_and_fit(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
            im = im.convert("RGB")
            iw, ih = im.size
            scale = max(target_w / iw, target_h / ih)
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            im_scaled = im.resize((nw, nh), Image.Resampling.LANCZOS)
            left = max(0, (nw - target_w) // 2)
            top = max(0, (nh - target_h) // 2)
            return im_scaled.crop((left, top, left + target_w, top + target_h))

        left_half = crop_and_fit(img1, 960, 1080)
        right_half = crop_and_fit(img2, 960, 1080)

        canvas = Image.new("RGB", (1920, 1080), (0, 0, 0))
        canvas.paste(left_half, (0, 0))
        canvas.paste(right_half, (960, 0))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(954, 0), (966, 1080)], fill=(0, 0, 0))
        draw.rectangle([(957, 0), (963, 1080)], fill=(255, 230, 0))

        return ThumbnailEngine.apply_cinematic_grading(canvas)

    @staticmethod
    def apply_cinema_poster_style(img: Image.Image) -> Image.Image:
        """
        Layout 3: Cinema Poster Art (1920x1080 Full HD).
        """
        img = img.convert("RGB")
        w, h = img.size
        scale = max(1920 / w, 1080 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = max(0, (nw - 1920) // 2)
        top = max(0, (nh - 1080) // 2)
        canvas = resized.crop((left, top, left + 1920, top + 1080))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (1920, 36)], fill=(0, 0, 0))
        draw.rectangle([(0, 1044), (1920, 1080)], fill=(0, 0, 0))

        enhanced = ImageEnhance.Contrast(canvas).enhance(1.22)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.24)
        return ThumbnailEngine.apply_cinematic_grading(enhanced)

    @staticmethod
    def fetch_youtube_viral_thumbnail(url: str, output_path: str) -> Optional[str]:
        """
        Downloads high-res YouTube thumbnail reference without crashing on errors.
        """
        import urllib.request
        pattern = r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, url)
        if not match:
            return None

        video_id = match.group(1)
        resolutions = ["maxresdefault.jpg", "sddefault.jpg", "hqdefault.jpg"]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for res in resolutions:
            thumb_url = f"https://img.youtube.com/vi/{video_id}/{res}"
            try:
                req = urllib.request.Request(thumb_url, headers=headers)
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        content = resp.read()
                        if len(content) > 3000:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(content)
                            return output_path
            except Exception:
                continue
        return None

    @staticmethod
    def render_typography(
        img: Image.Image,
        hook_text: str,
        lang: str = "en",
        badge_label: str = "🎬 RECAP"
    ) -> Image.Image:
        """
        Universal Multi-Line Typography with Dynamic Font Auto-Scaling and RTL Support.
        Prevents text overflow and off-screen clipping.
        """
        w, h = img.size
        draw = ImageDraw.Draw(img)

        is_rtl = SUPPORTED_LANGUAGES.get(lang, {}).get("rtl", False)
        display_text = hook_text.strip()

        # Font candidate resolution
        font_candidates = []
        if lang in ["ur", "ar"]:
            font_candidates.extend([
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\tahomabd.ttf",
                str(FONTS_DIR / "arabtype.ttf"),
                r"C:\Windows\Fonts\arabtype.ttf",
                "arialbd.ttf"
            ])
        elif lang == "hi":
            font_candidates.extend([
                str(FONTS_DIR / "NirmalaB.ttf"),
                r"C:\Windows\Fonts\NirmalaB.ttf",
                "mangal.ttf",
                "arialbd.ttf"
            ])
        else:
            font_candidates.extend([
                str(FONTS_DIR / "arialbd.ttf"),
                "arialbd.ttf",
                "impact.ttf",
                "Montserrat-Bold.ttf",
                "tahoma.ttf"
            ])

        base_font_size = int(h * 0.11)
        max_allowed_w = int(w * 0.88)

        def get_font(size: int):
            for cand in font_candidates:
                try:
                    if os.path.exists(cand) or not os.path.isabs(cand):
                        return ImageFont.truetype(cand, size)
                except Exception:
                    continue
            return ImageFont.load_default()

        # Word wrap if too long
        words = display_text.split()
        lines = [display_text]
        font = get_font(base_font_size)

        try:
            bbox = draw.textbbox((0, 0), display_text, font=font)
            line_w = bbox[2] - bbox[0]
        except Exception:
            line_w = len(display_text) * (base_font_size * 0.5)

        if line_w > max_allowed_w and len(words) > 1:
            mid = len(words) // 2
            lines = [" ".join(words[:mid]), " ".join(words[mid:])]

        # Auto-scale font size to guarantee 100% boundary safety
        current_font_size = base_font_size
        while current_font_size > 24:
            font = get_font(current_font_size)
            max_line_w = 0
            for l in lines:
                try:
                    b = draw.textbbox((0, 0), l, font=font)
                    lw = b[2] - b[0]
                except Exception:
                    lw = len(l) * (current_font_size * 0.5)
                if lw > max_line_w:
                    max_line_w = lw
            if max_line_w <= max_allowed_w:
                break
            current_font_size -= 4

        # RTL Shaping per line
        final_render_lines = []
        for l in lines:
            rendered_l = l
            if is_rtl:
                try:
                    import arabic_reshaper
                    from bidi.algorithm import get_display
                    reshaped = arabic_reshaper.reshape(l)
                    rendered_l = get_display(reshaped)
                except Exception:
                    pass
            final_render_lines.append(rendered_l)

        # Compute vertical positioning
        line_heights = []
        for l in final_render_lines:
            try:
                b = draw.textbbox((0, 0), l, font=font)
                line_heights.append(b[3] - b[1])
            except Exception:
                line_heights.append(int(current_font_size * 1.2))

        total_text_h = sum(line_heights) + (len(final_render_lines) - 1) * int(current_font_size * 0.25)
        start_y = h - total_text_h - int(h * 0.08)

        curr_y = start_y
        stroke_w = max(4, int(current_font_size * 0.11))
        text_color = (255, 230, 0) if "!" in display_text or "?" in display_text else (255, 255, 255)

        for idx, l in enumerate(final_render_lines):
            try:
                b = draw.textbbox((0, 0), l, font=font)
                lw = b[2] - b[0]
            except Exception:
                lw = int(w * 0.5)
            lx = (w - lw) // 2

            # Black stroke
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text((lx + dx, curr_y + dy), l, font=font, fill=(0, 0, 0))

            # Main text
            draw.text((lx, curr_y), l, font=font, fill=text_color)
            curr_y += line_heights[idx] + int(current_font_size * 0.25)

        # Top Badge
        if badge_label:
            badge_font_size = int(h * 0.055)
            clean_badge = re.sub(r'[^\w\s-]', '', badge_label).strip().upper()
            if not clean_badge:
                clean_badge = "RECAP"

            try:
                b_font = ImageFont.truetype("arialbd.ttf", badge_font_size)
            except Exception:
                b_font = font

            b_x = int(w * 0.05)
            b_y = int(h * 0.07)
            b_text = clean_badge
            try:
                b_bbox = draw.textbbox((0, 0), b_text, font=b_font)
                bw = b_bbox[2] - b_bbox[0] + 32
                bh = b_bbox[3] - b_bbox[1] + 20
                draw.rectangle([(b_x, b_y), (b_x + bw, b_y + bh)], fill=(220, 20, 60))
                draw.text((b_x + 16, b_y + 10), b_text, font=b_font, fill=(255, 255, 255))
            except Exception:
                pass

        return img

    @staticmethod
    def generate_viral_thumbnails(
        video_path: str,
        job_id: str,
        movie_title: str,
        lang: str = "en",
        custom_hooks: Optional[List[str]] = None,
        target_timestamps: Optional[List[float]] = None,
        plot_summary: str = "",
        ai_image_path: Optional[str] = None,
        youtube_url: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return ThumbnailEngine.generate_3_thumbnail_options(
            video_path=video_path,
            job_id=job_id,
            movie_title=movie_title,
            lang=lang,
            custom_hooks=custom_hooks,
            target_timestamps=target_timestamps,
            plot_summary=plot_summary,
            ai_image_path=ai_image_path,
            youtube_url=youtube_url,
            openai_api_key=openai_api_key
        )

    @staticmethod
    def generate_3_thumbnail_options(
        video_path: str,
        job_id: str,
        movie_title: str,
        lang: str = "en",
        custom_hooks: Optional[List[str]] = None,
        target_timestamps: Optional[List[float]] = None,
        plot_summary: str = "",
        ai_image_path: Optional[str] = None,
        youtube_url: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        temp_prefix = str(THUMBNAILS_DIR / f"{job_id}_raw")
        raw_candidates = ThumbnailEngine.extract_candidate_frames(
            video_path,
            temp_prefix,
            count=6,
            target_timestamps=target_timestamps
        )

        scored = []
        for cf in raw_candidates:
            s = ThumbnailEngine.calculate_sharpness(cf)
            scored.append((s, cf))

        selected_frames = []
        if len(scored) >= 6:
            act1 = sorted([scored[0], scored[1]], key=lambda x: x[0], reverse=True)
            selected_frames.append(act1[0][1])

            act2 = sorted([scored[2], scored[3]], key=lambda x: x[0], reverse=True)
            selected_frames.append(act2[0][1])

            act3 = sorted([scored[4], scored[5]], key=lambda x: x[0], reverse=True)
            selected_frames.append(act3[0][1])
        else:
            scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
            selected_frames = [item[1] for item in scored_sorted[:3]]
            while len(selected_frames) < 3 and raw_candidates:
                selected_frames.append(raw_candidates[0])

        if custom_hooks and len(custom_hooks) >= 3:
            hooks = custom_hooks
        else:
            try:
                from app.services.nine_router_client import generate_movie_specific_hooks
                hooks = generate_movie_specific_hooks(title=movie_title, plot_summary=plot_summary, lang=lang)
            except Exception:
                hooks = ["THE SHOCKING TWIST!", "THE TRUTH REVEALED!", "NOBODY SAW THIS COMING!"]

        yt_thumb_path = None
        if youtube_url:
            try:
                candidate_yt_path = str(THUMBNAILS_DIR / f"{job_id}_yt_ref.jpg")
                yt_thumb_path = ThumbnailEngine.fetch_youtube_viral_thumbnail(youtube_url, candidate_yt_path)
            except Exception:
                yt_thumb_path = None

        gen_ai_path = ai_image_path
        if not gen_ai_path or not os.path.exists(gen_ai_path):
            try:
                from app.services.openai_client import is_openai_available, generate_ai_image_with_dalle
                if is_openai_available() or openai_api_key:
                    from app.services.nine_router_client import craft_cinematic_thumbnail_prompt
                    d_prompt = craft_cinematic_thumbnail_prompt(title=movie_title, plot_summary=plot_summary)
                    gen_ai_path = generate_ai_image_with_dalle(prompt=d_prompt, api_key=openai_api_key)
            except Exception:
                gen_ai_path = None

        if not gen_ai_path or not os.path.exists(gen_ai_path):
            try:
                from app.services.nine_router_client import craft_cinematic_thumbnail_prompt, generate_ai_image_with_9router
                p_prompt = craft_cinematic_thumbnail_prompt(title=movie_title, plot_summary=plot_summary)
                gen_ai_path = generate_ai_image_with_9router(p_prompt, size="1792x1024")
            except Exception:
                gen_ai_path = None

        has_ai_poster = bool(gen_ai_path and os.path.exists(gen_ai_path))
        has_yt_remix = bool(yt_thumb_path and os.path.exists(yt_thumb_path))

        if has_ai_poster:
            badge_1 = "🔥 AI CINEMA POSTER"
            badge_3 = "🔥 OFFICIAL POSTER REMIX" if has_yt_remix else "🎬 CINEMA POSTER"
        elif has_yt_remix:
            badge_1 = "🔥 OFFICIAL POSTER REMIX"
            badge_3 = "🎬 CINEMA POSTER"
        else:
            badge_1 = "😱 CLIMAX ZOOM"
            badge_3 = "🎬 CINEMA POSTER"

        badges = [badge_1, "⚡ VS CONFRONTATION", badge_3]
        results = []

        for i in range(3):
            out_filename = f"thumb_{job_id}_v{i+1}.jpg"
            out_path = str(THUMBNAILS_DIR / out_filename)
            base_filename = f"thumb_{job_id}_v{i+1}_base.jpg"

            try:
                if i == 0:
                    if has_ai_poster:
                        with Image.open(gen_ai_path) as im:
                            base_img = ThumbnailEngine.apply_cinema_poster_style(im)
                    elif has_yt_remix:
                        with Image.open(yt_thumb_path) as im:
                            base_img = ThumbnailEngine.apply_cinema_poster_style(im)
                    elif selected_frames:
                        with Image.open(selected_frames[0]) as im:
                            base_img = ThumbnailEngine.apply_face_zoom_style(im)
                    else:
                        base_img = Image.new("RGB", (1920, 1080), color=(30, 30, 40))

                elif i == 1:
                    frame_a = selected_frames[0] if len(selected_frames) > 0 else None
                    frame_b = selected_frames[1] if len(selected_frames) > 1 else frame_a
                    if frame_a and frame_b:
                        with Image.open(frame_a) as im_a, Image.open(frame_b) as im_b:
                            base_img = ThumbnailEngine.apply_split_screen_style(im_a, im_b)
                    elif frame_a:
                        with Image.open(frame_a) as im_a:
                            base_img = ThumbnailEngine.apply_face_zoom_style(im_a)
                    else:
                        base_img = Image.new("RGB", (1920, 1080), color=(40, 20, 20))

                else:
                    if has_ai_poster and has_yt_remix:
                        with Image.open(yt_thumb_path) as im:
                            base_img = ThumbnailEngine.apply_cinema_poster_style(im)
                    elif selected_frames:
                        frame_c = selected_frames[-1]
                        with Image.open(frame_c) as im_c:
                            base_img = ThumbnailEngine.apply_cinema_poster_style(im_c)
                    else:
                        base_img = Image.new("RGB", (1920, 1080), color=(20, 30, 40))

                base_img.save(str(THUMBNAILS_DIR / base_filename), quality=95)

                final_thumb = ThumbnailEngine.render_typography(
                    base_img.copy(), hooks[i], lang=lang, badge_label=badges[i]
                )
                final_thumb.save(out_path, quality=94)

                results.append({
                    "variation": i + 1,
                    "badge": badges[i],
                    "hook_text": hooks[i],
                    "filename": out_filename,
                    "base_filename": base_filename,
                    "url": f"/outputs/thumbnails/{out_filename}",
                    "aspect_ratio": "16:9",
                    "width": 1920,
                    "height": 1080
                })
            except Exception as e:
                print(f"[ThumbnailGen Error v{i+1}] {e}")

        for cf in raw_candidates:
            if cf and os.path.exists(cf):
                try: os.remove(cf)
                except Exception: pass

        return results

    @staticmethod
    def re_render_thumbnail(
        filename: str,
        new_hook_text: str,
        lang: str = "en",
        badge_label: str = "🎬 RECAP"
    ) -> Optional[str]:
        clean_filename = os.path.basename(filename).strip()
        if not re.match(r"^[a-zA-Z0-9_-]+\.jpg$", clean_filename, re.IGNORECASE):
            return None

        base_name = clean_filename.replace(".jpg", "_base.jpg") if "_base" not in clean_filename else clean_filename
        base_path = (THUMBNAILS_DIR / base_name).resolve()
        target_path = (THUMBNAILS_DIR / clean_filename.replace("_base", "")).resolve()

        thumbnails_dir_resolved = THUMBNAILS_DIR.resolve()
        if not str(base_path).startswith(str(thumbnails_dir_resolved)) or not str(target_path).startswith(str(thumbnails_dir_resolved)):
            return None

        source_img_path = base_path if base_path.exists() else target_path
        if not source_img_path.exists():
            return None

        try:
            with Image.open(source_img_path) as im:
                im_clean = im.convert("RGB")
                updated = ThumbnailEngine.render_typography(im_clean, new_hook_text, lang=lang, badge_label=badge_label)
                updated.save(target_path, quality=94)
            return f"/outputs/thumbnails/{target_path.name}?t={int(os.path.getmtime(target_path))}"
        except Exception as e:
            print(f"[Thumbnail Re-render Error] {e}")
            return None
