import re
from typing import Dict, Any, List

class MetadataEngine:
    """
    YouTube, TikTok & Reels Viral Social Metadata Suite.
    Generates high-CTR titles, optimized descriptions, ranked SEO tags,
    and engagement pinned comments for creators.
    """

    @staticmethod
    def generate_viral_metadata(
        movie_title: str,
        script_snippet: str,
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Generates click-to-copy metadata package tailored to the video's language.
        """
        clean_title = movie_title.replace('_', ' ').replace('-', ' ').strip()
        if not clean_title or clean_title == "Movie Story Explanation":
            clean_title = "This Mystery Movie"

        # Clean title for hashtags (strip pipes, punctuation, special chars)
        clean_tag_title = re.sub(r'[^\w\u0600-\u06FF\u0900-\u097F]', '', clean_title.replace(' ', ''))
        if not clean_tag_title:
            clean_tag_title = "ViralStory"

        # Titles by Language
        if lang in ["ur", "hi"]:
            titles = [
                f"😱 {clean_title} کی یہ کہانی آپ کے ہوش اڑا دے گی! (Full Movie Explained)",
                f"⚠️ سب سے بڑا دھوکہ! {clean_title} Story Recap & Ending Explained",
                f"🔥 جب قاتل کا اصل چہرہ سامنے آیا! {clean_title} Full Hindi/Urdu Explanation"
            ]
            desc_hook = f"آج کی اس ویڈیو میں ہم آپ کو بتائیں گے {clean_title} کی سنسنی خیز اور مکمل کہانی۔ آخر تک دیکھیں اور لائک و سبسکرائب کرنا نہ بھولیں!"
            pinned = "💬 آپ کو اس کہانی میں سب سے بڑا موڑ کون سا لگا؟ کمنٹ میں اپنی رائے ضرور بتائیں! 👇"
            tags = [f"{clean_title} explained in urdu", f"{clean_title} hindi recap", "movie explainer", "ending explained", "viral story recap", "hindi movie summary"]
            hashtags = [f"#{clean_tag_title}", "#MovieRecap", "#UrduExplainer", "#HindiMovieRecap", "#TrendingRecap"]

        elif lang == "es":
            titles = [
                f"😱 ¡El SECRETO de {clean_title} que la policía ocultó! (Resumen Completo)",
                f"⚠️ ¡Nadie vio venir este final! {clean_title} Explicada en Minutos",
                f"🔥 ¡No confíes en nadie! La impactante historia de {clean_title}"
            ]
            desc_hook = f"En este video te explicamos la fascinante y aterradora historia de {clean_title}. ¡Mira hasta el final y suscríbete para más resúmenes de películas!"
            pinned = "💬 ¿Qué te pareció el final de esta película? ¡Déjamelo saber en los comentarios! 👇"
            tags = [f"{clean_title} explicacion", f"{clean_title} resumen", "resumen de peliculas", "final explicado", "peliculas en minutos"]
            hashtags = [f"#{clean_tag_title}", "#ResumenDePeliculas", "#PeliculasEnMinutos", "#CineRecap"]

        elif lang == "id":
            titles = [
                f"😱 RAHASIA KELAM di Balik {clean_title} yang Bikin Merinding! (Alur Cerita)",
                f"⚠️ Jangan Nonton Sendirian! Alur Cerita Lengkap {clean_title}",
                f"🔥 Akhir Paling Tak Terduga! {clean_title} Rangkuman Film"
            ]
            desc_hook = f"Berikut adalah alur cerita lengkap dan penjelasan ending dari film {clean_title}. Tonton sampai habis dan jangan lupa like & subscribe!"
            pinned = "💬 Menurut kalian, siapa karakter paling licik di film ini? Tulis di kolom komentar ya! 👇"
            tags = [f"alur cerita {clean_title}", f"rekap film {clean_title}", "alur cerita film", "rangkuman film", "movie recap indonesia"]
            hashtags = [f"#{clean_tag_title}", "#AlurCeritaFilm", "#RekapFilm", "#NontonFilm"]

        elif lang == "ar":
            titles = [
                f"😱 الصدمة الكبرى في قصة {clean_title}! (ملخص الفيلم كامل)",
                f"⚠️ النهاية التي صدمت الملايين! فيلم {clean_title} ملخص كامل",
                f"🔥 حقيقة غير متوقعة تقلب الموازين في {clean_title}!"
            ]
            desc_hook = f"نقدم لكم اليوم القصة الكاملة والمثيرة لفيلم {clean_title} مع شرح النهاية الصادمة. لا تنسوا الإعجاب والاشتراك بالقناة!"
            pinned = "💬 ما هو أكثر مشهد صدمك في هذا الفيلم؟ شاركنا رأيك في التعليقات! 👇"
            tags = [f"ملخص فيلم {clean_title}", f"شرح فيلم {clean_title}", "ملخص افلام", "نهاية فيلم", "قصة فيلم"]
            hashtags = [f"#{clean_tag_title}", "#ملخص_افلام", "#سينما", "#افلام_رعب"]

        else: # English & Universal
            titles = [
                f"😱 Nobody Expected the Twisted Ending of {clean_title}! (Full Movie Recap)",
                f"⚠️ THE WORST MISTAKE! {clean_title} Story & Ending Explained",
                f"🔥 He Thought He Was Safe Until... {clean_title} Movie Recap"
            ]
            desc_hook = f"In this video, we break down the shocking story and unexpected twists of {clean_title}. Watch until the end and subscribe for more viral movie recaps!"
            pinned = "💬 What was your favorite moment from this story? Let us know in the comments below! 👇"
            tags = [f"{clean_title} movie recap", f"{clean_title} explained", "movie recap", "ending explained", "story recap", "films in minutes"]
            hashtags = [f"#{clean_tag_title}", "#MovieRecap", "#StoryRecap", "#EndingExplained", "#TrendingMovies"]

        # Fair Use Disclaimer (Essential for YouTube monetization protection)
        disclaimer = (
            "\n\n---\n⚖️ COPYRIGHT & FAIR USE DISCLAIMER:\n"
            "This video is for educational, critical commentary, and storytelling purposes. "
            "Under Section 107 of the Copyright Act 1976, allowance is made for 'fair use' for purposes such as criticism, commentary, news reporting, teaching, and research."
        )

        full_description = f"{desc_hook}\n\n🎬 Movie: {clean_title}\n\n{' '.join(hashtags)}{disclaimer}"

        return {
            "movie_title": clean_title,
            "titles": titles,
            "description": full_description,
            "tags": tags,
            "hashtags": hashtags,
            "pinned_comment": pinned
        }
