import language_tool_python
from textstat import flesch_reading_ease # Flesch metrics
import math
from transformers import pipeline

grammar_tool = language_tool_python.LanguageTool('en-US')
# Initialize grammar correction model and sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

cv_keywords = cv_keywords = [
    "team", "managed", "developed", "analyzed", "designed", "implemented", "achieved", "adapted",
    "administered", "advised", "allocated", "assembled", "assessed", "built", "calculated", "coached",
    "collaborated", "communicated", "completed", "conceptualized", "conducted", "constructed", "consulted",
    "coordinated", "created", "delivered", "demonstrated", "designed", "devised", "directed", "earned",
    "enhanced", "established", "evaluated", "executed", "expanded", "facilitated", "formulated", "generated",
    "guided", "identified", "implemented", "improved", "increased", "initiated", "installed", "integrated",
    "launched", "led", "maintained", "managed", "mentored", "mobilized", "monitored", "negotiated",
    "optimized", "organized", "overcame", "oversaw", "performed", "planned", "prepared", "presented",
    "prioritized", "produced", "programmed", "promoted", "proposed", "provided", "published", "recruited",
    "redesigned", "reduced", "refined", "resolved", "reviewed", "scheduled", "secured", "streamlined",
    "strengthened", "supervised", "supported", "taught", "trained", "transformed", "updated", "utilized",
    "validated", "volunteered", "won", "wrote", "worked"
]

def heuristic_score(text):
    # Step 1: Grammar check
    matches = grammar_tool.check(text)
    grammar_penalty = len(matches)

    # Step 2: Readability score
    readability = flesch_reading_ease(text)

    # Readiblity score thats too high usually means its too simple
    if readability > 70:
        excess = readability - 70
        readability = readability - excess * 2.5

    # Step 3: Sentiment analysis (encourages positive, professional tone)
    sentiment_result = sentiment_analyzer(text)
    sentiment_score = sentiment_result[0]['score'] if sentiment_result[0]['label'] == "POSITIVE" else 1-sentiment_result[0]['score']

    # Step 4: Keyword presence for CVs (e.g., "team", "managed", "developed")
    keyword_score = sum(1 for word in text.lower().split() if word in cv_keywords)

    # Step 5: Weighting and combining scores
    grammar_weight = -2
    readability_weight = 1
    sentiment_weight = 100
    keyword_weight = 5

    weighted_score = (
        readability * readability_weight
        + sentiment_score * sentiment_weight
        + keyword_score * keyword_weight
        + grammar_penalty * grammar_weight
    )

    final_score = math.floor(weighted_score/2)

    # Debugging output
    print(f"Readability: {readability}, Grammar Penalty: {grammar_penalty}, Sentiment Score: {sentiment_score}, Keyword Score: {keyword_score}, Final Score: {final_score}")

    if final_score < 0:
        final_score = 0
    elif final_score > 100:
        final_score = 100

    return final_score