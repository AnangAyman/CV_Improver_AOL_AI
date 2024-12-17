import language_tool_python
from textstat import flesch_reading_ease # Flesch metrics
import math
from transformers import pipeline

grammar_tool = language_tool_python.LanguageTool('en-US')
# Initialize grammar correction model and sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def heuristic_score(text):
    # Step 1: Grammar check
    matches = grammar_tool.check(text)
    grammar_penalty = len(matches)

    # Step 2: Readability score
    readability = flesch_reading_ease(text)

    # Step 3: Sentiment analysis (encourages positive, professional tone)
    sentiment_result = sentiment_analyzer(text)
    sentiment_score = sentiment_result[0]['score'] if sentiment_result[0]['label'] == "POSITIVE" else -sentiment_result[0]['score']

    # Step 4: Keyword presence for CVs (e.g., "team", "managed", "developed")
    cv_keywords = ["team", "managed", "developed", "analyzed", "designed", "implemented"]
    stemmed_text = text.lower().split()  # Simplified for keywords
    keyword_score = sum(1 for word in stemmed_text if word in cv_keywords)

    # Step 5: Weighting and combining scores
    grammar_weight = -5
    readability_weight = 1
    sentiment_weight = 2
    keyword_weight = 3

    weighted_score = (
        readability * readability_weight
        + sentiment_score * sentiment_weight
        + keyword_score * keyword_weight
        + grammar_penalty * grammar_weight
    )

    final_score = math.floor(weighted_score)

    # Normalize final score between 0 and 100
    final_score = max(0, min(final_score, 100))

    # Debugging output
    # print(f"Readability: {readability}, Grammar Penalty: {grammar_penalty}, Sentiment Score: {sentiment_score}, Keyword Score: {keyword_score}, Final Score: {final_score}")

    return final_score