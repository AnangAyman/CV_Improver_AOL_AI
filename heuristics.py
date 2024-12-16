# import language_tool_python
# from textstat import flesch_reading_ease # Flesch metrics

# tool = language_tool_python.LanguageTool('en-US')

# def heuristic_score(text):
#     """
#     Heuristic: Combines grammar errors and readability.
#     """
#     # Grammar check
#     matches = tool.check(text)
#     grammar_penalty = len(matches)
    
#     # Readability score
#     readability = flesch_reading_ease(text)
    
#     # Combine scores
#     score = readability - grammar_penalty * 5  # Penalize grammar errors
#     return score