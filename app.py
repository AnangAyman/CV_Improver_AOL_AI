from flask import Flask, request, jsonify, render_template
import requests
from AI import improve_cv_text
from heuristics import heuristic_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/improve', methods=['POST'])
def improve_text():
    user_input = request.json.get('text', '')
    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400

    # Use GPT-2 function to improve the text
    try:
        improved_text = improve_cv_text(user_input)

        previous_score = heuristic_score (user_input)
        new_score = heuristic_score(improved_text)
        
        return jsonify({'improved_text': improved_text, 'previous_score' : previous_score, 'new_score': new_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
