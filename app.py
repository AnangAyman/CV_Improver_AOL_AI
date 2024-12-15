from flask import Flask, request, jsonify, render_template
import requests
from AI import improve_cv_text


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
        return jsonify({'improved_text': improved_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
