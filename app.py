
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
# Get your Hugging Face API token from https://huggingface.co/settings/tokens
# It's best practice to set this as an environment variable.
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

SUMMARIZATION_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# --- FLASK APP ---
app = Flask(__name__)
# This enables Cross-Origin Resource Sharing (CORS)
# and allows your frontend to make requests to this backend.
CORS(app) 

# --- HELPER FUNCTIONS ---
def get_article_text(url):
    """Fetches and extracts the main text from a given URL."""
    try:
        # Define a browser-like User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Add the headers to the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def summarize_text_free(text_to_summarize):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    truncated_text = text_to_summarize[:1024*4] # Truncate text

    # Create a clear instruction for the Flan-T5 model
    task_prompt = f"Summarize the following text:\n\n{truncated_text}"
    
    payload = {
        "inputs": task_prompt,
        "parameters": {"min_length": 30, "max_length": 150} 
    }
    
    try:
        response = requests.post(SUMMARIZATION_API_URL, headers=headers, json=payload, timeout=60)
        summary = response.json()
        return summary[0]['summary_text']
    except Exception as e:
        print(e)
        return {"error": "Model is loading or an issue occurred. Please try again in a moment."}
    
    try:
        response = requests.post(SUMMARIZATION_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        summary = response.json()
        return summary[0]['summary_text']
    except requests.RequestException as e:
        print(f"API Error: {e}")
        return {"error": "Failed to get summary from the API. The model might be loading, please try again in a minute."}

# --- API ENDPOINT ---
@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    """The main endpoint that receives a URL and returns a summary."""
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Step 1: Scrape the text from the article
    article_text = get_article_text(url)
    if not article_text:
        return jsonify({"error": "Failed to retrieve or parse article text"}), 500
        
    # Step 2: Summarize the text
    summary = summarize_text(article_text)
    if isinstance(summary, dict) and "error" in summary:
        return jsonify(summary), 500

    return jsonify({"summary": summary})

if __name__ == '__main__':
    # This runs the app in debug mode locally.
    # We use gunicorn to run it in production on Render.

    app.run(debug=True, port=5000)