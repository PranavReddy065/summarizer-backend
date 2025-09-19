import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# Imports for new features
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import fitz # PyMuPDF

# --- CONFIGURATION ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
SUMMARIZATION_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# --- FLASK APP ---
app = Flask(__name__)
CORS(app) 

# --- HELPER FUNCTIONS ---

def get_text_from_article(url):
    """Fetches and extracts text from a standard article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except requests.RequestException as e:
        print(f"Error fetching article URL: {e}")
        return None

# Replace the old function with this correct version in app.py

def get_text_from_youtube(url):
    """Fetches the transcript from a YouTube video URL."""
    try:
        video_id = parse_qs(urlparse(url).query).get('v', [None])[0]
        if not video_id: 
            return {"error": "Invalid YouTube URL. Could not find video ID."}
        
        # The library call is correct here, ensuring no typos
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"Could not get YouTube transcript: {e}")
        return {"error": f"Could not retrieve transcript. It may be disabled for this video or unavailable in your region. Details: {e}"}
    
def get_text_from_pdf(url):
    """Downloads a PDF from a URL and extracts its text."""
    try:
        pdf_response = requests.get(url, timeout=20)
        pdf_response.raise_for_status()
        with fitz.open(stream=pdf_response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Could not get PDF text: {e}")
        return {"error": f"Failed to read the PDF file. It might be corrupted or an image-based PDF. Details: {e}"}

def summarize_text_free(text_to_summarize):
    """Calls the Hugging Face API to summarize the text."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    truncated_text = text_to_summarize[:1024*4] # Truncate text to a reasonable length
    task_prompt = f"Summarize the following text in a detailed and structured way:\n\n{truncated_text}"
    payload = {"inputs": task_prompt, "parameters": {"min_length": 50, "max_length": 250}}
    try:
        response = requests.post(SUMMARIZATION_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        summary = response.json()
        return summary[0]['summary_text']
    except Exception as e:
        print(f"API Error: {e}")
        return {"error": "Model is loading or an API issue occurred. Please try again in a moment."}

# --- API ENDPOINT ---
@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    extracted_text = None
    
    # --- INTELLIGENT TOOL SELECTION ---
    if "youtube.com" in url or "youtu.be" in url:
        print("Source identified as YouTube.")
        extracted_text = get_text_from_youtube(url)
    elif url.lower().endswith('.pdf'):
        print("Source identified as PDF.")
        extracted_text = get_text_from_pdf(url)
    else:
        print("Source identified as Article.")
        extracted_text = get_text_from_article(url)

    # Check for errors from the extraction functions
    if isinstance(extracted_text, dict) and 'error' in extracted_text:
        return jsonify(extracted_text), 400
    
    if not extracted_text:
        return jsonify({"error": "Failed to extract any text from the source."}), 500
        
    summary = summarize_text_free(extracted_text)
    if isinstance(summary, dict) and 'error' in summary:
        return jsonify(summary), 500

    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, port=5000)