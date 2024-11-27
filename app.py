# Instagram Sentiment Analyzer 2
#pip install instaloader flask transformers torch torchvision pandas numpy requests pillow
#.env file needs variables INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD for instagrapi

import os
from instagrapi import Client
from transformers import pipeline
from PIL import Image
import io
import requests
from collections import defaultdict
from dotenv import load_dotenv
import time
from flask import Flask, request, jsonify, render_template
import torch

import numpy as np
import pandas as pd
import instaloader
from instaloader import Instaloader, Profile

app = Flask(__name__)
client = Client()

device = 0 if torch.cuda.is_available() else -1

# Log in using environment variables
load_dotenv()

loader = instaloader.Instaloader()

try:
    loader.login(os.getenv('INSTAGRAM_USERNAME'), os.getenv('INSTAGRAM_PASSWORD'))
    print('<<<Log in succuessful')
    loader.save_session_to_file()
except Exception as e:
    print(f"<<<An unexpected error occurred during login: {str(e)}")

def analyze_instagram_sentiment(username, num_posts=5):
    """
    Analyze brand sentiment from Instagram posts using various AI models.

    Parameters:
    username (str): Instagram username/handle
    num_posts (int): Number of recent posts to analyze

    Returns:
    dict: Sentiment analysis results
    """

    try:
        profile = Profile.from_username(loader.context, username)
    except Exception as e:
        print("Profile does not exist.")
        return {
            'error': 'Profile not found',
            'overall_caption_sentiment': 0,
            'overall_image_sentiment': 0,
            'overall_engagement_sentiment': 0,
            'overall_followers_sentiment': 0,
            'caption_sentiments': [],
            'image_sentiments': []
        }

    # Initialize sentiment analyzers
    text_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="pt",
    device=device)
    image_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50",
    framework="pt",
    device=device)


    count = 0
    followers_count = profile.followers
    results = defaultdict(list)

    for post in profile.get_posts():
        if count >= num_posts:
            break

        # Process caption sentiment
        if post.caption:
            caption_sentiment = text_sentiment(post.caption)[0]
            results['caption_sentiments'].append({
                'text': post.caption,
                'score': float(caption_sentiment['score']),
                'label': caption_sentiment['label']
            })

        # Process image sentiment
        image_url = post.url
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                image_pred = image_classifier(image)[0]
                results['image_sentiments'].append({
                    'url': image_url,
                    'label': image_pred['label'],
                    'score': float(image_pred['score'])
                })
            except Exception as e:
                print(f"Failed to process image for URL: {image_url}, Error: {e}")


        engagement_sentiment = {
            'likes': post.likes
        }

        results['engagement_sentiments'].append(engagement_sentiment)

        count = count + 1

    # Compile and return the analysis results

    # Calculate overall metrics
    caption_scores = [s['score'] for s in results['caption_sentiments']]
    image_scores = [s['score'] for s in results['image_sentiments']]
    engagement_scores = [eng['likes'] for eng in results['engagement_sentiments']]

    
    analysis = {
        'overall_caption_sentiment': np.mean(caption_scores) if caption_scores else 0,
        'overall_image_sentiment': np.mean(image_scores) if image_scores else 0,
        'overall_engagement_sentiment': np.mean(engagement_scores) if engagement_scores else 0,
        'overall_followers_sentiment': followers_count,
        'caption_sentiments': results['caption_sentiments'],
        'image_sentiments': results['image_sentiments'],
        'engagement_sentiments': results['engagement_sentiments']
    }

    return analysis

def safe_request(call, max_attempts=5, delay=60):
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                print("Rate limit exceeded, sleeping for", delay, "seconds")
                time.sleep(delay)
            elif "JSONDecodeError" in str(e):
                # Debug the response content
                print("Response content:", e.response.text if hasattr(e, "response") else "No response")
            else:
                break
    return None

# Flask routes
@app.route('/')
def home():
    """
    Render the input form for Instagram username submission.
    """
    return """
    <html>
        <head>
            <title>Instagram Sentiment Analysis</title>
        </head>
        <body>
            <h1>Instagram Sentiment Analysis</h1>
            <form action="/analyze" method="post">
                <label for="username">Instagram Username:</label>
                <input type="text" id="username" name="username" required>
                <br><br>
                <label for="num_posts">Number of Posts to Analyze:</label>
                <input type="number" id="num_posts" name="num_posts" value="5" min="1" max="20">
                <br><br>
                <button type="submit">Analyze</button>
            </form>
        </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handle form submission and return sentiment analysis results as HTML.
    """
    username = request.form['username']
    num_posts = int(request.form.get('num_posts', 5))
    
    if not username:
        return """
        <h1>Error</h1>
        <p>Username is required. Please go back and enter a username.</p>
        """

    analysis = analyze_instagram_sentiment(username, num_posts=num_posts)
    
    if analysis:
        # Render results as HTML
        return f"""
        <h1>Sentiment Analysis Results for @{username}</h1>
        <p><b>Overall Caption Sentiment:</b> {analysis.get('overall_caption_sentiment', 0):.2f}</p>
        <p><b>Overall Image Sentiment:</b> {analysis.get('overall_image_sentiment', 0):.2f}</p>
        <p><b>Overall Likes Engagement Sentiment:</b> {analysis.get('overall_engagement_sentiment', 0):.2f}</p>
        <p><b>Total Followers:</b> {analysis.get('overall_followers_sentiment', 0):,}</p>
        <h2>Caption Sentiments:</h2>
        <ul>
            {''.join(f"<li>{item['text']} - {item['label']} (Score: {item['score']:.2f})</li>" for item in analysis.get('caption_sentiments', []))}
        </ul>
    
        <h2>Image Sentiments:</h2>
        <ul>
            {''.join(f"<li><img src='{item['url']}' alt='Post Image' width='100'> - {item['label']} (Score: {item['score']:.2f})</li>" for item in analysis.get('image_sentiments', []))}
        </ul>
        """
    else:
        return "<h1>Analysis could not be performed. Please try again later.</h1>"

if __name__ == '__main__':
    print('http://127.0.0.1:5000/')
    app.run(debug=True)



