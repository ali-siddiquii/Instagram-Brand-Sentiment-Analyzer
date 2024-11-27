# Instagram-Brand-Sentiment-Analyzer

A Flask-based web application that leverages machine learning models to analyze the sentiment of Instagram posts' captions and images for a brand, providing insights into public perception and engagement metrics for any public Instagram profile


## .env File Configuration (For Instaloader access)

INSTAGRAM_USERNAME: Instagram username for login

INSTAGRAM_PASSWORD: Instagram password for login


## Required libraries
Instaloader: For downloading Instagram posts, metadata, and handling Instagram sessions.

Flask: To create and manage the web server and user interfaces.

Transformers: Provides access to pre-trained models for natural language processing tasks.

Torch: Machine learning library from PyTorch to support a wide range of deep learning tasks.

Pandas & numpy: For data manipulation and numerical operations.

Requests: To handle HTTP requests for external image processing.

Pillow (PIL Fork): For image processing tasks within Python.

Dotenv: Loads environment variables from a .env file.
