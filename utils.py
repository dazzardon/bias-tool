# utils.py

import yaml
import json
import re
import hashlib
import os
import requests
from bs4 import BeautifulSoup

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def is_valid_url(url):
    return re.match(r'^(http|https)://', url) is not None

def sanitize_text(text):
    return text.strip()

def load_users():
    try:
        if not os.path.exists('users.json'):
            return []
        with open('users.json', 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Handle corrupted JSON by resetting the file
        print("users.json is corrupted. Resetting the file.")
        return []
    except Exception as e:
        print(f"Error loading users: {e}")
        return []

def save_users(users):
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

def hash_password(password):
    try:
        return hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        print(f"Error hashing password: {e}")
        return None

def verify_password(hashed_password, input_password):
    return hashed_password == hashlib.sha256(input_password.encode()).hexdigest()

def is_valid_username(username):
    return re.match(r'^[a-zA-Z0-9]{3,30}$', username) is not None

def is_strong_password(password):
    return len(password) >= 8 and re.search(r'[^a-zA-Z0-9]', password)

def load_user_preferences(username, config):
    users = load_users()
    for user in users:
        if user['username'].lower() == username.lower():
            return {
                'bias_terms': user.get('bias_terms', config.get('bias_terms', []))
            }
    return {
        'bias_terms': config.get('bias_terms', [])
    }

def save_analysis_to_history(analysis_data, username):
    history_file = f'history_{username}.json'
    history = []
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                history = json.load(file)
    except json.JSONDecodeError:
        # Handle corrupted JSON by resetting the history
        print(f"{history_file} is corrupted. Resetting the file.")
        history = []
    except Exception as e:
        print(f"Error loading history: {e}")
        history = []
    
    history.append(analysis_data)
    try:
        with open(history_file, 'w') as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

def load_user_history(username):
    history_file = f'history_{username}.json'
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # Handle corrupted JSON by resetting the history
        print(f"{history_file} is corrupted. Resetting the file.")
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def fetch_article_text(url):
    """
    Fetches the main article text from the provided URL.
    This function uses requests and BeautifulSoup to scrape the article content.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MediaBiasTool/1.0; +https://example.com/bias-tool)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        html_content = response.text

        # Use BeautifulSoup to parse the HTML and extract the article text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Attempt to extract the main content of the article
        # This is a simplified example and may need adjustments based on the website's structure
        # Common tags or classes used for article content can be added here
        article_text = ''

        # Try common tags and classes
        article_tags = [
            {'name': 'article'},
            {'name': 'div', 'class_': 'article-content'},
            {'name': 'div', 'class_': 'entry-content'},
            {'name': 'div', 'class_': 'post-content'},
            {'name': 'div', 'id': 'article-body'},
            {'name': 'div', 'class_': 'story-body'},
            {'name': 'div', 'class_': 'main-content'},
            {'name': 'div', 'class_': 'content'},
        ]

        for tag in article_tags:
            elements = soup.find_all(tag.get('name'), class_=tag.get('class_'), id=tag.get('id'))
            if elements:
                for element in elements:
                    article_text += element.get_text(separator=' ', strip=True) + ' '
                if article_text:
                    break  # Exit if content is found

        # If no content found, fallback to extracting all paragraph texts
        if not article_text:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                article_text += p.get_text(separator=' ', strip=True) + ' '

        article_text = article_text.strip()
        return article_text if article_text else None
    except Exception as e:
        print(f"Error fetching article text from {url}: {e}")
        return None
