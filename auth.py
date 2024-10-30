# auth.py

import bcrypt
import logging
import re
import json
import os

logger = logging.getLogger(__name__)

def hash_password(password):
    """
    Hash a password for storing using bcrypt.
    """
    try:
        # Generate a salt and hash the password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        return None

def verify_password(stored_password, provided_password):
    """
    Verify a stored password against one provided by user using bcrypt.
    """
    try:
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def is_strong_password(password):
    """
    Check if the password meets strength requirements:
    - At least 8 characters
    - Contains at least one special character
    """
    if len(password) < 8:
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

def is_valid_username(username):
    """
    Validate that the username contains only alphanumeric characters and is between 3 to 30 characters.
    """
    return re.match(r'^[A-Za-z0-9]{3,30}$', username) is not None

def load_users(users_path='users.json'):
    """
    Load users from a JSON file.
    """
    if not os.path.exists(users_path):
        with open(users_path, 'w') as f:
            json.dump([], f)
        logger.info("Created new users.json file.")

    try:
        with open(users_path, 'r') as f:
            users = json.load(f)
        logger.info("Users loaded successfully.")
        return users
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return []

def save_users(users, users_path='users.json'):
    """
    Save users to a JSON file.
    """
    try:
        with open(users_path, 'w') as f:
            json.dump(users, f, indent=4)
        logger.info("Users saved successfully.")
    except Exception as e:
        logger.error(f"Error saving users: {e}")
