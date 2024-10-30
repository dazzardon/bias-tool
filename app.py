# app.py

import logging
import streamlit as st
import datetime
import os
import json
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BertTokenizerFast
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
import spacy
import yaml
import bcrypt
import plotly.express as px

# --- Configure Logging ---
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- User Management Functions ---

def hash_password(password):
    """
    Hash a password for storing using bcrypt.
    """
    try:
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

def register_user(config):
    st.title("Register")
    st.write("Create a new account to access personalized features.")

    with st.form("registration_form"):
        username = st.text_input("Choose a Username", key="register_username")
        password = st.text_input("Choose a Password", type='password', key="register_password")
        password_confirm = st.text_input("Confirm Password", type='password', key="register_password_confirm")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not password or not password_confirm:
                st.error("Please fill out all fields.")
                return
            if not is_valid_username(username):
                st.error("Username must be 3-30 characters long and contain only letters and numbers.")
                return
            if password != password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(password):
                st.error("Password must be at least 8 characters long and include at least one special character.")
                return
            users = load_users()
            if any(user['username'].lower() == username.lower() for user in users):
                st.error("Username already exists. Please choose a different one.")
                return
            hashed_pwd = hash_password(password)
            if not hashed_pwd:
                st.error("Error hashing password. Please try again.")
                return
            new_user = {
                "username": username,
                "password": hashed_pwd,
                "preferences": {},
                "bias_terms": config['bias_terms']
            }
            users.append(new_user)
            save_users(users)
            st.success("Registration successful. You can now log in.")
            logger.info(f"New user registered: {username}")

def login_user(config):
    st.title("Login")
    st.write("Access your account to view history and customize settings.")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type='password', key="login_password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not username or not password:
                st.error("Please enter both username and password.")
                return
            users = load_users()
            user = next((user for user in users if user['username'].lower() == username.lower()), None)
            if user and verify_password(user['password'], password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = user['username']
                st.session_state['bias_terms'] = user.get('bias_terms', config.get('bias_terms', []))
                st.success("Logged in successfully.")
                logger.info(f"User '{username}' logged in successfully.")
            else:
                st.error("Invalid username or password.")
                logger.warning(f"Failed login attempt for username: '{username}'.")

def logout_user():
    logger.info(f"User '{st.session_state['username']}' logged out.")
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''
    st.session_state['bias_terms'] = []
    st.sidebar.success("Logged out successfully.")

# --- Configuration Loader ---

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        # Create a default config.yaml if it doesn't exist
        default_config = {
            'performance': {
                'batch_size': 32,
                'enable_caching': True,
                'cache_size': 256
            },
            'api': {
                'endpoint': "https://api.yourapp.com/analyze",
                'timeout': 30,
                'retries': 3
            },
            'database': {
                'type': "postgresql",
                'host': "localhost",
                'port': 5432,
                'username': "your_username",
                'password': "your_password",
                'db_name': "biased_news_db"
            },
            'security': {
                'api_keys': [
                    "your_api_key_1",
                    "your_api_key_2"
                ]
            },
            'i18n': {
                'supported_languages': [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "zh"
                ],
                'default_language': "en"
            },
            'metadata': {
                'version': "1.0.0",
                'author': "Your Name",
                'last_updated': "2024-04-27",
                'description': "Configuration for biased media news app with expanded bias terms and scoring."
            },
            'bias_terms': [
                # Add your bias terms here
                "alarming",
                "allegations",
                "unfit",
                # ... other terms
            ],
            'scoring': {
                'sentiment_weight': 0.4,
                'bias_weight': 0.3,
                'propaganda_weight': 0.3,
                'max_bias': 20,
                'max_propaganda': 20
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        logger.info("Created default config.yaml file.")
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return None

# --- Model Definition ---

class BertForTokenAndSequenceJointClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels = config.num_labels
        self.num_sequence_labels = 2  # Propaganda or Non-Propaganda

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)

        self.init_weights()

        # Label mappings
        self.token_tags = {
            0: 'O',
            1: 'Appeal_to_Authority',
            2: 'Appeal_to_fear-prejudice',
            3: 'Bandwagon,Reductio_ad_hitlerum',
            4: 'Black-and-White_Fallacy',
            5: 'Causal_Oversimplification',
            6: 'Doubt',
            7: 'Exaggeration,Minimisation',
            8: 'Flag-Waving',
            9: 'Loaded_Language',
            10: 'Name_Calling,Labeling',
            11: 'Repetition',
            12: 'Slogans',
            13: 'Thought-terminating_Cliches',
            14: 'Whataboutism,Straw_Men,Red_Herring'
        }

        self.sequence_tags = {
            0: 'Non-Propaganda',
            1: 'Propaganda'
        }

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sequence_output = self.dropout(sequence_output)
        sequence_logits = self.sequence_classifier(sequence_output)

        token_output = outputs.last_hidden_state
        token_output = self.dropout(token_output)
        token_logits = self.token_classifier(token_output)

        return {
            'sequence_logits': sequence_logits,
            'token_logits': token_logits
        }

# --- Initialize Models ---

@st.cache_resource
def initialize_models(config):
    # Initialize Sentiment Analysis Model
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=config['models']['sentiment_model'],
        tokenizer=config['models']['sentiment_model'],
        device=-1  # Use CPU
    )
    # Initialize Propaganda Detection Model
    propaganda_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    propaganda_model = BertForTokenAndSequenceJointClassification.from_pretrained(
        "QCRI/PropagandaTechniquesAnalysis-en-BERT",
        revision="v0.1.0",
    )
    # Initialize SpaCy NLP Model
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading SpaCy model: {e}")
        nlp = spacy.blank("en")  # Fallback to a blank model

    models = {
        'sentiment': sentiment_pipeline,
        'propaganda_model': propaganda_model,
        'propaganda_tokenizer': propaganda_tokenizer,
        'nlp': nlp
    }
    return models

# --- Analysis Functions ---

def split_text_into_chunks(text, max_chars=500):
    """
    Split text into chunks that do not exceed max_chars.
    Splitting is done at sentence boundaries to preserve context.
    """
    try:
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}", exc_info=True)
        return [text.strip()]

def explain_bias(terms):
    """
    Provide user-friendly explanations for detected bias terms.
    """
    explanations = [f"The term '{term}' indicates potential bias." for term in terms]
    return " ".join(explanations)

def describe_propaganda_term(term):
    """
    Provide user-friendly descriptions for each propaganda technique.
    """
    descriptions = {
        "Appeal To Authority": "using references to influential people to support an argument without substantial evidence",
        "Appeal To Fear-Prejudice": "playing on people's fears or prejudices to influence their opinion",
        "Bandwagon": "suggesting that something is good because many people do it",
        "Reductio Ad Hitlerum": "comparing an opponent to Hitler or Nazis to discredit them",
        "Black-And-White Fallacy": "presenting only two options when more exist",
        "Causal Oversimplification": "reducing a complex issue to a single cause",
        "Doubt": "casting doubt on an idea without sufficient justification",
        "Exaggeration": "making something seem better or worse than it is",
        "Minimisation": "downplaying the significance of an issue",
        "Flag-Waving": "appealing to patriotism or nationalism to support an argument",
        "Loaded Language": "using emotionally charged words to influence opinion",
        "Name Calling": "attaching negative labels to individuals or groups without evidence",
        "Labeling": "applying simplistic labels to complex situations or people",
        "Repetition": "repeating a message multiple times to reinforce it",
        "Slogans": "using catchy phrases to simplify complex ideas",
        "Thought-Terminating Cliches": "using clichés to end debate or discussion",
        "Whataboutism": "distracting from the main issue with irrelevant points",
        "Straw Man": "misrepresenting an opponent's argument to make it easier to attack",
        "Red Herring": "introducing an irrelevant topic to divert attention from the original issue",
        "Propaganda": "the use of information, ideas, or rumors to influence public opinion",
        "Non-Propaganda": "no propaganda detected"
    }
    return descriptions.get(term, "a propaganda technique")

def explain_propaganda(techniques):
    """
    Provide detailed explanations for detected propaganda techniques.
    """
    explanations = []
    for term in techniques:
        description = describe_propaganda_term(term)
        explanations.append(f"The text involves {description}.")
    return " ".join(explanations)

def calculate_final_score(sentiment_score, bias_count, propaganda_count, config):
    """
    Calculate a final score out of 100.
    Higher scores indicate less bias and propaganda.
    """
    try:
        sentiment_weight = config.get('scoring', {}).get('sentiment_weight', 0.4)
        bias_weight = config.get('scoring', {}).get('bias_weight', 0.3)
        propaganda_weight = config.get('scoring', {}).get('propaganda_weight', 0.3)

        # Normalize sentiment_score from -1 to 1 to a 0 to 100 scale
        sentiment_subscore = ((sentiment_score + 1) / 2) * 100

        # Cap counts at max values to prevent excessive penalties
        max_bias = config.get('scoring', {}).get('max_bias', 10)
        max_propaganda = config.get('scoring', {}).get('max_propaganda', 10)
        bias_penalty = min(bias_count / max_bias, 1) * 100 if max_bias > 0 else 0
        propaganda_penalty = min(propaganda_count / max_propaganda, 1) * 100 if max_propaganda > 0 else 0

        # Calculate final score
        final_score = (
            sentiment_subscore * sentiment_weight +
            (100 - bias_penalty) * bias_weight +
            (100 - propaganda_penalty) * propaganda_weight
        )
        final_score = max(min(final_score, 100), 0)
        final_score = round(final_score, 2)
        logger.info(f"Final Score Calculated: {final_score}")
        return final_score
    except Exception as e:
        logger.error(f"Error calculating final score: {e}", exc_info=True)
        return 0.0

def perform_analysis(article_text, title, features, models, config, bias_terms=None):
    """
    Perform analysis on the provided article text.
    Handles long texts by splitting into chunks.
    """
    if not models:
        logger.error("Models are not loaded. Cannot perform analysis.")
        return None

    analysis_data = {
        'title': title if title else 'Untitled',
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sentiment_score': 0.0,
        'sentiment_label': 'Neutral',
        'bias_score': 0,
        'propaganda_score': 0,
        'final_score': 0.0,
        'entities': {},
        'biased_sentences': [],
        'propaganda_sentences': [],
        'username': 'guest'  # Default username
    }

    total_sentiments = []
    bias_count = 0
    propaganda_count = 0
    all_biased_sentences = []
    all_propaganda_sentences = []

    chunks = split_text_into_chunks(article_text, max_chars=500)

    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Analyzing chunk {idx}/{len(chunks)}")
        # Sentiment Analysis
        if "Sentiment Analysis" in features:
            try:
                sentiments = models['sentiment'](chunk)
                for sentiment in sentiments:
                    label = sentiment['label'].upper()
                    score = sentiment['score']
                    if label == 'NEGATIVE':
                        score = -score
                    elif label == 'POSITIVE':
                        score = score
                    else:
                        score = 0
                    total_sentiments.append(score)
            except Exception as e:
                logger.error(f"Error during sentiment analysis: {e}")

        # Bias Detection
        if "Bias Detection" in features:
            try:
                terms = bias_terms if bias_terms else models.get('bias_terms', [])
                detected = [term for term in terms if re.search(r'\b' + re.escape(term) + r'\b', chunk, re.IGNORECASE)]
                if detected:
                    unique_terms = set(detected)
                    bias_count += len(unique_terms)
                    biased_sentence = {
                        'sentence': chunk,
                        'detected_terms': list(unique_terms),
                        'explanation': explain_bias(list(unique_terms))
                    }
                    all_biased_sentences.append(biased_sentence)
                    logger.info(f"Bias detected in chunk {idx}: {unique_terms}")
            except Exception as e:
                logger.error(f"Error during bias detection: {e}")

        # Propaganda Detection
        if "Propaganda Detection" in features:
            try:
                propaganda_model = models.get('propaganda_model')
                propaganda_tokenizer = models.get('propaganda_tokenizer')
                if propaganda_model is None or propaganda_tokenizer is None:
                    logger.error("Propaganda model or tokenizer not loaded.")
                else:
                    inputs = propaganda_tokenizer.encode_plus(
                        chunk,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    outputs = propaganda_model(**inputs)

                    # Sequence Classification
                    sequence_logits = outputs['sequence_logits']
                    sequence_class_index = torch.argmax(sequence_logits, dim=-1)
                    sequence_class = propaganda_model.sequence_tags.get(sequence_class_index.item(), 'Non-Propaganda')

                    sequence_confidence = torch.softmax(sequence_logits, dim=-1)[0][sequence_class_index].item()

                    techniques = set()

                    # Define a confidence threshold to ensure accurate detection
                    confidence_threshold = 0.7

                    if sequence_class.lower() == 'propaganda' and sequence_confidence > confidence_threshold:
                        # Token Classification
                        token_logits = outputs['token_logits']
                        token_class_indices = torch.argmax(token_logits, dim=-1)

                        tokens = propaganda_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                        attention_mask = inputs['attention_mask'][0]

                        for token, tag_idx, mask in zip(tokens, token_class_indices[0], attention_mask):
                            if mask == 1 and token not in ["[CLS]", "[SEP]", "[PAD]"]:
                                tag = propaganda_model.token_tags.get(tag_idx.item(), 'O')
                                if tag != 'O':
                                    normalized_tag = tag.replace('_', ' ').title()
                                    techniques.add(normalized_tag)

                        if techniques:
                            propaganda_count += len(techniques)
                            propaganda_sentence = {
                                'sentence': chunk,
                                'detected_terms': list(techniques),
                                'explanation': explain_propaganda(list(techniques))
                            }
                            all_propaganda_sentences.append(propaganda_sentence)
                            logger.info(f"Propaganda techniques detected in chunk {idx}: {techniques}")
                        else:
                            propaganda_count += 1
                            propaganda_sentence = {
                                'sentence': chunk,
                                'detected_terms': [sequence_class],
                                'explanation': explain_propaganda([sequence_class]),
                            }
                            all_propaganda_sentences.append(propaganda_sentence)
                            logger.info(f"General propaganda detected in chunk {idx}: {sequence_class}")
            except Exception as e:
                logger.error(f"Error during propaganda detection: {e}")

        # Entity Detection
        if "Pattern Detection" in features:
            try:
                nlp = models['nlp']
                doc = nlp(chunk)
                for ent in doc.ents:
                    entities = analysis_data['entities']
                    if ent.text in entities:
                        entities[ent.text]['types'].add(ent.label_)
                        entities[ent.text]['count'] += 1
                    else:
                        entities[ent.text] = {
                            'types': set([ent.label_]),
                            'count': 1
                        }
            except Exception as e:
                logger.error(f"Error during entity extraction: {e}")

    # Aggregate Sentiment Scores
    if total_sentiments:
        average_sentiment = sum(total_sentiments) / len(total_sentiments)
        analysis_data['sentiment_score'] = average_sentiment
        if average_sentiment >= 0.05:
            analysis_data['sentiment_label'] = 'Positive'
        elif average_sentiment <= -0.05:
            analysis_data['sentiment_label'] = 'Negative'
        else:
            analysis_data['sentiment_label'] = 'Neutral'
        logger.info(f"Aggregated Sentiment: {analysis_data['sentiment_label']} with average score {average_sentiment}")

    analysis_data['bias_score'] = bias_count
    analysis_data['propaganda_score'] = propaganda_count
    analysis_data['biased_sentences'] = all_biased_sentences
    analysis_data['propaganda_sentences'] = all_propaganda_sentences

    # Final Score Calculation
    analysis_data['final_score'] = calculate_final_score(
        analysis_data['sentiment_score'],
        analysis_data['bias_score'],
        analysis_data['propaganda_score'],
        config
    )

    return analysis_data

# --- Display Functions ---

def display_results(data, unique_id='', is_nested=False, save_to_history=True):
    with st.container():
        st.markdown(f"## {data.get('title', 'Untitled Article')}")
        st.markdown(f"**Date:** {data.get('date', 'N/A')}")
        st.markdown(f"**Analyzed by:** {data.get('username', 'guest')}")

        # Overview Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sentiment_label = data.get('sentiment_label', 'Neutral')
            sentiment_score = data.get('sentiment_score', 0.0)
            if sentiment_label == "Positive":
                sentiment_color = "#28a745"  # Green
            elif sentiment_label == "Negative":
                sentiment_color = "#dc3545"  # Red
            else:
                sentiment_color = "#6c757d"  # Gray
            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{sentiment_label}</span>", unsafe_allow_html=True)
            st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")

        with col2:
            bias_count = data.get('bias_score', 0)
            st.markdown("**Bias Count**")
            st.metric(label="Bias Terms Detected", value=f"{int(bias_count)}")

        with col3:
            st.markdown("**Propaganda Count**")
            propaganda_count = data.get('propaganda_score', 0)
            st.metric(label="Propaganda Techniques Detected", value=f"{int(propaganda_count)}")

        with col4:
            final_score = data.get('final_score', 0.0)
            st.markdown("**Final Score**")
            st.metric(
                label="Final Score",
                value=f"{final_score:.2f}",
                help="A score out of 100 indicating overall article quality, with higher scores being better. Higher scores reflect positive sentiment and lower levels of bias and propaganda."
            )

        st.markdown("---")  # Separator

        # Tabs for Different Analysis Sections
        tabs = st.tabs(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"])

        # --- Sentiment Analysis Tab ---
        with tabs[0]:
            st.markdown("### Sentiment Analysis")
            st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
            st.write(f"**Sentiment Score:** {sentiment_score:.2f}")

        # --- Bias Detection Tab ---
        with tabs[1]:
            st.markdown("### Bias Detection")
            st.write(f"**Bias Count:** {int(bias_count)} bias terms detected.")

            if data.get('biased_sentences'):
                if not is_nested:
                    with st.expander("View Biased Sentences", expanded=False):
                        for idx, item in enumerate(data['biased_sentences'], 1):
                            st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                            st.markdown(f"   - **Detected Bias Terms:** {', '.join(item['detected_terms'])}")
                            st.markdown(f"   - **Explanation:** {item['explanation']}")
                else:
                    for idx, item in enumerate(data['biased_sentences'], 1):
                        st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                        st.markdown(f"   - **Detected Bias Terms:** {', '.join(item['detected_terms'])}")
                        st.markdown(f"   - **Explanation:** {item['explanation']}")
            else:
                st.write("No biased sentences detected.")

        # --- Propaganda Detection Tab ---
        with tabs[2]:
            st.markdown("### Propaganda Detection")
            st.write(f"**Propaganda Count:** {int(propaganda_count)} propaganda techniques detected.")

            if data.get('propaganda_sentences'):
                if not is_nested:
                    with st.expander("View Propaganda Sentences", expanded=False):
                        for idx, item in enumerate(data['propaganda_sentences'], 1):
                            st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                            st.markdown(f"   - **Detected Propaganda Techniques:** {', '.join(item['detected_terms'])}")
                            st.markdown(f"   - **Explanation:** {item['explanation']}")
                else:
                    for idx, item in enumerate(data['propaganda_sentences'], 1):
                        st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                        st.markdown(f"   - **Detected Propaganda Techniques:** {', '.join(item['detected_terms'])}")
                        st.markdown(f"   - **Explanation:** {item['explanation']}")
            else:
                st.write("No propaganda techniques detected.")

        st.markdown("---")  # Separator

        # Only save to history if save_to_history is True
        if save_to_history and st.session_state.get('logged_in', False):
            analysis_data = data.copy()
            analysis_data['username'] = st.session_state['username']
            save_analysis_to_history(analysis_data, st.session_state['username'])
            logger.info("Analysis saved to history automatically.")
            st.success("Analysis saved to your history.")

            # Provide Download Option for CSV
            csv_data = {
                'title': data.get('title', 'Untitled'),
                'date': data.get('date', ''),
                'sentiment_score': data.get('sentiment_score', 0.0),
                'sentiment_label': data.get('sentiment_label', 'Neutral'),
                'bias_count': data.get('bias_score', 0),
                'propaganda_count': data.get('propaganda_score', 0),
                'final_score': data.get('final_score', 0.0),
                'entities': data.get('entities', {}),
                'biased_sentences': data.get('biased_sentences', []),
                'propaganda_sentences': data.get('propaganda_sentences', [])
            }
            df_csv = pd.DataFrame([csv_data])
            csv_buffer = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Analysis as CSV",
                data=csv_buffer,
                file_name=f"analysis_{data.get('title', 'untitled').replace(' ', '_')}.csv",
                mime='text/csv',
                key=f"download_analysis_csv_{unique_id}"
            )

        # --- Feedback Section with Unique Key ---
        if not is_nested:
            st.markdown("---")
            st.markdown("### Provide Feedback")
            feedback = st.text_area(
                "Your Feedback",
                placeholder="Enter your feedback here...",
                height=100,
                key=f"feedback_text_area_{unique_id}"
            )
            if st.button("Submit Feedback", key=f"submit_feedback_{unique_id}"):
                if feedback:
                    # Save feedback to a JSON file
                    feedback_path = 'feedback.json'
                    feedback_entry = {
                        'username': st.session_state.get('username', 'guest'),
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'feedback': feedback
                    }
                    try:
                        if not os.path.exists(feedback_path):
                            with open(feedback_path, 'w') as f:
                                json.dump([], f)
                            logger.info("Created new feedback file.")
                        with open(feedback_path, 'r') as f:
                            feedback_data = json.load(f)
                        feedback_data.append(feedback_entry)
                        with open(feedback_path, 'w') as f:
                            json.dump(feedback_data, f, indent=4)
                        logger.info(f"Feedback saved from user '{st.session_state.get('username', 'guest')}'.")
                        st.success("Thank you for your feedback!")
                    except Exception as e:
                        logger.error(f"Error saving feedback: {e}", exc_info=True)
                        st.error("An error occurred while saving your feedback.")
                else:
                    st.warning("Please enter your feedback before submitting.")

def save_analysis_to_history(analysis_data, username):
    history_file = f'history_{username}.json'
    history = []
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                history = json.load(file)
    except json.JSONDecodeError:
        logger.error(f"{history_file} is corrupted. Resetting the file.")
        history = []
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        history = []

    history.append(analysis_data)
    try:
        with open(history_file, 'w') as file:
            json.dump(history, file, indent=4)
        logger.info("Analysis saved to history.")
    except Exception as e:
        logger.error(f"Error saving history: {e}")

def load_user_history(username):
    history_file = f'history_{username}.json'
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.error(f"{history_file} is corrupted. Resetting the file.")
        return []
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []

# --- Config Loader and Model Initialization ---

config = load_config('config.yaml')
if not config:
    st.error("Configuration file not found or invalid.")
    st.stop()

models = initialize_models(config)

# --- Helper Functions ---

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
        response.raise_for_status()
        html_content = response.text

        soup = BeautifulSoup(html_content, 'html.parser')

        article_text = ''

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
                    break

        if not article_text:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                article_text += p.get_text(separator=' ', strip=True) + ' '

        article_text = article_text.strip()
        return article_text if article_text else None
    except Exception as e:
        logger.error(f"Error fetching article text from {url}: {e}")
        return None

def sanitize_text(text):
    return text.strip()

# --- Analysis Functions (Already Defined Above) ---

# --- Main Application Functions ---

def single_article_analysis(features, config, models):
    st.header("Single Article Analysis")
    st.write("Enter the article URL or paste the article text below.")

    input_type = st.radio(
        "Select Input Type",
        ['Enter URL', 'Paste Article Text'],
        key="single_article_input_type"
    )
    if input_type == 'Enter URL':
        url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            key="single_article_url"
        ).strip()
        article_text = ''
    else:
        article_text = st.text_area(
            "Article Text",
            placeholder="Paste the article text here...",
            height=300,
            key="single_article_text"
        ).strip()
        url = ''

    title = st.text_input(
        "Article Title",
        value="Article",
        placeholder="Enter a title for the article",
        key="single_article_title"
    )

    if st.button("Analyze", key="analyze_single_article"):
        if input_type == 'Enter URL':
            if url:
                if fetch_article_text(url):
                    with st.spinner('Fetching the article...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            sanitized_text = sanitize_text(article_text_fetched)
                            st.success("Article text fetched successfully.")
                            article_text = sanitized_text  # Use sanitized text for analysis
                        else:
                            st.error("Failed to fetch article text. Please check the URL and try again.")
                            return
                else:
                    st.error("Please enter a valid URL.")
                    return
            else:
                st.error("Please enter a URL.")
                return
        else:
            if not article_text.strip():
                st.error("Please paste the article text.")
                return
            sanitized_text = sanitize_text(article_text)
            article_text = sanitized_text

        with st.spinner('Performing analysis...'):
            analysis_data = perform_analysis(
                article_text=article_text,
                title=title,
                features=features,
                models=models,
                config=config,
                bias_terms=st.session_state.get('bias_terms', config.get('bias_terms', []))
            )
            if st.session_state.get('logged_in', False):
                analysis_data['username'] = st.session_state['username']
            else:
                analysis_data['username'] = 'guest'

        if analysis_data:
            st.success("Analysis completed successfully.")
            display_results(analysis_data, unique_id=f"single_{title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        else:
            st.error("Failed to perform analysis on the provided article.")

def comparative_analysis(features, config, models):
    st.header("Comparative Analysis")
    st.write("Enter URLs or paste texts of multiple articles below for comparison.")

    num_articles = st.number_input("Number of Articles", min_value=2, max_value=5, value=2, step=1)

    articles = []
    for i in range(int(num_articles)):
        st.write(f"### Article {i+1}")
        input_type = st.radio(
            f"Input Type for Article {i+1}",
            ['Enter URL', 'Paste Article Text'],
            key=f"comp_input_type_{i}"
        )
        if input_type == 'Enter URL':
            url = st.text_input(
                f"Article {i+1} URL",
                placeholder="https://example.com/article",
                key=f"comp_article_url_{i}"
            ).strip()
            article_text = ''
        else:
            article_text = st.text_area(
                f"Article {i+1} Text",
                placeholder="Paste the article text here...",
                height=200,
                key=f"comp_article_text_{i}"
            ).strip()
            url = ''
        title = st.text_input(
            f"Article {i+1} Title",
            value=f"Article {i+1}",
            placeholder="Enter a title for the article",
            key=f"comp_article_title_{i}"
        )
        articles.append({
            'input_type': input_type,
            'url': url,
            'text': article_text,
            'title': title
        })

    if st.button("Analyze Comparative Articles", key="analyze_comparative_articles"):
        analyses = []
        for idx, article in enumerate(articles):
            st.write(f"### Analyzing Article {idx+1}")
            if article['input_type'] == 'Enter URL':
                if article['url']:
                    if fetch_article_text(article['url']):
                        with st.spinner(f'Fetching and analyzing Article {idx+1}...'):
                            article_text_fetched = fetch_article_text(article['url'])
                            if article_text_fetched:
                                sanitized_text = sanitize_text(article_text_fetched)
                                st.success(f"Article {idx+1} text fetched successfully.")
                                article_text = sanitized_text  # Use sanitized text for analysis
                            else:
                                st.error(f"Failed to fetch article text for Article {idx+1}.")
                                continue
                    else:
                        st.error(f"Please enter a valid URL for Article {idx+1}.")
                        continue
                else:
                    st.error(f"Please enter a URL for Article {idx+1}.")
                    continue
            else:
                if article['text'].strip():
                    sanitized_text = sanitize_text(article['text'])
                    article_text = sanitized_text
                else:
                    st.error(f"Please paste the text for Article {idx+1}.")
                    continue

            with st.spinner(f'Performing analysis on Article {idx+1}...'):
                analysis_data = perform_analysis(
                    article_text=article_text,
                    title=article['title'],
                    features=features,
                    models=models,
                    config=config,
                    bias_terms=st.session_state.get('bias_terms', config.get('bias_terms', []))
                )
                if st.session_state.get('logged_in', False):
                    analysis_data['username'] = st.session_state['username']
                else:
                    analysis_data['username'] = 'guest'

            if analysis_data:
                st.success(f"Analysis completed for Article {idx+1}.")
                analyses.append(analysis_data)
                display_results(analysis_data, unique_id=f"comp_{idx}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}", save_to_history=False)

        # Comparative Metrics
        if analyses:
            st.markdown("### Comparative Metrics")
            df = pd.DataFrame([{
                'Title': analysis['title'],
                'Sentiment Score': analysis['sentiment_score'],
                'Sentiment Label': analysis['sentiment_label'],
                'Bias Count': analysis['bias_score'],
                'Propaganda Count': analysis['propaganda_score'],
                'Final Score': analysis['final_score']
            } for analysis in analyses])

            # Sort the DataFrame by Final Score descending
            df_sorted = df.sort_values(by='Final Score', ascending=False)

            st.table(df_sorted)

            # Download Comparative Analysis Results as CSV
            csv = df_sorted.to_csv(index=False).encode('utf-8')
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            st.download_button(
                label="Download Comparative Analysis Results as CSV",
                data=csv,
                file_name=f"comparative_analysis_results_{timestamp}.csv",
                mime='text/csv',
                key=f"download_comparative_csv_{timestamp}"
            )

def display_history(features, config, models):
    st.header("Your Analysis History")
    username = st.session_state.get('username', '')
    history = load_user_history(username)

    if not history:
        st.info("No history available.")
        return

    # Convert history to DataFrame for sorting
    history_df = pd.DataFrame(history)

    # Sort history by date descending
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df_sorted = history_df.sort_values(by='date', ascending=False)

    for idx, entry in history_df_sorted.iterrows():
        unique_id = f"history_{idx}"
        with st.expander(f"{entry.get('title', 'Untitled')} - {entry.get('date', 'N/A')}", expanded=False):
            # Convert Series to dictionary
            entry_dict = entry.to_dict()
            # Ensure that nested lists and dictionaries are correctly parsed
            try:
                entry_dict['biased_sentences'] = entry_dict.get('biased_sentences', [])
                entry_dict['propaganda_sentences'] = entry_dict.get('propaganda_sentences', [])
                entry_dict['entities'] = entry_dict.get('entities', {})
                # If stored as strings, parse them
                if isinstance(entry_dict['biased_sentences'], str):
                    entry_dict['biased_sentences'] = json.loads(entry_dict['biased_sentences'])
                if isinstance(entry_dict['propaganda_sentences'], str):
                    entry_dict['propaganda_sentences'] = json.loads(entry_dict['propaganda_sentences'])
                if isinstance(entry_dict['entities'], str):
                    entry_dict['entities'] = json.loads(entry_dict['entities'])
            except json.JSONDecodeError:
                st.error("Failed to parse history entry due to invalid JSON format.")
                continue
            display_results(entry_dict, unique_id=unique_id, is_nested=True, save_to_history=False)

def settings_page(config, models):
    st.header("Settings")
    st.write("Customize your analysis settings.")

    # Manage Bias Terms
    st.subheader("Manage Bias Terms")

    # Input field to add a new bias term
    with st.form("add_bias_term_form"):
        new_bias_term = st.text_input(
            "Add a New Bias Term",
            placeholder="Enter new bias term",
            key="add_bias_term_input"
        )
        submitted = st.form_submit_button("Add Term")
        if submitted:
            if new_bias_term:
                if new_bias_term.lower() in [term.lower() for term in st.session_state['bias_terms']]:
                    st.warning("This bias term already exists.")
                else:
                    st.session_state['bias_terms'].append(new_bias_term)
                    st.success(f"Added new bias term: {new_bias_term}")
                    logger.info(f"Added new bias term: {new_bias_term}")
            else:
                st.warning("Please enter a valid bias term.")

    # Text area to edit bias terms
    st.subheader("Edit Bias Terms")
    bias_terms_str = '\n'.join(st.session_state['bias_terms'])
    edited_bias_terms_str = st.text_area(
        "Edit Bias Terms (one per line)",
        value=bias_terms_str,
        height=200,
        key="edit_bias_terms_textarea"
    )
    if st.button("Save Bias Terms", key="save_bias_terms_button"):
        updated_bias_terms = [term.strip() for term in edited_bias_terms_str.strip().split('\n') if term.strip()]
        # Remove duplicates and ensure uniqueness
        unique_terms = []
        seen = set()
        for term in updated_bias_terms:
            lower_term = term.lower()
            if lower_term not in seen:
                unique_terms.append(term)
                seen.add(lower_term)
        st.session_state['bias_terms'] = unique_terms
        # Save to user's preferences in the JSON file
        users = load_users()
        for user in users:
            if user['username'].lower() == st.session_state['username'].lower():
                user['bias_terms'] = st.session_state['bias_terms']
                break
        save_users(users)
        st.success("Bias terms updated successfully.")
        logger.info("Updated bias terms list.")

    # Button to reset bias terms to default
    if st.button("Reset Bias Terms to Default", key="reset_bias_terms"):
        st.session_state['bias_terms'] = config['bias_terms'].copy()
        # Save to user's preferences in the JSON file
        users = load_users()
        for user in users:
            if user['username'].lower() == st.session_state['username'].lower():
                user['bias_terms'] = st.session_state['bias_terms']
                break
        save_users(users)
        st.success("Bias terms have been reset to default.")
        logger.info("Reset bias terms list.")

    st.markdown("### Note:")
    st.markdown("Use the **'Add a New Bias Term'** form to introduce new terms. You can edit existing terms in the text area above. To reset to the default bias terms, click the **'Reset Bias Terms to Default'** button.")

def help_feature():
    st.header("Help")
    st.write("""
    **Media Bias Detection Tool** helps you analyze articles for sentiment, bias, and propaganda. Here's how to use the tool:

    ### **1. Single Article Analysis**
    - **Input Type**: Choose to either enter a URL or paste the article text directly.
    - **Article Title**: Provide a title for your reference.
    - **Analyze**: Click the "Analyze" button to perform the analysis.
    - **Save Analysis**: After analysis, it is automatically saved to your history.

    ### **2. Comparative Analysis**
    - **Number of Articles**: Specify how many articles you want to compare (2-5).
    - **Input Articles**: For each article, choose to enter a URL or paste the text, and provide a title.
    - **Analyze**: Click the "Analyze Comparative Articles" button to perform the analysis on all articles.
    - **Download Results**: Download the comparative analysis results as a CSV file.

    ### **3. History**
    - View all your saved analyses.
    - Expand each entry to see detailed results.

    ### **4. Settings**
    - **Manage Bias Terms**: Add new bias terms or edit existing ones to customize the analysis.
    - **Reset Terms**: Revert to the default bias terms if needed.

    ### **5. Login & Registration**
    - **Register**: Create a new account with a unique username and password.
    - **Login**: Access your personalized dashboard with your preferences and history.
    - **Logout**: Securely log out of your account.

    If you encounter any issues or have questions, please refer to the documentation or contact support.
    """)

# --- User Management in Sidebar ---

def register_user_sidebar(config):
    register_user(config)

def login_user_sidebar(config):
    login_user(config)

# --- Analysis and History ---

def save_analysis_to_history(analysis_data, username):
    history_file = f'history_{username}.json'
    history = []
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as file:
                history = json.load(file)
    except json.JSONDecodeError:
        logger.error(f"{history_file} is corrupted. Resetting the file.")
        history = []
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        history = []

    history.append(analysis_data)
    try:
        with open(history_file, 'w') as file:
            json.dump(history, file, indent=4)
        logger.info("Analysis saved to history.")
    except Exception as e:
        logger.error(f"Error saving history: {e}")

# --- Main Function ---

def main():
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ''
    if 'bias_terms' not in st.session_state:
        st.session_state['bias_terms'] = config.get('bias_terms', [])

    # Sidebar Navigation
    st.sidebar.title("Media Bias Detection Tool")
    st.sidebar.markdown("---")
    if not st.session_state['logged_in']:
        page = st.sidebar.radio(
            "Navigate to",
            ["Login", "Register", "Help"]
        )
    else:
        page = st.sidebar.radio(
            "Navigate to",
            ["Single Article Analysis", "Comparative Analysis", "History", "Settings", "Help"]
        )
    st.sidebar.markdown("---")

    # Page Routing
    if page == "Login":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already logged in as **{st.session_state['username']}**.")
        else:
            login_user(config)
    elif page == "Register":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already registered as **{st.session_state['username']}**.")
        else:
            register_user(config)
    elif page == "Single Article Analysis":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            single_article_analysis(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"], config, models)
    elif page == "Comparative Analysis":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            comparative_analysis(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"], config, models)
    elif page == "History":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            display_history(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"], config, models)
    elif page == "Settings":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            settings_page(config, models)
    elif page == "Help":
        help_feature()

    # Logout Option
    if st.session_state['logged_in'] and page not in ["Login", "Register"]:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout_user()

if __name__ == "__main__":
    main()
