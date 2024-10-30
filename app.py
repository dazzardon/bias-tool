# media_bias_detection.py

import streamlit as st
import spacy
from transformers import pipeline
import pandas as pd
import plotly.express as px
from collections import Counter
import datetime
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import warnings
import logging
from urllib.parse import urlparse
from sklearn.exceptions import ConvergenceWarning
import streamlit_authenticator as stauth
from pathlib import Path
import yaml
import unicodedata
import secrets
import sqlite3
import bcrypt

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Suppress Specific Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- CSS Styles ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Hide the default hamburger menu */
        .css-1v0mbdj.edgvbvh3 {visibility: hidden;}
        /* Customize the navigation menu */
        .css-18ni7ap {visibility: visible;}
        .css-18ni7ap::before {
            content: 'Menu';
            font-size: 18px;
            font-weight: bold;
            margin-right: 10px;
        }
        /* Adjust the sidebar when collapsed */
        .css-1d391kg {
            width: 200px;
        }
        /* Remove emojis from interface */
        .emoji {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper Function to Initialize Session State ---
def initialize_session_state():
    if 'default_bias_terms_list' not in st.session_state:
        st.session_state.default_bias_terms_list = [
            # Updated and refined list of bias terms
            'alarming', 'unfit', 'aggressive', 'alleged', 'apparently', 'arguably',
            'claims', 'controversial', 'disputed', 'insists', 'questionable',
            'reportedly', 'rumored', 'suggests', 'supposedly', 'unconfirmed', 'suspected',
            'reckless', 'radical', 'extremist', 'biased', 'manipulative', 'deceptive',
            'unbelievable', 'incredible', 'shocking', 'outrageous', 'bizarre', 'absurd',
            'ridiculous', 'disgraceful', 'disgusting', 'horrible', 'terrible', 'unacceptable',
            'unfair', 'scandalous', 'suspicious', 'illegal', 'illegitimate', 'immoral',
            'corrupt', 'criminal', 'dangerous', 'threatening', 'harmful', 'menacing',
            'disturbing', 'distressing', 'troubling', 'fearful', 'afraid', 'panic', 'terror',
            'catastrophe', 'disaster', 'chaos', 'crisis', 'collapse', 'failure',
            'ruin', 'devastation', 'suffering', 'misery', 'pain', 'dreadful', 'awful', 'nasty',
            'vile', 'vicious', 'brutal', 'violent', 'greedy', 'selfish',
            'arrogant', 'ignorant', 'stupid', 'unwise', 'illogical', 'unreasonable',
            'delusional', 'paranoid', 'obsessed', 'fanatical', 'zealous', 'militant',
            'dictator', 'regime', 'propaganda', 'brainwash', 'suppress', 'censor', 'exploit',
            'oppress', 'inequality', 'discrimination', 'prejudice', 'segregation',
            'marginalize', 'stereotype', 'xenophobic', 'racist', 'sexist', 'homophobic',
            'transphobic', 'intolerant', 'bigoted', 'divisive', 'polarizing', 'elitist',
            'dogmatic', 'authoritarian', 'totalitarian', 'despot', 'tyrant',
            # Propaganda-related terms
            'must', 'always', 'never', 'only', 'guaranteed', 'obviously', 'clearly',
            'undeniably', 'unquestionably', 'without doubt', 'we all know', 'naturally',
            'of course', 'absolutely', 'impossible', 'completely', 'totally',
            'utterly', 'extremely', 'forever', 'nothing', 'best', 'worst', 'perfect',
            'destroy', 'eliminate', 'threat', 'enemy', 'fight', 'battle', 'victory', 'defeat',
            'win', 'lose', 'victim', 'attack', 'kill', 'die', 'death', 'danger', 'risk',
            'safe', 'security', 'protect', 'save', 'freedom', 'liberty', 'justice', 'truth',
            'honor', 'patriot', 'hero', 'evil', 'crooked', 'liar', 'fraud', 'scam', 'hoax',
            'fake', 'phony', 'illegal', 'immoral', 'unethical', 'unjust', 'wrong',
            'bad', 'harmful', 'hurtful', 'hazardous', 'toxic',
            # Add more precise terms as needed...
        ]
        logger.info("Initialized default_bias_terms_list in session state.")
    
    if 'bias_terms_list' not in st.session_state:
        # Ensure uniqueness (case-insensitive)
        unique_terms = []
        seen = set()
        for term in st.session_state.default_bias_terms_list:
            lower_term = term.lower()
            if lower_term not in seen:
                unique_terms.append(term)
                seen.add(lower_term)
        st.session_state.bias_terms_list = unique_terms.copy()
        logger.info("Initialized bias_terms_list in session state.")
    
    if 'history' not in st.session_state:
        st.session_state.history = []
        logger.info("Initialized history in session state.")
    
    if 'feedback' not in st.session_state:
        st.session_state.feedback = []
        logger.info("Initialized feedback in session state.")
    
    # Initialize authentication state
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = False  # Default to False
    if 'username' not in st.session_state:
        st.session_state.username = 'guest'  # Default to 'guest'

# --- Function to Load spaCy Model ---
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model successfully.")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        logger.error(f"Error loading spaCy model: {e}")
        nlp = None
    return nlp

# --- Lazy Loading Sentiment Analysis Model ---
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1  # Use CPU
        )
        logger.info("Loaded sentiment analysis model successfully.")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment analysis model: {e}")
        logger.error(f"Error loading sentiment analysis model: {e}")
        return None

# --- Lazy Loading Zero-Shot Classifier ---
@st.cache_resource(show_spinner=False)
def load_zero_shot_classifier():
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU
        )
        logger.info("Loaded zero-shot classifier successfully.")
        return classifier
    except Exception as e:
        st.error(f"Error loading zero-shot classifier: {e}")
        logger.error(f"Error loading zero-shot classifier: {e}")
        return None

# --- Get Sentiment Explanation ---
def get_sentiment_explanation(score):
    if score >= 0.05:
        return "The overall sentiment of the article is **positive**."
    elif score <= -0.05:
        return "The overall sentiment of the article is **negative**."
    else:
        return "The overall sentiment of the article is **neutral**."

# --- Validate URL ---
def is_valid_url(url):
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

# --- Asynchronous Fetch Article Text ---
async def fetch_article_text_async(url):
    if not is_valid_url(url):
        st.error("Invalid URL format.")
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    st.error(f"HTTP Error: {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Extract text from common article tags
                article_text = ''
                main_content = soup.find('main')
                if main_content:
                    article_text = main_content.get_text(separator=' ', strip=True)
                else:
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                if not article_text.strip():
                    st.error("No content found at the provided URL.")
                    return None
                return article_text
    except Exception as e:
        st.error(f"Error fetching the article: {e}")
        logger.error(f"Error fetching the article: {e}")
        return None

# --- Fetch Article Text (Wrapper) ---
def fetch_article_text(url):
    try:
        article_text = asyncio.run(fetch_article_text_async(url))
        return article_text
    except Exception as e:
        st.error(f"Error in fetching article text: {e}")
        logger.error(f"Error in fetching article text: {e}")
        return None

# --- Preprocess Text ---
def preprocess_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Remove unwanted characters and correct encoding issues
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Additional cleaning steps can be added here
    return text

# --- Perform Analysis ---
def perform_analysis(text, title="Article", save=True):
    if not text:
        st.error("No text to analyze.")
        return

    # Load spaCy Model
    nlp = load_spacy_model()
    if not nlp:
        st.error("spaCy model is not loaded.")
        return

    # Load Sentiment Model
    sentiment_pipeline = load_sentiment_model()

    # Load Zero-Shot Classifier
    zero_shot_classifier = load_zero_shot_classifier()

    # Preprocess Text
    text = preprocess_text(text)

    # Split text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    total_sentences = len(sentences)

    # Sentiment Analysis
    try:
        if sentiment_pipeline is not None:
            sentiment_results = sentiment_pipeline(sentences, batch_size=4, truncation=True)
            sentiment_scores = []
            for result in sentiment_results:
                label = result['label']
                score = result['score']
                # Adjust sentiment scoring based on the model's labels
                if label in ['1 star', '2 stars']:
                    sentiment_scores.append(-score)
                elif label in ['4 stars', '5 stars']:
                    sentiment_scores.append(score)
                else:
                    sentiment_scores.append(0)  # Neutral or undefined
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            avg_sentiment = round(avg_sentiment, 4)
        else:
            avg_sentiment = 0
            st.error("Sentiment analysis model is not loaded.")
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        avg_sentiment = 0
        logger.error(f"Error during sentiment analysis: {e}")

    # Bias Detection with Explanations
    biased_sentences_keyword = []
    bias_terms = st.session_state.bias_terms_list
    for sentence in sentences:
        doc_sentence = nlp(sentence)
        sentence_tokens = set([token.text.lower() for token in doc_sentence])
        detected_terms = [term for term in bias_terms if term.lower() in sentence_tokens]
        if detected_terms:
            biased_sentences_keyword.append({'sentence': sentence, 'detected_terms': detected_terms})

    bias_score = len(biased_sentences_keyword) / total_sentences if total_sentences > 0 else 0
    bias_score = round(bias_score * 100, 2)

    # Propaganda Detection
    propaganda_sentences = []
    if zero_shot_classifier is not None:
        candidate_labels = ['propaganda']
        try:
            for i in range(0, len(sentences), 4):
                batch_sentences = sentences[i:i+4]
                classifications = zero_shot_classifier(batch_sentences, candidate_labels, multi_label=False)
                if isinstance(classifications, dict):
                    classifications = [classifications]
                for classification in classifications:
                    if classification['labels'][0] == 'propaganda' and classification['scores'][0] > 0.5:
                        propaganda_sentences.append({'sentence': classification['sequence'], 'score': classification['scores'][0]})
        except Exception as e:
            st.error(f"Error during propaganda classification: {e}")
            logger.error(f"Error during propaganda classification: {e}")

    propaganda_score = len(propaganda_sentences) / total_sentences if total_sentences > 0 else 0
    propaganda_score = round(propaganda_score * 100, 2)

    # Extract entities
    entities = extract_entities(text, nlp)
    entities_normalized = [entity.lower().strip("'s") for entity in entities]
    entity_counts = Counter(entities_normalized)

    # Enhanced Entity Sentiment Analysis
    entity_sentiments = {entity: [] for entity in set(entities_normalized)}
    for sentence in sentences:
        doc_sentence = nlp(sentence)
        sentiment = 0
        if sentiment_pipeline is not None:
            try:
                result = sentiment_pipeline(sentence)[0]
                label = result['label']
                score = result['score']
                if label in ['1 star', '2 stars']:
                    sentiment = -score
                elif label in ['4 stars', '5 stars']:
                    sentiment = score
                else:
                    sentiment = 0
            except Exception as e:
                st.error(f"Error during sentiment analysis of a sentence: {e}")
                logger.error(f"Error during sentiment analysis of a sentence: {e}")
                sentiment = 0
        for ent in doc_sentence.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                ent_text = ent.text.lower().strip("'s")
                if ent_text in entity_sentiments:
                    entity_sentiments[ent_text].append(sentiment)
                else:
                    entity_sentiments[ent_text] = [sentiment]

    # Calculate average sentiment per entity
    for entity, sentiments in entity_sentiments.items():
        if sentiments:
            avg_sentiment_entity = sum(sentiments) / len(sentiments)
            entity_sentiments[entity] = round(avg_sentiment_entity, 4)
        else:
            entity_sentiments[entity] = 0

    # Save analysis data
    analysis_data = {
        'title': title,
        'text': text,
        'sentiment_score': avg_sentiment,
        'bias_score': bias_score,
        'biased_sentences': biased_sentences_keyword,
        'propaganda_score': propaganda_score,
        'propaganda_sentences': propaganda_sentences,
        'entities': entity_counts,
        'entity_sentiments': entity_sentiments,
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'username': st.session_state.username
    }

    if save and st.session_state.authentication_status:
        save_to_history(analysis_data)
    return analysis_data

# --- Save Analysis to History ---
def save_to_history(data):
    st.session_state.history.append(data)
    logger.info(f"Saved analysis data for '{data['title']}' to history.")

# --- Load History from Database ---
def load_history():
    if st.session_state.authentication_status:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT data FROM history WHERE username = ?", (st.session_state.username,))
        rows = c.fetchall()
        st.session_state.history = [eval(row[0]) for row in rows]  # Use eval to convert string to dict
        conn.close()
        logger.info("Loaded history from database.")

# --- Save History to Database ---
def save_history_to_db():
    if st.session_state.authentication_status:
        conn = get_connection()
        c = conn.cursor()
        for data in st.session_state.history:
            c.execute("INSERT INTO history (username, data) VALUES (?, ?)", (st.session_state.username, str(data)))
        conn.commit()
        conn.close()
        logger.info("Saved history to database.")

# --- Extract Entities ---
def extract_entities(text, nlp):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
    return entities

# --- Detect Biased Entities ---
def detect_biased_entities(history):
    entity_bias = {}
    sentiment_pipeline = load_sentiment_model()
    nlp = load_spacy_model()
    for entry in history:
        doc = nlp(entry['text'])
        sentences = list(doc.sents)
        for sentence in sentences:
            try:
                sentiment = 0
                if sentiment_pipeline is not None:
                    result = sentiment_pipeline(sentence.text)[0]
                    label = result['label']
                    score = result['score']
                    if label in ['1 star', '2 stars']:
                        sentiment = -score
                    elif label in ['4 stars', '5 stars']:
                        sentiment = score
                    else:
                        sentiment = 0
                entities = [ent.text.lower().strip("'s") for ent in sentence.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
                for entity in entities:
                    if entity not in entity_bias:
                        entity_bias[entity] = []
                    entity_bias[entity].append(sentiment)
            except Exception as e:
                st.error(f"Error during sentiment analysis of a sentence: {e}")
                logger.error(f"Error during sentiment analysis of a sentence: {e}")
    biased_entities = []
    for entity, sentiments in entity_bias.items():
        avg_sentiment = sum(sentiments) / len(sentiments)
        if avg_sentiment <= -0.05:
            biased_entities.append(entity)
    return biased_entities

# --- Explanation Function for Bias ---
def explain_bias(sentence, bias_terms_detected):
    explanation = f"The sentence is flagged as biased due to the presence of the following terms: {', '.join(bias_terms_detected)}."
    return explanation

# --- Display Results ---
def display_results(data, features, in_expander=False, unique_id=''):
    if not in_expander:
        st.markdown("---")
        st.subheader(f"Analysis Results for '{data['title']}'")
    else:
        st.markdown("---")
        st.markdown(f"### Analysis Results for '{data['title']}'")

    # Sentiment, Bias, and Propaganda Scores
    if any(feature in features for feature in ["Sentiment Analysis", "Bias Detection", "Propaganda Detection"]):
        if in_expander:
            st.write("### Analysis Scores")
            cols = st.columns(3)
            if "Sentiment Analysis" in features:
                with cols[0]:
                    st.metric("Sentiment Score", data['sentiment_score'])
                    sentiment_explanation = get_sentiment_explanation(data['sentiment_score'])
                    st.write(sentiment_explanation)
            if "Bias Detection" in features:
                with cols[1]:
                    st.metric("Bias Score", f"{data['bias_score']}%")
                    st.write("Bias score represents the percentage of sentences detected as biased.")
            if "Propaganda Detection" in features:
                with cols[2]:
                    st.metric("Propaganda Score", f"{data['propaganda_score']}%")
                    st.write("Propaganda score represents the percentage of sentences detected as propaganda.")
        else:
            st.markdown("## Analysis Scores")
            cols = st.columns(3)
            if "Sentiment Analysis" in features:
                with cols[0]:
                    st.metric("Sentiment Score", data['sentiment_score'])
                    sentiment_explanation = get_sentiment_explanation(data['sentiment_score'])
                    st.write(sentiment_explanation)
            if "Bias Detection" in features:
                with cols[1]:
                    st.metric("Bias Score", f"{data['bias_score']}%")
                    st.write("Bias score represents the percentage of sentences detected as biased.")
            if "Propaganda Detection" in features:
                with cols[2]:
                    st.metric("Propaganda Score", f"{data['propaganda_score']}%")
                    st.write("Propaganda score represents the percentage of sentences detected as propaganda.")

    # Biased Sentences
    if "Bias Detection" in features:
        if in_expander:
            st.write("### Biased Sentences Detected")
            if data['biased_sentences']:
                for item in data['biased_sentences']:
                    st.markdown(f"- **Sentence:** {item['sentence']}")
                    if item['detected_terms']:
                        explanation = explain_bias(item['sentence'], item['detected_terms'])
                        st.markdown(f"  - **Detected Bias Terms:** {', '.join(item['detected_terms'])}")
                        st.markdown(f"  - **Explanation:** {explanation}")
            else:
                st.write("No biased sentences detected.")
        else:
            with st.expander("Biased Sentences Detected"):
                if data['biased_sentences']:
                    for item in data['biased_sentences']:
                        st.markdown(f"- **Sentence:** {item['sentence']}")
                        if item['detected_terms']:
                            explanation = explain_bias(item['sentence'], item['detected_terms'])
                            st.markdown(f"  - **Detected Bias Terms:** {', '.join(item['detected_terms'])}")
                            st.markdown(f"  - **Explanation:** {explanation}")
                else:
                    st.write("No biased sentences detected.")

                # Bias Sentences Frequency Chart
                bias_sentences = [item['sentence'] for item in data['biased_sentences']]
                bias_counter = Counter(bias_sentences)
                if bias_counter:
                    st.write("### Bias Sentences Frequency")
                    fig = px.bar(
                        x=list(bias_counter.keys()),
                        y=list(bias_counter.values()),
                        labels={'x': 'Biased Sentences', 'y': 'Frequency'},
                        color_discrete_sequence=['#636EFA'],
                        title='Bias Sentences Frequency',
                        height=400
                    )
                    fig.update_layout(
                        xaxis_title='Biased Sentences',
                        yaxis_title='Frequency',
                        width=700,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig, key=f"bias_chart_{unique_id}")
                else:
                    st.write("No biased sentences were detected in the article.")

    # Propaganda Sentences
    if "Propaganda Detection" in features:
        if in_expander:
            st.write("### Propaganda Sentences Detected")
            if data['propaganda_sentences']:
                for item in data['propaganda_sentences']:
                    st.markdown(f"- **Sentence:** {item['sentence']} (Confidence: {round(item['score'], 2)})")
            else:
                st.write("No propaganda sentences detected.")
        else:
            with st.expander("Propaganda Sentences Detected"):
                if data['propaganda_sentences']:
                    for item in data['propaganda_sentences']:
                        st.markdown(f"- **Sentence:** {item['sentence']} (Confidence: {round(item['score'], 2)})")
                else:
                    st.write("No propaganda sentences detected.")

                # Propaganda Sentences Frequency Chart
                propaganda_sentences = [item['sentence'] for item in data['propaganda_sentences']]
                propaganda_counter = Counter(propaganda_sentences)
                if propaganda_counter:
                    st.write("### Propaganda Sentences Frequency")
                    fig = px.bar(
                        x=list(propaganda_counter.keys()),
                        y=list(propaganda_counter.values()),
                        labels={'x': 'Propaganda Sentences', 'y': 'Frequency'},
                        color_discrete_sequence=['#EF553B'],
                        title='Propaganda Sentences Frequency',
                        height=400
                    )
                    fig.update_layout(
                        xaxis_title='Propaganda Sentences',
                        yaxis_title='Frequency',
                        width=700,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig, key=f"propaganda_chart_{unique_id}")
                else:
                    st.write("No propaganda sentences were detected in the article.")

    # Entity Sentiment Analysis
    if "Pattern Detection" in features:
        st.write("### Entity Sentiment Analysis")
        if data['entity_sentiments']:
            entity_df = pd.DataFrame.from_dict(data['entity_sentiments'], orient='index', columns=['Average Sentiment'])
            entity_df = entity_df.reset_index().rename(columns={'index': 'Entity'})
            fig_entity = px.bar(
                entity_df,
                x='Entity',
                y='Average Sentiment',
                color='Average Sentiment',
                color_continuous_scale='RdBu',
                labels={'Average Sentiment': 'Average Sentiment'},
                title='Entity Sentiment Analysis',
                height=400
            )
            fig_entity.update_layout(
                xaxis_title='Entity',
                yaxis_title='Average Sentiment',
                width=700,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_entity, key=f"entity_sentiment_chart_{unique_id}")
        else:
            st.write("No entities found for sentiment analysis.")

    # Entities with Consistent Negative Bias
    if "Pattern Detection" in features and st.session_state.authentication_status:
        st.write("### Entities with Consistent Negative Bias")
        biased_entities = detect_biased_entities(st.session_state.history)
        if biased_entities:
            for entity in biased_entities:
                st.markdown(f"- **{entity.capitalize()}**")
        else:
            st.write("No entities with consistent negative bias detected.")

    # Download Analysis Results
    download_results(data)

    # --- User Feedback Form ---
    st.markdown("---")
    st.subheader("Provide Feedback")
    with st.form(key=f"feedback_form_{unique_id}"):
        feedback_text = st.text_area(
            "Your Feedback",
            help="Please describe any inaccuracies or issues you encountered during the analysis.",
            label_visibility="collapsed",
            key=f"feedback_text_{unique_id}"
        )
        feedback_type = st.selectbox(
            "Feedback Type",
            ["Inaccuracy in Sentiment Analysis", "Inaccuracy in Bias Detection", "Inaccuracy in Propaganda Detection", "Other"],
            help="Select the type of feedback you're providing.",
            key=f"feedback_type_{unique_id}"
        )
        submit_feedback = st.form_submit_button("Submit Feedback")
        if submit_feedback:
            if feedback_text.strip():
                feedback_entry = {
                    'username': st.session_state.username,
                    'title': data['title'],
                    'date': data['date'],
                    'feedback_type': feedback_type,
                    'feedback_text': feedback_text.strip()
                }
                st.session_state.feedback.append(feedback_entry)
                st.success("Thank you for your feedback!")
                logger.info("User submitted feedback.")
            else:
                st.warning("Please enter your feedback before submitting.")

    # --- Save Feedback (Optional) ---
    if st.button("Save Feedback", key=f"save_feedback_button_{unique_id}"):
        save_feedback()

# --- Save Feedback ---
def save_feedback():
    if st.session_state.feedback:
        df_feedback = pd.DataFrame(st.session_state.feedback)
        try:
            # Append to existing CSV or create a new one
            feedback_path = Path("user_feedback.csv")
            if feedback_path.exists():
                df_feedback.to_csv(feedback_path, mode='a', header=False, index=False)
            else:
                df_feedback.to_csv(feedback_path, mode='w', header=True, index=False)
            st.success("Feedback saved successfully.")
            st.session_state.feedback = []
            logger.info("Saved user feedback to CSV.")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
            logger.error(f"Error saving feedback: {e}")
    else:
        st.warning("No feedback to save.")

# --- Download Analysis Results ---
def download_results(data):
    if data:
        # Convert data to DataFrame for better structuring
        df = pd.DataFrame({
            'Title': [data['title']],
            'Date': [data['date']],
            'Sentiment Score': [data['sentiment_score']],
            'Bias Score (%)': [data['bias_score']],
            'Propaganda Score (%)': [data['propaganda_score']],
            'Number of Biased Sentences': [len(data['biased_sentences'])],
            'Number of Propaganda Sentences': [len(data['propaganda_sentences'])],
            'Entities': [", ".join([f"{k}: {v}" for k, v in data['entities'].items()])],
            'Entity Sentiments': [", ".join([f"{k}: {v}" for k, v in data['entity_sentiments'].items()])],
        })
        csv = df.to_csv(index=False).encode('utf-8')
        # Use a unique key, possibly with timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        st.download_button(
            label="Download Analysis Results as CSV",
            data=csv,
            file_name=f"{data['title'].replace(' ', '_')}_{timestamp}_analysis.csv",
            mime='text/csv',
            key=f"download_button_{data['title']}_{timestamp}"
        )
    else:
        st.warning("No data available to download.")

# --- Display History ---
def display_history(features):
    if st.session_state.authentication_status:
        st.subheader("Analysis History")
        load_history()  # Load history from database
        if not st.session_state.history:
            st.info("No history available.")
            return

        for idx, entry in enumerate(reversed(st.session_state.history)):
            unique_id = f"history_{idx}"
            with st.expander(f"{entry['title']} - {entry['date']}", expanded=False):
                display_results(entry, features, in_expander=True, unique_id=unique_id)

        # Allow downloading all history
        if st.button("Download All History as CSV", key="download_history_csv"):
            history_data = []
            for entry in st.session_state.history:
                history_data.append({
                    'Title': entry['title'],
                    'Date': entry['date'],
                    'Sentiment Score': entry['sentiment_score'],
                    'Bias Score (%)': entry['bias_score'],
                    'Propaganda Score (%)': entry['propaganda_score'],
                    'Number of Biased Sentences': len(entry['biased_sentences']),
                    'Number of Propaganda Sentences': len(entry['propaganda_sentences']),
                    'Entities': ", ".join([f"{k}: {v}" for k, v in entry['entities'].items()]),
                    'Entity Sentiments': ", ".join([f"{k}: {v}" for k, v in entry['entity_sentiments'].items()]),
                })
            df_history = pd.DataFrame(history_data)
            csv_history = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download History as CSV",
                data=csv_history,
                file_name="analysis_history.csv",
                mime='text/csv',
                key="download_history_csv_button"
            )
    else:
        st.info("Please log in to access your analysis history.")

# --- Settings Page ---
def settings_page():
    st.subheader("Settings")
    st.write("Customize your analysis settings.")

    # Manage Bias Terms
    st.write("### Manage Bias Terms")

    # Input field to add a new bias term
    with st.form("add_bias_term_form"):
        new_bias_term = st.text_input(
            "Add a New Bias Term",
            placeholder="Enter new bias term",
            key="new_bias_term_input"
        )
        submitted = st.form_submit_button("Add Term")
        if submitted:
            if new_bias_term:
                if new_bias_term.lower() in [term.lower() for term in st.session_state.bias_terms_list]:
                    st.warning("This bias term already exists.")
                else:
                    st.session_state.bias_terms_list.append(new_bias_term)
                    st.success(f"Added new bias term: {new_bias_term}")
                    logger.info(f"Added new bias term: {new_bias_term}")
            else:
                st.warning("Please enter a valid bias term.")

    # Text area to edit bias terms
    st.write("### Edit Bias Terms")
    bias_terms_str = '\n'.join(st.session_state.bias_terms_list)
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
        st.session_state.bias_terms_list = unique_terms
        st.success("Bias terms updated successfully.")
        logger.info("Updated bias terms list.")

    # Button to reset bias terms to default
    if st.button("Reset Bias Terms to Default", key="reset_bias_terms"):
        st.session_state.bias_terms_list = st.session_state.default_bias_terms_list.copy()
        st.success("Bias terms have been reset to default.")
        logger.info("Reset bias terms to default.")

    st.write("### Note:")
    st.write("Use the **'Add a New Bias Term'** form to introduce new bias terms. You can edit existing terms in the text area above. To reset to the default list, click 'Reset Bias Terms to Default'.")

# --- Help Feature ---
def help_feature():
    st.markdown("---")
    st.subheader("Help & How to Use")
    st.write("""
        Welcome to the **Media Bias Detection Tool**! This tool helps you analyze media articles for sentiment, bias, propaganda techniques, and patterns in entity sentiment.
        
        **Features:**
        
        - **Single Article Analysis**: Analyze a single article by entering its URL or pasting the text. You'll get insights on sentiment, bias, propaganda, and patterns.
        - **Comparative Analysis**: Compare multiple articles to see differences and similarities in sentiment, bias, propaganda, and key entities.
        - **History**: View your previous analyses and revisit the results at any time (available for logged-in users).
        - **Settings**: Customize your analysis by managing bias terms and settings.
        - **Help**: Access guidance on how to use the tool and understand the features.
        
        **Analysis Features:**
        
        - **Sentiment Analysis**: Determines the overall sentiment (positive, negative, neutral) of the article.
        - **Bias Detection**: Identifies sentences containing bias based on a list of bias terms.
        - **Propaganda Detection**: Detects sentences that may contain propaganda using advanced NLP techniques.
        - **Pattern Detection**: Analyzes entities mentioned in the text and their associated sentiments.
        
        **Getting Started:**
        
        1. **Use the Tool**: You can immediately start analyzing articles by selecting the desired analysis type from the sidebar.
        2. **Log In (Optional)**: If you want to access your saved history and data, you can log in using your credentials.
        3. **Select Analysis Type**: Choose between Single Article Analysis or Comparative Analysis from the sidebar.
        4. **Input Article(s)**: Enter the URL or paste the text of the article(s) you wish to analyze.
        5. **Customize Features**: In the sidebar, select which analysis features you want to enable.
        6. **Run Analysis**: Click the "Analyze" button to perform the analysis.
        7. **View Results**: The results will display sentiment scores, bias scores, propaganda detection, and entity sentiment analysis.
        8. **Download Results**: You can download the analysis results as a CSV file for further examination.
        9. **Provide Feedback**: Use the feedback form to help us improve the tool.
        
        **Managing Bias Terms:**
        
        - You can add, remove, or reset bias terms in the Settings page to customize bias detection according to your needs.
        
        **Understanding the Results:**
        
        - **Sentiment Score**: Indicates the overall emotional tone of the article.
        - **Bias Score**: Represents the percentage of sentences detected as biased.
        - **Propaganda Score**: Shows the percentage of sentences detected as containing propaganda.
        - **Biased/Propaganda Sentences**: Lists the sentences identified as biased or containing propaganda.
        - **Entity Sentiment Analysis**: Displays the average sentiment associated with key entities mentioned in the article.
        
        **Need Further Assistance?**
        
        - If you have any questions or need help, feel free to reach out through the feedback form.
    """)

# --- Single Article Analysis ---
def single_article_analysis(features):
    # Input Section
    st.subheader("Single Article Analysis")
    st.write("Enter the article URL or paste the article text.")

    input_type = st.radio(
        "Input Type",
        ['Enter URL', 'Paste Article Text'],
        label_visibility="collapsed",
        key="single_article_input_type"
    )
    if input_type == 'Enter URL':
        url = st.text_input(
            "Article URL",
            placeholder="Enter the URL of the article",
            label_visibility="collapsed",
            key="single_article_url"
        )
        article_text = ''
    else:
        article_text = st.text_area(
            "Article Text",
            placeholder="Paste the article text here",
            label_visibility="collapsed",
            height=300,
            key="single_article_text"
        )
        url = ''

    title = st.text_input(
        "Article Title",
        value="Article",
        placeholder="Enter a title for the article",
        label_visibility="collapsed",
        key="single_article_title"
    )

    if st.button("Analyze", key="analyze_single_article"):
        if input_type == 'Enter URL':
            if url:
                if is_valid_url(url):
                    with st.spinner('Fetching and analyzing the article...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            data = perform_analysis(article_text_fetched, title)
                            if data:
                                display_results(data, features, unique_id=f"single_{title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
                else:
                    st.error("Invalid URL format.")
            else:
                st.error("Please enter a valid URL.")
        else:
            if article_text:
                data = perform_analysis(article_text, title)
                if data:
                    display_results(data, features, unique_id=f"single_{title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            else:
                st.error("Please paste the article text.")

# --- Comparative Analysis ---
def comparative_analysis(features):
    st.subheader("Comparative Analysis")
    st.write("Compare multiple articles side by side.")

    num_articles = st.number_input(
        "Number of Articles",
        min_value=2,
        max_value=5,
        value=2,
        step=1,
        key="num_articles",
        label_visibility="collapsed"
    )

    articles = []
    for i in range(int(num_articles)):
        st.markdown(f"#### Article {i+1}")
        input_type = st.radio(
            f"Input Type for Article {i+1}",
            ['Enter URL', 'Paste Article Text'],
            key=f'comp_input_type_{i}',
            label_visibility="collapsed"
        )
        if input_type == 'Enter URL':
            url = st.text_input(
                f"Article {i+1} URL",
                placeholder=f"Enter the URL of Article {i+1}",
                key=f'comp_url_{i}',
                label_visibility="collapsed"
            )
            article_text = ''
        else:
            article_text = st.text_area(
                f"Article {i+1} Text",
                placeholder=f"Paste the text for Article {i+1} here",
                key=f'comp_article_text_{i}',
                label_visibility="collapsed",
                height=200
            )
            url = ''
        title = st.text_input(
            f"Article {i+1} Title",
            value=f"Article {i+1}",
            placeholder=f"Enter a title for Article {i+1}",
            key=f'comp_title_{i}',
            label_visibility="collapsed"
        )
        articles.append({'title': title, 'url': url, 'text': article_text, 'input_type': input_type})

    if st.button("Analyze Articles", key="analyze_comparative_articles"):
        analyzed_articles = []
        for idx, article in enumerate(articles):
            if article['input_type'] == 'Enter URL':
                if article['url']:
                    if is_valid_url(article['url']):
                        with st.spinner(f"Fetching and analyzing {article['title']}..."):
                            article_text_fetched = fetch_article_text(article['url'])
                            if article_text_fetched:
                                data = perform_analysis(article_text_fetched, article['title'], save=False)
                                if data:
                                    analyzed_articles.append(data)
                    else:
                        st.error(f"Invalid URL format for {article['title']}.")
                else:
                    st.error(f"Please enter a valid URL for {article['title']}.")
            else:
                if article['text']:
                    data = perform_analysis(article['text'], article['title'], save=False)
                    if data:
                        analyzed_articles.append(data)
                else:
                    st.error(f"Please paste the article text for {article['title']}.")

        if analyzed_articles:
            display_comparative_results(analyzed_articles, features)

# --- Display Comparative Results ---
def display_comparative_results(articles, features):
    st.markdown("---")
    st.subheader("Comparative Analysis Results")

    # Sentiment Comparison
    if "Sentiment Analysis" in features:
        st.markdown("### Sentiment Score Comparison")
        df_sentiment = pd.DataFrame({
            'Article': [a['title'] for a in articles],
            'Sentiment Score': [a['sentiment_score'] for a in articles]
        })
        fig1 = px.bar(
            df_sentiment,
            x='Article',
            y='Sentiment Score',
            color='Sentiment Score',
            color_continuous_scale='Viridis',
            labels={'Sentiment Score': 'Sentiment Score'},
            title='Sentiment Score Comparison',
            height=400
        )
        fig1.update_layout(
            xaxis_title='Article',
            yaxis_title='Sentiment Score',
            width=700,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig1, use_container_width=True, key='sentiment_comparison')

    # Bias Score Comparison
    if "Bias Detection" in features:
        st.markdown("### Bias Score Comparison")
        df_bias = pd.DataFrame({
            'Article': [a['title'] for a in articles],
            'Bias Score (%)': [a['bias_score'] for a in articles]
        })
        fig2 = px.bar(
            df_bias,
            x='Article',
            y='Bias Score (%)',
            color='Bias Score (%)',
            color_continuous_scale='Reds',
            labels={'Bias Score (%)': 'Bias Score (%)'},
            title='Bias Score Comparison',
            height=400
        )
        fig2.update_layout(
            xaxis_title='Article',
            yaxis_title='Bias Score (%)',
            width=700,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True, key='bias_comparison')

    # Propaganda Score Comparison
    if "Propaganda Detection" in features:
        st.markdown("### Propaganda Score Comparison")
        df_propaganda = pd.DataFrame({
            'Article': [a['title'] for a in articles],
            'Propaganda Score (%)': [a['propaganda_score'] for a in articles]
        })
        fig3 = px.bar(
            df_propaganda,
            x='Article',
            y='Propaganda Score (%)',
            color='Propaganda Score (%)',
            color_continuous_scale='Blues',
            labels={'Propaganda Score (%)': 'Propaganda Score (%)'},
            title='Propaganda Score Comparison',
            height=400
        )
        fig3.update_layout(
            xaxis_title='Article',
            yaxis_title='Propaganda Score (%)',
            width=700,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True, key='propaganda_comparison')

    # Frequent Entities Comparison
    if "Pattern Detection" in features:
        st.markdown("### Frequent Entities Across Articles")
        all_entities = []
        for article in articles:
            entities = article['entities']
            all_entities.extend(list(entities.elements()))
        entity_counts = Counter(all_entities)
        df_entities = pd.DataFrame(entity_counts.items(), columns=['Entity', 'Count']).sort_values(by='Count', ascending=False)
        st.dataframe(df_entities.head(20))

        # Entity Sentiment Comparison
        st.markdown("### Entity Sentiment Comparison")
        entity_sentiments = {}
        for article in articles:
            for entity, sentiment in article['entity_sentiments'].items():
                if entity not in entity_sentiments:
                    entity_sentiments[entity] = []
                entity_sentiments[entity].append(sentiment)
        entity_avg_sentiments = {entity: round(sum(sentiments)/len(sentiments), 4) for entity, sentiments in entity_sentiments.items()}
        df_entity_sentiments = pd.DataFrame.from_dict(entity_avg_sentiments, orient='index', columns=['Average Sentiment']).reset_index().rename(columns={'index': 'Entity'})
        if not df_entity_sentiments.empty:
            fig_entity = px.bar(
                df_entity_sentiments,
                x='Entity',
                y='Average Sentiment',
                color='Average Sentiment',
                color_continuous_scale='RdBu',
                labels={'Average Sentiment': 'Average Sentiment'},
                title='Entity Sentiment Comparison',
                height=400
            )
            fig_entity.update_layout(
                xaxis_title='Entity',
                yaxis_title='Average Sentiment',
                width=700,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_entity, use_container_width=True, key='entity_sentiment_comparative_chart')
        else:
            st.write("No entities found for sentiment comparison.")

    # --- Download Comparative Analysis Results ---
    if st.button("Download Comparative Analysis Results as CSV", key="download_comparative_csv"):
        # Prepare comparative data
        data = []
        for article in articles:
            data.append({
                'Title': article['title'],
                'Date': article['date'],
                'Sentiment Score': article['sentiment_score'],
                'Bias Score (%)': article['bias_score'],
                'Propaganda Score (%)': article['propaganda_score'],
                'Number of Biased Sentences': len(article['biased_sentences']),
                'Number of Propaganda Sentences': len(article['propaganda_sentences']),
                'Entities': ", ".join([f"{k}: {v}" for k, v in article['entities'].items()]),
                'Entity Sentiments': ", ".join([f"{k}: {v}" for k, v in article['entity_sentiments'].items()]),
            })
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False).encode('utf-8')
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        st.download_button(
            label="Download Comparative Analysis Results as CSV",
            data=csv,
            file_name=f"comparative_analysis_results_{timestamp}.csv",
            mime='text/csv',
            key=f"download_comparative_csv_{timestamp}"
        )

# --- Database Functions ---
DB_PATH = Path("users.db")

def get_connection():
    """
    Establishes a connection to the SQLite database.
    Creates the users and history tables if they don't exist.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Create users table if it doesn't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        # Create history table if it doesn't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        conn.commit()
        logger.info("Connected to the database and ensured users and history tables exist.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None

def get_user(username):
    """
    Retrieves a user from the database by username.
    Returns the user record if found, else None.
    """
    try:
        conn = get_connection()
        if conn is None:
            return None
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user:
            logger.info(f"User '{username}' retrieved successfully.")
        else:
            logger.info(f"User '{username}' not found.")
        return user
    except Exception as e:
        logger.error(f"Error fetching user '{username}': {e}")
        return None

def create_user(username, name, email, password):
    """
    Creates a new user with the provided details.
    Passwords are hashed before storing.
    Returns True if successful, else False.
    """
    try:
        conn = get_connection()
        if conn is None:
            return False
        c = conn.cursor()
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # Insert the new user
        c.execute("INSERT INTO users (username, name, email, password) VALUES (?, ?, ?, ?)",
                  (username, name, email, hashed_password))
        conn.commit()
        conn.close()
        logger.info(f"User '{username}' created successfully.")
        return True
    except sqlite3.IntegrityError as ie:
        if 'UNIQUE constraint failed: users.username' in str(ie):
            logger.error(f"Username '{username}' already exists.")
        elif 'UNIQUE constraint failed: users.email' in str(ie):
            logger.error(f"Email '{email}' is already registered.")
        else:
            logger.error(f"Integrity Error: {ie}")
        return False
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        return False

def verify_password(username, password):
    """
    Verifies a user's password.
    Returns True if the password is correct, else False.
    """
    try:
        user = get_user(username)
        if user:
            stored_password = user[4]  # Assuming password is the 5th column
            is_correct = bcrypt.checkpw(password.encode('utf-8'), stored_password)
            if is_correct:
                logger.info(f"Password for user '{username}' verified successfully.")
            else:
                logger.info(f"Password verification failed for user '{username}'.")
            return is_correct
        else:
            logger.info(f"User '{username}' does not exist for password verification.")
            return False
    except Exception as e:
        logger.error(f"Error verifying password for user '{username}': {e}")
        return False

# --- User Registration ---
def register_user():
    st.subheader("Register")
    st.write("Create a new account to save your analysis history.")
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        if submit:
            if password != password_confirm:
                st.error("Passwords do not match.")
            else:
                success = create_user(username, name, email, password)
                if success:
                    st.success("Registration successful. You can now log in.")
                else:
                    st.error("Registration failed. Username or email may already be in use.")

# --- User Login ---
def login_user():
    st.subheader("Login")
    st.write("Log in to access and save your analysis history.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if verify_password(username, password):
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.success(f"Logged in as {username}")
                logger.info(f"User '{username}' logged in successfully.")
            else:
                st.error("Login failed. Incorrect username or password.")
                logger.info(f"Login failed for user '{username}'.")

# --- User Logout ---
def logout_user():
    st.session_state.authentication_status = False
    st.session_state.username = 'guest'
    st.session_state.history = []
    st.success("Logged out successfully.")
    logger.info("User logged out.")

# --- Run the App ---
def main():
    # Inject Custom CSS
    inject_custom_css()

    # Initialize session state variables
    initialize_session_state()

    # Title Section
    st.title("Media Bias Detection Tool")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Login", "Register", "Single Article Analysis", "Comparative Analysis", "History", "Settings", "Help"],
        index=2
    )

    # Sidebar Features
    if page not in ["Settings", "Help"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Analysis Features")
        features = st.sidebar.multiselect(
            "Choose the features you want to enable:",
            ["Sentiment Analysis", "Bias Detection", "Propaganda Detection", "Pattern Detection"],
            default=["Sentiment Analysis", "Bias Detection", "Propaganda Detection", "Pattern Detection"],
            label_visibility="visible"
        )
    else:
        features = ["Sentiment Analysis", "Bias Detection", "Propaganda Detection", "Pattern Detection"]

    # Page Routing
    if page == "Login":
        login_user()
    elif page == "Register":
        register_user()
    elif page == "Single Article Analysis":
        single_article_analysis(features)
    elif page == "Comparative Analysis":
        comparative_analysis(features)
    elif page == "History":
        display_history(features)
    elif page == "Settings":
        settings_page()
    elif page == "Help":
        help_feature()

    # Logout Option
    if st.session_state.authentication_status and page != "Login":
        if st.sidebar.button("Logout"):
            logout_user()

# --- Run the Application ---
if __name__ == "__main__":
    main()
