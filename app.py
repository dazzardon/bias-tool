# app.py

import logging
import streamlit as st
import datetime
import re
import os
import json
import pandas as pd
import plotly.express as px
from transformers import pipeline, BertTokenizerFast
from model import BertForTokenAndSequenceJointClassification  # Ensure this is correctly imported
import utils  # Ensure utils.py is in the same directory
import analysis  # Ensure analysis.py is in the same directory

# --- Configure Logging ---
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    models = {
        'sentiment': sentiment_pipeline,
        'propaganda_model': propaganda_model,
        'propaganda_tokenizer': propaganda_tokenizer,
        'nlp': nlp
    }
    return models

# Load Configuration
config = utils.load_config('config.yaml')
if not config:
    st.error("Configuration file not found or invalid.")
    st.stop()

# Load Models
models = initialize_models(config)

# --- Helper Functions ---

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
            utils.save_analysis_to_history(data, st.session_state['username'])
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

# --- User Management Functions ---

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
            if not utils.is_valid_username(username):
                st.error("Username must be 3-30 characters long and contain only letters and numbers.")
                return
            if password != password_confirm:
                st.error("Passwords do not match.")
                return
            if not utils.is_strong_password(password):
                st.error("Password must be at least 8 characters long and include at least one special character.")
                return
            users = utils.load_users()
            if any(user['username'].lower() == username.lower() for user in users):
                st.error("Username already exists. Please choose a different one.")
                return
            hashed_pwd = utils.hash_password(password)
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
            utils.save_users(users)
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
            users = utils.load_users()
            user = next((user for user in users if user['username'].lower() == username.lower()), None)
            if user and utils.verify_password(user['password'], password):
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

# --- Analysis Functions ---

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
                if utils.is_valid_url(url):
                    with st.spinner('Fetching the article...'):
                        article_text_fetched = utils.fetch_article_text(url)
                        if article_text_fetched:
                            sanitized_text = utils.sanitize_text(article_text_fetched)
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
            sanitized_text = utils.sanitize_text(article_text)
            article_text = sanitized_text

        with st.spinner('Performing analysis...'):
            analysis_data = analysis.perform_analysis(
                article_text=article_text,
                title=title,
                features=features,
                models=models,
                config=config,
                bias_terms=st.session_state.get('bias_terms', config.get('bias_terms', []))
            )

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
                    if utils.is_valid_url(article['url']):
                        with st.spinner(f'Fetching and analyzing Article {idx+1}...'):
                            article_text_fetched = utils.fetch_article_text(article['url'])
                            if article_text_fetched:
                                sanitized_text = utils.sanitize_text(article_text_fetched)
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
                    sanitized_text = utils.sanitize_text(article['text'])
                    article_text = sanitized_text
                else:
                    st.error(f"Please paste the text for Article {idx+1}.")
                    continue

            with st.spinner(f'Performing analysis on Article {idx+1}...'):
                analysis_data = analysis.perform_analysis(
                    article_text=article_text,
                    title=article['title'],
                    features=features,
                    models=models,
                    config=config,
                    bias_terms=st.session_state.get('bias_terms', config.get('bias_terms', []))
                )

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
    history = utils.load_user_history(username)

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
        users = utils.load_users()
        for user in users:
            if user['username'].lower() == st.session_state['username'].lower():
                user['bias_terms'] = st.session_state['bias_terms']
                break
        utils.save_users(users)
        st.success("Bias terms updated successfully.")
        logger.info("Updated bias terms list.")

    # Button to reset bias terms to default
    if st.button("Reset Bias Terms to Default", key="reset_bias_terms"):
        st.session_state['bias_terms'] = config['bias_terms'].copy()
        # Save to user's preferences in the JSON file
        users = utils.load_users()
        for user in users:
            if user['username'].lower() == st.session_state['username'].lower():
                user['bias_terms'] = st.session_state['bias_terms']
                break
        utils.save_users(users)
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
