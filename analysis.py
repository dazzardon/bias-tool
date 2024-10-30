# analysis.py

import logging
import datetime
import re
import torch

logger = logging.getLogger(__name__)

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
                        'explanation': explain_bias(unique_terms)
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

def explain_bias(terms):
    """
    Provide user-friendly explanations for detected bias terms.
    """
    explanations = [f"The term '{term}' indicates potential bias." for term in terms]
    return " ".join(explanations)

def explain_propaganda(techniques):
    """
    Provide detailed explanations for detected propaganda techniques.
    """
    explanations = []
    for term in techniques:
        description = describe_propaganda_term(term)
        explanations.append(f"The text involves {description}.")
    return " ".join(explanations)

def describe_propaganda_term(term):
    """
    Provide user-friendly descriptions for each propaganda technique.
    """
    descriptions = {
        "Appeal To Authority": "using references to influential people to support an argument without substantial evidence",
        "Directive Statement": "issuing clear instructions or orders to influence behavior",
        "Policy Declaration": "formally announcing new policies or regulations to inform or direct public action",
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
        "Thought-Terminating Clichés": "using clichés to end debate or discussion",
        "Whataboutism": "distracting from the main issue with irrelevant points",
        "Straw Man": "misrepresenting an opponent's argument to make it easier to attack",
        "Red Herring": "introducing an irrelevant topic to divert attention from the original issue",
        "Propaganda": "the use of information, ideas, or rumors to influence public opinion",
        "Non-Propaganda": "no propaganda detected"
    }
    return descriptions.get(term, "a propaganda technique")
