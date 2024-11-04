'''
This code finds the top 25 unigram, bigrams, and trigrams for each identity 
that occurs for more than 2 generations for that identity.
Exports information to data/identity_marker_counts_25.json
'''

import json
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the data from JSON
with open("data/intersectional_identity_dataset_full.json", "r") as file:
    data = json.load(file)

# Define base stop words to filter out common but non-meaningful words
stop_words = set(stopwords.words("english"))

# Function to remove repeating sequences from text
def remove_repeating_sequences(text):
    sentences = text.split('.')
    seen = set()
    unique_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)

    cleaned_text = '. '.join(unique_sentences)
    
    # Remove immediate repeating words
    words = cleaned_text.split()
    cleaned_words = []
    
    for i in range(len(words)):
        if i == 0 or words[i] != words[i - 1]:  # Check for immediate repetition
            cleaned_words.append(words[i])
    
    return ' '.join(cleaned_words)

# Dictionary to store counts of common markers (words and phrases) for each intersectional identity
identity_marker_counts = {}

# Define top K phrases to display
K = 25

# Process each entry in the dataset
for idx, entry in enumerate(data):
    # Create a unique identity key based on religion, gender, language, and role
    identity_key = (entry["religion"], entry["gender"], entry["language"], entry["role"])
    
    # Remove repeating sequences from initial_output
    cleaned_initial_output = remove_repeating_sequences(entry["initial_output"])
    
    # Tokenize the prompt and add its words to the set of stop words
    prompt_tokens = set(word_tokenize(entry["prompt"].lower()))
    custom_stop_words = stop_words.union(prompt_tokens)
    
    # Tokenize the cleaned `initial_output` text and filter out stopwords and punctuation
    tokens = word_tokenize(cleaned_initial_output.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in custom_stop_words]
    
    # Remove immediate repetitions from filtered tokens
    filtered_tokens_no_repeats = []
    for i in range(len(filtered_tokens)):
        if i == 0 or filtered_tokens[i] != filtered_tokens[i - 1]:
            filtered_tokens_no_repeats.append(filtered_tokens[i])
    
    # Initialize counts for this identity combination if not already present
    if identity_key not in identity_marker_counts:
        identity_marker_counts[identity_key] = {
            "unigrams": Counter(),
            "bigrams": Counter(),
            "trigrams": Counter(),
            "entry_count": {
                "unigrams": defaultdict(set),
                "bigrams": defaultdict(set),
                "trigrams": defaultdict(set)
            }
        }
    
    # Count unigrams, bigrams, and trigrams from the filtered tokens without immediate repetitions
    unigrams = filtered_tokens_no_repeats
    bigrams = list(ngrams(filtered_tokens_no_repeats, 2))
    trigrams = list(ngrams(filtered_tokens_no_repeats, 3))
    
    identity_marker_counts[identity_key]["unigrams"].update(unigrams)
    identity_marker_counts[identity_key]["bigrams"].update(bigrams)
    identity_marker_counts[identity_key]["trigrams"].update(trigrams)
    
    # Track entry occurrence for each n-gram
    for unigram in unigrams:
        identity_marker_counts[identity_key]["entry_count"]["unigrams"][unigram].add(idx)
    for bigram in bigrams:
        identity_marker_counts[identity_key]["entry_count"]["bigrams"][bigram].add(idx)
    for trigram in trigrams:
        identity_marker_counts[identity_key]["entry_count"]["trigrams"][trigram].add(idx)

ENTRY_COUNT_THRESHOLD = 2#5

# Prepare the final JSON structure for output, filtering by unqieu entries entryCount > threshold
output_data = {}
for identity, markers in identity_marker_counts.items():
    identity_str = f"{identity[0]}_{identity[1]}_{identity[2]}_{identity[3]}"  # Combine tuple as string for JSON keys
    output_data[identity_str] = {
        "top_unigrams": [
            {"term": term, "count": count, "entryCount": len(markers["entry_count"]["unigrams"][term])}
            for term, count in markers["unigrams"].most_common(K)
            if len(markers["entry_count"]["unigrams"][term]) > ENTRY_COUNT_THRESHOLD
        ],
        "top_bigrams": [
            {"term": " ".join(phrase), "count": count, "entryCount": len(markers["entry_count"]["bigrams"][phrase])}
            for phrase, count in markers["bigrams"].most_common(K)
            if len(markers["entry_count"]["bigrams"][phrase]) > ENTRY_COUNT_THRESHOLD
        ],
        "top_trigrams": [
            {"term": " ".join(phrase), "count": count, "entryCount": len(markers["entry_count"]["trigrams"][phrase])}
            for phrase, count in markers["trigrams"].most_common(K)
            if len(markers["entry_count"]["trigrams"][phrase]) > ENTRY_COUNT_THRESHOLD
        ]
    }

# Export the results to a JSON file
with open(f"data/identity_marker_counts_{K}.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"Top K common markers for each identity combination have been saved to 'data/identity_marker_counts_{K}.json'.")
