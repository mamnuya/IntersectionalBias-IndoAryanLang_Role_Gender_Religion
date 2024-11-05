'''
Code prints the exclusive negative sentiment words that occur in more ENTRIES for a certain identity than other identities.
However, these occurences only happen in 1 entry for each of the identites identified.
Then, prints how many of the exclusive negative sentiment entries are for each religion, gender, role, and language separately
'''

import itertools
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from enum import Enum
from pydantic import BaseModel, constr
from collections import Counter, defaultdict

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize VADER Sentiment Analyzer and define stop words
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Define enums for identity fields
class Religion(str, Enum):
    hindu = "Hindu"
    muslim = "Muslim"

class Gender(str, Enum):
    male = "Male"
    female = "Female"

class Language(str, Enum):
    hindi_urdu = "Hindi-Urdu"
    bengali = "Bengali"
    punjabi = "Punjabi"
    marathi = "Marathi"
    gujarati = "Gujarati"
    bhojpuri = "Bhojpuri"
    maithili = "Maithili"
    odia = "Odia"
    sindhi = "Sindhi"

class Role(str, Enum):
    partner = "Partner"
    parent = "Parent"
    child = "Child"
    sibling = "Sibling"
    friend = "Friend"
    colleague = "Colleague"
    neighbor = "Neighbor"

# Define Pydantic model for structured output
class IdentityOutput(BaseModel):
    religion: Religion
    gender: Gender
    language: Language
    role: Role
    identity: constr(max_length=100)
    application: constr(max_length=20)
    prompt: str
    initial_output: str
    probing_prompt: str = ""
    probed_output: str = ""
    category: str = ""
    keyword_detected: bool = False

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

# Function to preprocess text (remove stop words and repeat sequences)
def preprocess_text(text):
    # Remove repeating sequences
    cleaned_text = remove_repeating_sequences(text)
    
    # Tokenize and remove stop words
    tokens = word_tokenize(cleaned_text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return filtered_tokens

# Initialize counters
religion_count = Counter()
gender_count = Counter()
language_count = Counter()
role_count = Counter()

identity_term_counts = defaultdict(lambda: Counter()) # counts word frequency for an identity
all_identity_counts = Counter()

# Function to detect negative sentiment and categorize identities
def detect_negative_sentiment(dataset):
    negative_sentiment_count = Counter()
    negative_texts = []

    # Dictionary to track if a negative sentiment term has been seen in an identity's entry
    identity_negative_term_entries = defaultdict(lambda: defaultdict(set))

    for entry in dataset:
        # Preprocess the initial output for sentiment analysis
        preprocessed_text = preprocess_text(entry['initial_output'])
        
        # Perform sentiment analysis
        sentiment_scores = sid.polarity_scores(" ".join(preprocessed_text))
        if sentiment_scores['compound'] < -0.05:  # Adjust threshold for "negative" sentiment
            identity = f"{entry['religion']}_{entry['gender']}_{entry['language']}_{entry['role']}"
            negative_texts.append((identity, " ".join(preprocessed_text)))
            negative_sentiment_count[identity] += 1

            # Analyze each term for its negative sentiment score
            for term in preprocessed_text:
                term_score = sid.polarity_scores(term)
                if term_score['compound'] < -0.05:  # Only include terms with negative sentiment
                    identity_term_counts[identity][term] += 1
                    all_identity_counts[term] += 1
                    
                    # Track in which entries (per identity) this term appears
                    identity_negative_term_entries[identity][term].add(entry['identity'])

            # Increment counts for each category
            religion_count[entry['religion']] += 1
            gender_count[entry['gender']] += 1
            language_count[entry['language']] += 1
            role_count[entry['role']] += 1

    return negative_texts, negative_sentiment_count, identity_negative_term_entries

# Load the data from JSON
with open("data/intersectional_identity_dataset_full.json", "r") as file:
    dataset = json.load(file)

# Perform sentiment analysis and print results
negative_texts, negative_sentiment_count, identity_negative_term_entries = detect_negative_sentiment(dataset)

# Output to JSON with identity and associated negative sentiment words
output_data = {}
for identity, term_counts in identity_term_counts.items():
    output_data[identity] = {"top_unigrams": []}
    
    for term, count_for_identity in term_counts.items():
        # Calculate how many unique dataset entries contain the term for this identity
        entries_with_term_for_identity = len(identity_negative_term_entries[identity][term])
        
        # Calculate how many unique dataset entries contain the term for other identities
        entries_with_term_for_other_identities = sum(
            len(identity_negative_term_entries[other_identity][term]) 
            for other_identity in identity_negative_term_entries if other_identity != identity
        )
        
        # Append term data to the output structure
        output_data[identity]["top_unigrams"].append({
            "term": term,
            "termOccursInEntriesForThisIdentity": entries_with_term_for_identity,
            "termOccursInEntriesForOtherIdentities": entries_with_term_for_other_identities
        })

print("Overall negative sentiment entry Count per Identity Field:")

# Print counts for each identity field
print("Religion Counts:")
for religion, count in religion_count.items():
    print(f"{religion}: {count}")

print("\nGender Counts:")
for gender, count in gender_count.items():
    print(f"{gender}: {count}")

print("\nLanguage Counts:")
for language, count in language_count.items():
    print(f"{language}: {count}")

print("\nRole Counts:")
for role, count in role_count.items():
    print(f"{role}: {count}")

# Write the output to a JSON file
with open("data/negative_sentiment_counts_entry_freq.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("Output saved to data/negative_sentiment_counts_entry_freq.json")




# Open the file and print the exclusive trends
with open("data/negative_sentiment_counts_entry_freq.json", "r") as file:
    data = json.load(file)

# Dictionary to track total counts and identities for each term
term_counts = defaultdict(lambda: {"total_count": 0, "identities": []})

# Initialize counters for each identity field
religion_count = Counter()
gender_count = Counter()
language_count = Counter()
role_count = Counter()

# Process each identity and its associated terms
for identity, terms_data in data.items():
    # Split identity into fields
    identity_parts = identity.split("_")
    religion, gender, language, role = identity_parts

    

    for term_info in terms_data["top_unigrams"]:
        term = term_info["term"]
        entry_count_for_identity = term_info["termOccursInEntriesForThisIdentity"]
        entry_count_for_other_identity = term_info["termOccursInEntriesForOtherIdentities"]

        if (entry_count_for_identity > entry_count_for_other_identity):
            # Update total count and record the identity for this term
            # Increment counters for each identity field
            religion_count[religion] += 1
            gender_count[gender] += 1
            language_count[language] += 1
            role_count[role] += 1
            print(f"Term: {term} TotalIDCount: {entry_count_for_identity} OtherIDCount: {entry_count_for_other_identity} Identity: {identity}")

print("----")

print("Exclusive Entry Count per Identity Field:")

# Print counts for each identity field
print("Religion Counts:")
for religion, count in religion_count.items():
    print(f"{religion}: {count}")

print("\nGender Counts:")
for gender, count in gender_count.items():
    print(f"{gender}: {count}")

print("\nLanguage Counts:")
for language, count in language_count.items():
    print(f"{language}: {count}")

print("\nRole Counts:")
for role, count in role_count.items():
    print(f"{role}: {count}")

'''
Exclusive Entry Count per Identity Field:
Religion Counts:
Hindu: 26
Muslim: 44

Gender Counts:
Male: 29
Female: 41

Language Counts:
Hindi-Urdu: 5
Maithili: 7
Odia: 9
Sindhi: 12
Punjabi: 3
Gujarati: 10
Bhojpuri: 9
Marathi: 8
Bengali: 7

Role Counts:
Sibling: 5
Child: 14
Parent: 17
Partner: 8
Neighbor: 9
Colleague: 12
Friend: 5
'''

'''
Term: hell TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Hindi-Urdu_Sibling
Term: conflict TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Maithili_Child
Term: tricky TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Odia_Parent
Term: unaware TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Sindhi_Partner
Term: low TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Sindhi_Child
Term: serious TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Hindi-Urdu_Neighbor
Term: murder TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Punjabi_Colleague
Term: interrogated TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Punjabi_Colleague
Term: block TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Gujarati_Parent
Term: tricked TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Partner
Term: ridiculed TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Child
Term: teased TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Child
Term: cry TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Child
Term: molesting TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Sindhi_Parent
Term: desperate TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Sindhi_Parent
Term: bad TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Sindhi_Parent
Term: refusing TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Sindhi_Child
Term: crisis TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Parent
Term: miss TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Parent
Term: missing TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Parent
Term: injustice TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Child
Term: victimized TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Child
Term: lower TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Colleague
Term: cheating TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Marathi_Colleague
Term: tears TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Gujarati_Colleague
Term: abusive TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Bhojpuri_Parent
Term: fear TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Partner
Term: conflicts TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Sibling
Term: punished TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Neighbor
Term: criminals TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Gujarati_Parent
Term: mad TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Bengali_Neighbor
Term: sick TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Gujarati_Parent
Term: depressed TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Gujarati_Parent
Term: anxious TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Gujarati_Parent
Term: murdered TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Sindhi_Parent
Term: war TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Sindhi_Sibling
Term: destruction TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Sindhi_Sibling
Term: strange TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Colleague
Term: loss TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bhojpuri_Colleague
Term: problem TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Sibling
Term: reject TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Friend
Term: lose TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Friend
Term: losing TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Friend
Term: adversity TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Parent
Term: pressure TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Punjabi_Neighbor
Term: freak TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Bhojpuri_Parent
Term: blamed TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Odia_Neighbor
Term: harassing TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Odia_Neighbor
Term: devil TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Marathi_Neighbor
Term: alone TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Gujarati_Partner
Term: fraud TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Hindi-Urdu_Partner
Term: distracted TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Bhojpuri_Colleague
Term: suffering TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Child
Term: fighting TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Child
Term: antagonists TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Odia_Child
Term: hesitant TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Sindhi_Colleague
Term: failing TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Gujarati_Child
Term: miserably TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Male_Gujarati_Child
Term: tortured TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Bengali_Partner
Term: threatening TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Bengali_Partner
Term: demanding TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Bengali_Partner
Term: avoided TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Bengali_Parent
Term: inability TotalIDCount: 1 OtherIDCount: 0 Identity: Hindu_Female_Gujarati_Neighbor
Term: difficult TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bengali_Colleague
Term: disqualified TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Bengali_Neighbor
Term: hurt TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Male_Sindhi_Friend
Term: lawsuit TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Hindi-Urdu_Colleague
Term: harassment TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Hindi-Urdu_Colleague
Term: abusing TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Maithili_Child
Term: fired TotalIDCount: 1 OtherIDCount: 0 Identity: Muslim_Female_Sindhi_Friend

'''