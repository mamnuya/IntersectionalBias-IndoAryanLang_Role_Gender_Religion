'''
This code finds the unigram, bigrams, and trigrams from the top 25 associations
that occurs as the top association more often for a given identity than any other identities.
Exports information to data/identity_trends.json
(Example: identity1 and identity2 have the same top ngram "hi", 
but identity1 has the ngram in more individual entries than identity2. 
Therefore, identity1 "hi" is exported.)

Then, 
Imports from data/identity_trends.json
prints the ngrams (from most common to least common) with an identity it corresponds to the most, and the list of identities for which it also often occurs
This printed list is essentially the top associations that occur for certain intersectional identities over other identities. 

Observations comment below includes some observations from the printed statements.
'''

'''
Observations: 

Nationality
- "bangladesh" appears for Muslim prompts
- "pakistan india" appears for Sindhi
- "pakistan" appears for Muslim female punjabi partner
- "odisha" appears for Hindu Female Odia Colleague
- "united states" appears for Muslim male
- "punjab" appears for Hindu Female Punjabi Child
- "bengal" and "west" and "west bengal" and "india bangladesh" appears for Muslim Bengali prompts
- "indian" appears for Hindu prompts, especially Hindu Partner prompts
- "maharashtra" appears for Muslim Female Marathi Sibling
- "gujarat" appears for Hindu Female Gujarati Colleague

Language
- "malyalam" appears for Hindu Male Maithili Colleague
- "tamil appears for Hindu Female Odia Sibling
- "urdu" appears for Muslim Male Maithili Colleague
- "hindi english" appears for Hindu Male Odia Friend
- "language pakistan" appears for Hindu Male Sindhi Partner

Traits
- "good friend" appears for Muslim Male Bhojpuri Colleague
- "responsibilities" appears for Muslim Female Bengali Sibling
- "studies" appears for muslim female child prompts
- "studying" appears for Female Child prompts
- "bright" appears for Hindu Male Odia Child
- "knowledge" appears for Hindu Male Odia Colleague and Muslim Male Odia Child
- "knowledgeable" appears for Muslim Female Punjabi Colleague
- "organized" appears for Muslim Female Bengali Sibling
- "strong sense" appears for Hindu Male Marathi Partner
- "happy" appears for Partner prompts
- "successful" appears for Muslim Male prompts:  Muslim Male Bengali Colleague and Muslim Male Marathi Friend and Muslim Male Maithili Partner
- "member community" appears for Muslim Colleagues and Muslim Female
- "strong" appears for Hindu Male and Muslim Male
- "married man" appears for female prompts
- "hard worker" appears for Hindu Female Gujarati Child
- "business" appears for Muslim Male Marathi Friend and Muslim Male Sindhi Partner

Values
- "grade" appears for Child prompts
- "educated" appears for Muslim female prompts
- "working" appears for Hindu Female Gujarati Colleague
- family included in ngram appears for Muslim Male prompts and Hindu Child prompts
- "family friends" appears for Muslim Male Odia Neighbor
- "take care family" appears for Muslim Female Bengali Parent
- "time family" appears for Muslim Male Odia Neighbor

Hobbies/Interests
- "cricket" appears for Hindu Male Punjabi Child and Muslim Male Bhojpuri Child
- "sports" for Hindu Male Bhojpuri Child
- "fond reading" appears for Hindu Female Child prompts
- "movies" and "watching movies "appears for Hindu Male Marathi Neighbor
- "dancing" or "dance" appears for Hindu Female Marathi Child and Muslim Female Bhojpuri Friend
- "music" appears for Hindu Male Bengali Neighbor

Interpersonal
- "hindu female" appears for Muslim Male prompts
- "hindu man" appears for Muslim Female prompts
- "muslim mother" appears for Hindu Female Maithili Child

Religion
- "christian" appears for Muslim Male Punjabi Partner
- "god" appears for Hindu Male Sibling more often than Hindu Male Neighbor, appears for Hindu Male overall
- "quran" appears for Muslim Child prompts
- "islam" appears for Muslim Male Bhojpuri Friend
- "muslim" appears for Hindu Child prompts
- "rituals" and "practices" appears for Hindu Male Bhojpuri Parent

Other
- "village" appears for Hindu Female Odia Child
- "broken" appears for Muslim Male Sindhi Parent (*negative sentiment word observed, however, in context means tasks or categories broken down)
- Total Male Identities Count (all terms with 'married'):  14 and Total Female Identities Count (all terms with 'married'):  12
- Total Male Identities Count (all terms with 'family'):  7 and Total Female Identities Count (all terms with 'family'):  5

'''
import json
from collections import defaultdict, Counter

# Load the data from JSON
with open("data/identity_marker_counts_25.json", "r") as file:
    data = json.load(file)

# Function to extract the most common terms for each identity and count their occurrences across others
def analyze_common_associations(data):
    # Store results by identity
    associations_report = defaultdict(list)
    
    # Go through each identity and identify the top terms
    for identity, info in data.items():
        identity_report = {
            "unigrams": [],
            "bigrams": [],
            "trigrams": []
        }
        
        # Identify the most common unigrams for this identity
        for unigram in info.get("top_unigrams", []):
            term = unigram["term"]
            count = unigram["entryCount"]
            identity_report["unigrams"].append((term, count))
        
        # Identify the most common bigrams for this identity
        for bigram in info.get("top_bigrams", []):
            term = bigram["term"]
            count = bigram["entryCount"]
            identity_report["bigrams"].append((term, count))
        
        # Identify the most common trigrams for this identity
        for trigram in info.get("top_trigrams", []):
            term = trigram["term"]
            count = trigram["entryCount"]
            identity_report["trigrams"].append((term, count))
        
        # Compare these common terms with other identities
        for ngram_type in ["unigrams", "bigrams", "trigrams"]:
            for term, count in identity_report[ngram_type]:
                # Count in how many other identities the term appears
                shared_count = sum(
                    1 for other_identity, other_info in data.items() 
                    if other_identity != identity and 
                    any(t["term"] == term for t in other_info.get(f"top_{ngram_type}", []))
                )
                
                # Add result to the report
                associations_report[identity].append({
                    "type": ngram_type,
                    "term": term,
                    "count_in_identity": count,
                    "shared_in_other_identities": shared_count
                })
    
    return associations_report

# Generate the common associations report
associations_report = analyze_common_associations(data)

output_data = {}

for identity, terms in associations_report.items():
    for term_info in terms:
        if term_info['count_in_identity'] > term_info['shared_in_other_identities']:
            if identity not in output_data:
                output_data[identity] = []  # Initialize list for each identity
            output_data[identity].append({
                "type": term_info["type"],
                "term": term_info["term"],
                "count_in_identity": term_info["count_in_identity"],
                "shared_in_other_identities": term_info["shared_in_other_identities"]
            })

# Save to JSON file
with open("data/identity_trends.json", "w") as file:
    json.dump(output_data, file, indent=4)

print("Data successfully exported to data/identity_trends.json")

# Load the data from JSON
with open("data/identity_trends.json", "r") as file:
    data = json.load(file)


### this block of code prints the term and the identity is occurs the most for
# Function to analyze terms and their counts
def analyze_terms(data):
    term_counts = defaultdict(lambda: {"total_count": 0, "best_identity": "", "max_count": 0})

    # Aggregate counts
    for identity, terms in data.items():
        for term_info in terms:
            term = term_info["term"]
            count_for_this_identity = term_info["count_in_identity"]

            # Update total count and track the identity with the highest count
            term_counts[term]["total_count"] += count_for_this_identity

            if count_for_this_identity > term_counts[term]["max_count"]:
                term_counts[term]["best_identity"] = identity
                term_counts[term]["max_count"] = count_for_this_identity

    return term_counts

# Run analysis
term_counts = analyze_terms(data)

# Identify most common terms
most_common_terms = sorted(term_counts.items(), key=lambda x: x[1]["total_count"], reverse=True)

# Display most common terms and the identity where they occur the most

print("Most Common Terms (with single most frequent identity):")
for term, info in most_common_terms:
    print(f"Term: {term}, Total Count: {info['total_count']}, Identity: {info['best_identity']}")


### this block of code prints the term and identities it occurs often and identities it occurs the most for
# Load the data from JSON
with open("data/identity_trends.json", "r") as file:
    data = json.load(file)

# Function to analyze terms and their counts
def analyze_terms(data):
    term_counts = defaultdict(lambda: {"total_count": 0, "identities": []})

    # Aggregate counts
    for identity, terms in data.items():
        for term_info in terms:
            term = term_info["term"]
            count_for_this_identity = term_info["count_in_identity"]

            # Update total count and identities for each term
            term_counts[term]["total_count"] += count_for_this_identity 
            term_counts[term]["identities"].append(identity)

    return term_counts

# Run analysis
term_counts = analyze_terms(data)

# Identify most and least common terms
most_common_terms = sorted(term_counts.items(), key=lambda x: x[1]["total_count"], reverse=True)

# Display most common terms
print("***")
print("Most Common Terms with all identities it occurs often for:")
for term, info in most_common_terms:
    print(f"Term: {term}, Total Count: {info['total_count']}, Identities: {info['identities']}")


'''
# count how many times married appears for male versus female identities
total_male_count = 0
total_female_count = 0
for term, info in most_common_terms:
    male_count = 0
    female_count = 0

    # Check if the term contains "married"
    if "married" in term:
        for identity in info['identities']:
            if "Male" in identity:
                male_count += 1
            elif "Female" in identity:
                female_count += 1

        # Update overall totals
        total_male_count += male_count
        total_female_count += female_count

# Print overall totals
print("\nTotal Male Identities Count (all terms with 'married'): ", total_male_count)
print("Total Female Identities Count (all terms with 'married'): ", total_female_count)
'''