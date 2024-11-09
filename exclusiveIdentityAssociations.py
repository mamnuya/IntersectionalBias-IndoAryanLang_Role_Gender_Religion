'''
This code finds the unigram, bigrams, and trigrams from the top 25 associations
that occurs as the top association that occur in more entries for a given identity than any other identities.
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
- is overall observations
* is the singlemost common identity the term appears for 

Values
- "grade" appears for Child prompts
* Term: grade, Total Count: 13, Identity: Muslim_Male_Marathi_Child
- "educated" appears for Muslim female prompts
* Term: educated, Total Count: 6, Identity: Muslim_Female_Bengali_Friend
- "working" appears for Hindu Female Gujarati Colleague
* Term: working, Total Count: 6, Identity: Hindu_Female_Gujarati_Colleague
- family included in ngram appears for Muslim Male prompts and Hindu Child prompts
* Term: family also, Total Count: 3, Identity: Hindu_Female_Bhojpuri_Child
* Term: family parents, Total Count: 3, Identity: Hindu_Male_Gujarati_Child
* Term: family friends, Total Count: 4, Identity: Muslim_Male_Odia_Neighbor
- "family friends" appears for Muslim Male Odia Neighbor
- "take care family" appears for Muslim Female Bengali Parent
* Term: take care family, Total Count: 3, Identity: Muslim_Female_Bengali_Parent

Hobbies/Interests
- "cricket" appears for Hindu Male Punjabi Child and Muslim Male Bhojpuri Child
* Term: cricket, Total Count: 7, Identity: Muslim_Male_Bhojpuri_Child
- "sports" for Hindu Male Bhojpuri Child
* Term: sports, Total Count: 3, Identity: Hindu_Male_Bhojpuri_Child
- "fond reading" appears for Hindu Female Child prompts
* Term: fond reading, Total Count: 10, Identity: Hindu_Female_Marathi_Child
- "movies" and "watching movies "appears for Hindu Male Marathi Neighbor
* Term: movies, Total Count: 3, Identity: Hindu_Male_Marathi_Neighbor
- "dancing" or "dance" appears for Hindu Female Marathi Child and Muslim Female Bhojpuri Friend
* Term: dancing, Total Count: 3, Identity: Hindu_Female_Marathi_Child
* Term: dance, Total Count: 3, Identity: Muslim_Female_Bhojpuri_Friend
- "music" appears for Hindu Male Bengali Neighbor
* Term: music, Total Count: 4, Identity: Hindu_Male_Bengali_Neighbor

Nationality
- "bangladesh" appears for Muslim prompts
* bangladesh, Total Count: 14, Identity: Muslim_Male_Bengali_Parent
- "pakistan india" appears for Sindhi
* pakistan india, Total Count: 6, Identity: Hindu_Male_Sindhi_Partner
- "odisha" appears for Hindu Female Odia Colleague
* Term: odisha, Total Count: 6, Identity: Hindu_Female_Odia_Colleague
- "united states" appears for Muslim male
* Term: united states, Total Count: 6, Identity: Muslim_Male_Punjabi_Sibling
- "punjab" appears for Hindu Female Punjabi Child
* Term: punjab, Total Count: 3, Identity: Hindu_Female_Punjabi_Child
- "bengal" and "west" and "west bengal" and "india bangladesh" appears for Muslim Bengali prompts
* Terms appear for Identity: Muslim_Male_Bengali_Parent
- "indian" appears for Hindu prompts, especially Hindu Partner prompts
* indian, Total Count: 9, Identity: Hindu_Male_Hindi-Urdu_Partner
- "maharashtra" appears for Muslim Female Marathi Sibling
* Term: maharashtra, Total Count: 3, Identity: Muslim_Female_Marathi_Sibling
- "gujarat" appears for Hindu Female Gujarati Colleague
* Term: gujarat, Total Count: 3, Identity: Hindu_Female_Gujarati_Colleague

Interpersonal
- "hindu female" appears for Muslim Male prompts
- "hindu man" appears for Muslim Female prompts
- "muslim mother" appears for Hindu Female Maithili Child

Religion
- "christian" appears for Muslim Male Punjabi Partner
- "god" appears for Hindu Male Sibling more often than Hindu Male Neighbor, appears for Hindu Male overall
- "quran" appears for Muslim Child prompts
* Term: quran, Total Count: 9, Identity: Muslim_Male_Hindi-Urdu_Child
- "islam" appears for Muslim Male Bhojpuri Friend
* Term: islam, Total Count: 3, Identity: Muslim_Male_Bhojpuri_Friend
- "muslim" appears for Hindu Child prompts
* Term: muslim mother, Total Count: 3, Identity: Hindu_Female_Maithili_Child
* Term: muslim, Total Count: 8, Identity: Hindu_Male_Hindi-Urdu_Child
- "rituals" and "practices" appears for Hindu Male Bhojpuri Parent
* Term: rituals, Total Count: 3, Identity: Hindu_Male_Bhojpuri_Parent
* Term: practices, Total Count: 3, Identity: Hindu_Male_Bhojpuri_Parent

Language 
- "malyalam" appears for Hindu Male Maithili Colleague
* Term: malayalam, Total Count: 3, Identity: Hindu_Male_Maithili_Colleague
- "tamil appears for Hindu Female Odia Sibling
* Term: tamil, Total Count: 3, Identity: Hindu_Female_Odia_Sibling
- "urdu" appears for Muslim Male Maithili Colleague
* Term: urdu, Total Count: 6, Identity: Muslim_Male_Maithili_Colleague
- "hindi english" appears for Hindu Male Odia Friend
* Term: hindi english, Total Count: 3, Identity: Hindu_Male_Odia_Friend
- "language pakistan" appears for Hindu Male Sindhi Partner
* Term: language pakistan, Total Count: 3, Identity: Hindu_Male_Sindhi_Partner

Traits 
- "good friend" appears for Muslim Male Bhojpuri Colleague
* Term: good friend, Total Count: 4, Identity: Muslim_Male_Bhojpuri_Colleague
- "responsibilities" appears for Muslim Female Bengali Sibling
* Term: responsibilities, Total Count: 4, Identity: Muslim_Female_Bengali_Sibling
- "studies" appears for muslim female child prompts
* Term: studies, Total Count: 9, Identity: Muslim_Female_Marathi_Child
* Term: studies good, Total Count: 3, Identity: Muslim_Female_Marathi_Child
- "studying" appears for Female Child prompts
* Term: studying class, Total Count: 3, Identity: Muslim_Male_Odia_Child
* Term: studying, Total Count: 10, Identity: Muslim_Female_Gujarati_Child
- "bright" appears for Hindu Male Odia Child
* Term: bright, Total Count: 4, Identity: Hindu_Male_Odia_Child
- "knowledge" appears for Hindu Male Odia Colleague and Muslim Male Odia Child
* Term: knowledge, Total Count: 6, Identity: Hindu_Male_Odia_Colleague
- "knowledgeable" appears for Muslim Female Punjabi Colleague
* Term: knowledgeable, Total Count: 3, Identity: Muslim_Female_Punjabi_Colleague
- "organized" appears for Muslim Female Bengali Sibling
* Term: organized, Total Count: 4, Identity: Muslim_Female_Bengali_Sibling
- "strong sense" appears for Hindu Male Marathi Partner
* Term: strong sense, Total Count: 3, Identity: Hindu_Male_Marathi_Partner
- "happy" appears for Partner prompts
* Term: happy, Total Count: 9, Identity: Hindu_Female_Maithili_Partner
- "successful" appears for Muslim Male prompts:  Muslim Male Bengali Colleague and Muslim Male Marathi Friend and Muslim Male Maithili Partner
* Term: successful, Total Count: 10, Identity: Muslim_Male_Maithili_Partner
- "member community" appears for Muslim Colleagues and Muslim Female
* Term: member community, Total Count: 10, Identity: Muslim_Female_Sindhi_Parent
- "strong" appears for Hindu Male and Muslim Male
* Term: strong, Total Count: 11, Identity: Hindu_Male_Marathi_Partner
- "married man" appears for female prompts
* Term: married man, Total Count: 10, Identity: Hindu_Female_Bhojpuri_Partner
- "hard worker" appears for Hindu Female Gujarati Child
* Term: hard worker, Total Count: 3, Identity: Hindu_Female_Gujarati_Child
- "business" appears for Muslim Male Marathi Friend and Muslim Male Sindhi Partner
* Term: business, Total Count: 6, Identity: Muslim_Male_Marathi_Friend

Other 
- "village" appears for Hindu Female Odia Child
* Term: village, Total Count: 3, Identity: Hindu_Female_Odia_Child
- "broken" appears for Muslim Male Sindhi Parent (*negative sentiment word observed, however, in context means tasks or categories broken down)
* Term: broken, Total Count: 3, Identity: Muslim_Male_Sindhi_Parent
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