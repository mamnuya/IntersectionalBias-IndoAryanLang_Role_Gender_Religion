# Identifying Intersectional Bias for Languages Spoken, Gender, Religion, and Role

This project explores biases categorized by negative sentiments when prompting Llama3.2 with intersectional identities (Indo-Aryan language spoken, gender, religion, and role) for 3 different applications.

Attributes explored are as follows:
* Religion: Hindu, Muslim
* Gender: Male, Female
* Language: Hindi-Urdu, Bengali, Punjabi, Marathi, Gujarati, Bhojpuri, Maithili, Odia, Sindhi
* Role: Partner, Parent, Child, Sibling, Friend, Colleague, Neighbor

Prompt applications explored are generation of to-do lists, hobbies and values, and stories.

This project includes a written report evaluating results, methods, motivations, previous studies, and more.

## Files
data/intersectional_identity_dataset_full.json holds our full raw dataset.

### Finding Biased Associations
sentimentAnalysis_entryCount.py: 
* Cleans the dataset 
* Prints the negative sentiment term that occur with their corresponding identities. Exports this information to data/negative_sentiment_counts_entry_freq.json. 
* Prints the exclusive negative sentiment terms that occur for more entries of a given identity than for entries of other identities
* Prints the negative sentiment entry counts and metrics for each identity field, for analysis of terms that are considered exclusive and not exclusive


### Finding Top Associations (no bias)
findIdentityAssociations.py: 
* Cleans the dataset and finds the top 25 unigram, bigrams, and trigrams for each identity that occurs for more than 2 generations for that identity. Exports this information to data/identity_marker_counts_25.json. 
* Some of these top n-grams are exported to data/associations.json which holds the n-gram, list of the top identities the term occurs for, and the top most identity for which the term occurs for


exclusiveIdentityAssociations.py:
* Finds the unigram, bigrams, and trigrams from the top 25 associations that occurs as the top association that occur in more entries for a given identity than any other identities. Exports thisinformation to data/identity_trends.json.  
* Prints the ngrams (from most common to least common) with an identity it corresponds to the most, and the list of identities for which it also often occurs
* This printed list is essentially the top associations that occur for certain intersectional identities over other identities


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

```
pip install nltk
pip install pydantic
```

## Usage
Keep the intersectional_identity_dataset.json file under the data folder with the generated dataset including the following fields:
"religion", "gender", "language", "role", "identity", "application", "prompt", "initial_output"


Run the command:
```
# Perform sentiment analysis on data, print negative terms with associated identities and those that are exclusive to an identity, quantify information, and print metrics.
python sentimentAnalysis_entryCount.py
```

## Metric and Results
For easier access, view our metric results and interpretation information in metricInterpretation.txt.
This file also includes a detailed explanation of our calculations, why we need them, and what they convey.

The full code for calculating and generating our metric results is within sentiemntAnalysis_entryCount.py

# Authors Version 1
Mamnuya Rinki (mrinki@gmu.edu), Aksh Patel (apatel66@gmu.edu), Sai Sharanya Garika (sgarika@gmu.edu)
