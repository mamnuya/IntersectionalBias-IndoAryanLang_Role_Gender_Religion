# Credit Score Classification

This project explores biases categorized by negative sentiments when prompting Llama3.2 with intersectional identities (Indo-Aryan language spoken, gender, religion, and role) for 3 different applications (generation of to-do lists, hobbies and values, and stories).

This project includes a written report evaluating results, methods, motivations, previous studies, and more.

## Files
data/intersectional_identity_dataset_full.json holds our full raw dataset.

### Finding Biased Associations
sentimentAnalysis_entryCount.py: prints the negative sentiment term that occur with their corresponding identities. Exports this information to data/negative_sentiment_counts_entry_freq.json. Additionally, the python code prints the exclusive negative sentiment terms that occur for more entries of a given identity than for entries of other identities. At the end, it prints the counts and metrics for each identity field. 


### Finding Top Associations (no bias)
findIdentityAssociations.py: finds the top 25 unigram, bigrams, and trigrams for each identity that occurs for more than 2 generations for that identity. This exports information to data/identity_marker_counts_25.json

exclusiveIdentityAssociations.py finds the finds the unigram, bigrams, and trigrams from the top 25 associations
that occurs as the top association that occur in more entries for a given identity than any other identities. exports information to data/identity_trends.json. Then, prints the ngrams (from most common to least common) with an identity it corresponds to the most, and the list of identities for which it also often occurs.
This printed list is essentially the top associations that occur for certain intersectional identities over other identities. 


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

```
pip install nltk
pip install pydantic
```

## Usage
Keep the intersectional_identity_dataset.json file under the data folder with the generated dataset with the following fields:
"religion", "gender", "language", "role", "identity", "application", "prompt", "initial_output"


Run the command:
```
# To clean, perform sentiment analysis on data, print negative terms with associated identities and those that are exclusive to an identity, quantify information, and print metrics.
python sentimentAnalysis_entryCount.py
```

# Authors Version 1
Mamnuya Rinki (mrinki@gmu.edu), Aksh Patel (apatel66@gmu.edu), Sai Sharanya Garika (sgarika@gmu.edu)