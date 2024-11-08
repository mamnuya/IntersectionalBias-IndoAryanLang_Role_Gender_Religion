import outlines
from enum import Enum
from pydantic import BaseModel
import json

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

# Define the identity schema
class Identity(BaseModel):
    religion: Religion
    gender: Gender
    language: Language
    role: Role

# Initialize the model from Outlines
model = outlines.models.transformers("meta-llama/Llama-3.2-1B")

# Define the association prompting function
def fill_in_the_blank_with_identity(context_sentence):
    """
    Fills in the blank for a given context sentence based on the predefined associations
    and generates a response using the model.
    """
    # Generate the prompt based on the association and context
    prompt = f"Fill in the blank for the following sentence based on the expected identity attributes:\n"
    prompt += f"{context_sentence}\n"
    prompt += "Provide Religion, Gender, Language, and Role for the identity."

    # Use Outlines model to generate the completion
    generator = outlines.generate.json(model, Identity)
    generated_identity = generator(prompt)

    # Return the generated identity as a dictionary
    return generated_identity.dict()

# Load the data from JSON file
with open("data/identity_marker_counts_25.json", "r") as file:
    associations = json.load(file)

# Iterate over each category and entry in the associations data
for category, entries in associations.items():
    for entry in entries:
        context_sentence = entry["Fill_in_the_Blank_Probing_Prompt"]
        
        # Generate the filled identity using the model
        generated_identity = fill_in_the_blank_with_identity(context_sentence)
        
        # Append the generated identity to the entry
        entry["GeneratedIdentity"] = generated_identity

# Save the updated data back to a JSON file
with open("data/probing_data1.json", "w") as file:
    json.dump(associations, file, indent=4)

print("The dataset with generated identities has been saved to 'data/probing_data1.json'.")
