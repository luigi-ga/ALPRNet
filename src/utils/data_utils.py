import string

# This function returns the vocabulary for license plates
# that includes digits and uppercase ascii characters
def get_vocabularies():
    # Define characters
    characters = string.digits + string.ascii_letters + string.punctuation
    # Define the characters you want to include in the vocabulary as a list
    valid_characters = list(characters) + ['EOS', 'PADDING', 'UNKNOWN']
    # Generate the vocabulary dictionary using dictionary comprehension
    voc = {char: idx for idx, char in enumerate(valid_characters)}
    # Invert keys and values to create a new dictionary with indices as keys and characters as values
    voc_inverted = {idx: char for char, idx in voc.items()}
    # Return the vocabulary and it's inverted
    return voc, voc_inverted


# Convert labels from array to string
def label_to_string(array, vocabulary=get_vocabularies()[1]):
    result = ""
    for num in array:
        if num.item() in vocabulary and vocabulary[num.item()] != 'EOS':
            result += vocabulary[num.item()]
        else:
            return result
    return result