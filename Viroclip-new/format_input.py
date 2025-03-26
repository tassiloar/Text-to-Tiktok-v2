import re
import random

# extract_keywords
# Input: A script string generated by claud
# Function: Get the keywords for every sentnce
# Output: A list of strings where each string is the keywords of that sentence
def extract_keywords(text):
    
    # Remove all characters except for letters (A-Z, a-z) and full stops
    cleaned_text = re.sub(r'[^A-Za-z\.]', ' ', text)
    
    # Split the input string into sentences using '.'
    sentences = cleaned_text.split('. ')
    
    result = []
    
    for sentence in sentences:
        # Strip leading/trailing spaces
        sentence = sentence.strip()
        
        # Skip empty sentences
        if not sentence:
            continue
        
        # Split the sentence into words
        words = sentence.split()

        # Extract capitalized words except the first word of the sentence
        capitalized_words = [word for word in words[1:] if word.istitle()]

        # Join the capitalized words into a single string separated by spaces
        capitalized_string = ' '.join(capitalized_words)
        
        # Append the list of capitalized words for the sentence to the result
        result.append(capitalized_string)
    
    return result


# remove_format_prefix
# Input: A list strings of content generated by claud tool
# Function: Remove the format of the content 
# Output: A list of strings 
def remove_format_prefix(content_list):
    # Define a regular expression pattern to match variations of 'videos of' and 'images of'
    pattern = re.compile(r"^(videos|video|images|image|photo|photos) of\s+", re.IGNORECASE)

    # Process each string in the list
    cleaned_list = [re.sub(pattern, '', item).strip() for item in content_list]

    return cleaned_list


# generate_content_list
# Input: A list strings of content generated by claud tool
# and a list of keywords for each sentence
# Function: Create a list of content strings for each sentences 
# Output: A list of strings 
def generate_content_list(cleaned_content_items, keywords_list):
    # Initialize a new list to store the results
    new_list = []

    # Iterate over each item in keywords_list
    for keyword in keywords_list:
        if not keyword:
            # If the keyword is empty, choose a random item from cleaned_content_items
            new_list.append(random.choice(cleaned_content_items))
        else:
            # If the keyword is not empty, randomly decide what to insert (50% chance)
            if random.random() < 0.5:
                new_list.append(keyword)
            else:
                new_list.append(random.choice(cleaned_content_items))

    return new_list
    
