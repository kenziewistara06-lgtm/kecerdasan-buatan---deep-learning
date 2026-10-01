# Exercise 4.18
# Text Analysis using NLTK

import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Different text for analysis
sentence = """
Google developed TensorFlow, an open-source machine learning platform.
The platform is widely used for artificial intelligence applications.
In 2024, many developers used TensorFlow to build computer vision
and natural language processing systems.
"""

# Tokenization
tokens = nltk.word_tokenize(sentence)
print("Tokens:")
print(tokens)

# Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("\nPOS Tags:")
print(tagged)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)
print("\nNamed Entities:")
print(entities)