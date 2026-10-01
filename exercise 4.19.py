# Exercise 4.19
# Sentiment Analysis using Transformers

from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    "This movie is amazing and I really enjoyed it.",
    "The food was delicious and the service was excellent.",
    "I love this new application.",
    "This movie is boring and disappointing.",
    "The food was terrible and the service was very slow.",
    "I hate this application because it is difficult to use."
]

for sentence in sentences:
    result = classifier(sentence)
    print("Sentence :", sentence)
    print("Result   :", result)
    print()