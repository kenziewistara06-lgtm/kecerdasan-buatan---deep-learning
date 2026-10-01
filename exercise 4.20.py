# Exercise 4.20
# Transformers - Question Answering

from transformers import pipeline

question_answerer = pipeline('question-answering')

context = """
Politeknik Elektronika Negeri Surabaya, also known as PENS,
is an engineering-focused institution in Surabaya, Indonesia.
PENS offers various programs in electronics, information technology,
multimedia engineering, and related fields.
The institution was established in 1988.
"""

questions = [
    "What is the name of the institution?",
    "Where is PENS located?",
    "When was PENS established?",
    "What fields does PENS offer?"
]

for question in questions:
    result = question_answerer({
        'question': question,
        'context': context
    })

    print("Question :", question)
    print("Answer   :", result['answer'])
    print("Score    :", result['score'])
    print()