#!/usr/bin/env python3
"""Test person query detection fix."""

from src.news_reporter.tools.header_vocab import extract_person_names_and_mode

test_queries = [
    'who does Kevin work with?',
    'Does Kevin have experience with Java?',
    'What are John Smith skills?',
    'Show me Alexis Torres colleagues',
    'Tell me about the project timeline',
    'Where does Maria work?',
    'Maria skills',
    'Kevin experience',
    'Who worked on the database project',
    'What is the timeline',
]

print('Testing final person query detection:')
print('=' * 90)
print(f"{'Query':<50} {'Names':<25} {'is_person'}")
print('=' * 90)
for query in test_queries:
    names, is_person = extract_person_names_and_mode(query)
    print(f"{query:<50} {str(names):<25} {is_person}")
