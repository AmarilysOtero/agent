#!/usr/bin/env python3
"""
Test script to verify person name extraction fix.

This validates that common query words (like "has", "with", "all") are not
mistakenly extracted as person names.
"""

def extract_person_names(query: str):
    """Extract potential person names from query (fallback implementation)."""
    common_words = {
        'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'am', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'will', 'would',
        'could', 'should', 'can', 'shall', 'may', 'might', 'must', 'the', 'a', 'an',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'about', 'show', 'tell', 'find', 'get', 'give', 'as', 'if', 'this', 'that',
        'these', 'those', 'me', 'you', 'him', 'her', 'us', 'them', 'it', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'all', 'each', 'every', 'both', 'any',
        'some', 'more', 'most', 'other', 'another', 'such', 'no', 'not', 'only',
        'then', 'now', 'just', 'also', 'still', 'up', 'down', 'out', 'over', 'under',
        'through', 'before', 'after', 'during', 'including', 'without', 'between',
    }
    
    tech_terms = {
        'python', 'java', 'javascript', 'rust', 'golang', 'csharp', 'ruby', 'php',
        'swift', 'kotlin', 'typescript', 'scala', 'haskell', 'clojure', 'perl',
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'rails', 'node',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
        'mongodb', 'postgresql', 'mysql', 'oracle', 'cassandra', 'redis', 'elasticsearch',
        'sql', 'html', 'css', 'xml', 'json', 'yaml', 'graphql', 'rest', 'soap',
    }
    
    words = query.split()
    names = []
    for w in words:
        # Strip punctuation and handle possessive form
        clean = w.strip('.,!?;:()')
        # Remove possessive 's ONLY if it's a standalone 's (not part of the word like "Alexis")
        if clean.endswith("'s"):
            clean = clean[:-2]  # Remove "'s"
        
        # Extract only if: (1) starts with uppercase, (2) length > 2, 
        # (3) not a common word, (4) not a tech term
        if (clean and clean[0].isupper() and len(clean) > 2 and 
            clean.lower() not in common_words and 
            clean.lower() not in tech_terms):
            names.append(clean)
    return names


# Test cases
test_cases = [
    ("How many years of experience in total Alexis has", ["Alexis"]),
    ("Alexis Torres skills", ["Alexis", "Torres"]),
    ("Tell me about Alexis", ["Alexis"]),
    ("Show me Kevin and Alexis skills", ["Kevin", "Alexis"]),
    ("Kevin's industry experience", ["Kevin"]),
    ("What does Sarah do?", ["Sarah"]),
    ("All Python skills", []),  # "All" is filtered out
    ("The company background", []),  # "The" is filtered out
    ("Alexis with Kevin", ["Alexis", "Kevin"]),  # "with" is filtered
    ("How many projects did Maria work on", ["Maria"]),  # "How" filtered
]

print("Testing person name extraction...\n")
all_pass = True

for query, expected in test_cases:
    result = extract_person_names(query)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    if result != expected:
        all_pass = False
    print(f"{status}: '{query}'")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}\n")

if all_pass:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed")
    exit(1)
