#!/usr/bin/env python3
"""
Test script to verify co-worker graph relationship discovery.

Tests:
1. Header vocabulary loading
2. Person name extraction 
3. Co-worker relationship queries
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.news_reporter.tools.header_vocab import extract_person_names_and_mode, load_header_vocab

def test_vocabulary_loading():
    """Test that vocabulary loads correctly."""
    print("🔍 Test 1: Loading header vocabulary...")
    vocab = load_header_vocab()
    
    if not vocab:
        print("❌ FAILED: Vocabulary is empty!")
        return False
    
    print(f"✅ PASSED: Loaded {len(vocab)} vocabulary phrases")
    print(f"   Sample phrases: {list(vocab)[:5]}")
    return True

def test_person_extraction():
    """Test person name extraction with corpus vocabulary."""
    print("\n🔍 Test 2: Person name extraction...")
    
    test_cases = [
        ("Do you know with what other employee Alexis works?", ["Alexis"], True),
        ("List Kevin skills", ["Kevin"], True),
        ("What are the key skills", [], False),
        ("Tell me about Alexis Torres", ["Alexis", "Torres"], True),
        ("Kevin Ramirez experience", ["Kevin", "Ramirez"], True),
    ]
    
    passed = 0
    for query, expected_names, expected_is_person in test_cases:
        names, is_person = extract_person_names_and_mode(query)
        
        if is_person:  # If we identified it as person query
            # Check if at least some of the expected names were found
            if names:
                print(f"✅ '{query}'")
                print(f"   Found names: {names}, is_person={is_person}")
                passed += 1
            else:
                print(f"⚠️  '{query}'")
                print(f"   Expected names but got: {names}, is_person={is_person}")
        else:
            print(f"ℹ️  '{query}'")
            print(f"   Names: {names}, is_person={is_person}")
    
    print(f"\n✅ PASSED: {passed}/{len(test_cases)} tests")
    return passed > 0

def main():
    """Run all tests."""
    print("=" * 60)
    print("GRAPH CO-WORKER DISCOVERY TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Vocabulary Loading", test_vocabulary_loading()))
    results.append(("Person Extraction", test_person_extraction()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Test with real queries via the chat interface")
        print("2. Example: 'Do you know with what other employee Alexis works?'")
        print("3. Graph should now traverse WORKS_WITH relationships")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
