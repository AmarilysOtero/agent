# Query: "How many years of experience in total Alexis has"

## BEFORE FIX (BUGGY)

```
Step 1: extract_person_names()
  Words: ["How", "many", "years", "of", "experience", "in", "total", "Alexis", "has"]
  Filter: All capitalized + length > 2
  ├─ "How" → 3 chars ✓
  ├─ "Alexis" → 6 chars ✓
  └─ "has" → 3 chars ✓ (BUT IT'S A VERB, NOT A NAME!)
  
  Result: ["How", "Alexis", "has"] → cleaned to lowercase → ["how", "alexis", "has"]

Step 2: filter_results_by_exact_match()
  first_name = "alexis"
  last_name = "has"  ← PROBLEM: Treated as surname!
  
  Kevin's resume contains:
    ✓ "alexis" → Not in Kevin's resume ✗ (but maybe from hybrid search)
    ✓ "has" → Very common word, definitely in Kevin's text ✓
  
  Result: ❌ WRONG - Kevin's resume passes the filter
```

## AFTER FIX (CORRECT)

```
Step 1: extract_person_names() [IMPROVED]
  Words: ["How", "many", "years", "of", "experience", "in", "total", "Alexis", "has"]
  
  Filter Rules:
    1. Must be capitalized ✓
    2. Must be length > 2 ✓
    3. Must NOT be in common_words set ✗
    4. Must NOT be in tech_terms set ✓
  
  Processing:
    ├─ "How" → in common_words {how} → ❌ SKIP
    ├─ "Alexis" → NOT in common_words → ✓ KEEP
    └─ "has" → in common_words {has} → ❌ SKIP
  
  Result: ["Alexis"]

Step 2: filter_results_by_exact_match()
  first_name = "alexis"
  last_name = None  (only one name extracted)
  
  Search Results:
    - Alexis's resume:
      ✓ "alexis" found in text → name_match = True → KEEP
    
    - Kevin's resume:
      ✗ "alexis" NOT found in text → name_match = False → SKIP
  
  Result: ✓ CORRECT - Only Alexis's resume returned
```

## Key Improvement

**Before**: Extracted ["Alexis", "has"] → Required both to match → Kevin's resume matched (wrong!)

**After**: Extracted ["Alexis"] → Required only "alexis" to match → Only Alexis's resume matches (correct!)

The fix adds intelligent filtering to distinguish real person names from query words and technical terms.
