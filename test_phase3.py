#!/usr/bin/env python3
"""Test Phase 3: Full File Expansion API - Expand entry chunks to all chunks per file."""

import requests
import json
import time
from datetime import datetime

# Configuration
AGENT_URL = "http://localhost:8787"
SESSION_ID = "test_phase3_session"

def test_phase3():
    """Test Phase 3 by making queries that trigger file expansion in RLM mode."""
    
    print("=" * 100)
    print("Phase 3 (File Expansion API) Test")
    print("=" * 100)
    print(f"Test Started: {datetime.now().isoformat()}")
    print()
    
    # Query content - designed to trigger RLM and file expansion
    query = "What technical skills and certifications does Kelvin have?"
    
    print("\n" + "=" * 100)
    print("TEST: Query with RLM ENABLED (rlm_enabled=True) - Phase 3 File Expansion")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("Expected behavior:")
    print("  1. Phase 2 retrieves entry chunks with high_recall_mode")
    print("  2. Phase 3 expands entry chunks to full files")
    print("  3. Logs show: 'Expanded X → Y chunks across Z files'")
    
    try:
        response = requests.post(
            f"{AGENT_URL}/api/chat/sessions/{SESSION_ID}/messages",
            json={
                "content": query,
                "rlm_enabled": True
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Request succeeded (status 200)")
            print(f"Response length: {len(data.get('response', ''))} characters")
            print(f"Sources found: {len(data.get('sources', []))}")
            
            # Show response preview
            response_text = data.get('response', '')[:500]
            print(f"\nResponse preview:\n{response_text}...")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Wait between queries
    print("\nWaiting 5 seconds before next query...")
    time.sleep(5)
    
    # Test 2: Different query to test expansion on different files
    query2 = "List all projects Alexis has worked on and describe each one"
    
    print("\n" + "=" * 100)
    print("TEST 2: Different query with RLM ENABLED (rlm_enabled=True)")
    print("=" * 100)
    print(f"Query: {query2}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        response2 = requests.post(
            f"{AGENT_URL}/api/chat/sessions/{SESSION_ID}/messages",
            json={
                "content": query2,
                "rlm_enabled": True
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"\n✅ Request succeeded (status 200)")
            print(f"Response length: {len(data2.get('response', ''))} characters")
            print(f"Sources found: {len(data2.get('sources', []))}")
        else:
            print(f"❌ Request failed with status {response2.status_code}")
            print(f"Response: {response2.text[:500]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE - Check the Docker logs for:")
    print("  docker logs rag-agent | grep -i 'Phase 3|Expansion|expanded'")
    print()
    print("Expected log patterns:")
    print("  '🔄 Phase 3: Starting file expansion for X entry chunks'")
    print("  '📍 Phase 3.1: Identifying source files from entry chunks...'")
    print("  '✅ Phase 3.1: Found X source files'")
    print("  '📍 Phase 3.2: Fetching full chunk sets per file...'")
    print("  '✅ Phase 3: Expansion complete - X entry chunks → Y chunks across Z files'")
    print("=" * 100)

if __name__ == "__main__":
    test_phase3()
