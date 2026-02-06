#!/usr/bin/env python3
"""Test Phase 2 RLM functionality by making queries with and without RLM enabled."""

import requests
import json
import time
from datetime import datetime

# Configuration
AGENT_URL = "http://localhost:8787"
SESSION_ID = "test_phase2_session"

def test_phase2():
    """Test Phase 2 by making two queries - one with RLM disabled, one with RLM enabled."""
    
    print("=" * 100)
    print("Phase 2 (High-Recall Mode) Test")
    print("=" * 100)
    print(f"Test Started: {datetime.now().isoformat()}")
    print()
    
    # Query content
    query = "the me the lskill you can find for Kelvin"
    
    # Test 1: RLM Disabled
    print("\n" + "=" * 100)
    print("TEST 1: Query with RLM DISABLED (rlm_enabled=False)")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        response1 = requests.post(
            f"{AGENT_URL}/api/chat/sessions/{SESSION_ID}/messages",
            json={
                "content": query,
                "rlm_enabled": False
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"\n✅ Request succeeded")
            print(f"Response length: {len(data1.get('response', ''))} characters")
            print(f"Sources found: {len(data1.get('sources', []))}")
        else:
            print(f"❌ Request failed with status {response1.status_code}")
            print(f"Response: {response1.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Wait between queries
    print("\nWaiting 5 seconds before next query...")
    time.sleep(5)
    
    # Test 2: RLM Enabled
    print("\n" + "=" * 100)
    print("TEST 2: Query with RLM ENABLED (rlm_enabled=True)")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        response2 = requests.post(
            f"{AGENT_URL}/api/chat/sessions/{SESSION_ID}/messages",
            json={
                "content": query,
                "rlm_enabled": True
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"\n✅ Request succeeded")
            print(f"Response length: {len(data2.get('response', ''))} characters")
            print(f"Sources found: {len(data2.get('sources', []))}")
        else:
            print(f"❌ Request failed with status {response2.status_code}")
            print(f"Response: {response2.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE - Check the Docker logs for:")
    print("  docker logs rag-agent | grep -i 'high_recall\\|Calling graphrag_search'")
    print("=" * 100)

if __name__ == "__main__":
    test_phase2()
