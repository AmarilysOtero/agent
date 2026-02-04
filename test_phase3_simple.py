#!/usr/bin/env python3
"""Simple test for Phase 3 execution without Unicode issues."""

import requests
import json
import time
from datetime import datetime

AGENT_URL = "http://localhost:8787"
SESSION_ID = f"test_phase3_{int(time.time())}"

def test_phase3():
    """Test Phase 3 by making a query with RLM enabled."""
    
    print("=" * 80)
    print("Phase 3 File Expansion Test")
    print("=" * 80)
    print(f"Session ID: {SESSION_ID}")
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    query = "What technical skills does Kelvin have?"
    
    print(f"Query: {query}")
    print(f"RLM Enabled: True")
    print()
    
    try:
        response = requests.post(
            f"{AGENT_URL}/api/chat/sessions/{SESSION_ID}/messages",
            json={
                "content": query,
                "rlm_enabled": True
            },
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response received successfully")
            print(f"Response length: {len(data.get('response', ''))} characters")
        else:
            print(f"Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"Exception: {str(e)}")
    
    print()
    print("Check Docker logs for Phase 3 execution:")
    print(f"  docker logs rag-agent | grep 'Phase 3'")
    print()

if __name__ == "__main__":
    test_phase3()
