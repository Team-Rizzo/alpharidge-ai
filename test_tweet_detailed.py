#!/usr/bin/env python3
"""Detailed test script to check SN13 API response for a tweet."""
import os
import sys
import json
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from talisman_ai import config

def test_tweet_detailed(post_id: str):
    """Test SN13 API response in detail."""
    print(f"Testing tweet ID: {post_id}")
    print("=" * 60)
    
    api_key = getattr(config, "SN13_API_KEY", None)
    api_url = getattr(
        config,
        "SN13_API_URL",
        "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData",
    )
    
    if not api_key or api_key == "null":
        print("✗ SN13_API_KEY not set")
        return
    
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:20]}...{api_key[-10:]}")
    print()
    
    # Build the request
    post_url = f"https://x.com/i/web/status/{post_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"source": "X", "url": post_url}
    
    print(f"Request URL: {post_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    print("Sending request to SN13 API (timeout=60s)...")
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print(f"✓ HTTP Status: {resp.status_code}")
        print()
        
        json_resp = resp.json()
        print("Raw API Response:")
        print(json.dumps(json_resp, indent=2))
        print()
        
        # Analyze response
        status = json_resp.get("status")
        data_list = json_resp.get("data") or []
        meta = json_resp.get("meta", {})
        
        print("Analysis:")
        print(f"  Status: {status}")
        print(f"  Data items: {len(data_list)}")
        if meta:
            print(f"  Meta:")
            for key, value in meta.items():
                print(f"    {key}: {value}")
        print()
        
        if status != "success":
            print(f"✗ API returned status != 'success': {status}")
        elif not data_list:
            print(f"✗ API returned empty data list")
            print(f"  This means the tweet was not found by SN13's miners")
        else:
            print(f"✓ Tweet found! Data available")
            item = data_list[0]
            print(f"  Tweet ID: {item.get('tweet', {}).get('id', 'N/A')}")
            print(f"  Text (first 100 chars): {item.get('text', '')[:100]}...")
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out after 60 seconds")
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the tweet ID from the error message
    post_id = "1988779897620427125"
    
    if len(sys.argv) > 1:
        post_id = sys.argv[1]
    
    test_tweet_detailed(post_id)

