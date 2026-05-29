#!/usr/bin/env python3
"""Test different URL formats for SN13 API."""
import os
import sys
import json
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from talisman_ai import config

def test_url_format(post_id: str, url_format: str, description: str):
    """Test a specific URL format."""
    print(f"\n{description}")
    print(f"URL: {url_format}")
    print("-" * 60)
    
    api_key = getattr(config, "SN13_API_KEY", None)
    api_url = getattr(
        config,
        "SN13_API_URL",
        "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData",
    )
    
    if not api_key or api_key == "null":
        print("✗ SN13_API_KEY not set")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"source": "X", "url": url_format}
    
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        json_resp = resp.json()
        
        status = json_resp.get("status")
        data_list = json_resp.get("data") or []
        
        print(f"Status: {status}")
        print(f"Data items: {len(data_list)}")
        
        if status == "success" and data_list:
            print("✓ SUCCESS - Tweet found!")
            return True
        elif status == "success" and not data_list:
            print("⚠ SUCCESS status but empty data")
            return False
        else:
            print(f"✗ Failed - Status: {status}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    post_id = "1988779897620427125"
    
    if len(sys.argv) > 1:
        post_id = sys.argv[1]
    
    print(f"Testing different URL formats for post ID: {post_id}")
    print("=" * 60)
    
    # Test format 1: /i/web/status/ (what we currently use)
    url1 = f"https://x.com/i/web/status/{post_id}"
    result1 = test_url_format(post_id, url1, "Format 1: /i/web/status/ (current)")
    
    # Test format 2: Try to get username format (we'd need username for this)
    # For now, just test if /i/web/status/ works
    
    print("\n" + "=" * 60)
    if result1:
        print("✓ Current format works!")
    else:
        print("✗ Current format didn't work - may need to try username format")

