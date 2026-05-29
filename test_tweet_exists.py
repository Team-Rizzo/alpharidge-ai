#!/usr/bin/env python3
"""Test script to check if a tweet exists in SN13 API."""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from talisman_ai.validator.sn13_api_client import create_client

def test_tweet_exists(post_id: str):
    """Test if a tweet exists in SN13 API."""
    print(f"Testing tweet ID: {post_id}")
    print("=" * 60)
    
    try:
        client = create_client()
        print(f"✓ SN13 API client created")
        print(f"  API URL: {client.api_url}")
        print()
        
        print(f"Fetching post from SN13 API...")
        post_record = client.fetch_post(post_id, attempts=1)
        
        if post_record:
            print(f"✓ Tweet found!")
            print(f"  Post ID: {post_record.id}")
            print(f"  Author: @{post_record.author.username if post_record.author else 'N/A'}")
            print(f"  Text (first 100 chars): {post_record.text[:100]}...")
            print(f"  Created at: {post_record.created_at}")
            print(f"  Likes: {post_record.public_metrics.like_count}")
            print(f"  Retweets: {post_record.public_metrics.retweet_count}")
            print(f"  Replies: {post_record.public_metrics.reply_count}")
        else:
            print(f"✗ Tweet not found (API returned None)")
            print(f"  This could mean:")
            print(f"  - The tweet doesn't exist")
            print(f"  - The tweet is inaccessible")
            print(f"  - SN13 API couldn't fetch it from their miners")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the tweet ID from the error message
    post_id = "1988779897620427125"
    
    if len(sys.argv) > 1:
        post_id = sys.argv[1]
    
    test_tweet_exists(post_id)

