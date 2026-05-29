#!/usr/bin/env python3
"""Test script to check if a tweet exists in X API."""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from talisman_ai.validator.x_api_client import create_client

def test_tweet_x_api(post_id: str):
    """Test if a tweet exists in X API."""
    print(f"Testing tweet ID: {post_id}")
    print("=" * 60)
    
    try:
        client = create_client()
        print(f"✓ X API client created")
        print()
        
        start = time.time()
        print(f"Fetching post from X API...")
        post_record = client.fetch_post(post_id, attempts=1)
        elapsed = time.time() - start
        
        if post_record:
            print(f"✓ Tweet found! (took {elapsed:.2f}s)")
            print(f"  Post ID: {post_record.id}")
            print(f"  Author: @{post_record.author.username if post_record.author else 'N/A'}")
            print(f"  Display Name: {post_record.author.display_name if post_record.author else 'N/A'}")
            print(f"  Text (first 200 chars): {post_record.text[:200]}...")
            print(f"  Created at: {post_record.created_at}")
            print(f"  Likes: {post_record.public_metrics.like_count}")
            print(f"  Retweets: {post_record.public_metrics.retweet_count}")
            print(f"  Replies: {post_record.public_metrics.reply_count}")
            print(f"  Followers: {post_record.author.followers_count if post_record.author else 'N/A'}")
            if post_record.author and post_record.author.created_at:
                print(f"  Author account created: {post_record.author.created_at}")
        else:
            print(f"✗ Tweet not found (took {elapsed:.2f}s)")
            print(f"  This could mean:")
            print(f"  - The tweet doesn't exist")
            print(f"  - The tweet is private/deleted")
            print(f"  - The tweet is inaccessible")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the tweet ID from the error message
    post_id = "1988779897620427125"
    
    if len(sys.argv) > 1:
        post_id = sys.argv[1]
    
    test_tweet_x_api(post_id)

