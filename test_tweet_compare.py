#!/usr/bin/env python3
"""Compare SN13 API vs X API for fetching a tweet."""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from talisman_ai.validator.sn13_api_client import create_client as create_sn13_client
from talisman_ai.validator.x_api_client import create_client as create_x_client

def test_sn13(post_id_or_url: str, full_url: str = None):
    """Test SN13 API."""
    print("=" * 60)
    print("SN13 API Test")
    print("=" * 60)
    
    try:
        from talisman_ai import config
        import requests
        
        # Print SN13 API key (masked for security)
        api_key = getattr(config, "SN13_API_KEY", None)
        if api_key and api_key != "null":
            masked_key = api_key[:8] + "..." + api_key[-8:] if len(api_key) > 16 else "***"
            print(f"SN13 API Key: {masked_key}")
            print(f"SN13 API URL: {getattr(config, 'SN13_API_URL', 'N/A')}")
        else:
            print(f"✗ SN13_API_KEY not set or is 'null'")
            return False
        print()
        
        client = create_sn13_client()
        print(f"✓ SN13 API client created")
        
        # Use full URL if provided, otherwise try to get it from X API first
        if full_url:
            post_url = full_url
            print(f"Using provided full URL for SN13")
        elif post_id_or_url.startswith("http"):
            post_url = post_id_or_url
            print(f"Using provided URL as-is")
        else:
            # If we only have post ID, try to get username from X API first
            print(f"Only post ID provided, trying to get full URL from X API...")
            try:
                x_client = create_x_client()
                x_post = x_client.fetch_post(post_id_or_url, attempts=1)
                if x_post and x_post.author and x_post.author.username:
                    post_url = f"https://x.com/{x_post.author.username}/status/{post_id_or_url}"
                    print(f"✓ Got username from X API: @{x_post.author.username}")
                else:
                    # Fallback to /i/web/status/ format
                    post_url = f"https://x.com/i/web/status/{post_id_or_url}"
                    print(f"⚠ Could not get username, using /i/web/status/ format")
            except Exception as e:
                # Fallback to /i/web/status/ format
                post_url = f"https://x.com/i/web/status/{post_id_or_url}"
                print(f"⚠ Error getting username from X API: {e}")
                print(f"  Falling back to /i/web/status/ format")
        
        print(f"Post ID/URL input: {post_id_or_url}")
        print(f"URL being sent to SN13: {post_url}")
        print()
        
        start = time.time()
        print(f"Fetching post from SN13 API...")
        print(f"  (Note: SN13 can take 2+ minutes to respond)")
        
        # Make direct API call to SN13 with the full URL
        try:
            headers = {
                "Authorization": f"Bearer {client.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"source": "X", "url": post_url}
            resp = requests.post(client.api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            json_resp = resp.json()
            
            elapsed = time.time() - start
            
            # SN13 API may not always return a "status" field, but if there's data, we should process it
            status = json_resp.get("status")
            data_list = json_resp.get("data") or []
            
            if status is not None and status != "success":
                print(f"✗ SN13 API returned status: {status}")
                print(f"  Response: {json_resp}")
                return False
            
            if not data_list:
                print(f"✗ Tweet not found (took {elapsed:.2f}s)")
                print(f"  API returned empty data - tweet may not exist in SN13")
                return False
            
            # Find the tweet with matching ID in the data list
            target_tweet = None
            for item in data_list:
                tweet_data = item.get("tweet", {})
                if tweet_data.get("id") == post_id_or_url or tweet_data.get("id") == str(post_id_or_url):
                    target_tweet = item
                    break
            
            # If not found by ID, use first item (SN13 might return multiple tweets)
            if not target_tweet and data_list:
                print(f"⚠ Tweet ID not found in response, using first item from data list")
                target_tweet = data_list[0]
            
            if not target_tweet:
                print(f"✗ Tweet not found in SN13 response")
                return False
            
            item = target_tweet
            
            # Parse the response similar to SN13APIClient
            from dateutil.parser import isoparse
            from talisman_ai.validator.x_post_models import PostRecord, PublicMetrics, AuthorInfo
            
            item = data_list[0]
            dt = item.get("datetime")
            created_at = isoparse(dt) if dt and isinstance(dt, str) else None
            
            if created_at is None:
                print(f"✗ SN13 response missing datetime")
                return False
            
            post_data = item.get("tweet", {}) or {}
            user_data = item.get("user", {}) or {}
            
            metrics = PublicMetrics(
                like_count=int(post_data.get("like_count", 0) or 0),
                retweet_count=int(post_data.get("retweet_count", 0) or 0),
                reply_count=int(post_data.get("reply_count", 0) or 0),
            )
            
            author_info = AuthorInfo(
                id=str(user_data["id"]) if user_data.get("id") is not None else None,
                username=user_data.get("username", "") or "",
                display_name=user_data.get("display_name", "") or "",
                followers_count=int(user_data.get("followers_count", 0) or 0),
                created_at=None,
            )
            
            post_record = PostRecord(
                id=str(post_data.get("id") or ""),
                text=item.get("text", "") or "",
                created_at=created_at,
                public_metrics=metrics,
                author=author_info,
            )
            
            elapsed = time.time() - start
            
        except Exception as fetch_error:
            elapsed = time.time() - start
            print(f"✗ Fetch error (took {elapsed:.2f}s): {fetch_error}")
            import traceback
            traceback.print_exc()
            return False
        
        if post_record:
            print(f"✓ Tweet found! (took {elapsed:.2f}s)")
            print(f"  Post ID: {post_record.id}")
            print(f"  Author: @{post_record.author.username if post_record.author else 'N/A'}")
            print(f"  Display Name: {post_record.author.display_name if post_record.author else 'N/A'}")
            print(f"  Text (first 150 chars): {post_record.text[:150]}...")
            print(f"  Created at: {post_record.created_at}")
            print(f"  Likes: {post_record.public_metrics.like_count}")
            print(f"  Retweets: {post_record.public_metrics.retweet_count}")
            print(f"  Replies: {post_record.public_metrics.reply_count}")
            print(f"  Followers: {post_record.author.followers_count if post_record.author else 'N/A'}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_x_api(post_id: str):
    """Test X API."""
    print()
    print("=" * 60)
    print("X API Test")
    print("=" * 60)
    
    try:
        client = create_x_client()
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
            print(f"  Text (first 150 chars): {post_record.text[:150]}...")
            print(f"  Created at: {post_record.created_at}")
            print(f"  Likes: {post_record.public_metrics.like_count}")
            print(f"  Retweets: {post_record.public_metrics.retweet_count}")
            print(f"  Replies: {post_record.public_metrics.reply_count}")
            print(f"  Followers: {post_record.author.followers_count if post_record.author else 'N/A'}")
            return True
        else:
            print(f"✗ Tweet not found (took {elapsed:.2f}s)")
            print(f"  API returned None - tweet may not exist")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_results(sn13_result, x_result):
    """Compare results from both APIs."""
    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    
    if sn13_result and x_result:
        print("✓ Both APIs found the tweet")
        print()
        print("Comparing data...")
        # Note: We'd need to store the results to compare, but for now just show status
        print("  Both APIs returned valid PostRecord objects")
    elif sn13_result and not x_result:
        print("⚠ SN13 found the tweet, but X API did not")
        print("  This suggests the tweet exists but X API couldn't access it")
    elif not sn13_result and x_result:
        print("⚠ X API found the tweet, but SN13 did not")
        print("  This suggests SN13's miners couldn't find the tweet")
    else:
        print("✗ Neither API found the tweet")
        print("  The tweet may not exist or be inaccessible")

if __name__ == "__main__":
    # Test the tweet ID from the error message
    input_arg = "1988779897620427125"
    
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]
    
    # Determine if it's a full URL or just post ID
    full_url = None
    post_id = input_arg
    
    if input_arg.startswith("http"):
        # It's a full URL - extract post ID but keep URL for SN13
        full_url = input_arg
        if "status/" in input_arg:
            post_id = input_arg.split("status/")[-1].split("?")[0].split("/")[0]
    elif "status/" in input_arg:
        # It's a URL-like format but missing https://
        full_url = input_arg if input_arg.startswith("https://") else f"https://{input_arg}"
        post_id = input_arg.split("status/")[-1].split("?")[0].split("/")[0]
    
    print(f"Input: {input_arg}")
    if full_url:
        print(f"Full URL: {full_url}")
    print(f"Post ID: {post_id}")
    print()
    
    sn13_found = test_sn13(post_id, full_url=full_url)
    x_found = test_x_api(post_id)
    
    compare_results(sn13_found, x_found)

