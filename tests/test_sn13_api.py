"""
Test file for SN13 API endpoint.
Helps diagnose 443 errors and test API connectivity.
"""
import os
import pytest
import requests
import time
import socket
import ssl
from unittest.mock import patch, MagicMock
from urllib.parse import urlparse

from talisman_ai.validator.sn13_api_client import SN13APIClient, create_client, _make_x_url_from_post_id
from talisman_ai import config


class TestSN13APIConnectivity:
    """Tests for SN13 API connectivity and 443 error diagnosis."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or config."""
        return os.getenv("SN13_API_KEY") or getattr(config, "SN13_API_KEY", None)

    @pytest.fixture
    def api_url(self):
        """Get API URL from environment or config."""
        return os.getenv("SN13_API_URL") or getattr(
            config,
            "SN13_API_URL",
            "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData",
        )

    def test_api_url_format(self, api_url):
        """Test that the API URL is properly formatted."""
        parsed = urlparse(api_url)
        assert parsed.scheme == "https", f"Expected HTTPS, got {parsed.scheme}"
        assert parsed.netloc, f"API URL missing hostname: {api_url}"
        print(f"✓ API URL format is valid: {api_url}")

    def test_dns_resolution(self, api_url):
        """Test DNS resolution for the API hostname."""
        parsed = urlparse(api_url)
        hostname = parsed.hostname
        
        try:
            ip_address = socket.gethostbyname(hostname)
            print(f"✓ DNS resolution successful: {hostname} -> {ip_address}")
            assert ip_address, "DNS resolution returned empty IP"
        except socket.gaierror as e:
            pytest.fail(f"✗ DNS resolution failed for {hostname}: {e}")

    def test_port_443_connectivity(self, api_url):
        """Test if port 443 is reachable."""
        parsed = urlparse(api_url)
        hostname = parsed.hostname
        port = parsed.port or 443
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                print(f"✓ Port {port} is reachable on {hostname}")
            else:
                pytest.fail(f"✗ Port {port} is not reachable on {hostname} (error code: {result})")
        except Exception as e:
            pytest.fail(f"✗ Failed to test port connectivity: {e}")

    def test_ssl_certificate(self, api_url):
        """Test SSL certificate validity."""
        parsed = urlparse(api_url)
        hostname = parsed.hostname
        port = parsed.port or 443
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    print(f"✓ SSL certificate is valid for {hostname}")
                    print(f"  Certificate subject: {cert.get('subject', 'N/A')}")
                    # Don't return cert, just assert it exists
                    assert cert is not None
        except ssl.SSLError as e:
            pytest.fail(f"✗ SSL certificate error: {e}")
        except Exception as e:
            pytest.fail(f"✗ Failed to verify SSL certificate: {e}")

    def test_basic_https_connection(self, api_url):
        """Test basic HTTPS connection without authentication."""
        try:
            response = requests.get(api_url, timeout=10, verify=True)
            print(f"✓ Basic HTTPS connection successful (status: {response.status_code})")
        except requests.exceptions.SSLError as e:
            pytest.fail(f"✗ SSL error: {e}")
        except requests.exceptions.ConnectionError as e:
            # 443 errors often manifest as ConnectionError
            error_msg = str(e).lower()
            if "443" in error_msg or "connection refused" in error_msg:
                pytest.fail(f"✗ Connection error (possible 443 issue): {e}")
            pytest.fail(f"✗ Connection error: {e}")
        except requests.exceptions.Timeout:
            pytest.fail(f"✗ Connection timeout (server may be unreachable)")
        except Exception as e:
            # For POST endpoints, GET might return 405 Method Not Allowed, which is OK
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 405:
                    print(f"✓ HTTPS connection successful (405 Method Not Allowed is expected for GET)")
                    return
            pytest.fail(f"✗ Unexpected error: {e}")

    def test_api_key_present(self, api_key):
        """Test that API key is configured."""
        if not api_key or api_key == "null":
            pytest.skip("SN13_API_KEY not set in environment")
        assert len(api_key) > 0, "API key is empty"
        print(f"✓ API key is configured (length: {len(api_key)})")

    def test_api_authentication_headers(self, api_key, api_url):
        """Test API request with authentication headers."""
        if not api_key or api_key == "null":
            pytest.skip("SN13_API_KEY not set in environment")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"source": "X", "url": "https://x.com/i/web/status/1234567890"}
        
        parsed = urlparse(api_url)
        print(f"\n🔍 Testing API endpoint: {api_url}")
        print(f"   Hostname: {parsed.hostname}")
        print(f"   Port: {parsed.port or 443}")
        print(f"   Path: {parsed.path}")
        print(f"   Payload: {payload}")
        
        # First, test with a shorter timeout to see if connection is established
        print("\n⏱️  Testing connection with 5s timeout...")
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=5,
                verify=True
            )
            print(f"✓ Quick connection test successful (status: {response.status_code})")
        except requests.exceptions.Timeout:
            print("⚠ Quick timeout test failed - connection may be slow")
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e).lower()
            print(f"⚠ Quick connection test failed: {e}")
            if "443" in error_msg or "connection refused" in error_msg:
                pytest.fail(f"✗ 443 Connection error: {e}\n"
                           f"  This could indicate:\n"
                           f"  - Firewall blocking port 443\n"
                           f"  - SSL/TLS handshake failure\n"
                           f"  - Network connectivity issue\n"
                           f"  - Proxy configuration needed")
        except Exception as e:
            print(f"⚠ Quick connection test error: {e}")
        
        # Now test with full timeout
        print("\n⏱️  Testing with full 30s timeout...")
        start_time = time.time()
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30,
                verify=True
            )
            elapsed = time.time() - start_time
            print(f"✓ API request sent successfully (status: {response.status_code}, elapsed: {elapsed:.2f}s)")
            print(f"  Response headers: {dict(response.headers)}")
            if response.status_code == 200:
                print(f"  Response body (first 200 chars): {response.text[:200]}")
            else:
                print(f"  Response body: {response.text}")
        except requests.exceptions.SSLError as e:
            elapsed = time.time() - start_time
            pytest.fail(f"✗ SSL error during API request (after {elapsed:.2f}s): {e}")
        except requests.exceptions.ConnectionError as e:
            elapsed = time.time() - start_time
            error_msg = str(e).lower()
            if "443" in error_msg:
                pytest.fail(f"✗ 443 Connection error (after {elapsed:.2f}s): {e}\n"
                           f"  This could indicate:\n"
                           f"  - Firewall blocking port 443\n"
                           f"  - SSL/TLS handshake failure\n"
                           f"  - Network connectivity issue\n"
                           f"  - Proxy configuration needed")
            pytest.fail(f"✗ Connection error (after {elapsed:.2f}s): {e}")
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            pytest.fail(f"✗ Request timeout after {elapsed:.2f}s (server may be slow or unreachable)\n"
                       f"  Troubleshooting steps:\n"
                       f"  1. Check if the API endpoint is correct: {api_url}\n"
                       f"  2. Verify network connectivity: ping {parsed.hostname}\n"
                       f"  3. Test with curl: curl -X POST {api_url} -H 'Authorization: Bearer ...' -H 'Content-Type: application/json' -d '{payload}'\n"
                       f"  4. Check firewall/proxy settings\n"
                       f"  5. The server may be experiencing high load")
        except Exception as e:
            elapsed = time.time() - start_time
            pytest.fail(f"✗ Unexpected error (after {elapsed:.2f}s): {e}")

    def test_endpoint_diagnosis(self, api_key, api_url):
        """Comprehensive endpoint diagnosis to help debug timeout/443 issues."""
        if not api_key or api_key == "null":
            pytest.skip("SN13_API_KEY not set in environment")
        
        parsed = urlparse(api_url)
        hostname = parsed.hostname
        port = parsed.port or 443
        
        print(f"\n🔍 Comprehensive Endpoint Diagnosis")
        print(f"=" * 60)
        print(f"Endpoint: {api_url}")
        print(f"Hostname: {hostname}")
        print(f"Port: {port}")
        print(f"Path: {parsed.path}")
        
        # Test 1: DNS Resolution
        print(f"\n1️⃣  Testing DNS resolution...")
        try:
            ip = socket.gethostbyname(hostname)
            print(f"   ✓ DNS resolved: {hostname} -> {ip}")
        except socket.gaierror as e:
            pytest.fail(f"   ✗ DNS resolution failed: {e}")
        
        # Test 2: TCP Connection
        print(f"\n2️⃣  Testing TCP connection to {hostname}:{port}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((hostname, port))
            sock.close()
            if result == 0:
                print(f"   ✓ TCP connection successful")
            else:
                pytest.fail(f"   ✗ TCP connection failed (error code: {result})")
        except Exception as e:
            pytest.fail(f"   ✗ TCP connection error: {e}")
        
        # Test 3: SSL Handshake
        print(f"\n3️⃣  Testing SSL/TLS handshake...")
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    print(f"   ✓ SSL handshake successful")
                    print(f"   ✓ Protocol: {ssock.version()}")
                    print(f"   ✓ Cipher: {ssock.cipher()[0]}")
        except ssl.SSLError as e:
            pytest.fail(f"   ✗ SSL handshake failed: {e}")
        except Exception as e:
            pytest.fail(f"   ✗ SSL connection error: {e}")
        
        # Test 4: HTTP HEAD request (if supported)
        print(f"\n4️⃣  Testing HTTP HEAD request...")
        try:
            response = requests.head(api_url, timeout=10, verify=True, allow_redirects=True)
            print(f"   ✓ HTTP HEAD successful (status: {response.status_code})")
        except requests.exceptions.Timeout:
            print(f"   ⚠ HTTP HEAD timed out (endpoint may not support HEAD)")
        except requests.exceptions.ConnectionError as e:
            print(f"   ⚠ HTTP HEAD connection error: {e}")
        except Exception as e:
            print(f"   ⚠ HTTP HEAD error: {e}")
        
        # Test 5: POST with minimal payload
        print(f"\n5️⃣  Testing POST with minimal payload...")
        print(f"   Note: This endpoint is known to be slow (20-30s response time)")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        minimal_payload = {"source": "X", "url": "https://x.com/i/web/status/1234567890"}
        
        start = time.time()
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=minimal_payload,
                timeout=35,  # Increased to 35s to account for slow endpoint
                verify=True
            )
            elapsed = time.time() - start
            print(f"   ✓ POST request successful (status: {response.status_code}, time: {elapsed:.2f}s)")
            print(f"   Response size: {len(response.content)} bytes")
        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print(f"   ✗ POST request timed out after {elapsed:.2f}s")
            print(f"\n   💡 Possible causes:")
            print(f"      - Server is overloaded or slow")
            print(f"      - Network latency is high")
            print(f"      - Firewall/proxy is interfering")
            print(f"      - API endpoint may require different authentication")
        except requests.exceptions.ConnectionError as e:
            error_str = str(e).lower()
            if "443" in error_str:
                print(f"   ✗ POST request failed with 443 error: {e}")
                print(f"\n   💡 443 errors typically indicate:")
                print(f"      - Firewall blocking HTTPS traffic")
                print(f"      - SSL/TLS handshake failure")
                print(f"      - Proxy configuration needed")
                print(f"      - Network routing issue")
            else:
                print(f"   ✗ POST request connection error: {e}")
        except Exception as e:
            print(f"   ✗ POST request error: {e}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ Diagnosis complete")


class TestSN13APIClient:
    """Tests for SN13APIClient class."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or config."""
        return os.getenv("SN13_API_KEY") or getattr(config, "SN13_API_KEY", None)

    @pytest.fixture
    def api_url(self):
        """Get API URL from environment or config."""
        return os.getenv("SN13_API_URL") or getattr(
            config,
            "SN13_API_URL",
            "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData",
        )

    def test_make_x_url_from_post_id(self):
        """Test URL generation from post ID."""
        # Test with numeric post ID
        url = _make_x_url_from_post_id("1234567890")
        assert url == "https://x.com/i/web/status/1234567890"
        
        # Test with full URL
        url = _make_x_url_from_post_id("https://x.com/i/web/status/1234567890")
        assert url == "https://x.com/i/web/status/1234567890"
        
        # Test with username/post format
        url = _make_x_url_from_post_id("username/1234567890")
        assert url == "https://x.com/username/1234567890"
        
        print("✓ URL generation works correctly")

    def test_client_initialization(self, api_key, api_url):
        """Test SN13APIClient initialization."""
        if not api_key or api_key == "null":
            pytest.skip("SN13_API_KEY not set in environment")
        
        client = SN13APIClient(api_key, api_url)
        assert client.api_key == api_key
        assert client.api_url == api_url
        assert client._session is not None
        print("✓ SN13APIClient initialized successfully")

    def test_create_client_from_config(self):
        """Test create_client function."""
        try:
            client = create_client()
            assert isinstance(client, SN13APIClient)
            print("✓ create_client() works correctly")
        except ValueError as e:
            if "SN13_API_KEY not set" in str(e):
                pytest.skip("SN13_API_KEY not set in environment")
            raise

    @pytest.mark.skip(reason="Requires valid API key and may make actual API calls")
    def test_fetch_post_real(self, api_key, api_url):
        """Test fetching a real post (skipped by default to avoid API usage)."""
        if not api_key or api_key == "null":
            pytest.skip("SN13_API_KEY not set in environment")
        
        client = SN13APIClient(api_key, api_url)
        # Use a known post ID for testing
        post_id = "1234567890"  # Replace with a real post ID
        
        try:
            result = client.fetch_post(post_id)
            if result:
                print(f"✓ Successfully fetched post: {result.id}")
                assert result.id is not None
            else:
                print("⚠ Post not found or API returned no data")
        except requests.exceptions.ConnectionError as e:
            if "443" in str(e).lower():
                pytest.fail(f"✗ 443 Connection error: {e}")
            raise

    def test_fetch_post_mock_success(self):
        """Test fetch_post with mocked successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": [{
                "id": "1234567890",
                "text": "Test tweet",
                "datetime": "2024-01-01T00:00:00Z",
                "tweet": {
                    "id": "1234567890",
                    "like_count": 10,
                    "retweet_count": 5,
                    "reply_count": 2
                },
                "user": {
                    "id": "987654321",
                    "username": "testuser",
                    "display_name": "Test User",
                    "followers_count": 1000
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('requests.Session.post', return_value=mock_response):
            client = SN13APIClient("test_key", "https://test.api.com/endpoint")
            result = client.fetch_post("1234567890")
            
            assert result is not None
            assert result.id == "1234567890"
            assert result.text == "Test tweet"
            print("✓ Mock fetch_post works correctly")

    def test_fetch_post_mock_connection_error(self):
        """Test fetch_post with mocked 443 connection error."""
        mock_error = requests.exceptions.ConnectionError("Connection refused on port 443")
        
        with patch('requests.Session.post', side_effect=mock_error):
            client = SN13APIClient("test_key", "https://test.api.com/endpoint")
            
            with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
                client._fetch_once("1234567890")
            
            assert "443" in str(exc_info.value) or "Connection" in str(exc_info.value)
            print("✓ Connection error handling works correctly")

    def test_fetch_post_mock_ssl_error(self):
        """Test fetch_post with mocked SSL error."""
        mock_error = requests.exceptions.SSLError("SSL certificate verification failed")
        
        with patch('requests.Session.post', side_effect=mock_error):
            client = SN13APIClient("test_key", "https://test.api.com/endpoint")
            
            with pytest.raises(requests.exceptions.SSLError):
                client._fetch_once("1234567890")
            
            print("✓ SSL error handling works correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

