#!/usr/bin/env python3
"""
Check which validators on subnet 45 have their axon ports open.
"""
import socket
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import bittensor as bt

NETUID = 45
TIMEOUT_SECONDS = 3.0
STAKE_THRESHOLD = 1000.0  # Minimum stake to be considered a validator


def check_port(ip: str, port: int, timeout: float = TIMEOUT_SECONDS) -> bool:
    """Check if a TCP port is open."""
    if not ip or ip == "0.0.0.0" or port == 0:
        return False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def main():
    print(f"\n🔍 Checking validator axon ports for Subnet {NETUID}...\n")
    
    # Connect to subtensor and get metagraph
    print("📡 Connecting to subtensor...")
    subtensor = bt.Subtensor()
    
    print("📊 Fetching metagraph...")
    metagraph = subtensor.metagraph(NETUID)
    
    # Find validators
    validators: List[Tuple[int, str, str, int, float]] = []
    for uid in range(metagraph.n):
        is_validator = bool(metagraph.validator_permit[uid])
        stake = float(metagraph.S[uid])
        
        if is_validator and stake >= STAKE_THRESHOLD:
            axon = metagraph.axons[uid]
            hotkey = metagraph.hotkeys[uid]
            ip = axon.ip if hasattr(axon, 'ip') else ""
            port = axon.port if hasattr(axon, 'port') else 0
            validators.append((uid, hotkey, ip, port, stake))
    
    print(f"\n✅ Found {len(validators)} validators with stake >= {STAKE_THRESHOLD}\n")
    
    # Check ports in parallel
    print("🔌 Checking axon ports...\n")
    
    results = {
        "open": [],
        "closed": [],
        "no_axon": [],
    }
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for uid, hotkey, ip, port, stake in validators:
            future = executor.submit(check_port, ip, port)
            futures.append((uid, hotkey, ip, port, stake, future))
        
        for uid, hotkey, ip, port, stake, future in futures:
            is_open = future.result()
            
            if not ip or ip == "0.0.0.0" or port == 0:
                results["no_axon"].append((uid, hotkey, ip, port, stake))
            elif is_open:
                results["open"].append((uid, hotkey, ip, port, stake))
            else:
                results["closed"].append((uid, hotkey, ip, port, stake))
    
    # Print results
    print("=" * 80)
    print(f"{'UID':<6} {'HOTKEY':<16} {'IP:PORT':<24} {'STAKE':<12} {'STATUS'}")
    print("=" * 80)
    
    # Open ports first (green)
    for uid, hotkey, ip, port, stake in sorted(results["open"], key=lambda x: -x[4]):
        print(f"\033[92m{uid:<6} {hotkey[:14]+'...':<16} {ip}:{port:<14} {stake:>10.0f}τ   OPEN\033[0m")
    
    # Closed ports (red)
    for uid, hotkey, ip, port, stake in sorted(results["closed"], key=lambda x: -x[4]):
        print(f"\033[91m{uid:<6} {hotkey[:14]+'...':<16} {ip}:{port:<14} {stake:>10.0f}τ   CLOSED\033[0m")
    
    # No axon configured (yellow)
    for uid, hotkey, ip, port, stake in sorted(results["no_axon"], key=lambda x: -x[4]):
        addr = f"{ip or 'N/A'}:{port or 'N/A'}"
        print(f"\033[93m{uid:<6} {hotkey[:14]+'...':<16} {addr:<24} {stake:>10.0f}τ   NO AXON\033[0m")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"  🟢 OPEN:     {len(results['open']):>3} validators")
    print(f"  🔴 CLOSED:   {len(results['closed']):>3} validators")
    print(f"  🟡 NO AXON:  {len(results['no_axon']):>3} validators")
    print(f"  📈 TOTAL:    {len(validators):>3} validators")
    print()
    
    # Calculate stake percentages
    total_stake = sum(v[4] for v in validators)
    open_stake = sum(v[4] for v in results["open"])
    closed_stake = sum(v[4] for v in results["closed"])
    
    if total_stake > 0:
        print(f"  Stake with OPEN ports:   {open_stake:>12.0f}τ ({100*open_stake/total_stake:.1f}%)")
        print(f"  Stake with CLOSED ports: {closed_stake:>12.0f}τ ({100*closed_stake/total_stake:.1f}%)")
    print()


if __name__ == "__main__":
    main()





