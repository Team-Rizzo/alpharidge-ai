#!/usr/bin/env python3
"""
Diagnostic script to investigate why a miner is only receiving batches from validator UID 21.

This script checks:
1. Miner's metagraph state - how many validators it sees
2. Whether the miner's metagraph is stale
3. How validators select miners (checking if this miner would be selected)
4. Whether there's an issue with hotkey lookup
"""

import bittensor as bt
import sys
from typing import List, Dict
from alpharidge_ai import config

def get_validator_info(metagraph) -> Dict[int, Dict]:
    """Extract validator information from metagraph."""
    validators = {}
    for uid in range(metagraph.n.item()):
        if metagraph.validator_permit[uid]:
            validators[uid] = {
                'hotkey': metagraph.hotkeys[uid],
                'stake': float(metagraph.S[uid]),
                'is_serving': metagraph.axons[uid].is_serving,
                'axon': metagraph.axons[uid],
            }
    return validators

def check_miner_metagraph_state(miner_config):
    """Check the miner's metagraph state."""
    print("=" * 80)
    print("MINER METAGRAPH STATE CHECK")
    print("=" * 80)
    
    # Initialize miner metagraph
    subtensor = bt.Subtensor(config=miner_config)
    metagraph = subtensor.metagraph(miner_config.netuid)
    metagraph.sync(subtensor=subtensor)
    
    print(f"\nMetagraph synced at block: {metagraph.block.item()}")
    print(f"Total UIDs in metagraph: {metagraph.n.item()}")
    
    # Get all validators
    validators = get_validator_info(metagraph)
    print(f"\nTotal validators found: {len(validators)}")
    
    if len(validators) == 0:
        print("⚠️  WARNING: No validators found in metagraph!")
        return None
    
    print("\nValidator details:")
    print("-" * 80)
    for uid, info in sorted(validators.items()):
        serving_status = "✓" if info['is_serving'] else "✗"
        print(f"UID {uid:3d}: {info['hotkey'][:20]}... | Stake: {info['stake']:12.2f} | Serving: {serving_status}")
    
    # Check for UID 21 specifically
    if 21 in validators:
        print(f"\n✓ Validator UID 21 found: {validators[21]['hotkey']}")
        print(f"  - Serving: {validators[21]['is_serving']}")
        print(f"  - Stake: {validators[21]['stake']}")
    else:
        print("\n⚠️  WARNING: Validator UID 21 NOT found in metagraph!")
    
    return metagraph, validators

def check_hotkey_lookup_issue(metagraph, validators):
    """Check if there's an issue with hotkey lookup."""
    print("\n" + "=" * 80)
    print("HOTKEY LOOKUP TEST")
    print("=" * 80)
    
    # Test looking up each validator's hotkey
    print("\nTesting hotkey lookup for each validator:")
    print("-" * 80)
    
    lookup_results = {}
    for uid, info in validators.items():
        try:
            found_uid = metagraph.hotkeys.index(info['hotkey'])
            lookup_results[uid] = found_uid
            if found_uid == uid:
                status = "✓ CORRECT"
            else:
                status = f"✗ WRONG! Found UID {found_uid} instead of {uid}"
            print(f"UID {uid:3d}: {status}")
        except ValueError:
            print(f"UID {uid:3d}: ✗ NOT FOUND in metagraph.hotkeys!")
            lookup_results[uid] = None
    
    # Check if all validators map to UID 21
    all_map_to_21 = all(uid == 21 or result == 21 for uid, result in lookup_results.items() if result is not None)
    if all_map_to_21 and len(validators) > 1:
        print("\n⚠️  CRITICAL: All validators are mapping to UID 21!")
        print("   This would explain why the miner always sees UID 21.")
    elif not all_map_to_21:
        print("\n✓ Hotkey lookups are correct - each validator maps to its own UID")
    
    return lookup_results

def check_miner_selection(miner_config, miner_hotkey, validators):
    """Check if this miner would be selected by validators."""
    print("\n" + "=" * 80)
    print("MINER SELECTION CHECK")
    print("=" * 80)
    
    subtensor = bt.Subtensor(config=miner_config)
    metagraph = subtensor.metagraph(miner_config.netuid)
    metagraph.sync(subtensor=subtensor)
    
    # Find miner UID
    try:
        miner_uid = metagraph.hotkeys.index(miner_hotkey)
        print(f"\nMiner hotkey: {miner_hotkey}")
        print(f"Miner UID: {miner_uid}")
    except ValueError:
        print(f"\n✗ ERROR: Miner hotkey {miner_hotkey} not found in metagraph!")
        return
    
    # Check miner status
    print(f"\nMiner status:")
    print(f"  - Is serving: {metagraph.axons[miner_uid].is_serving}")
    print(f"  - Stake: {float(metagraph.S[miner_uid]):.2f}")
    print(f"  - Validator permit: {bool(metagraph.validator_permit[miner_uid])}")
    
    # Check if miner would be available for selection
    from alpharidge_ai.utils.uids import check_uid_availability
    is_available = check_uid_availability(
        metagraph, 
        miner_uid, 
        miner_config.neuron.vpermit_tao_limit
    )
    print(f"  - Available for selection: {is_available}")
    
    if not is_available:
        print("\n⚠️  WARNING: Miner is NOT available for selection!")
        if not metagraph.axons[miner_uid].is_serving:
            print("   Reason: Miner axon is not serving")
        if metagraph.validator_permit[miner_uid] and metagraph.S[miner_uid] > miner_config.neuron.vpermit_tao_limit:
            print(f"   Reason: Validator permit with stake > {miner_config.neuron.vpermit_tao_limit}")
    
    # Simulate validator selection
    print(f"\nSimulating validator selection (using get_random_uids):")
    print("-" * 80)
    
    # Create a mock validator object for get_random_uids
    class MockValidator:
        def __init__(self, metagraph, config):
            self.metagraph = metagraph
            self.config = config
            self.uid = None  # Will be set per validator
    
    from alpharidge_ai.utils.uids import get_random_uids
    
    for validator_uid, validator_info in list(validators.items())[:5]:  # Test first 5 validators
        mock_validator = MockValidator(metagraph, miner_config)
        mock_validator.uid = validator_uid
        
        try:
            selected_uids = list(get_random_uids(mock_validator, k=10, exclude=[validator_uid]))
            if miner_uid in selected_uids:
                print(f"Validator UID {validator_uid:3d}: ✓ Would select miner UID {miner_uid}")
            else:
                print(f"Validator UID {validator_uid:3d}: ✗ Would NOT select miner UID {miner_uid}")
        except Exception as e:
            print(f"Validator UID {validator_uid:3d}: ✗ Error during selection: {e}")

def check_metagraph_staleness(miner_config):
    """Check if the miner's metagraph might be stale."""
    print("\n" + "=" * 80)
    print("METAGRAPH STALENESS CHECK")
    print("=" * 80)
    
    subtensor = bt.Subtensor(config=miner_config)
    metagraph = subtensor.metagraph(miner_config.netuid)
    
    # Sync to get fresh data
    print("\nSyncing metagraph...")
    metagraph.sync(subtensor=subtensor)
    current_block = metagraph.block.item()
    print(f"Current block: {current_block}")
    
    # Check last_update for each validator
    validators = get_validator_info(metagraph)
    print(f"\nLast update times for validators:")
    print("-" * 80)
    
    for uid in sorted(validators.keys()):
        last_update = metagraph.last_update[uid].item()
        blocks_since_update = current_block - last_update
        print(f"UID {uid:3d}: Last update at block {last_update:8d} ({blocks_since_update:6d} blocks ago)")

def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("MINER VALIDATOR ISSUE DIAGNOSTIC")
    print("=" * 80)
    print("\nThis script helps diagnose why a miner only receives batches from validator UID 21.")
    print("It checks the miner's metagraph state, hotkey lookups, and selection criteria.\n")
    
    # Get miner config
    miner_config = config()
    
    # Check if miner hotkey is provided
    if len(sys.argv) > 1:
        miner_hotkey = sys.argv[1]
    else:
        print("Usage: python diagnose_validator_issue.py <miner_hotkey>")
        print("\nIf you want to check without a specific miner hotkey, the script will")
        print("still check metagraph state and validator information.")
        miner_hotkey = None
    
    # Run diagnostics
    result = check_miner_metagraph_state(miner_config)
    if result is None:
        print("\n⚠️  Cannot proceed - no validators found in metagraph")
        return
    
    metagraph, validators = result
    
    check_hotkey_lookup_issue(metagraph, validators)
    check_metagraph_staleness(miner_config)
    
    if miner_hotkey:
        check_miner_selection(miner_config, miner_hotkey, validators)
    else:
        print("\n" + "=" * 80)
        print("MINER SELECTION CHECK SKIPPED")
        print("=" * 80)
        print("\nTo check if your miner would be selected by validators,")
        print("run this script with your miner's hotkey as an argument:")
        print(f"  python diagnose_validator_issue.py <your_miner_hotkey>")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nPossible causes for only seeing validator UID 21:")
    print("1. Miner's metagraph is stale (not syncing frequently enough)")
    print("2. Only validator UID 21 is selecting this miner (low stake, not serving, etc.)")
    print("3. Miner's metagraph has a bug where all validators map to UID 21")
    print("4. Other validators are not running or not selecting miners properly")
    print("\nRecommendations:")
    print("- Check miner logs for metagraph sync messages")
    print("- Verify miner is serving (axon.is_serving = True)")
    print("- Check miner's stake and validator permit status")
    print("- Verify other validators are actually running and selecting miners")

if __name__ == "__main__":
    main()













