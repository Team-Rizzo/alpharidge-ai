#!/usr/bin/env python3
"""
Validator Stats Script
Run this to see current validation statistics from PM2 logs.

Usage:
    python scripts/vali_stats.py
    python scripts/vali_stats.py --log /path/to/custom.log
    python scripts/vali_stats.py --minutes 30
    python scripts/vali_stats.py --hotkey 5GTZqEMYsmwt   # Report for specific miner
"""

import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Default PM2 log path for sn45_vali
DEFAULT_LOG = Path("/home/rizzo/.pm2/logs/sn45-vali-out.log")

# ANSI escape code pattern
ANSI = re.compile(r'\x1b\[[0-9;]*m')
TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')

# Regex patterns for parsing log lines
FAILED_FIELDS_RE = re.compile(r"failed_fields=\['([^]]+)'\]")
MISMATCH_DETAIL_RE = re.compile(r"Failed fields: (.+)$")
FIELD_MISMATCH_RE = re.compile(r"(\w+) \(miner=([^ ]+) vs validator=([^)]+)\)")
# New format: field(m=value|v=value)
NEW_MISMATCH_RE = re.compile(r"Mismatch for (\w+):\s*(.+?)\s*\|\s*preview=(.*)$")
NEW_FIELD_COMPARE_RE = re.compile(r"(\w+)\(m=([^|]+)\|v=([^)]+)\)")
MINER_CLASS_RE = re.compile(r"Miner classification: subnet_id=(\d+), sentiment=(\w+), content_type=(\w+), tech=(\w+), market=(\w+), impact=(\w+)")
VALIDATOR_CLASS_RE = re.compile(r"Validator classification: subnet_id=(\d+), sentiment=(\w+), content_type=(\w+), tech=(\w+), market=(\w+), impact=(\w+)")


def parse_ts(line: str):
    """Extract timestamp from log line."""
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        return None


def analyze_logs(log_path: Path, cutoff: datetime):
    """Analyze validation logs since cutoff time."""
    stats = {
        'received': 0,
        'sampling': 0,
        'accepted': 0,
        'rejected': 0,
        'timeout': 0,
        'field_failures': defaultdict(int),  # field_name -> count
        'field_mismatches': defaultdict(lambda: defaultdict(int)),  # field -> (miner_val, vali_val) -> count
        'lazy_miner_count': 0,  # miners returning all defaults
        'miner_hotkeys': defaultdict(lambda: {'accepted': 0, 'rejected': 0}),
    }
    
    last_accepted = None
    
    for raw in log_path.open('r', errors='ignore'):
        line = ANSI.sub('', raw).rstrip('\n')
        ts = parse_ts(line)
        if ts is None or ts < cutoff:
            continue
        
        low = line.lower()
        
        # Count basic events
        if 'received tweetbatch' in low:
            stats['received'] += 1
        if 'sampling' in low and 'from batch' in low:
            stats['sampling'] += 1
        if 'batch accepted' in low:
            stats['accepted'] += 1
            last_accepted = line
        if 'batch rejected' in low:
            stats['rejected'] += 1
        if 'request timeout after' in low or 'adding penalty to hotkey' in low:
            stats['timeout'] += 1
            
        # Parse failed fields from validator.py logs
        # Example: failed_fields=['content_type', 'market_analysis', 'impact_potential']
        failed_match = FAILED_FIELDS_RE.search(line)
        if failed_match:
            fields_str = failed_match.group(1)
            fields = [f.strip().strip("'") for f in fields_str.split(",")]
            for field in fields:
                field = field.strip().strip("'")
                if field:
                    stats['field_failures'][field] += 1
        
        # Parse detailed mismatch info from scoring.py logs
        # Example: Failed fields: content_type (miner=other vs validator=community), sentiment (miner=neutral vs validator=bullish)
        mismatch_match = MISMATCH_DETAIL_RE.search(line)
        if mismatch_match:
            field_matches = FIELD_MISMATCH_RE.findall(mismatch_match.group(1))
            for field, miner_val, vali_val in field_matches:
                stats['field_mismatches'][field][(miner_val.lower(), vali_val.lower())] += 1
        
        # Detect "lazy miner" pattern - all defaults
        miner_match = MINER_CLASS_RE.search(line)
        if miner_match:
            sentiment, content_type, tech, market, impact = miner_match.groups()[1:]
            if (sentiment.lower() == 'neutral' and 
                content_type.lower() == 'other' and 
                tech.lower() == 'none' and 
                market.lower() == 'other' and 
                impact.lower() == 'none'):
                stats['lazy_miner_count'] += 1
        
        # Track per-miner stats by hotkey
        if 'batch validation failed for miner' in low:
            # Match full hotkey (48 chars) or truncated (12 chars with ..)
            hotkey_match = re.search(r'miner (\w{48}|\w{12}\.\.)[ \n]', line)
            if hotkey_match:
                hotkey = hotkey_match.group(1)[:12]  # Use first 12 chars for grouping
                stats['miner_hotkeys'][hotkey]['rejected'] += 1
        if 'batch validation passed for miner' in low:
            hotkey_match = re.search(r'miner (\w{48}|\w{12}\.\.)[ \n]', line)
            if hotkey_match:
                hotkey = hotkey_match.group(1)[:12]
                stats['miner_hotkeys'][hotkey]['accepted'] += 1
    
    stats['last_accepted'] = last_accepted
    return stats


def analyze_hotkey(log_path: Path, cutoff: datetime, hotkey_prefix: str):
    """Analyze logs for a specific miner hotkey."""
    stats = {
        'accepted': 0,
        'rejected': 0,
        'field_failures': defaultdict(int),
        'field_mismatches': defaultdict(list),  # field -> [(miner_val, validator_val, count)]
        'lazy_count': 0,
        'recent_failures': [],  # List of {ts, fields, preview, comparisons}
        'recent_successes': [],  # List of timestamps
    }
    
    hotkey_lower = hotkey_prefix.lower()
    current_rejection = None  # Track context for multi-line parsing
    
    for raw in log_path.open('r', errors='ignore'):
        line = ANSI.sub('', raw).rstrip('\n')
        ts = parse_ts(line)
        if ts is None or ts < cutoff:
            continue
        
        low = line.lower()
        
        # Check if this line mentions the hotkey
        if hotkey_lower not in low:
            continue
        
        # Track accepts/rejects
        if 'batch validation passed for miner' in low:
            stats['accepted'] += 1
            stats['recent_successes'].append(ts)
            
        if 'batch validation failed for miner' in low:
            stats['rejected'] += 1
            current_rejection = {'ts': ts, 'fields': [], 'preview': '', 'comparisons': []}
        
        # Parse new format: Mismatch for HOTKEY: field(m=val|v=val), ... | preview=...
        mismatch_match = NEW_MISMATCH_RE.search(line)
        if mismatch_match and hotkey_lower in low:
            fields_str = mismatch_match.group(2)
            preview = mismatch_match.group(3)[:80]
            
            # Parse each field comparison
            field_comparisons = NEW_FIELD_COMPARE_RE.findall(fields_str)
            comparisons = []
            for field, miner_val, validator_val in field_comparisons:
                stats['field_failures'][field] += 1
                stats['field_mismatches'][field].append((miner_val, validator_val))
                comparisons.append({'field': field, 'miner': miner_val, 'validator': validator_val})
            
            if current_rejection:
                current_rejection['fields'] = [c['field'] for c in comparisons]
                current_rejection['preview'] = preview
                current_rejection['comparisons'] = comparisons
                stats['recent_failures'].append(current_rejection)
                current_rejection = None
            else:
                # No prior "FAILED" line, create entry anyway
                stats['recent_failures'].append({
                    'ts': ts,
                    'fields': [c['field'] for c in comparisons],
                    'preview': preview,
                    'comparisons': comparisons
                })
        
        # Also parse old format: rejection detail with failed_fields=[...]
        elif 'rejection detail' in low and hotkey_lower in low:
            failed_match = FAILED_FIELDS_RE.search(line)
            if failed_match:
                fields_str = failed_match.group(1)
                fields = [f.strip().strip("'") for f in fields_str.split(",")]
                for field in fields:
                    field = field.strip().strip("'")
                    if field:
                        stats['field_failures'][field] += 1
                        if current_rejection:
                            current_rejection['fields'].append(field)
            
            # Extract preview
            preview_match = re.search(r'preview=(.+)$', line)
            if preview_match and current_rejection:
                current_rejection['preview'] = preview_match.group(1)[:80]
                stats['recent_failures'].append(current_rejection)
                current_rejection = None
    
    return stats


def get_recent_accepted(log_path: Path, limit: int = 10):
    """Get the most recent ACCEPTED batch lines."""
    accepted_lines = []
    for raw in log_path.open('r', errors='ignore'):
        line = ANSI.sub('', raw).rstrip('\n')
        if 'batch accepted' in line.lower():
            accepted_lines.append(line)
    return accepted_lines[-limit:]


def report_hotkey(log_path: Path, cutoff: datetime, hotkey: str, minutes: int):
    """Generate a report for a specific miner hotkey."""
    print("=" * 80)
    print(f"  MINER REPORT: {hotkey}")
    print(f"  Time Window: Last {minutes} minutes")
    print("=" * 80)
    print()
    
    stats = analyze_hotkey(log_path, cutoff, hotkey)
    
    total = stats['accepted'] + stats['rejected']
    if total == 0:
        print(f"  ⚠️  No validation events found for hotkey '{hotkey}' in the last {minutes} minutes.")
        print()
        print("  Tips:")
        print("    - Check if the hotkey prefix is correct (first 12+ chars)")
        print("    - Try increasing --minutes to search further back")
        print("    - Run without --hotkey to see all active miners")
        print()
        return
    
    accept_rate = stats['accepted'] / total * 100 if total > 0 else 0
    
    # Summary
    print("📊 SUMMARY")
    print("-" * 40)
    print(f"  Total Validations:   {total:>6}")
    print(f"  ✅ Accepted:         {stats['accepted']:>6}  ({accept_rate:.1f}%)")
    print(f"  ❌ Rejected:         {stats['rejected']:>6}  ({100-accept_rate:.1f}%)")
    print()
    
    # Field failure breakdown for this miner
    if stats['field_failures']:
        print("🔍 FIELD FAILURES FOR THIS MINER")
        print("-" * 60)
        sorted_failures = sorted(stats['field_failures'].items(), key=lambda x: -x[1])
        total_failures = sum(stats['field_failures'].values())
        for field, count in sorted_failures:
            pct = count / total_failures * 100 if total_failures > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {field:<20} {count:>5}  {bar} {pct:>5.1f}%")
        print()
    
    # Miner vs Validator value patterns
    if stats['field_mismatches']:
        print("📊 MINER VS VALIDATOR VALUES (what miner sent → what validator found)")
        print("-" * 80)
        for field, pairs in sorted(stats['field_mismatches'].items()):
            # Count occurrences of each miner→validator pattern
            pair_counts = defaultdict(int)
            for miner_val, validator_val in pairs:
                pair_counts[(miner_val, validator_val)] += 1
            
            # Show top patterns for this field
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:3]
            for (miner_val, validator_val), count in sorted_pairs:
                print(f"  {field:<18}  {miner_val:<12} → {validator_val:<15} ({count}x)")
        print()
    
    # Recent failures with details
    if stats['recent_failures']:
        print("📋 RECENT FAILURES (last 10)")
        print("-" * 90)
        for failure in stats['recent_failures'][-10:]:
            ts_str = failure['ts'].strftime('%H:%M:%S')
            preview = failure['preview'][:50] + "..." if len(failure['preview']) > 50 else failure['preview']
            
            # Show field comparisons if available
            comparisons = failure.get('comparisons', [])
            if comparisons:
                comp_strs = [f"{c['field']}({c['miner']}→{c['validator']})" for c in comparisons]
                print(f"  {ts_str} | {', '.join(comp_strs)}")
            else:
                fields = ', '.join(failure['fields']) if failure['fields'] else 'unknown'
                print(f"  {ts_str} | Fields: {fields}")
            
            if preview:
                print(f"           | Preview: {preview}")
        print()
    
    # Recent successes
    if stats['recent_successes']:
        print("✅ RECENT SUCCESSES (last 5)")
        print("-" * 40)
        for ts in stats['recent_successes'][-5:]:
            print(f"  {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Diagnosis
    print("🔎 DIAGNOSIS")
    print("-" * 60)
    if accept_rate == 0:
        print("  ❌ This miner has 0% acceptance rate")
        if stats['field_failures']:
            top_failure = sorted(stats['field_failures'].items(), key=lambda x: -x[1])[0]
            print(f"  → Most common failure: {top_failure[0]} ({top_failure[1]} times)")
        
        # Check for lazy miner pattern
        lazy_fields = {'content_type', 'sentiment', 'market_analysis', 'impact_potential', 'technical_quality'}
        failing_fields = set(stats['field_failures'].keys())
        if failing_fields >= lazy_fields or len(failing_fields) >= 4:
            print("  → Pattern suggests miner is returning default values (lazy miner)")
            print("  → Miner may not be running LLM analysis properly")
    elif accept_rate < 50:
        print(f"  ⚠️  Low acceptance rate ({accept_rate:.1f}%)")
        print("  → Miner's classifications frequently differ from validator")
    else:
        print(f"  ✅ Good acceptance rate ({accept_rate:.1f}%)")
    print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Show validator statistics')
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG,
                        help=f'Path to validator log file (default: {DEFAULT_LOG})')
    parser.add_argument('--minutes', type=int, default=60,
                        help='Analyze logs from the last N minutes (default: 60)')
    parser.add_argument('--hotkey', type=str, default=None,
                        help='Show detailed report for a specific miner hotkey (prefix match)')
    args = parser.parse_args()
    
    log_path = args.log
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1
    
    now = datetime.now()
    cutoff = now - timedelta(minutes=args.minutes)
    
    # If hotkey specified, show miner-specific report
    if args.hotkey:
        report_hotkey(log_path, cutoff, args.hotkey, args.minutes)
        return 0
    
    print("=" * 80)
    print(f"  VALIDATOR STATS  |  {now.strftime('%Y-%m-%d %H:%M:%S')}  |  Last {args.minutes} min")
    print("=" * 80)
    print()
    
    # Analyze logs
    stats = analyze_logs(log_path, cutoff)
    
    # Summary table
    total_validated = stats['accepted'] + stats['rejected']
    accept_rate = (stats['accepted'] / total_validated * 100) if total_validated > 0 else 0
    
    print("📊 SUMMARY")
    print("-" * 40)
    print(f"  Batches Received:    {stats['received']:>6}")
    print(f"  Batches Validated:   {total_validated:>6}")
    print(f"  ✅ Accepted:         {stats['accepted']:>6}  ({accept_rate:.1f}%)")
    print(f"  ❌ Rejected:         {stats['rejected']:>6}  ({100-accept_rate:.1f}%)")
    print(f"  ⏱️  Timeouts:         {stats['timeout']:>6}")
    print()
    
    # Field failure breakdown
    if stats['field_failures']:
        print("🔍 FIELD FAILURE BREAKDOWN (which fields caused rejections)")
        print("-" * 60)
        sorted_failures = sorted(stats['field_failures'].items(), key=lambda x: -x[1])
        total_failures = sum(stats['field_failures'].values())
        for field, count in sorted_failures:
            pct = count / total_failures * 100 if total_failures > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {field:<20} {count:>5}  {bar} {pct:>5.1f}%")
        print()
    
    # Common mismatch patterns
    if stats['field_mismatches']:
        print("🎯 TOP MISMATCH PATTERNS (miner → validator)")
        print("-" * 60)
        
        # Flatten and sort all mismatches
        all_mismatches = []
        for field, pairs in stats['field_mismatches'].items():
            for (miner_val, vali_val), count in pairs.items():
                all_mismatches.append((field, miner_val, vali_val, count))
        
        all_mismatches.sort(key=lambda x: -x[3])
        
        # Show top 15
        for field, miner_val, vali_val, count in all_mismatches[:15]:
            print(f"  {field:<18} {miner_val:<12} → {vali_val:<15} ({count}x)")
        print()
    
    # Lazy miner detection
    if stats['lazy_miner_count'] > 0:
        lazy_pct = stats['lazy_miner_count'] / stats['rejected'] * 100 if stats['rejected'] > 0 else 0
        print("⚠️  LAZY MINER DETECTION")
        print("-" * 60)
        print(f"  Miners returning all defaults (neutral/other/none): {stats['lazy_miner_count']}")
        print(f"  This accounts for {lazy_pct:.1f}% of rejections")
        print()
    
    # Per-miner stats (top offenders)
    if stats['miner_hotkeys']:
        print("👤 MINER PERFORMANCE (top 10 by volume)")
        print("-" * 60)
        
        # Sort by total volume
        miner_list = [(hotkey, data['accepted'], data['rejected']) 
                      for hotkey, data in stats['miner_hotkeys'].items()]
        miner_list.sort(key=lambda x: -(x[1] + x[2]))
        
        print(f"  {'Hotkey':<14} {'Accepted':>10} {'Rejected':>10} {'Rate':>8}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8}")
        for hotkey, accepted, rejected in miner_list[:10]:
            total = accepted + rejected
            rate = accepted / total * 100 if total > 0 else 0
            status = "✅" if rate > 50 else "❌" if rate < 20 else "⚠️"
            print(f"  {hotkey}.. {accepted:>10} {rejected:>10} {rate:>6.1f}% {status}")
        print()
        print(f"  💡 Tip: Run with --hotkey <prefix> for detailed miner report")
        print()
    
    # Recent accepted
    print("🏆 RECENT ACCEPTED BATCHES")
    print("-" * 60)
    recent = get_recent_accepted(log_path, limit=5)
    if recent:
        for line in recent:
            ts_match = TS_RE.match(line)
            if ts_match:
                ts_str = ts_match.group(1)
                print(f"  {ts_str} - Batch ACCEPTED")
    else:
        print("  (none found)")
    
    print()
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
