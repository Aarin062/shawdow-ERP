"""
===========================================================================
STEPS 4 & 5: TEMPORAL TOPOLOGY ENGINE + FEATURE EXTRACTION
Project: The Self-Auditing Ledger
Purpose: Detect fraud structures in the Shadow Graph and extract features.
===========================================================================

WHAT THIS SCRIPT DOES (The Core Innovation):
---------------------------------------------
This is the most important script in the entire project. It implements
the two-phase detection approach that makes our system unique:

  PHASE 1: TOPOLOGY DETECTION ("The Shape Finder")
  -------------------------------------------------
  We run 5 Cypher queries against Neo4j that look for specific
  structural patterns known to indicate fraud:
    Detector 1: Circular Payment Patterns
    Detector 2: Lapping Indicators
    Detector 3: Abnormal Transaction Density
    Detector 4: Rapid Transfer Chains
    Detector 5: Amount Anomalies

  These detectors DON'T declare fraud — they find SHAPES.

  PHASE 2: FEATURE EXTRACTION ("The Translator")
  -------------------------------------------------
  We convert those shapes into numerical features suitable for
  machine learning:
    - cycle_score, density_score, lapping_score, etc.
    - Plus graph-structural features (degree, centrality)
    - Plus temporal features (burst counts, time deltas)

  The output is a CSV where each row = one Document, and each
  column = one structural feature. This CSV feeds into Step 5
  (AI Model Training).

WHY THIS APPROACH IS BETTER THAN PURE ML:
-------------------------------------------
Traditional ML:  Raw transaction → [Black Box Model] → Fraud/NotFraud
Our approach:    Raw transaction → [Graph Topology] → Structural Features
                                                           ↓
                                                   [Interpretable ML] → Fraud Score
                                                           ↓
                                                   "Found circular payment loop
                                                    of length 3, amount $50K,
                                                    within 5-minute window"

The graph layer provides EXPLAINABILITY — an auditor can see exactly
WHY the system flagged something, not just a probability number.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from neo4j import GraphDatabase

sys.stdout.reconfigure(encoding='utf-8')

# ===================================================================
# CONFIGURATION
# ===================================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "shadowgraph2026"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import"
)

# Time window for sliding window analysis (in seconds)
# WHY 300 SECONDS (5 MINUTES)?
# In real ERP fraud, suspicious patterns often occur within minutes.
# A circular payment scheme typically executes within a single session.
# 5 minutes is a reasonable starting window for ERPsim data.
TIME_WINDOW_SECONDS = 300

# ===================================================================
# NEO4J CONNECTION
# ===================================================================
def create_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"  ✅ Connected to Neo4j at {NEO4J_URI}")
    return driver

def run_query(driver, query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


# ===================================================================
# PHASE 1: TOPOLOGY DETECTORS
# ===================================================================

# -----------------------------------------------------------------
# DETECTOR 1: CIRCULAR PAYMENT / VENDOR CYCLE DETECTION
# -----------------------------------------------------------------
def detector_circular_payments(driver):
    """
    WHAT IT FINDS:
    Documents that share the same vendor in a cycle-like pattern
    within a sliding time window.

    THE PATTERN:
      Doc_1 → Vendor_A (time T)
      Doc_2 → Vendor_B (time T + delta1)
      Doc_3 → Vendor_A (time T + delta2)  ← same vendor again!
    All within TIME_WINDOW_SECONDS.

    WHY THIS MATTERS:
    Circular payments are when money flows A→B→C→A to disguise
    its origin. In our graph, this appears as the same vendor
    appearing in multiple documents within a short time window,
    with different vendors in between.

    In a real ERP system with hundreds of vendors, this would find
    actual multi-hop cycles. With ERPsim's 5 vendors, we detect
    the structural pattern that would indicate circular flow.
    """
    print("\n  🔍 Detector 1: Circular Payment Patterns...")

    # OPTIMIZED: Process per source_file to avoid cartesian explosion.
    # The original 3-way MATCH across all 4,000+ vendor edges caused
    # Neo4j to run out of memory. By scoping to one source_file at a
    # time, we keep the working set small.

    query_per_source = """
    // Find vendor cycle patterns within a single simulation run.
    // Step 1: Get all vendor-document pairs for this source file
    // Step 2: Find cases where the same vendor reappears after
    //         a different vendor within the time window.

    MATCH (d1:Document)-[:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(v1:Vendor)
    WHERE d1.source_file = $source_file
    WITH d1, v1
    ORDER BY d1.timestamp

    // Find a second doc with a DIFFERENT vendor, after d1
    MATCH (d2:Document)-[:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(v2:Vendor)
    WHERE d2.source_file = $source_file
      AND v1 <> v2
      AND d2.timestamp > d1.timestamp
      AND duration.between(d1.timestamp, d2.timestamp).seconds <= $time_window
    WITH d1, v1, d2, v2

    // Find a third doc that goes BACK to v1 (the cycle)
    MATCH (d3:Document)-[:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(v1)
    WHERE d3.source_file = $source_file
      AND d3 <> d1 AND d3 <> d2
      AND d3.timestamp > d2.timestamp
      AND duration.between(d1.timestamp, d3.timestamp).seconds <= $time_window
    RETURN d1.doc_id AS start_doc,
           d2.doc_id AS middle_doc,
           d3.doc_id AS end_doc,
           v1.vendor_id AS cycle_vendor,
           v2.vendor_id AS middle_vendor,
           duration.between(d1.timestamp, d3.timestamp).seconds AS cycle_duration_seconds,
           d1.total_amount + d2.total_amount + d3.total_amount AS cycle_total_amount
    ORDER BY cycle_duration_seconds ASC
    LIMIT 100
    """

    # Get source files
    sources = run_query(driver, "MATCH (d:Document) RETURN DISTINCT d.source_file AS sf")
    source_files = [s['sf'] for s in sources]

    all_results = []
    for sf in source_files:
        print(f"    Scanning {sf}...", end=" ")
        results = run_query(driver, query_per_source, {
            "source_file": sf,
            "time_window": TIME_WINDOW_SECONDS
        })
        print(f"{len(results)} patterns")
        all_results.extend(results)

    print(f"    Total circular patterns found: {len(all_results)}")

    # Collect all document IDs involved in cycles
    cycle_docs = {}
    for r in all_results:
        for doc_key in ['start_doc', 'middle_doc', 'end_doc']:
            doc_id = r[doc_key]
            if doc_id not in cycle_docs:
                cycle_docs[doc_id] = {
                    'cycle_count': 0,
                    'min_cycle_duration': float('inf'),
                    'max_cycle_amount': 0,
                }
            cycle_docs[doc_id]['cycle_count'] += 1
            cycle_docs[doc_id]['min_cycle_duration'] = min(
                cycle_docs[doc_id]['min_cycle_duration'],
                r['cycle_duration_seconds']
            )
            cycle_docs[doc_id]['max_cycle_amount'] = max(
                cycle_docs[doc_id]['max_cycle_amount'],
                r['cycle_total_amount'] or 0
            )

    if all_results:
        print(f"    Unique documents in cycles: {len(cycle_docs)}")
        print(f"    Sample: {all_results[0]}")

    return cycle_docs


# -----------------------------------------------------------------
# DETECTOR 2: LAPPING INDICATORS
# -----------------------------------------------------------------
def detector_lapping(driver):
    """
    WHAT IT FINDS:
    Sequential documents to the same vendor where amounts are
    suspiciously similar — a hallmark of lapping fraud.

    THE PATTERN (Lapping):
      Customer A owes $5,000 → Employee steals the payment
      Customer B pays $5,000 → Employee uses B's payment to cover A's
      Customer C pays $5,000 → Employee uses C's payment to cover B's
    
    In the graph, this shows up as rapid, same-amount transactions
    to the same vendor.

    WHAT WE ACTUALLY DETECT:
    Documents involving the same vendor, close in time, with similar
    amounts. We compute a "lapping score" based on amount similarity
    and temporal proximity.
    """
    print("\n  🔍 Detector 2: Lapping Indicators...")

    query = """
    // Find pairs of documents involving the same vendor within
    // a time window, with similar amounts (within 20% of each other).

    MATCH (d1:Document)-[r1:INVOICED_BY|RECEIVED_FROM]->(v:Vendor)
    MATCH (d2:Document)-[r2:INVOICED_BY|RECEIVED_FROM]->(v)
    WHERE d1 <> d2
      AND d1.source_file = d2.source_file
      AND d1.timestamp < d2.timestamp
      AND duration.between(d1.timestamp, d2.timestamp).seconds <= $time_window
      AND d1.total_amount > 0 AND d2.total_amount > 0
    WITH d1, d2, v,
         abs(d1.total_amount - d2.total_amount) / 
         CASE WHEN (d1.total_amount + d2.total_amount) / 2 = 0 THEN 1 
              ELSE (d1.total_amount + d2.total_amount) / 2 END AS amount_diff_ratio,
         duration.between(d1.timestamp, d2.timestamp).seconds AS time_gap
    WHERE amount_diff_ratio < 0.2  // Amounts within 20% of each other
    RETURN d1.doc_id AS doc1,
           d2.doc_id AS doc2,
           v.vendor_id AS vendor,
           d1.total_amount AS amount1,
           d2.total_amount AS amount2,
           amount_diff_ratio,
           time_gap
    ORDER BY time_gap ASC
    LIMIT 500
    """

    results = run_query(driver, query, {"time_window": TIME_WINDOW_SECONDS})
    print(f"    Found {len(results)} lapping indicator pairs")

    lapping_docs = {}
    for r in results:
        for doc_key in ['doc1', 'doc2']:
            doc_id = r[doc_key]
            if doc_id not in lapping_docs:
                lapping_docs[doc_id] = {
                    'lapping_pair_count': 0,
                    'min_amount_diff_ratio': 1.0,
                    'min_time_gap': float('inf'),
                }
            lapping_docs[doc_id]['lapping_pair_count'] += 1
            lapping_docs[doc_id]['min_amount_diff_ratio'] = min(
                lapping_docs[doc_id]['min_amount_diff_ratio'],
                r['amount_diff_ratio']
            )
            lapping_docs[doc_id]['min_time_gap'] = min(
                lapping_docs[doc_id]['min_time_gap'],
                r['time_gap']
            )

    if results:
        print(f"    Unique documents in lapping patterns: {len(lapping_docs)}")
        print(f"    Sample: {results[0]}")

    return lapping_docs


# -----------------------------------------------------------------
# DETECTOR 3: ABNORMAL TRANSACTION DENSITY
# -----------------------------------------------------------------
def detector_density(driver):
    """
    WHAT IT FINDS:
    Documents in time windows with unusually high transaction volume.

    THE PATTERN:
    Normal: ~5-10 documents per minute
    Suspicious: 50+ documents in 60 seconds involving the same entity

    WHY THIS MATTERS:
    Automated fraud scripts or rapid manual manipulation creates
    bursts of activity. Our project description mentions:
    "Vendor_A receives 200 payments within 60 seconds — potential
    automated abuse."

    WHAT WE COMPUTE:
    For each document, count how many other documents exist within
    1-minute and 5-minute windows. High counts = high density score.
    """
    print("\n  🔍 Detector 3: Abnormal Transaction Density...")

    # OPTIMIZED: Instead of expensive variable-length path traversal,
    # we use a simple approach: for each document, count how many
    # TEMPORAL_NEXT edges with small time_delta_seconds exist in its
    # immediate forward chain. We process per source_file.
    #
    # WHY THIS WORKS: If a doc has 15 neighbors all within 60 seconds,
    # then the first 15 TEMPORAL_NEXT edges from that doc all have
    # cumulative time_delta < 60s. We accumulate time_delta along the chain.

    query_density = """
    // For each document, walk forward through TEMPORAL_NEXT and
    // count how many docs are reachable within the time window.
    // We use a fixed hop limit of 20 to bound memory.
    MATCH (d:Document)
    WHERE d.source_file = $source_file
    OPTIONAL MATCH path = (d)-[:TEMPORAL_NEXT*1..20]->(d_next:Document)
    WHERE d_next.source_file = $source_file
      AND duration.between(d.timestamp, d_next.timestamp).seconds <= $window
    WITH d.doc_id AS doc_id, count(DISTINCT d_next) AS forward_neighbors
    RETURN doc_id, forward_neighbors
    """

    sources = run_query(driver, "MATCH (d:Document) RETURN DISTINCT d.source_file AS sf")
    source_files = [s['sf'] for s in sources]

    density_docs = {}

    # Pass 1: 1-minute window
    print("    Computing 1-minute density...")
    for sf in source_files:
        print(f"      {sf}...", end=" ")
        results = run_query(driver, query_density, {"source_file": sf, "window": 60})
        print(f"{len(results)} docs")
        for r in results:
            density_docs[r['doc_id']] = {
                'density_1min': r['forward_neighbors'],
                'density_5min': 0,
            }

    # Pass 2: 5-minute window
    print("    Computing 5-minute density...")
    for sf in source_files:
        print(f"      {sf}...", end=" ")
        results = run_query(driver, query_density, {"source_file": sf, "window": 300})
        print(f"{len(results)} docs")
        for r in results:
            if r['doc_id'] in density_docs:
                density_docs[r['doc_id']]['density_5min'] = r['forward_neighbors']
            else:
                density_docs[r['doc_id']] = {
                    'density_1min': 0,
                    'density_5min': r['forward_neighbors'],
                }

    # Find high-density documents
    high_density = {k: v for k, v in density_docs.items() if v['density_1min'] >= 15}
    print(f"    High-density documents (>=15 in 1 min): {len(high_density)}")

    return density_docs


# -----------------------------------------------------------------
# DETECTOR 4: RAPID TRANSFER CHAINS
# -----------------------------------------------------------------
def detector_rapid_chains(driver):
    """
    WHAT IT FINDS:
    Long chains of documents executed in very rapid succession.

    THE PATTERN:
      Doc_1 (T+0s) → Doc_2 (T+1s) → Doc_3 (T+2s) → Doc_4 (T+3s) → Doc_5 (T+4s)
    Five documents in 4 seconds = potential layering activity.

    WHY THIS MATTERS:
    Money laundering "layering" involves rapidly moving funds through
    multiple accounts/entities to obscure the trail. In the graph,
    this appears as long chains where each TEMPORAL_NEXT edge has
    a very small time_delta_seconds.

    WHAT WE COMPUTE:
    For each document, find the longest rapid chain it participates in
    (where every step is < 5 seconds apart).
    """
    print("\n  🔍 Detector 4: Rapid Transfer Chains...")

    # OPTIMIZED: Process per source_file and limit chain depth to 8
    # to avoid memory explosion from combinatorial path expansion.

    query = """
    // Find chains of 3+ documents where each step is < 5 seconds apart
    MATCH path = (d_start:Document)-[:TEMPORAL_NEXT*3..8]->(d_end:Document)
    WHERE ALL(r IN relationships(path) WHERE r.time_delta_seconds < 5)
      AND d_start.source_file = $source_file
    WITH d_start, d_end, length(path) AS chain_length,
         duration.between(d_start.timestamp, d_end.timestamp).seconds AS total_seconds,
         [n IN nodes(path) | n.doc_id] AS chain_docs
    RETURN chain_docs,
           chain_length,
           total_seconds,
           d_start.source_file AS source_file
    ORDER BY chain_length DESC
    LIMIT 100
    """

    sources = run_query(driver, "MATCH (d:Document) RETURN DISTINCT d.source_file AS sf")
    source_files = [s['sf'] for s in sources]

    results = []
    for sf in source_files:
        print(f"    Scanning {sf}...", end=" ")
        sf_results = run_query(driver, query, {"source_file": sf})
        print(f"{len(sf_results)} chains")
        results.extend(sf_results)

    print(f"    Total rapid chains found: {len(results)}")

    chain_docs = {}
    for r in results:
        for doc_id in r['chain_docs']:
            if doc_id not in chain_docs:
                chain_docs[doc_id] = {
                    'max_chain_length': 0,
                    'chain_count': 0,
                    'min_chain_seconds': float('inf'),
                }
            chain_docs[doc_id]['max_chain_length'] = max(
                chain_docs[doc_id]['max_chain_length'],
                r['chain_length']
            )
            chain_docs[doc_id]['chain_count'] += 1
            chain_docs[doc_id]['min_chain_seconds'] = min(
                chain_docs[doc_id]['min_chain_seconds'],
                r['total_seconds']
            )

    if results:
        print(f"    Unique documents in rapid chains: {len(chain_docs)}")
        print(f"    Longest chain: {results[0]['chain_length']} steps in {results[0]['total_seconds']}s")

    return chain_docs


# -----------------------------------------------------------------
# DETECTOR 5: AMOUNT ANOMALIES (per vendor/account context)
# -----------------------------------------------------------------
def detector_amount_anomalies(driver):
    """
    WHAT IT FINDS:
    Documents with amounts that are statistical outliers compared
    to their vendor's typical transaction amounts.

    THE PATTERN:
    Vendor_A's normal invoices: $5,000 - $15,000
    Suspicious invoice: $64,000,000  ← Invoice Kickback!

    WHY THIS MATTERS:
    Invoice kickback fraud involves inflating invoice amounts.
    Our Step 1 analysis already showed that fraud_1_5000000038
    (Invoice Kickback) had an amount of $63.9M — wildly above normal.

    WHAT WE COMPUTE:
    Z-score of each document's amount relative to:
    1. All documents in the same source file
    2. Documents involving the same vendor
    """
    print("\n  🔍 Detector 5: Amount Anomalies...")

    # Compute global amount statistics per source file
    query_global = """
    MATCH (d:Document)
    WHERE d.total_amount IS NOT NULL
    WITH d.source_file AS source_file,
         avg(d.total_amount) AS mean_amount,
         stDev(d.total_amount) AS std_amount
    RETURN source_file, mean_amount, std_amount
    """
    global_stats = run_query(driver, query_global)
    print(f"    Global stats computed for {len(global_stats)} source files")

    # Build stats lookup
    stats_map = {}
    for s in global_stats:
        stats_map[s['source_file']] = {
            'mean': s['mean_amount'],
            'std': s['std_amount'] if s['std_amount'] and s['std_amount'] > 0 else 1
        }

    # Compute per-vendor stats
    query_vendor = """
    MATCH (d:Document)-[:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(v:Vendor)
    WHERE d.total_amount IS NOT NULL
    WITH v.vendor_id AS vendor_id, d.source_file AS source_file,
         avg(d.total_amount) AS vendor_mean,
         stDev(d.total_amount) AS vendor_std,
         count(d) AS vendor_doc_count
    RETURN vendor_id, source_file, vendor_mean, vendor_std, vendor_doc_count
    """
    vendor_stats = run_query(driver, query_vendor)
    print(f"    Vendor stats computed for {len(vendor_stats)} vendor-source combos")

    vendor_stats_map = {}
    for vs in vendor_stats:
        key = f"{vs['source_file']}_{vs['vendor_id']}"
        vendor_stats_map[key] = {
            'mean': vs['vendor_mean'],
            'std': vs['vendor_std'] if vs['vendor_std'] and vs['vendor_std'] > 0 else 1,
            'count': vs['vendor_doc_count']
        }

    # Now compute z-scores for each document
    query_docs = """
    MATCH (d:Document)
    WHERE d.total_amount IS NOT NULL
    OPTIONAL MATCH (d)-[:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(v:Vendor)
    RETURN d.doc_id AS doc_id,
           d.total_amount AS amount,
           d.source_file AS source_file,
           v.vendor_id AS vendor_id
    """
    all_docs = run_query(driver, query_docs)
    print(f"    Computing z-scores for {len(all_docs)} documents...")

    anomaly_docs = {}
    for doc in all_docs:
        doc_id = doc['doc_id']
        amount = doc['amount']
        source = doc['source_file']

        # Global z-score
        if source in stats_map:
            global_zscore = (amount - stats_map[source]['mean']) / stats_map[source]['std']
        else:
            global_zscore = 0

        # Vendor z-score (if vendor exists)
        vendor_zscore = 0
        vendor_doc_count = 0
        if doc['vendor_id']:
            vkey = f"{source}_{doc['vendor_id']}"
            if vkey in vendor_stats_map:
                vs = vendor_stats_map[vkey]
                vendor_zscore = (amount - vs['mean']) / vs['std']
                vendor_doc_count = vs['count']

        anomaly_docs[doc_id] = {
            'amount_global_zscore': round(global_zscore, 4),
            'amount_vendor_zscore': round(vendor_zscore, 4),
            'vendor_doc_count': vendor_doc_count,
        }

    # Count extreme anomalies
    extreme = sum(1 for v in anomaly_docs.values() if abs(v['amount_global_zscore']) > 3)
    print(f"    Extreme anomalies (|z| > 3): {extreme}")

    return anomaly_docs


# ===================================================================
# PHASE 2: GRAPH-STRUCTURAL FEATURE EXTRACTION
# ===================================================================

def extract_structural_features(driver):
    """
    Extract basic graph-structural features for each document.
    
    These features describe the document's "position" in the graph:
    - How many entities it connects to (degree)
    - What types of connections it has
    - Time to neighboring documents
    
    WHY THESE FEATURES MATTER:
    A normal document might have 2-3 edges (post to account, receive material).
    A fraudulent document might have unusual connectivity — e.g., posting to
    accounts it shouldn't, or connecting to a vendor that rarely appears.
    """
    print("\n  📊 Extracting structural features...")

    query = """
    MATCH (d:Document)
    
    // Count edges by type
    OPTIONAL MATCH (d)-[rv:INVOICED_BY|CREDITED_BY|RECEIVED_FROM|POSTED_VIA]->(:Vendor)
    WITH d, count(rv) AS vendor_edges
    
    OPTIONAL MATCH (d)-[ra:POSTED_TO]->(:Account)
    WITH d, vendor_edges, count(ra) AS account_edges
    
    OPTIONAL MATCH (d)-[rm:MATERIAL_RECEIVED|MATERIAL_WITHDRAWN|INVOLVES_MATERIAL]->(:Material)
    WITH d, vendor_edges, account_edges, count(rm) AS material_edges
    
    // Get temporal neighbors
    OPTIONAL MATCH (d_prev:Document)-[t_prev:TEMPORAL_NEXT]->(d)
    OPTIONAL MATCH (d)-[t_next:TEMPORAL_NEXT]->(d_next:Document)
    
    RETURN d.doc_id AS doc_id,
           d.doc_number AS doc_number,
           d.total_amount AS total_amount,
           d.line_count AS line_count,
           d.timestamp AS timestamp,
           d.transaction_type AS transaction_type,
           d.label AS label,
           d.is_fraud AS is_fraud,
           d.source_file AS source_file,
           d.dataset_type AS dataset_type,
           vendor_edges,
           account_edges,
           material_edges,
           vendor_edges + account_edges + material_edges AS total_degree,
           t_prev.time_delta_seconds AS time_from_prev,
           t_next.time_delta_seconds AS time_to_next
    """

    results = run_query(driver, query)
    print(f"    Extracted features for {len(results)} documents")
    return results


# ===================================================================
# PHASE 3: COMBINE ALL FEATURES INTO FINAL DATASET
# ===================================================================

def build_feature_dataset(structural_features, cycle_data, lapping_data,
                          density_data, chain_data, anomaly_data):
    """
    Merge all detector outputs and structural features into one DataFrame.
    
    Each row = one Document
    Each column = one feature
    
    This is the dataset that feeds into Step 5 (AI Model Training).
    The model will learn: "Given these structural features, what's the
    probability this document is fraudulent?"
    """
    print("\n" + "=" * 70)
    print("PHASE 3: BUILDING FINAL FEATURE DATASET")
    print("=" * 70)

    # Start with structural features as the base
    df = pd.DataFrame(structural_features)

    # --- Add Detector 1: Circular Payment features ---
    df['cycle_count'] = df['doc_id'].map(
        lambda x: cycle_data.get(x, {}).get('cycle_count', 0)
    )
    df['min_cycle_duration'] = df['doc_id'].map(
        lambda x: cycle_data.get(x, {}).get('min_cycle_duration', 0)
    )
    df['max_cycle_amount'] = df['doc_id'].map(
        lambda x: cycle_data.get(x, {}).get('max_cycle_amount', 0)
    )
    # Replace inf with 0
    df['min_cycle_duration'] = df['min_cycle_duration'].replace([float('inf')], 0)

    # Binary: is this document part of ANY cycle?
    df['in_cycle'] = (df['cycle_count'] > 0).astype(int)

    # --- Add Detector 2: Lapping features ---
    df['lapping_pair_count'] = df['doc_id'].map(
        lambda x: lapping_data.get(x, {}).get('lapping_pair_count', 0)
    )
    df['lapping_min_diff_ratio'] = df['doc_id'].map(
        lambda x: lapping_data.get(x, {}).get('min_amount_diff_ratio', 1.0)
    )
    df['lapping_min_time_gap'] = df['doc_id'].map(
        lambda x: lapping_data.get(x, {}).get('min_time_gap', 0)
    )
    df['lapping_min_time_gap'] = df['lapping_min_time_gap'].replace([float('inf')], 0)

    # Composite lapping score (0 to 1, higher = more suspicious)
    # WHY THIS FORMULA: More pairs + smaller amount differences + shorter gaps = higher score
    df['lapping_score'] = np.where(
        df['lapping_pair_count'] > 0,
        np.clip(
            (df['lapping_pair_count'] / df['lapping_pair_count'].max()) *
            (1 - df['lapping_min_diff_ratio']) *
            np.clip(1 - (df['lapping_min_time_gap'] / TIME_WINDOW_SECONDS), 0, 1),
            0, 1
        ),
        0
    )

    # --- Add Detector 3: Density features ---
    df['density_1min'] = df['doc_id'].map(
        lambda x: density_data.get(x, {}).get('density_1min', 0)
    )
    df['density_5min'] = df['doc_id'].map(
        lambda x: density_data.get(x, {}).get('density_5min', 0)
    )

    # Normalized density score (relative to max in dataset)
    max_1min = df['density_1min'].max() if df['density_1min'].max() > 0 else 1
    max_5min = df['density_5min'].max() if df['density_5min'].max() > 0 else 1
    df['density_score'] = (
        (df['density_1min'] / max_1min) * 0.6 +
        (df['density_5min'] / max_5min) * 0.4
    ).round(4)

    # --- Add Detector 4: Rapid Chain features ---
    df['max_chain_length'] = df['doc_id'].map(
        lambda x: chain_data.get(x, {}).get('max_chain_length', 0)
    )
    df['chain_count'] = df['doc_id'].map(
        lambda x: chain_data.get(x, {}).get('chain_count', 0)
    )
    df['min_chain_seconds'] = df['doc_id'].map(
        lambda x: chain_data.get(x, {}).get('min_chain_seconds', 0)
    )
    df['min_chain_seconds'] = df['min_chain_seconds'].replace([float('inf')], 0)

    # --- Add Detector 5: Amount Anomaly features ---
    df['amount_global_zscore'] = df['doc_id'].map(
        lambda x: anomaly_data.get(x, {}).get('amount_global_zscore', 0)
    )
    df['amount_vendor_zscore'] = df['doc_id'].map(
        lambda x: anomaly_data.get(x, {}).get('amount_vendor_zscore', 0)
    )
    df['vendor_doc_count'] = df['doc_id'].map(
        lambda x: anomaly_data.get(x, {}).get('vendor_doc_count', 0)
    )

    # Absolute z-scores (magnitude of anomaly regardless of direction)
    df['amount_anomaly_score'] = np.abs(df['amount_global_zscore']).round(4)

    # --- Fill NaN values ---
    df['time_from_prev'] = df['time_from_prev'].fillna(0)
    df['time_to_next'] = df['time_to_next'].fillna(0)
    df = df.fillna(0)

    # --- Convert is_fraud to proper binary ---
    df['is_fraud'] = df['is_fraud'].map(
        lambda x: 1 if x in [True, 'true', 'True', 1] else 0
    )

    print(f"\n  📋 Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Feature columns ({df.shape[1] - 4} features + 4 metadata):")

    # Categorize columns for display
    metadata_cols = ['doc_id', 'doc_number', 'source_file', 'dataset_type']
    target_cols = ['label', 'is_fraud']
    feature_cols = [c for c in df.columns if c not in metadata_cols + target_cols + ['timestamp', 'transaction_type']]

    print(f"\n    METADATA:  {metadata_cols}")
    print(f"    TARGET:    {target_cols}")
    print(f"    FEATURES:  {feature_cols}")

    # --- Summary of fraud vs non-fraud ---
    print(f"\n  📊 CLASS DISTRIBUTION:")
    for label_val, count in df['is_fraud'].value_counts().items():
        pct = count / len(df) * 100
        tag = "FRAUD" if label_val == 1 else "CLEAN"
        print(f"    {tag}: {count:,} ({pct:.2f}%)")

    # --- Check feature distributions for fraud vs clean ---
    print(f"\n  📊 FEATURE MEANS (Fraud vs. Clean):")
    print(f"    {'Feature':<30} {'Clean':>12} {'Fraud':>12} {'Ratio':>8}")
    print(f"    {'-'*62}")
    for col in feature_cols:
        if col in ['timestamp', 'transaction_type']:
            continue
        try:
            clean_mean = df[df['is_fraud'] == 0][col].mean()
            fraud_mean = df[df['is_fraud'] == 1][col].mean()
            ratio = fraud_mean / clean_mean if clean_mean != 0 else float('inf')
            if abs(ratio) > 1.5 or abs(ratio) < 0.67:
                marker = " ← DISCRIMINATIVE"
            else:
                marker = ""
            print(f"    {col:<30} {clean_mean:>12.4f} {fraud_mean:>12.4f} {ratio:>8.2f}{marker}")
        except (TypeError, ValueError):
            pass

    return df


# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL TOPOLOGY ENGINE + FEATURE EXTRACTION")
    print("=" * 70)

    start_time = time.time()
    driver = create_driver()

    # ─── PHASE 1: RUN ALL DETECTORS ─────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: RUNNING TOPOLOGY DETECTORS")
    print("=" * 70)

    cycle_data = detector_circular_payments(driver)
    lapping_data = detector_lapping(driver)
    density_data = detector_density(driver)
    chain_data = detector_rapid_chains(driver)
    anomaly_data = detector_amount_anomalies(driver)

    # ─── PHASE 2: EXTRACT STRUCTURAL FEATURES ──────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: EXTRACTING STRUCTURAL FEATURES")
    print("=" * 70)

    structural_features = extract_structural_features(driver)

    # ─── PHASE 3: BUILD COMBINED DATASET ────────────────────────
    df = build_feature_dataset(
        structural_features, cycle_data, lapping_data,
        density_data, chain_data, anomaly_data
    )

    # ─── SAVE OUTPUT ────────────────────────────────────────────
    output_file = os.path.join(OUTPUT_DIR, 'graph_features.csv')
    df.to_csv(output_file, index=False)
    print(f"\n  ✅ Saved to: {output_file}")

    # Also save a fraud-only subset for easy inspection
    fraud_df = df[df['is_fraud'] == 1]
    fraud_file = os.path.join(OUTPUT_DIR, 'graph_features_fraud_only.csv')
    fraud_df.to_csv(fraud_file, index=False)
    print(f"  ✅ Fraud-only subset: {fraud_file} ({len(fraud_df)} rows)")

    driver.close()

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f} seconds")
    print(f"\n✅ Steps 4 & 5 complete!")
    print(f"   Feature dataset ready for AI Model Training (Step 5).")
    print(f"   File: {output_file}")
