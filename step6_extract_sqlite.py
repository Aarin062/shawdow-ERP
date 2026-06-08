"""
===========================================================================
STEP 6: DATA EXTRACTION & SYNTHETIC FRAUD INJECTION (Dataset 2)
Project: The Self-Auditing Ledger
Purpose: Extract raw SAP IDES data from SQLite, format it for Neo4j,
         and inject structural fraud patterns to create ML labels.
===========================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import random

# ===================================================================
# CONFIGURATION
# ===================================================================
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "erp_fraud_data_2", "sap.sqlite"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import_dataset_2"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAUD_RATIO = 0.005  # 0.5% fraud injection to ensure enough samples

# ===================================================================
# STEP 6A: EXTRACT FROM SQLITE
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6A: EXTRACTING SAP IDES DATA")
print("=" * 70)

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database not found at {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

# We join BKPF (Header) and BSEG (Items) on Document Keys
query = """
    SELECT 
        h.belnr AS Belegnummer,
        h.cpudt AS Date,
        h.cputm AS Time,
        i.hkont AS Sachkonto,
        i.lifnr AS Kreditor,
        i.matnr AS Material,
        i.dmbtr AS "Betrag Hauswaehr",
        i.shkzg AS "Soll/Haben-Kennz_",
        i.buzei AS Position,
        h.blart AS Transaktionsart,
        i.ktosl AS Buchungsschluessel
    FROM bkpf h
    JOIN bseg i ON h.mandt = i.mandt 
                AND h.bukrs = i.bukrs 
                AND h.belnr = i.belnr 
                AND h.gjahr = i.gjahr
    WHERE i.lifnr IS NOT NULL AND i.lifnr != ''
"""

print("  Executing SQL join across BKPF and BSEG...")
df = pd.read_sql_query(query, conn)
conn.close()

print(f"  Extracted {len(df)} line items.")

# Format Timestamp
df['Erfassungsuhrzeit'] = df['Date'] + ' ' + df['Time']
df.drop(columns=['Date', 'Time'], inplace=True)

# Fill NA
df['Material'] = df['Material'].fillna('UNKNOWN')
df['Sachkonto'] = df['Sachkonto'].fillna('UNKNOWN')

# Add target label (default to 0)
df['is_fraud'] = 0
df['Label'] = 'NonFraud'

# Assign a dummy source file
df['source_file'] = 'sap_ides_1.csv'

# Sort by timestamp
df = df.sort_values('Erfassungsuhrzeit').reset_index(drop=True)

# ===================================================================
# STEP 6B: SYNTHETIC FRAUD INJECTION (STRUCTURAL)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 6B: INJECTING STRUCTURAL FRAUD PATTERNS")
print("=" * 70)

# We need to inject structural fraud (Cycles and Lapping) so the 
# ML models have a target variable to learn/evaluate against.

num_fraud_docs = int(df['Belegnummer'].nunique() * FRAUD_RATIO)
print(f"  Targeting {num_fraud_docs} documents for fraud injection.")

unique_docs = df['Belegnummer'].unique()
fraud_docs_pool = np.random.choice(unique_docs, size=num_fraud_docs, replace=False)

# Split into two types of fraud
cycle_docs = fraud_docs_pool[:len(fraud_docs_pool)//2]
lapping_docs = fraud_docs_pool[len(fraud_docs_pool)//2:]

# 1. Inject Circular Payments (A -> B -> A)
print(f"  Injecting Circular Payments into {len(cycle_docs)} documents...")
vendors = df['Kreditor'].unique()

# We need sequences of 3 docs to form a cycle. We will pick consecutive docs.
for i in range(0, len(cycle_docs) - 2, 3):
    d1, d2, d3 = cycle_docs[i], cycle_docs[i+1], cycle_docs[i+2]
    
    # Pick two random vendors
    v1, v2 = np.random.choice(vendors, size=2, replace=False)
    
    # Modify the dataset
    df.loc[df['Belegnummer'] == d1, 'Kreditor'] = v1
    df.loc[df['Belegnummer'] == d2, 'Kreditor'] = v2
    df.loc[df['Belegnummer'] == d3, 'Kreditor'] = v1  # The loop!
    
    # Label them
    df.loc[df['Belegnummer'].isin([d1, d2, d3]), 'is_fraud'] = 1
    df.loc[df['Belegnummer'].isin([d1, d2, d3]), 'Label'] = 'Circular Payment'
    
    # Inflate amounts
    df.loc[df['Belegnummer'].isin([d1, d2, d3]), 'Betrag Hauswaehr'] *= np.random.uniform(10, 50)

# 2. Inject Lapping (A -> A rapidly with similar amounts)
print(f"  Injecting Lapping into {len(lapping_docs)} documents...")
for i in range(0, len(lapping_docs) - 1, 2):
    d1, d2 = lapping_docs[i], lapping_docs[i+1]
    
    v = np.random.choice(vendors)
    
    df.loc[df['Belegnummer'] == d1, 'Kreditor'] = v
    df.loc[df['Belegnummer'] == d2, 'Kreditor'] = v
    
    # Label them
    df.loc[df['Belegnummer'].isin([d1, d2]), 'is_fraud'] = 1
    df.loc[df['Belegnummer'].isin([d1, d2]), 'Label'] = 'Lapping'
    
    # Make amounts similar
    base_amount = df.loc[df['Belegnummer'] == d1, 'Betrag Hauswaehr'].values[0]
    df.loc[df['Belegnummer'] == d2, 'Betrag Hauswaehr'] = base_amount * np.random.uniform(0.95, 1.05)


actual_fraud = df[df['is_fraud'] == 1]['Belegnummer'].nunique()
print(f"\n  Fraud Injection Complete.")
print(f"  Total unique documents: {df['Belegnummer'].nunique()}")
print(f"  Total fraudulent documents: {actual_fraud} ({(actual_fraud/df['Belegnummer'].nunique())*100:.2f}%)")

# ===================================================================
# STEP 6C: SAVE TO CSV FOR NEO4J
# ===================================================================
output_file = os.path.join(OUTPUT_DIR, 'sap_ides_1.csv')
df.to_csv(output_file, index=False)
print(f"\n  Saved extracted data to {output_file}")

# Save expls file for reference
expls_file = os.path.join(OUTPUT_DIR, 'sap_ides_1_expls.csv')
fraud_df = df[df['is_fraud'] == 1].copy()
fraud_df.to_csv(expls_file, index=False)
print(f"  Saved fraud ground-truth to {expls_file}")

print("\n  NEXT STEP: Update step3 and step4 to point to `data/neo4j_import_dataset_2`")
