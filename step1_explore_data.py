"""
===========================================================================
STEP 1: DATA EXPLORATION SCRIPT
Project: The Self-Auditing Ledger
Purpose: Explore the Würzburg DeepScan ERP Fraud dataset to understand
         its structure before we build the Shadow Graph.
===========================================================================

WHY THIS SCRIPT EXISTS:
-----------------------
Before we can stream ERP data into Neo4j as a graph, we need to answer:
  1. What files are in the dataset?
  2. What columns does each file have?
  3. Which columns map to NODES (accounts, vendors, etc.)?
  4. Which columns map to EDGES (transactions, payments)?
  5. Which columns provide PROPERTIES (timestamps, amounts)?

We'll explore the "joint_datasets" folder because those are the
pre-processed, ready-to-use CSV files (as stated in the dataset README).
The "raw_data" folder contains zipped SAP table dumps that we don't need
right now — the joint datasets already combine the relevant SAP tables
(RBKP, RSEG, BKPF, BSEG).

WHAT ARE RBKP, RSEG, BKPF, BSEG?
-----------------------------------
These are standard SAP ERP tables:
  - BKPF = Accounting Document Header (one row per document)
  - BSEG = Accounting Document Line Items (individual postings)
  - RBKP = Invoice Document Header
  - RSEG = Invoice Document Line Items
The joint datasets merge these into a single flat file per simulation run.
"""

import pandas as pd
import os
import sys

# Fix Windows console encoding (PowerShell defaults to cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# ===================================================================
# CONFIGURATION
# ===================================================================
# Path to the joint_datasets folder (pre-joined SAP tables)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "erp_fraud_data (1)", "erp_fraud_data", "joint_datasets"
)

# The column_information.csv maps German SAP column names to English
COL_INFO_FILE = os.path.join(DATA_DIR, "column_information.csv")

# ===================================================================
# STEP 1A: Load the column information (German -> English mapping)
# ===================================================================
print("=" * 70)
print("STEP 1A: COLUMN NAME MAPPING (German → English)")
print("=" * 70)

# This CSV has a special structure:
#   Row 0 = column indices (0-57)
#   Row 1 = German names
#   Row 2 = English names
#   Row 3 = categorical indicator (1.0 = categorical, -1.0 = metadata)
#   Row 4 = numerical indicator (1.0 = numerical, -1.0 = metadata)

col_info = pd.read_csv(COL_INFO_FILE, header=None)

# Extract German and English column names
german_names = col_info.iloc[1, :].tolist()    # Row index 1
english_names = col_info.iloc[2, :].tolist()   # Row index 2
col_types = col_info.iloc[3, :].tolist()       # Row index 3 (cat indicator)

print(f"\nTotal columns in dataset: {len(german_names)}")
print(f"\n{'Index':<6} {'German Name':<35} {'English Name':<35} {'Type'}")
print("-" * 110)
for i, (ger, eng, ct) in enumerate(zip(german_names, english_names, col_types)):
    # Determine type label
    if ct == -1.0 or str(ct) == '-1.0':
        type_label = "METADATA"
    elif ct == 1.0 or str(ct) == '1.0':
        type_label = "Categorical"
    else:
        type_label = "Numerical"
    print(f"{i:<6} {str(ger):<35} {str(eng):<35} {type_label}")


# ===================================================================
# STEP 1B: List all CSV data files
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1B: AVAILABLE DATA FILES")
print("=" * 70)

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f != 'column_information.csv']
csv_files.sort()

for f in csv_files:
    filepath = os.path.join(DATA_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  📄 {f:<25} ({size_mb:.2f} MB)")


# ===================================================================
# STEP 1C: Load and explore each main dataset (fraud + normal)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1C: EXPLORING EACH DATASET")
print("=" * 70)

# We only look at the main data files (not _expls files)
# _expls files contain which columns were manipulated for fraud injection
main_files = [f for f in csv_files if '_expls' not in f]

for filename in main_files:
    filepath = os.path.join(DATA_DIR, filename)
    
    print(f"\n{'─' * 70}")
    print(f"📊 FILE: {filename}")
    print(f"{'─' * 70}")
    
    # Load the CSV
    # The dataset uses German column names as headers
    df = pd.read_csv(filepath)
    
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # --- Show the first 5 rows of KEY columns ---
    # We pick the columns most relevant to our graph:
    key_columns_ger = [
        'Kreditor',          # Vendor (NODE)
        'Sachkonto',         # G/L Account (NODE)
        'Buchungsschluessel', # Posting Key (EDGE type indicator)
        'Soll/Haben-Kennz_', # Debit/Credit indicator
        'Betrag Hauswaehr',  # Amount in local currency (PROPERTY)
        'Betrag',            # Amount (PROPERTY)
        'Belegnummer',       # Document Number (unique transaction ID)
        'Position',          # Line item position within document
        'Transaktionsart',   # Transaction type (Invoice, Credit, etc.)
        'Erfassungsuhrzeit', # Recording time (TIMESTAMP!)
        'Label',             # Fraud label (ground truth)
    ]
    
    # Filter to columns that actually exist in this file
    available_keys = [c for c in key_columns_ger if c in df.columns]
    
    print(f"\n  🔑 KEY COLUMNS (first 5 rows):")
    print(df[available_keys].head(5).to_string(index=False))
    
    # --- Fraud label distribution ---
    if 'Label' in df.columns:
        print(f"\n  🏷️  FRAUD LABEL DISTRIBUTION:")
        label_counts = df['Label'].value_counts()
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"     {label:<30} {count:>6} ({pct:.2f}%)")
    
    # --- Transaction type distribution ---
    if 'Transaktionsart' in df.columns:
        print(f"\n  📋 TRANSACTION TYPE DISTRIBUTION:")
        txn_counts = df['Transaktionsart'].value_counts()
        for txn, count in txn_counts.items():
            pct = (count / len(df)) * 100
            print(f"     {txn:<30} {count:>6} ({pct:.2f}%)")
    
    # --- Vendor (Kreditor) statistics ---
    if 'Kreditor' in df.columns:
        unique_vendors = df['Kreditor'].nunique()
        print(f"\n  👤 UNIQUE VENDORS (Kreditor): {unique_vendors}")
        print(f"     Top 5 vendors by frequency:")
        top_vendors = df['Kreditor'].value_counts().head(5)
        for vendor, count in top_vendors.items():
            print(f"       {vendor}: {count} transactions")
    
    # --- Amount statistics ---
    if 'Betrag Hauswaehr' in df.columns:
        amounts = pd.to_numeric(df['Betrag Hauswaehr'], errors='coerce')
        print(f"\n  💰 AMOUNT STATISTICS (Betrag Hauswaehr / Amount in Local Currency):")
        print(f"     Min:    {amounts.min():>15,.2f}")
        print(f"     Max:    {amounts.max():>15,.2f}")
        print(f"     Mean:   {amounts.mean():>15,.2f}")
        print(f"     Median: {amounts.median():>15,.2f}")
    
    # --- Timestamp info ---
    if 'Erfassungsuhrzeit' in df.columns:
        print(f"\n  ⏰ TIMESTAMP COLUMN (Erfassungsuhrzeit / Recording Time):")
        print(f"     Sample values: {df['Erfassungsuhrzeit'].dropna().head(5).tolist()}")
        print(f"     Unique timestamps: {df['Erfassungsuhrzeit'].nunique()}")
        print(f"     Null count: {df['Erfassungsuhrzeit'].isnull().sum()}")

    # --- G/L Account info ---
    if 'Sachkonto' in df.columns:
        print(f"\n  📒 UNIQUE G/L ACCOUNTS (Sachkonto): {df['Sachkonto'].nunique()}")

    # --- Document Number info ---
    if 'Belegnummer' in df.columns:
        print(f"\n  📄 UNIQUE DOCUMENTS (Belegnummer): {df['Belegnummer'].nunique()}")


# ===================================================================
# STEP 1D: Examine the fraud explanation files
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1D: FRAUD EXPLANATION FILES")
print("=" * 70)
print("""
These '_expls' files tell us WHICH columns were modified to inject fraud.
An 'X' in a column means that column was altered for the fraud scenario.
This is important for understanding the fraud injection methodology.
""")

expls_files = [f for f in csv_files if '_expls' in f]
for filename in expls_files:
    filepath = os.path.join(DATA_DIR, filename)
    df_expls = pd.read_csv(filepath)
    
    print(f"\n  📋 {filename}:")
    print(f"     Fraud scenarios: {len(df_expls)} manipulated line items")
    if 'Label' in df_expls.columns:
        print(f"     Fraud types present:")
        for label in df_expls['Label'].unique():
            count = (df_expls['Label'] == label).sum()
            print(f"       - {label}: {count} line items")


# ===================================================================
# STEP 1E: Check the fraud labels master file
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1E: MASTER FRAUD LABELS FILE")
print("=" * 70)

labels_file = os.path.join(
    os.path.dirname(DATA_DIR), "fraud_labels_all_data.xlsx"
)
if os.path.exists(labels_file):
    df_labels = pd.read_excel(labels_file)
    print(f"  Shape: {df_labels.shape}")
    print(f"  Columns: {df_labels.columns.tolist()}")
    print(f"\n  First 10 rows:")
    print(df_labels.head(10).to_string())
else:
    print("  ⚠️ fraud_labels_all_data.xlsx not found!")


# ===================================================================
# SUMMARY: GRAPH MAPPING PREVIEW
# ===================================================================
print("\n" + "=" * 70)
print("STEP 1F: GRAPH MAPPING PREVIEW (What goes into Neo4j)")
print("=" * 70)
print("""
Based on this exploration, here's our preliminary Neo4j mapping:

  ┌──────────────────────────────────────────────────────────────┐
  │                     NODES (Entities)                         │
  ├──────────────────────────────────────────────────────────────┤
  │  • Vendor    ← 'Kreditor' column                            │
  │  • G/L Account ← 'Sachkonto' column                        │
  │  • Document  ← 'Belegnummer' column (transaction ID)        │
  │  • Cost Centre ← 'Kostenstelle' column                     │
  ├──────────────────────────────────────────────────────────────┤
  │                   EDGES (Relationships)                      │
  ├──────────────────────────────────────────────────────────────┤
  │  • POSTED_TO (Document → G/L Account)                       │
  │  • PAID_BY   (Document → Vendor)                            │
  │  • TRANSFERRED_TO (Vendor → Vendor, via document chain)     │
  ├──────────────────────────────────────────────────────────────┤
  │                 PROPERTIES (on Edges)                        │
  ├──────────────────────────────────────────────────────────────┤
  │  • amount      ← 'Betrag Hauswaehr'                         │
  │  • timestamp   ← 'Erfassungsuhrzeit'                        │
  │  • debit_credit ← 'Soll/Haben-Kennz_'                      │
  │  • posting_key  ← 'Buchungsschluessel'                      │
  │  • label        ← 'Label' (fraud ground truth)              │
  └──────────────────────────────────────────────────────────────┘

  This mapping will be refined in Step 2 after we see the actual data.
""")

print("✅ Exploration complete! Review the output above.")
