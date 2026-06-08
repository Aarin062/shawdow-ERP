"""
===========================================================================
STEP 2: DATA CLEANING & NEO4J FORMATTING PIPELINE
Project: The Self-Auditing Ledger
Purpose: Transform raw Würzburg ERP CSVs into Neo4j-ready import files.
===========================================================================

WHAT THIS SCRIPT DOES (Big Picture):
--------------------------------------
Neo4j doesn't read SAP tables directly. It needs data in a specific format:
  - NODE files:  Each row = one entity (Vendor, Document, Account, etc.)
  - EDGE files:  Each row = one relationship between two entities

Think of it like this:
  Raw ERP Data (messy German SAP table rows)
        ↓  [This Script]
  Clean Neo4j Files (separate nodes + edges CSVs)
        ↓  [Step 3]
  Neo4j Graph Database (visual network of transactions)

WHY WE STRUCTURE THE GRAPH THIS WAY:
--------------------------------------
Our fraud detectors need to traverse paths like:
  Vendor_A → Document_1 → Account_X → Document_2 → Vendor_B → ...

If Vendor_B then connects back to Vendor_A through more documents,
that's a CIRCULAR PAYMENT pattern — our Detector #1.

The Document node is the "hub" that connects all other entities.
Every financial event in SAP creates a Document, and that Document
touches Vendors, Accounts, and Materials. By making Documents the
center of our graph, we can trace money flow through the entire system.

GRAPH MODEL:
--------------------------------------

    [Vendor:V01]                    [Account:800000]
         ↑                              ↑
    INVOICED_BY                     POSTED_TO
         ↑                              ↑
    [Document:5000036] ──TEMPORAL_NEXT──→ [Document:5000038]
         ↓                              ↓
    INVOLVES_MATERIAL               POSTED_TO
         ↓                              ↓
    [Material:FERT01]              [Account:191100]

Each edge carries: amount, timestamp, debit/credit indicator
"""

import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ===================================================================
# CONFIGURATION
# ===================================================================
# Where the raw CSVs live
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import_dataset_2"
)

# Where we'll write the Neo4j-ready files
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import_dataset_2"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# COLUMN RENAMING MAP
# ===================================================================
# WHY: The dataset uses German SAP column names. We rename them to
# English for readability. Only the 11 columns we need are included.
# Everything else gets dropped — we don't need 57 columns in Neo4j.

COLUMN_RENAME_MAP = {
    # --- NODE source columns ---
    'Kreditor':             'vendor_id',        # Vendor node identifier
    'Sachkonto':            'gl_account_id',    # General Ledger Account node
    'Belegnummer':          'doc_number',        # Document node identifier
    'Material':             'material_id',       # Material node identifier
    'Kostenstelle':         'cost_centre',       # Cost Centre (optional node)
    'Hauptbuchkonto':       'gl_account_alt',    # Alternative G/L account field

    # --- EDGE type indicators ---
    'Buchungsschluessel':   'posting_key',       # SAP posting key (defines transaction type)
    'Soll/Haben-Kennz_':    'debit_credit',      # S = Debit, H = Credit
    'Transaktionsart':      'transaction_type',   # Human-readable type (Invoice, Credit, etc.)

    # --- PROPERTY columns ---
    'Betrag Hauswaehr':     'amount',            # Amount in local currency
    'Betrag':               'amount_alt',        # Alternative amount field
    'Erfassungsuhrzeit':    'timestamp',          # Recording time (HH:MM:SS)
    'Position':             'line_position',      # Line item position within document
    'Label':                'label',              # Fraud ground truth label
}

# The columns we want to keep (in order)
KEEP_COLUMNS = list(COLUMN_RENAME_MAP.values())


# ===================================================================
# STEP 2A: LOAD AND MERGE ALL DATASETS
# ===================================================================
print("=" * 70)
print("STEP 2A: LOADING AND MERGING ALL DATASETS")
print("=" * 70)

# Define our 5 dataset files with metadata
# WHY we track source_file and dataset_type:
#   - source_file: so we can trace any node/edge back to its origin
#   - dataset_type: 'fraud' or 'normal' — useful for model training later
DATASETS = [
    ('sap_ides_1.csv',  'ides'),
]

all_frames = []

for filename, dtype in DATASETS:
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n  Loading {filename}...", end=" ")

    # low_memory=False prevents mixed-type warnings on some columns
    df = pd.read_csv(filepath, low_memory=False)

    # Only keep columns that exist in our rename map
    existing_cols = [c for c in COLUMN_RENAME_MAP.keys() if c in df.columns]
    df = df[existing_cols].copy()

    # Rename German → English
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Add metadata columns
    df['source_file'] = filename
    df['dataset_type'] = dtype

    print(f"✓ {len(df)} rows, {len(df.columns)} columns")
    all_frames.append(df)

# Concatenate all datasets into one master DataFrame
# WHY: Having one unified DataFrame makes it easier to:
#   1. Extract all unique nodes across ALL datasets
#   2. Create consistent node IDs
#   3. Generate temporal edges across the full timeline
master_df = pd.concat(all_frames, ignore_index=True)

print(f"\n  ✅ MERGED DATASET: {len(master_df):,} total rows")
print(f"     Columns: {list(master_df.columns)}")


# ===================================================================
# STEP 2B: DATA CLEANING
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2B: CLEANING THE DATA")
print("=" * 70)

# --- 2B.1: Handle the vendor_id column ---
# WHY: ~98% of rows have NaN for vendor_id because only vendor-related
# transactions (Kreditorische Rechnung/Gutschrift) involve vendors.
# We keep NaN rows — they still create Document→Account edges.
# We just won't create vendor edges for them.
vendor_count_before = master_df['vendor_id'].notna().sum()
print(f"\n  Rows with a vendor: {vendor_count_before:,} / {len(master_df):,} "
      f"({vendor_count_before/len(master_df)*100:.2f}%)")

# --- 2B.2: Clean the amount column ---
# Convert to numeric (some might be strings due to CSV parsing)
master_df['amount'] = pd.to_numeric(master_df['amount'], errors='coerce')
null_amounts = master_df['amount'].isna().sum()
print(f"  Null amounts after conversion: {null_amounts:,}")

# --- 2B.3: Clean the timestamp column ---
# The timestamps are in HH:MM:SS format (time only, no date).
# Since each dataset is from a single simulation session, we need to
# create a synthetic full datetime to enable temporal ordering.
#
# STRATEGY: We assign each source_file a different synthetic date.
# This way, documents from fraud_1 come before fraud_2, etc.
# This is NOT the real date — it's a synthetic ordering mechanism.

print(f"\n  Parsing datetime...")
master_df['datetime'] = pd.to_datetime(
    master_df['timestamp'].astype(str),
    errors='coerce'
)

null_datetimes = master_df['datetime'].isna().sum()
print(f"  Null datetimes after conversion: {null_datetimes:,}")

# --- 2B.4: Normalize the label column ---
# Some datasets use spaces (e.g., "Invoice Kickback 1") and others use
# underscores (e.g., "Invoice_Kickback_I"). Let's standardize.
master_df['label'] = master_df['label'].fillna('NonFraud')
master_df['is_fraud'] = (master_df['label'] != 'NonFraud').astype(int)

print(f"\n  Label distribution after cleaning:")
label_counts = master_df['label'].value_counts()
for label, count in label_counts.items():
    print(f"    {label:<30} {count:>6}")

# --- 2B.5: Create a unique row ID ---
# WHY: Each row is a line item. We create a globally unique ID by combining
# source_file + doc_number + line_position. This prevents collisions
# across different datasets that reuse the same document numbers.
master_df['line_item_id'] = (
    master_df['source_file'].str.replace('.csv', '', regex=False) + '_' +
    master_df['doc_number'].astype(str) + '_' +
    master_df['line_position'].astype(str)
)

print(f"\n  Unique line item IDs: {master_df['line_item_id'].nunique():,}")
print(f"  Total rows: {len(master_df):,}")

# Verify no duplicates
dupes = master_df['line_item_id'].duplicated().sum()
if dupes > 0:
    print(f"  ⚠️ WARNING: {dupes} duplicate line_item_ids found!")
else:
    print(f"  ✅ No duplicate line_item_ids — perfect!")


# ===================================================================
# STEP 2C: GENERATE NEO4J NODE FILES
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2C: GENERATING NEO4J NODE FILES")
print("=" * 70)
print("""
  WHY SEPARATE FILES?
  Neo4j's bulk import tool (neo4j-admin import) expects separate CSV
  files for each node type and edge type. Each node file has a special
  column header format:

    nodeId:ID(NodeLabel)  — the unique identifier for this node type
    name                  — any additional properties

  This format tells Neo4j: "Create a node with label ':NodeLabel'
  and use this value as its unique ID."
""")

# --- 2C.1: VENDOR NODES ---
# Extract all unique vendor IDs across all datasets
vendors = master_df['vendor_id'].dropna().unique()
vendors_df = pd.DataFrame({
    'vendor_id:ID(Vendor)': vendors,
    'name': vendors,  # For display purposes in Neo4j Browser
})
vendors_file = os.path.join(OUTPUT_DIR, 'nodes_vendors.csv')
vendors_df.to_csv(vendors_file, index=False)
print(f"  ✅ nodes_vendors.csv: {len(vendors_df)} vendors → {vendors_df['vendor_id:ID(Vendor)'].tolist()}")


# --- 2C.2: DOCUMENT NODES ---
# Each document (Belegnummer) becomes a node. We aggregate info from
# its line items to create document-level properties.
#
# WHY AGGREGATE? A document has multiple line items. We want ONE node
# per document, with summary properties like total_amount, timestamp
# of the first line item, and the fraud label.

# We need to make doc_number unique per source file (different runs reuse numbers)
master_df['unique_doc_id'] = (
    master_df['source_file'].str.replace('.csv', '', regex=False) + '_' +
    master_df['doc_number'].astype(str)
)

doc_agg = master_df.groupby('unique_doc_id').agg(
    doc_number=('doc_number', 'first'),
    total_amount=('amount', 'sum'),
    line_count=('line_position', 'count'),
    timestamp=('datetime', 'min'),         # Earliest line item time
    transaction_type=('transaction_type', 'first'),
    label=('label', lambda x: x[x != 'NonFraud'].iloc[0] if (x != 'NonFraud').any() else 'NonFraud'),
    is_fraud=('is_fraud', 'max'),           # 1 if ANY line item is fraud
    source_file=('source_file', 'first'),
    dataset_type=('dataset_type', 'first'),
).reset_index()

documents_df = pd.DataFrame({
    'doc_id:ID(Document)':  doc_agg['unique_doc_id'],
    'doc_number':           doc_agg['doc_number'],
    'total_amount:double':  doc_agg['total_amount'].round(2),
    'line_count:int':       doc_agg['line_count'],
    'timestamp':            doc_agg['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
    'transaction_type':     doc_agg['transaction_type'],
    'label':                doc_agg['label'],
    'is_fraud:boolean':     doc_agg['is_fraud'].map({1: 'true', 0: 'false'}),
    'source_file':          doc_agg['source_file'],
    'dataset_type':         doc_agg['dataset_type'],
})
documents_file = os.path.join(OUTPUT_DIR, 'nodes_documents.csv')
documents_df.to_csv(documents_file, index=False)
print(f"  ✅ nodes_documents.csv: {len(documents_df):,} documents")
print(f"     Fraud documents: {(doc_agg['is_fraud'] == 1).sum()}")
print(f"     Clean documents: {(doc_agg['is_fraud'] == 0).sum()}")


# --- 2C.3: ACCOUNT NODES ---
# G/L accounts from the Sachkonto column
# We also prefix with source file to avoid collisions
accounts_raw = master_df[['gl_account_id', 'source_file']].dropna(subset=['gl_account_id'])
accounts_raw['unique_acct_id'] = (
    accounts_raw['source_file'].str.replace('.csv', '', regex=False) + '_' +
    accounts_raw['gl_account_id'].astype(str)
)
unique_accounts = accounts_raw['unique_acct_id'].unique()

# Also get the raw account IDs for display
acct_display = accounts_raw.drop_duplicates(subset=['unique_acct_id'])

accounts_df = pd.DataFrame({
    'account_id:ID(Account)': acct_display['unique_acct_id'].values,
    'gl_account':             acct_display['gl_account_id'].values,
})
accounts_file = os.path.join(OUTPUT_DIR, 'nodes_accounts.csv')
accounts_df.to_csv(accounts_file, index=False)
print(f"  ✅ nodes_accounts.csv: {len(accounts_df)} accounts")


# --- 2C.4: MATERIAL NODES ---
materials_raw = master_df[['material_id', 'source_file']].dropna(subset=['material_id'])
materials_raw['unique_mat_id'] = (
    materials_raw['source_file'].str.replace('.csv', '', regex=False) + '_' +
    materials_raw['material_id'].astype(str)
)
mat_display = materials_raw.drop_duplicates(subset=['unique_mat_id'])

materials_df = pd.DataFrame({
    'material_id:ID(Material)': mat_display['unique_mat_id'].values,
    'material_name':            mat_display['material_id'].values,
})
materials_file = os.path.join(OUTPUT_DIR, 'nodes_materials.csv')
materials_df.to_csv(materials_file, index=False)
print(f"  ✅ nodes_materials.csv: {len(materials_df)} materials")


# ===================================================================
# STEP 2D: GENERATE NEO4J EDGE FILES
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2D: GENERATING NEO4J EDGE FILES")
print("=" * 70)
print("""
  EDGE FILE FORMAT:
  Neo4j edge CSVs need these special headers:
    :START_ID(StartNodeLabel)  — which node this edge starts from
    :END_ID(EndNodeLabel)      — which node this edge points to
    :TYPE                      — the relationship type (e.g., INVOICED_BY)

  Plus any property columns like amount, timestamp, etc.
""")

# --- 2D.1: DOCUMENT → VENDOR edges ---
# These edges represent vendor-related transactions:
#   - Kreditorische Rechnung (Vendor Invoice) → INVOICED_BY
#   - Kreditorische Gutschrift (Vendor Credit) → CREDITED_BY
#
# WHY THIS MATTERS FOR FRAUD:
# Invoice Kickback fraud shows up as unusual Document→Vendor patterns.
# If a vendor appears in documents with inflated amounts, or if the same
# vendor appears in suspiciously rapid succession, our topology engine
# will flag it.

vendor_rows = master_df[master_df['vendor_id'].notna()].copy()

# Map German transaction types to edge relationship types
VENDOR_EDGE_MAP = {
    'Kreditorische Rechnung':    'INVOICED_BY',
    'Kreditorische Gutschrift':  'CREDITED_BY',
    'Materialzugang':           'RECEIVED_FROM',   # Material received from vendor
    'Sachkontenbuchung':        'POSTED_VIA',       # G/L posting involving vendor
}

vendor_rows['edge_type'] = vendor_rows['transaction_type'].map(VENDOR_EDGE_MAP)
vendor_rows['edge_type'] = vendor_rows['edge_type'].fillna('RELATED_TO')  # Fallback

# Create unique vendor IDs per source file
vendor_rows['unique_vendor_id'] = (
    vendor_rows['source_file'].str.replace('.csv', '', regex=False) + '_' +
    vendor_rows['vendor_id'].astype(str)
)

edges_doc_vendor = pd.DataFrame({
    ':START_ID(Document)':  vendor_rows['unique_doc_id'],
    ':END_ID(Vendor)':      vendor_rows['vendor_id'],   # Vendors are global (V01, V02)
    ':TYPE':                vendor_rows['edge_type'],
    'amount:double':        vendor_rows['amount'].round(2),
    'debit_credit':         vendor_rows['debit_credit'],
    'posting_key':          vendor_rows['posting_key'],
    'timestamp':            vendor_rows['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
    'label':                vendor_rows['label'],
    'is_fraud:boolean':     vendor_rows['is_fraud'].map({1: 'true', 0: 'false'}),
})

edges_vendor_file = os.path.join(OUTPUT_DIR, 'edges_doc_vendor.csv')
edges_doc_vendor.to_csv(edges_vendor_file, index=False)
print(f"  ✅ edges_doc_vendor.csv: {len(edges_doc_vendor):,} edges")
print(f"     Edge type breakdown:")
for etype, count in edges_doc_vendor[':TYPE'].value_counts().items():
    print(f"       {etype}: {count}")


# --- 2D.2: DOCUMENT → ACCOUNT edges ---
# Every line item posts to a G/L account.
# These edges carry the debit/credit indicator and amount.
#
# WHY THIS MATTERS FOR FRAUD:
# Larceny fraud often involves posting to unusual accounts or posting
# abnormal amounts. By tracking all Document→Account edges, our topology
# engine can detect anomalous posting patterns.

acct_rows = master_df[master_df['gl_account_id'].notna()].copy()
acct_rows['unique_acct_id'] = (
    acct_rows['source_file'].str.replace('.csv', '', regex=False) + '_' +
    acct_rows['gl_account_id'].astype(str)
)

edges_doc_account = pd.DataFrame({
    ':START_ID(Document)':  acct_rows['unique_doc_id'],
    ':END_ID(Account)':     acct_rows['unique_acct_id'],
    ':TYPE':                'POSTED_TO',
    'amount:double':        acct_rows['amount'].round(2),
    'debit_credit':         acct_rows['debit_credit'],
    'posting_key':          acct_rows['posting_key'],
    'timestamp':            acct_rows['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
    'label':                acct_rows['label'],
})

edges_account_file = os.path.join(OUTPUT_DIR, 'edges_doc_account.csv')
edges_doc_account.to_csv(edges_account_file, index=False)
print(f"  ✅ edges_doc_account.csv: {len(edges_doc_account):,} edges")


# --- 2D.3: DOCUMENT → MATERIAL edges ---
# Material-related transactions (receipt/withdrawal)
mat_rows = master_df[master_df['material_id'].notna()].copy()
mat_rows['unique_mat_id'] = (
    mat_rows['source_file'].str.replace('.csv', '', regex=False) + '_' +
    mat_rows['material_id'].astype(str)
)

# Map transaction type to edge type
MAT_EDGE_MAP = {
    'Materialzugang':  'MATERIAL_RECEIVED',
    'Materialabgang':  'MATERIAL_WITHDRAWN',
}
mat_rows['edge_type'] = mat_rows['transaction_type'].map(MAT_EDGE_MAP)
mat_rows['edge_type'] = mat_rows['edge_type'].fillna('INVOLVES_MATERIAL')

edges_doc_material = pd.DataFrame({
    ':START_ID(Document)':  mat_rows['unique_doc_id'],
    ':END_ID(Material)':    mat_rows['unique_mat_id'],
    ':TYPE':                mat_rows['edge_type'],
    'amount:double':        mat_rows['amount'].round(2),
    'debit_credit':         mat_rows['debit_credit'],
    'timestamp':            mat_rows['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S'),
    'label':                mat_rows['label'],
})

edges_material_file = os.path.join(OUTPUT_DIR, 'edges_doc_material.csv')
edges_doc_material.to_csv(edges_material_file, index=False)
print(f"  ✅ edges_doc_material.csv: {len(edges_doc_material):,} edges")


# --- 2D.4: TEMPORAL EDGES (Document → Document) ---
# THIS IS THE MOST IMPORTANT EDGE TYPE FOR FRAUD DETECTION.
#
# WHY: Fraud patterns like circular payments and lapping require
# understanding the TIME SEQUENCE of transactions. We create
# TEMPORAL_NEXT edges between consecutive documents to enable
# sliding time window traversal in Neo4j.
#
# HOW: We sort all documents by timestamp within each source_file,
# then create an edge from each document to the next one.
# We also store the time delta (in seconds) between them.

print(f"\n  Building temporal edges (document sequence)...")

temporal_edges_list = []

# Process each source file separately (they represent independent simulation runs)
for source in master_df['source_file'].unique():
    source_docs = doc_agg[doc_agg['source_file'] == source].copy()
    source_docs = source_docs.sort_values('timestamp').reset_index(drop=True)

    for i in range(len(source_docs) - 1):
        current_doc = source_docs.iloc[i]
        next_doc = source_docs.iloc[i + 1]

        # Calculate time delta in seconds
        if pd.notna(current_doc['timestamp']) and pd.notna(next_doc['timestamp']):
            time_delta = (next_doc['timestamp'] - current_doc['timestamp']).total_seconds()
        else:
            time_delta = None

        temporal_edges_list.append({
            ':START_ID(Document)': current_doc['unique_doc_id'],
            ':END_ID(Document)':   next_doc['unique_doc_id'],
            ':TYPE':               'TEMPORAL_NEXT',
            'time_delta_seconds:double': time_delta,
            'source_file':         source,
        })

edges_temporal = pd.DataFrame(temporal_edges_list)
edges_temporal_file = os.path.join(OUTPUT_DIR, 'edges_temporal.csv')
edges_temporal.to_csv(edges_temporal_file, index=False)
print(f"  ✅ edges_temporal.csv: {len(edges_temporal):,} temporal edges")


# ===================================================================
# STEP 2E: SAVE THE CLEANED MASTER DATASET
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2E: SAVING CLEANED MASTER DATASET")
print("=" * 70)

# Save the full cleaned dataset for reference and future use
master_output_file = os.path.join(OUTPUT_DIR, 'master_cleaned.csv')
master_df.to_csv(master_output_file, index=False)
print(f"  ✅ master_cleaned.csv: {len(master_df):,} rows")


# ===================================================================
# STEP 2F: SUMMARY REPORT
# ===================================================================
print("\n" + "=" * 70)
print("STEP 2F: FINAL SUMMARY REPORT")
print("=" * 70)

print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │                    NEO4J IMPORT FILES                       │
  ├────────────────────────────────────────────────────────────┤
  │  NODE FILES:                                               │
  │    • nodes_vendors.csv      {len(vendors_df):>6} vendor nodes           │
  │    • nodes_documents.csv    {len(documents_df):>6} document nodes        │
  │    • nodes_accounts.csv     {len(accounts_df):>6} account nodes          │
  │    • nodes_materials.csv    {len(materials_df):>6} material nodes         │
  │                                                            │
  │  EDGE FILES:                                               │
  │    • edges_doc_vendor.csv   {len(edges_doc_vendor):>6} doc→vendor edges     │
  │    • edges_doc_account.csv  {len(edges_doc_account):>6} doc→account edges    │
  │    • edges_doc_material.csv {len(edges_doc_material):>6} doc→material edges   │
  │    • edges_temporal.csv     {len(edges_temporal):>6} temporal next edges   │
  │                                                            │
  │  REFERENCE:                                                │
  │    • master_cleaned.csv     {len(master_df):>6} cleaned line items    │
  ├────────────────────────────────────────────────────────────┤
  │  Output directory: {OUTPUT_DIR:<39} │
  └────────────────────────────────────────────────────────────┘

  TOTAL GRAPH SIZE:
    Nodes: {len(vendors_df) + len(documents_df) + len(accounts_df) + len(materials_df):,}
    Edges: {len(edges_doc_vendor) + len(edges_doc_account) + len(edges_doc_material) + len(edges_temporal):,}
""")

print("✅ Step 2 complete! Files are ready for Neo4j import in Step 3.")
