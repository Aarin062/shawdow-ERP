"""
===========================================================================
STEP 3: SHADOW GRAPH BUILDER — Load Data into Neo4j
Project: The Self-Auditing Ledger
Purpose: Stream our cleaned ERP data into Neo4j to build the Shadow Graph.
===========================================================================

WHAT THIS SCRIPT DOES:
-----------------------
This is the "Shadow Graph Builder" from our architecture diagram:

  ERP System → [Shadow Graph Builder] → Temporal Graph Database

In our case:
  CSV Files → [This Script] → Neo4j Graph Database

The script uses the official Neo4j Python driver to:
  1. Connect to a running Neo4j instance
  2. Create constraints and indexes (for performance)
  3. Load node CSVs (Vendors, Documents, Accounts, Materials)
  4. Load edge CSVs (all relationships)
  5. Verify the graph was built correctly

WHY WE USE THE PYTHON DRIVER (not neo4j-admin import):
-------------------------------------------------------
neo4j-admin import is faster for bulk loading but requires the
database to be OFFLINE. The Python driver lets us:
  - Load data into a RUNNING database (like a real Shadow System would)
  - Handle errors gracefully
  - Show progress as we load
  - Easily re-run to update the graph
  
In a real enterprise deployment, the Shadow Graph Builder would
intercept live ERP events via Kafka/RabbitMQ and stream them into
Neo4j in real time. Our CSV-based approach simulates this.

PRE-REQUISITES:
-----------------
1. Neo4j Desktop is installed and running
2. A database is created and started (default: neo4j)
3. The neo4j Python package is installed (pip install neo4j)

CONNECTION DETAILS (Neo4j Desktop defaults):
  URL:      bolt://localhost:7687
  Username: neo4j
  Password: (you set this when creating the database)
"""

import pandas as pd
import os
import sys
import time
from neo4j import GraphDatabase

sys.stdout.reconfigure(encoding='utf-8')

# ===================================================================
# CONFIGURATION — Update these if your Neo4j setup differs
# ===================================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "shadowgraph2026"  # Change this to your password!

# Where our Neo4j-ready CSVs are
IMPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import"
)

# Batch size for loading (how many rows to send per transaction)
# WHY: Sending all 200k+ rows in one transaction would consume too
# much memory. Batching keeps memory usage manageable.
BATCH_SIZE = 1000


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def create_driver():
    """
    Create a Neo4j driver connection.
    
    WHY A DRIVER?
    The Neo4j Python driver manages a connection pool to the database.
    Think of it like a database connection in SQL — you create it once,
    use it for all your queries, and close it when done.
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"  ✅ Connected to Neo4j at {NEO4J_URI}")
        return driver
    except Exception as e:
        print(f"\n  ❌ FAILED to connect to Neo4j!")
        print(f"     Error: {e}")
        print(f"\n  TROUBLESHOOTING:")
        print(f"  1. Is Neo4j Desktop running?")
        print(f"  2. Is your database started (green 'Active' status)?")
        print(f"  3. Is the password correct? (currently set to: '{NEO4J_PASSWORD}')")
        print(f"  4. Is the bolt port 7687 available?")
        print(f"\n  HOW TO FIX:")
        print(f"  1. Open Neo4j Desktop")
        print(f"  2. Create a new project (or use default)")
        print(f"  3. Click 'Add' → 'Local DBMS'")
        print(f"  4. Set password to: {NEO4J_PASSWORD}")
        print(f"  5. Click 'Start' on the database")
        print(f"  6. Re-run this script")
        sys.exit(1)


def run_query(driver, query, parameters=None):
    """Execute a single Cypher query and return results."""
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def batch_load(driver, query, data_list, batch_size=BATCH_SIZE, label="items"):
    """
    Load data in batches using UNWIND.
    
    WHY UNWIND?
    UNWIND is Neo4j's way of "for each item in this list, do X".
    Instead of running 1000 separate CREATE statements, we send
    one UNWIND query with 1000 items — much faster!
    
    Think of it like:
      SQL:   INSERT INTO ... VALUES (...), (...), (...)  -- bulk insert
      Neo4j: UNWIND $batch AS row CREATE (n:Label {...}) -- bulk create
    """
    total = len(data_list)
    loaded = 0
    
    for i in range(0, total, batch_size):
        batch = data_list[i:i + batch_size]
        with driver.session() as session:
            session.run(query, {"batch": batch})
        loaded += len(batch)
        
        # Progress indicator
        pct = (loaded / total) * 100
        print(f"\r    Loading {label}: {loaded:,}/{total:,} ({pct:.1f}%)", end="", flush=True)
    
    print(f"\r    ✅ Loaded {total:,} {label}" + " " * 30)


# ===================================================================
# STEP 3A: CLEAR EXISTING DATA (clean slate)
# ===================================================================
def step_3a_clear_database(driver):
    print("\n" + "=" * 70)
    print("STEP 3A: CLEARING EXISTING DATA")
    print("=" * 70)
    print("  Removing any previously loaded graph data...")
    
    # Drop all nodes and relationships
    # WHY: We want a clean slate each time we run the script.
    # In production, you'd do incremental updates instead.
    run_query(driver, "MATCH (n) DETACH DELETE n")
    
    # Verify it's clean
    result = run_query(driver, "MATCH (n) RETURN count(n) AS count")
    count = result[0]['count']
    print(f"  ✅ Database cleared. Node count: {count}")


# ===================================================================
# STEP 3B: CREATE CONSTRAINTS AND INDEXES
# ===================================================================
def step_3b_create_constraints(driver):
    print("\n" + "=" * 70)
    print("STEP 3B: CREATING CONSTRAINTS AND INDEXES")
    print("=" * 70)
    print("""
  WHY CONSTRAINTS?
  Constraints serve two purposes:
    1. UNIQUENESS: Ensure no duplicate nodes (e.g., two V01 vendors)
    2. PERFORMANCE: Neo4j automatically creates an index on constrained
       properties, making lookups O(1) instead of full scans.
  
  Without indexes, finding "Document fraud_1_5000000036" would require
  scanning all 59,852 document nodes. With an index, it's instant.
""")
    
    constraints = [
        ("Vendor",   "vendor_id",   "CREATE CONSTRAINT vendor_id_unique IF NOT EXISTS FOR (v:Vendor) REQUIRE v.vendor_id IS UNIQUE"),
        ("Document", "doc_id",      "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"),
        ("Account",  "account_id",  "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE"),
        ("Material", "material_id", "CREATE CONSTRAINT material_id_unique IF NOT EXISTS FOR (m:Material) REQUIRE m.material_id IS UNIQUE"),
    ]
    
    for label, prop, query in constraints:
        try:
            run_query(driver, query)
            print(f"  ✅ Constraint on :{label}.{prop}")
        except Exception as e:
            print(f"  ⚠️ Constraint on :{label}.{prop} — {e}")


# ===================================================================
# STEP 3C: LOAD NODE FILES
# ===================================================================
def step_3c_load_nodes(driver):
    print("\n" + "=" * 70)
    print("STEP 3C: LOADING NODES INTO NEO4J")
    print("=" * 70)
    
    # --- 3C.1: VENDOR NODES ---
    print("\n  📦 Loading Vendor nodes...")
    vendors_df = pd.read_csv(os.path.join(IMPORT_DIR, 'nodes_vendors.csv'))
    vendor_data = vendors_df.rename(columns={
        'vendor_id:ID(Vendor)': 'vendor_id',
    }).to_dict('records')
    
    # MERGE = "create if doesn't exist, match if it does"
    # This prevents duplicates if we run the script twice.
    batch_load(driver,
        """
        UNWIND $batch AS row
        MERGE (v:Vendor {vendor_id: row.vendor_id})
        SET v.name = row.name
        """,
        vendor_data, label="vendors"
    )
    
    # --- 3C.2: DOCUMENT NODES ---
    print("\n  📦 Loading Document nodes...")
    docs_df = pd.read_csv(os.path.join(IMPORT_DIR, 'nodes_documents.csv'))
    doc_data = docs_df.rename(columns={
        'doc_id:ID(Document)': 'doc_id',
        'total_amount:double': 'total_amount',
        'line_count:int': 'line_count',
        'is_fraud:boolean': 'is_fraud',
    }).to_dict('records')
    
    # For documents, we use CREATE instead of MERGE for speed
    # (we cleared the database in Step 3A, so no duplicates possible)
    batch_load(driver,
        """
        UNWIND $batch AS row
        CREATE (d:Document {
            doc_id: row.doc_id,
            doc_number: toString(row.doc_number),
            total_amount: toFloat(row.total_amount),
            line_count: toInteger(row.line_count),
            timestamp: datetime(row.timestamp),
            transaction_type: row.transaction_type,
            label: row.label,
            is_fraud: row.is_fraud,
            source_file: row.source_file,
            dataset_type: row.dataset_type
        })
        """,
        doc_data, label="documents"
    )
    
    # --- 3C.3: ACCOUNT NODES ---
    print("\n  📦 Loading Account nodes...")
    accounts_df = pd.read_csv(os.path.join(IMPORT_DIR, 'nodes_accounts.csv'))
    account_data = accounts_df.rename(columns={
        'account_id:ID(Account)': 'account_id',
    }).to_dict('records')
    
    batch_load(driver,
        """
        UNWIND $batch AS row
        MERGE (a:Account {account_id: row.account_id})
        SET a.gl_account = toString(row.gl_account)
        """,
        account_data, label="accounts"
    )
    
    # --- 3C.4: MATERIAL NODES ---
    print("\n  📦 Loading Material nodes...")
    materials_df = pd.read_csv(os.path.join(IMPORT_DIR, 'nodes_materials.csv'))
    material_data = materials_df.rename(columns={
        'material_id:ID(Material)': 'material_id',
        'material_name': 'material_name',
    }).to_dict('records')
    
    batch_load(driver,
        """
        UNWIND $batch AS row
        MERGE (m:Material {material_id: row.material_id})
        SET m.material_name = row.material_name
        """,
        material_data, label="materials"
    )


# ===================================================================
# STEP 3D: LOAD EDGE FILES
# ===================================================================
def step_3d_load_edges(driver):
    print("\n" + "=" * 70)
    print("STEP 3D: LOADING EDGES (RELATIONSHIPS) INTO NEO4J")
    print("=" * 70)
    print("""
  HOW EDGES WORK IN NEO4J:
  An edge (relationship) connects two existing nodes. The Cypher syntax:
    MATCH (a:Document {doc_id: "X"})
    MATCH (b:Vendor {vendor_id: "V01"})
    CREATE (a)-[:INVOICED_BY {amount: 5000}]->(b)
  
  This finds Document X, finds Vendor V01, and draws an arrow between
  them labeled "INVOICED_BY" with the amount as a property.
""")
    
    # --- 3D.1: DOCUMENT → VENDOR edges ---
    print("\n  🔗 Loading Document → Vendor edges...")
    edges_vendor_df = pd.read_csv(os.path.join(IMPORT_DIR, 'edges_doc_vendor.csv'))
    
    # We need to handle different edge types (INVOICED_BY, CREDITED_BY, etc.)
    # Neo4j doesn't support dynamic relationship types in a single query easily,
    # so we load each type separately.
    for edge_type in edges_vendor_df[':TYPE'].unique():
        subset = edges_vendor_df[edges_vendor_df[':TYPE'] == edge_type].copy()
        edge_data = subset.rename(columns={
            ':START_ID(Document)': 'start_id',
            ':END_ID(Vendor)': 'end_id',
            'amount:double': 'amount',
            'is_fraud:boolean': 'is_fraud',
        }).to_dict('records')
        
        # Build the appropriate Cypher query for each edge type
        # WHY SEPARATE QUERIES? Neo4j Cypher requires relationship types
        # to be hardcoded in the query (can't use a variable for the type
        # in CREATE). So we generate a query per type.
        cypher = f"""
        UNWIND $batch AS row
        MATCH (d:Document {{doc_id: row.start_id}})
        MATCH (v:Vendor {{vendor_id: row.end_id}})
        CREATE (d)-[:{edge_type} {{
            amount: toFloat(row.amount),
            debit_credit: row.debit_credit,
            posting_key: toString(row.posting_key),
            timestamp: datetime(row.timestamp),
            label: row.label,
            is_fraud: row.is_fraud
        }}]->(v)
        """
        batch_load(driver, cypher, edge_data, label=f"{edge_type} edges")
    
    
    # --- 3D.2: DOCUMENT → ACCOUNT edges ---
    print("\n  🔗 Loading Document → Account edges (POSTED_TO)...")
    edges_acct_df = pd.read_csv(os.path.join(IMPORT_DIR, 'edges_doc_account.csv'))
    acct_edge_data = edges_acct_df.rename(columns={
        ':START_ID(Document)': 'start_id',
        ':END_ID(Account)': 'end_id',
        'amount:double': 'amount',
    }).to_dict('records')
    
    batch_load(driver,
        """
        UNWIND $batch AS row
        MATCH (d:Document {doc_id: row.start_id})
        MATCH (a:Account {account_id: row.end_id})
        CREATE (d)-[:POSTED_TO {
            amount: toFloat(row.amount),
            debit_credit: row.debit_credit,
            posting_key: toString(row.posting_key),
            timestamp: datetime(row.timestamp),
            label: row.label
        }]->(a)
        """,
        acct_edge_data, label="POSTED_TO edges"
    )
    
    
    # --- 3D.3: DOCUMENT → MATERIAL edges ---
    print("\n  🔗 Loading Document → Material edges...")
    edges_mat_df = pd.read_csv(os.path.join(IMPORT_DIR, 'edges_doc_material.csv'))
    
    for edge_type in edges_mat_df[':TYPE'].unique():
        subset = edges_mat_df[edges_mat_df[':TYPE'] == edge_type].copy()
        mat_edge_data = subset.rename(columns={
            ':START_ID(Document)': 'start_id',
            ':END_ID(Material)': 'end_id',
            'amount:double': 'amount',
        }).to_dict('records')
        
        cypher = f"""
        UNWIND $batch AS row
        MATCH (d:Document {{doc_id: row.start_id}})
        MATCH (m:Material {{material_id: row.end_id}})
        CREATE (d)-[:{edge_type} {{
            amount: toFloat(row.amount),
            debit_credit: row.debit_credit,
            timestamp: datetime(row.timestamp),
            label: row.label
        }}]->(m)
        """
        batch_load(driver, cypher, mat_edge_data, label=f"{edge_type} edges")
    
    
    # --- 3D.4: TEMPORAL EDGES (Document → Document) ---
    print("\n  🔗 Loading Temporal edges (TEMPORAL_NEXT)...")
    print("    (This is the largest edge set — may take a minute)")
    edges_temp_df = pd.read_csv(os.path.join(IMPORT_DIR, 'edges_temporal.csv'))
    temp_edge_data = edges_temp_df.rename(columns={
        ':START_ID(Document)': 'start_id',
        ':END_ID(Document)': 'end_id',
        'time_delta_seconds:double': 'time_delta',
    }).to_dict('records')
    
    batch_load(driver,
        """
        UNWIND $batch AS row
        MATCH (d1:Document {doc_id: row.start_id})
        MATCH (d2:Document {doc_id: row.end_id})
        CREATE (d1)-[:TEMPORAL_NEXT {
            time_delta_seconds: toFloat(row.time_delta),
            source_file: row.source_file
        }]->(d2)
        """,
        temp_edge_data, label="TEMPORAL_NEXT edges"
    )


# ===================================================================
# STEP 3E: VERIFY THE GRAPH
# ===================================================================
def step_3e_verify(driver):
    print("\n" + "=" * 70)
    print("STEP 3E: VERIFYING THE SHADOW GRAPH")
    print("=" * 70)
    
    # Count nodes by label
    print("\n  📊 NODE COUNTS:")
    for label in ['Vendor', 'Document', 'Account', 'Material']:
        result = run_query(driver, f"MATCH (n:{label}) RETURN count(n) AS count")
        count = result[0]['count']
        print(f"    :{label:<12} → {count:>10,}")
    
    # Count total nodes
    result = run_query(driver, "MATCH (n) RETURN count(n) AS count")
    total_nodes = result[0]['count']
    
    # Count edges by type
    print("\n  📊 EDGE COUNTS:")
    result = run_query(driver, """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)
    total_edges = 0
    for record in result:
        print(f"    :{record['rel_type']:<20} → {record['count']:>10,}")
        total_edges += record['count']
    
    # Summary
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  SHADOW GRAPH SUMMARY            │")
    print(f"  │  Total Nodes: {total_nodes:>10,}         │")
    print(f"  │  Total Edges: {total_edges:>10,}         │")
    print(f"  └─────────────────────────────────┘")
    
    # Check fraud nodes
    print("\n  🔍 FRAUD DOCUMENT SAMPLES:")
    fraud_docs = run_query(driver, """
        MATCH (d:Document)
        WHERE d.is_fraud = 'true' OR d.is_fraud = true
        RETURN d.doc_id AS doc_id, d.label AS label, 
               d.total_amount AS amount, d.timestamp AS timestamp
        LIMIT 10
    """)
    for doc in fraud_docs:
        print(f"    📄 {doc['doc_id']} | {doc['label']} | ${doc['amount']:,.2f} | {doc['timestamp']}")
    
    if not fraud_docs:
        # Try alternate field check
        fraud_docs2 = run_query(driver, """
            MATCH (d:Document)
            WHERE d.label <> 'NonFraud'
            RETURN d.doc_id AS doc_id, d.label AS label,
                   d.total_amount AS amount
            LIMIT 10
        """)
        if fraud_docs2:
            print("    (Found via label check:)")
            for doc in fraud_docs2:
                print(f"    📄 {doc['doc_id']} | {doc['label']} | ${doc['amount']:,.2f}")
    
    print("\n✅ Shadow Graph Builder complete!")
    print("   You can now open Neo4j Browser (http://localhost:7474)")
    print("   and explore the graph visually.")
    print("\n   Try this Cypher query to see fraud documents:")
    print("   MATCH (d:Document) WHERE d.label <> 'NonFraud' RETURN d LIMIT 25")


# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 3: SHADOW GRAPH BUILDER")
    print("Loading ERP data into Neo4j...")
    print("=" * 70)
    
    start_time = time.time()
    
    # Connect
    driver = create_driver()
    
    # Execute pipeline
    step_3a_clear_database(driver)
    step_3b_create_constraints(driver)
    step_3c_load_nodes(driver)
    step_3d_load_edges(driver)
    step_3e_verify(driver)
    
    # Cleanup
    driver.close()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f} seconds")
