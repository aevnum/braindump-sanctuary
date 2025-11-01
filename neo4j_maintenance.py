"""
Neo4j Database Maintenance Script
Use this to manage your Brain Dump Sanctuary Neo4j database
"""

import os
from dotenv import load_dotenv
from braindump_core import BrainDumpDB

load_dotenv()

def clear_all_dumps():
    """Clear all dumps from the database."""
    db = BrainDumpDB()
    with db.driver.session() as session:
        # Delete all relationships and nodes
        session.run("MATCH (d:Dump) DETACH DELETE d")
        session.run("MATCH (c:Cluster) DETACH DELETE c")
    print("✓ All dumps and clusters cleared from Neo4j")
    db.close()

def show_database_stats():
    """Show current database statistics."""
    db = BrainDumpDB()
    with db.driver.session() as session:
        # Count dumps
        result_dumps = session.run("MATCH (d:Dump) RETURN COUNT(d) as count")
        dump_count = result_dumps.single()["count"]
        
        # Count clusters
        result_clusters = session.run("MATCH (c:Cluster) RETURN COUNT(c) as count")
        cluster_count = result_clusters.single()["count"]
        
        # Count relationships
        result_rels = session.run("MATCH ()-[r:IN_CLUSTER]->() RETURN COUNT(r) as count")
        rel_count = result_rels.single()["count"]
    
    print("\n=== Neo4j Database Statistics ===")
    print(f"Total Brain Dumps: {dump_count}")
    print(f"Total Clusters: {cluster_count}")
    print(f"Dump-to-Cluster Relationships: {rel_count}")
    
    db.close()

def remove_duplicates():
    """Remove duplicate brain dumps (same text)."""
    db = BrainDumpDB()
    with db.driver.session() as session:
        # Find duplicates
        result = session.run("""
            MATCH (d:Dump)
            WITH d.text as text, COLLECT(d.id) as ids
            WHERE SIZE(ids) > 1
            RETURN text, ids
        """)
        
        duplicates = list(result)
        if not duplicates:
            print("✓ No duplicates found!")
        else:
            print(f"Found {len(duplicates)} duplicate texts:")
            for row in duplicates:
                text = row["text"]
                ids = row["ids"]
                print(f"  - '{text[:50]}...' appears {len(ids)} times")
                
                # Keep first, delete rest
                for dup_id in ids[1:]:
                    session.run("""
                        MATCH (d:Dump {id: $dump_id})
                        DETACH DELETE d
                    """, dump_id=dup_id)
            
            print(f"✓ Removed duplicate entries")
    
    db.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python neo4j_maintenance.py <command>")
        print("\nCommands:")
        print("  stats      - Show database statistics")
        print("  clear      - Clear all dumps and clusters")
        print("  dedup      - Remove duplicate brain dumps")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "stats":
        show_database_stats()
    elif command == "clear":
        confirm = input("Are you sure? This will delete all data. (yes/no): ")
        if confirm.lower() == "yes":
            clear_all_dumps()
        else:
            print("Cancelled.")
    elif command == "dedup":
        remove_duplicates()
    else:
        print(f"Unknown command: {command}")
