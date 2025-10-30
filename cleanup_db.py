"""
Quick script to clean up duplicate entries in braindump.db
Run this once to fix the current database state
"""

import sqlite3

# Connect to the database
conn = sqlite3.connect("braindump.db")
cursor = conn.cursor()

# Get all dumps
cursor.execute("SELECT id, text, COUNT(*) as count FROM dumps GROUP BY text HAVING count > 1")
duplicates = cursor.fetchall()

if duplicates:
    print(f"Found {len(duplicates)} duplicate texts:")
    for text_id, text, count in duplicates:
        print(f"  - '{text[:50]}...' appears {count} times")
        
        # Keep only the first occurrence, delete the rest
        cursor.execute("""
            DELETE FROM dumps 
            WHERE text = ? AND id NOT IN (
                SELECT MIN(id) FROM dumps WHERE text = ?
            )
        """, (text, text))
    
    conn.commit()
    print(f"\n✓ Removed duplicate entries")
else:
    print("No duplicates found!")

# Show current state
cursor.execute("SELECT COUNT(*) FROM dumps")
total = cursor.fetchone()[0]
print(f"\nTotal unique brain dumps: {total}")

# Optionally reset the auto-increment counter
reset = input("\nReset ID counter to start from 1? (y/n): ")
if reset.lower() == 'y':
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='dumps'")
    conn.commit()
    print("✓ ID counter reset")

conn.close()
print("\nDone! You can now run the Streamlit app.")
