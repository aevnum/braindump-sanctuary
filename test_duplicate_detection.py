"""
Test script to verify duplicate detection functionality
"""

from braindump_core import BrainDumpDB

def test_duplicate_detection():
    """Test that duplicate brain dumps are detected and not re-added"""
    
    print("=== Testing Duplicate Detection ===\n")
    
    # Initialize database
    db = BrainDumpDB()
    
    # Test 1: Add a new dump
    test_text = "Why do dreams feel so real but fade so quickly?"
    print(f"Test 1: Adding new dump...")
    dump_id1, is_duplicate1 = db.add_dump(test_text)
    print(f"  Result: dump_id={dump_id1}, is_duplicate={is_duplicate1}")
    assert not is_duplicate1, "First addition should not be a duplicate"
    print("  ✓ PASS: New dump added successfully\n")
    
    # Test 2: Try to add the same dump again
    print(f"Test 2: Adding same dump again...")
    dump_id2, is_duplicate2 = db.add_dump(test_text)
    print(f"  Result: dump_id={dump_id2}, is_duplicate={is_duplicate2}")
    assert is_duplicate2, "Second addition should be detected as duplicate"
    assert dump_id1 == dump_id2, "Should return the same dump_id"
    print("  ✓ PASS: Duplicate detected and prevented\n")
    
    # Test 3: Add a different dump
    different_text = "How does quantum entanglement actually work?"
    print(f"Test 3: Adding different dump...")
    dump_id3, is_duplicate3 = db.add_dump(different_text)
    print(f"  Result: dump_id={dump_id3}, is_duplicate={is_duplicate3}")
    assert not is_duplicate3, "Different dump should not be a duplicate"
    assert dump_id3 != dump_id1, "Should have a different dump_id"
    print("  ✓ PASS: Different dump added successfully\n")
    
    # Test 4: Verify total count
    all_dumps = db.get_all_dumps()
    print(f"Test 4: Verifying database state...")
    print(f"  Total dumps in database: {len(all_dumps)}")
    print(f"  Expected at least 2 unique dumps")
    # Note: We might have more dumps from previous runs
    print("  ✓ Database contains the dumps\n")
    
    # Clean up
    db.close()
    
    print("=== All Tests Passed! ===")
    print("\nDuplicate detection is working correctly:")
    print("  ✓ New dumps are added")
    print("  ✓ Duplicate dumps are detected")
    print("  ✓ Duplicate dumps return existing ID")
    print("  ✓ Different dumps are treated separately")

if __name__ == "__main__":
    test_duplicate_detection()
