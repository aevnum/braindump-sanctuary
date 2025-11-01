from braindump_core import BrainDumpDB

db = BrainDumpDB()

# Template categories - fill in your own questions
categories = {
    "AI/Tech": [
        "Why do transformers need positional encodings?",
        "How does backpropagation actually update weights?",
        # Add 5-6 more
    ],
    "Psychology": [
        "Why do we procrastinate on important tasks?",
        "What causes déjà vu?",
        # Add 5-6 more
    ],
    "Nature/Biology": [
        "How do octopuses change color so fast?",
        "Why do cats purr?",
        # Add 5-6 more
    ],
    "Philosophy": [
        "Is consciousness substrate-independent?",
        "Can we prove we're not in a simulation?",
        # Add 3-4 more
    ],
    "Random": [
        "Why are manhole covers round?",
        "What makes sourdough different?",
        # Add 2-3 more
    ]
}

for category, dumps in categories.items():
    print(f"Adding {category}...")
    for dump in dumps:
        db.add_dump(dump)

print(f"\n✅ Added {sum(len(d) for d in categories.values())} brain dumps!")