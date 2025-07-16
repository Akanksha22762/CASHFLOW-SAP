import json
import re
from collections import defaultdict, Counter

LOG_FILE = "gpt_classification_log.jsonl"

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

def main():
    category_keyword_counts = defaultdict(Counter)

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            desc = entry.get("description", "")
            category = entry.get("gpt_category", "").strip()

            if category.lower() == "other" or not category:
                continue

            tokens = tokenize(desc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                category_keyword_counts[category][token] += 1

    print("=== Suggested new keywords ===")
    for category, counter in category_keyword_counts.items():
        print(f"\nCategory: {category}")
        for keyword, count in counter.most_common(10):
            print(f"  '{keyword}': {count} times")

if __name__ == "__main__":
    main()
