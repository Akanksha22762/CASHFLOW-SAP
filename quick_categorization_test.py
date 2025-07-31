#!/usr/bin/env python3
"""
Quick test for categorization fixes
"""

def test_categorization():
    """Test categorization with key examples"""
    print("🧪 Quick Categorization Test...")
    
    # Import the function
    from app1 import categorize_transaction_perfect
    
    # Test key problematic cases
    test_cases = [
        ("Infrastructure Development - Warehouse Construction", 1000000, "Investing Activities"),
        ("VIP Customer Payment - Construction Company", 1000000, "Operating Activities"),
        ("Investment Liquidation - Mutual Fund Units", 1000000, "Financing Activities"),
        ("Penalty Payment - Late Payment Charges", 1000000, "Financing Activities"),
        ("Customer Payment - Shipbuilding Yard", 1000000, "Operating Activities"),
    ]
    
    for desc, amount, expected in test_cases:
        result = categorize_transaction_perfect(desc, amount)
        status = "✅" if result == expected else "❌"
        print(f"{status} {desc}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        print()

if __name__ == "__main__":
    test_categorization() 