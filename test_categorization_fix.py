#!/usr/bin/env python3
"""
Test script to verify categorization fixes
"""

def test_categorization_fix():
    """Test the fixed categorization logic"""
    print("🧪 Testing Fixed Categorization Logic...")
    print("=" * 50)
    
    # Import the fixed function
    from app1 import categorize_transaction_perfect
    
    # Test cases based on your data
    test_cases = [
        # Investing Activities (should be correctly categorized)
        ("Infrastructure Development - Warehouse Construction - 4717 sq ft", 3709289.81, "Investing Activities"),
        ("Equipment Purchase - Rolling Mill Upgrade - Advanced Technology", 2948803.60, "Investing Activities"),
        ("Software Investment - ERP System - Digital Transformation", 2009087.02, "Investing Activities"),
        ("Plant Expansion - New Production Line - Capacity Increase", 2573125.58, "Investing Activities"),
        ("Machinery Purchase - Quality Testing Equipment - ISO Standards", 4466666.70, "Investing Activities"),
        ("Technology Investment - Automation System - Industry 4.0", 2761314.40, "Investing Activities"),
        ("Investment Liquidation - Mutual Fund Units - Capital Gains", 4206942.95, "Investing Activities"),
        ("Asset Sale Proceeds - Old Machinery - Blast Furnace Equipment", 4439568.72, "Investing Activities"),
        
        # Financing Activities (should be correctly categorized)
        ("Penalty Payment - Late Payment Charges - Overdue Interest", 2443546.41, "Financing Activities"),
        ("Interest Payment - Working Capital Loan - Monthly Interest", 3563362.88, "Financing Activities"),
        ("Bridge Loan - Project Funding - 25 Crores - 12% Interest", 4591837.18, "Financing Activities"),
        ("Bank Loan Disbursement - Working Capital - 13 Crores - 12% Interest", 2213490.68, "Financing Activities"),
        ("Term Loan - Plant Expansion - 22 Crores - 8% Interest", 3374591.26, "Financing Activities"),
        ("Loan EMI Payment - Principal + Interest - EMI #44", 4274790.04, "Financing Activities"),
        ("Bank Charges - Processing Fee - Loan Maintenance", 2609363.17, "Financing Activities"),
        ("Equipment Financing - New Machinery - 4 Crores - 10% Interest", 2779755.01, "Financing Activities"),
        
        # Operating Activities (should be correctly categorized)
        ("VIP Customer Payment - Construction Company - Steel Angles", 2141283.32, "Operating Activities"),
        ("Customer Payment - Shipbuilding Yard - Hot Rolled Coils", 4520025.10, "Operating Activities"),
        ("Salary Payment - Employee Payroll - 137 Employees", 2875106.26, "Operating Activities"),
        ("Cleaning Payment - Housekeeping Services - Monthly", 3670968.34, "Operating Activities"),
        ("Transport Payment - Logistics Services - Freight Charges", 4029308.51, "Operating Activities"),
        ("Utility Payment - Electricity Bill - 2870 MWh - Monthly", 4319957.11, "Operating Activities"),
        ("Telephone Payment - Landline & Mobile - Monthly Charges", 4720307.98, "Operating Activities"),
        ("Scrap Metal Sale - Excess Steel Scrap - 161 Tonnes", 4284497.34, "Operating Activities"),
        ("Export Payment - Automotive Manufacturer - Steel Sheets", 4654438.29, "Operating Activities"),
        ("Renovation Payment - Plant Modernization - Energy Efficiency", 1470457.46, "Operating Activities"),
    ]
    
    print("📊 Testing Categorization Results:")
    print("-" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for description, amount, expected in test_cases:
        result = categorize_transaction_perfect(description, amount)
        status = "✅" if result == expected else "❌"
        print(f"{status} {description[:50]}...")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        print()
        
        if result == expected:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"📈 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 90:
        print("✅ Categorization fix is working correctly!")
    else:
        print("❌ Categorization still needs improvement")
    
    return accuracy >= 90

if __name__ == "__main__":
    success = test_categorization_fix()
    if success:
        print("\n✅ Categorization is now working correctly!")
    else:
        print("\n❌ Categorization needs more fixes!") 