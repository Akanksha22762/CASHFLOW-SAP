#!/usr/bin/env python3
"""
Test customer counting logic with improved regex
"""

import pandas as pd

def test_customer_counting():
    """Test the customer counting logic"""
    
    # Create test data with multiple customers
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Description': ['Customer A Payment', 'Customer B Invoice', 'Customer C Service', 'Customer A Payment', 'Customer D Order'],
        'Amount': [1000, 2000, 1500, 500, 3000],
        'Type': ['INWARD', 'INWARD', 'INWARD', 'INWARD', 'INWARD']
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Test data created: {df.shape}")
    print(f"✅ Columns: {list(df.columns)}")
    
    # Filter revenue transactions
    revenue_transactions = df[df['Type'].str.contains('INWARD|CREDIT', case=False, na=False)]
    print(f"✅ Revenue transactions: {len(revenue_transactions)}")
    
    # Count unique customers from descriptions - IMPROVED REGEX
    customer_extractions = revenue_transactions['Description'].str.extract(r'(Customer\s+[A-Z]|Client\s+[A-Z]|Customer\s+\w+|Client\s+\w+)')
    print(f"✅ Customer extractions shape: {customer_extractions.shape}")
    
    if not customer_extractions.empty:
        unique_customers = customer_extractions.iloc[:, 0].dropna().unique()
        total_contracts = len(unique_customers)
        print(f"✅ Unique customers found: {unique_customers}")
        print(f"✅ Total contracts: {total_contracts}")
    else:
        total_contracts = max(1, len(revenue_transactions) // 4)
        print(f"✅ Estimated contracts: {total_contracts}")
    
    # Test customer segmentation
    enterprise_count = max(1, int(total_contracts * 0.2))
    mid_market_count = max(1, int(total_contracts * 0.5))
    small_business_count = max(1, total_contracts - enterprise_count - mid_market_count)
    
    if small_business_count < 0:
        small_business_count = 0
        mid_market_count = total_contracts - enterprise_count
    
    print(f"✅ Customer segmentation:")
    print(f"   Enterprise: {enterprise_count}")
    print(f"   Mid-market: {mid_market_count}")
    print(f"   Small business: {small_business_count}")
    print(f"   Total: {enterprise_count + mid_market_count + small_business_count}")
    
    # Verify it matches total_contracts
    total_calculated = enterprise_count + mid_market_count + small_business_count
    print(f"✅ Verification: Total contracts ({total_contracts}) = Calculated ({total_calculated}): {total_contracts == total_calculated}")

if __name__ == "__main__":
    test_customer_counting() 