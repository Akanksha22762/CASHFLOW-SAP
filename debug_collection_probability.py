#!/usr/bin/env python3
"""
Debug Collection Probability Calculation
Simulate the exact calculation that's causing 5000%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_exact_calculation():
    """Simulate the exact calculation from the system"""
    print("🔍 DEBUGGING COLLECTION PROBABILITY CALCULATION")
    print("=" * 60)
    
    # Create sample data that might cause the issue
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    
    # Create extreme revenue data that could cause 5000%
    revenue_data = []
    for i in range(180):  # 6 months
        if i % 30 == 0:  # Every month
            revenue_data.append(1000000)  # High revenue
        else:
            revenue_data.append(100)  # Low revenue
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates[:len(revenue_data)],
        'Amount (INR)': revenue_data
    })
    
    print("📊 Sample Revenue Data:")
    print(f"Total transactions: {len(df)}")
    print(f"Revenue range: ₹{df['Amount (INR)'].min():,.0f} to ₹{df['Amount (INR)'].max():,.0f}")
    print(f"Revenue mean: ₹{df['Amount (INR)'].mean():,.0f}")
    print(f"Revenue std: ₹{df['Amount (INR)'].std():,.0f}")
    
    # Simulate the exact calculation from the system
    amount_column = 'Amount (INR)'
    revenue_data = df[df[amount_column] > 0].copy()
    
    print(f"\n📊 Filtered Revenue Data:")
    print(f"Transactions with positive amounts: {len(revenue_data)}")
    
    # Calculate monthly revenue (as in the system)
    revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])
    monthly_revenue = revenue_data.groupby([revenue_data['Date'].dt.year, revenue_data['Date'].dt.month])[amount_column].sum()
    
    print(f"\n📊 Monthly Revenue:")
    print(f"Monthly revenue: {monthly_revenue.tolist()}")
    print(f"Monthly revenue mean: {monthly_revenue.mean():,.0f}")
    print(f"Monthly revenue std: {monthly_revenue.std():,.0f}")
    
    # Calculate revenue consistency (as in the system)
    revenue_consistency = monthly_revenue.std() / monthly_revenue.mean() if monthly_revenue.mean() > 0 else 0
    print(f"\n📊 Revenue Consistency: {revenue_consistency:.6f}")
    
    # Calculate collection probability (as in the system)
    collection_probability = max(0.5, 1 - revenue_consistency)
    print(f"Original Collection Probability: {collection_probability:.6f}")
    
    # Apply the fix (as in the system)
    collection_probability_fixed = min(collection_probability, 1.0)
    print(f"Fixed Collection Probability: {collection_probability_fixed:.6f}")
    
    # Convert to percentage for display
    original_percentage = collection_probability * 100
    fixed_percentage = collection_probability_fixed * 100
    
    print(f"\n📊 Results:")
    print(f"Original Collection Probability (%): {original_percentage:.1f}%")
    print(f"Fixed Collection Probability (%): {fixed_percentage:.1f}%")
    
    # Check if this could cause 5000%
    if original_percentage > 100:
        print(f"❌ This could cause the 5000% issue!")
        print(f"   The calculation produces {original_percentage:.1f}%")
        print(f"   But the fix should cap it at {fixed_percentage:.1f}%")
    else:
        print(f"✅ This calculation looks normal")
    
    return original_percentage, fixed_percentage

def test_extreme_scenarios():
    """Test various extreme scenarios"""
    print("\n🔍 TESTING EXTREME SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        ("Very Volatile", [100, 1000000, 100, 1000000, 100, 1000000]),
        ("Extreme Volatility", [1, 1000000, 1, 1000000, 1, 1000000]),
        ("Zero Mean", [0, 1000000, 0, 1000000, 0, 1000000]),
        ("Negative Values", [-1000, 1000000, -1000, 1000000, -1000, 1000000])
    ]
    
    for scenario_name, revenue_data in scenarios:
        print(f"\n📊 {scenario_name}:")
        monthly_revenue = pd.Series(revenue_data)
        
        # Calculate as in the system
        revenue_mean = monthly_revenue.mean()
        revenue_std = monthly_revenue.std()
        revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
        collection_probability = max(0.5, 1 - revenue_consistency)
        collection_probability_fixed = min(collection_probability, 1.0)
        
        # Convert to percentage
        original_percentage = collection_probability * 100
        fixed_percentage = collection_probability_fixed * 100
        
        print(f"  Revenue data: {revenue_data}")
        print(f"  Revenue mean: {revenue_mean:,.0f}")
        print(f"  Revenue std: {revenue_std:,.0f}")
        print(f"  Revenue consistency: {revenue_consistency:.6f}")
        print(f"  Original collection probability: {original_percentage:.1f}%")
        print(f"  Fixed collection probability: {fixed_percentage:.1f}%")
        
        if original_percentage > 100:
            print(f"  ❌ This scenario could cause the issue!")
        else:
            print(f"  ✅ This scenario is normal")

def check_fix_effectiveness():
    """Check if the fix is effective"""
    print("\n✅ CHECKING FIX EFFECTIVENESS")
    print("=" * 60)
    
    # Test the exact scenario that could cause 5000%
    extreme_revenue = [100, 1000000, 100, 1000000, 100, 1000000]
    monthly_revenue = pd.Series(extreme_revenue)
    
    revenue_mean = monthly_revenue.mean()
    revenue_std = monthly_revenue.std()
    revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
    collection_probability = max(0.5, 1 - revenue_consistency)
    collection_probability_fixed = min(collection_probability, 1.0)
    
    original_percentage = collection_probability * 100
    fixed_percentage = collection_probability_fixed * 100
    
    print(f"Extreme scenario test:")
    print(f"  Original: {original_percentage:.1f}%")
    print(f"  Fixed: {fixed_percentage:.1f}%")
    
    if fixed_percentage <= 100:
        print(f"  ✅ Fix is working correctly!")
    else:
        print(f"  ❌ Fix is not working!")

def main():
    """Main debug function"""
    print("🚀 DEBUGGING COLLECTION PROBABILITY ISSUE")
    print("=" * 60)
    
    # Simulate the exact calculation
    original, fixed = simulate_exact_calculation()
    
    # Test extreme scenarios
    test_extreme_scenarios()
    
    # Check fix effectiveness
    check_fix_effectiveness()
    
    print("\n🎯 CONCLUSION:")
    if original > 100:
        print("The calculation can produce values > 100%, but the fix should cap it.")
        print("If you're still seeing 5000%, it might be:")
        print("1. Cached results from before the fix")
        print("2. A different calculation path being used")
        print("3. The fix not being applied to the specific function")
    else:
        print("The calculation looks normal. The 5000% might be from cached data.")

if __name__ == "__main__":
    main() 