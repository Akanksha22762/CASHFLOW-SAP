#!/usr/bin/env python3
"""
Debug Actual Calculation
Test the actual calculation with real data to see why 5000% is still appearing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_extreme_scenario():
    """Test with extreme data that could cause 5000%"""
    print("🔍 TESTING EXTREME SCENARIO")
    print("=" * 60)
    
    # Create extreme data that could cause 5000%
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    
    # Create very volatile revenue data
    revenue_data = []
    for i in range(180):
        if i % 30 == 0:  # Every month
            revenue_data.append(1000000)  # Very high revenue
        else:
            revenue_data.append(1)  # Very low revenue
    
    df = pd.DataFrame({
        'Date': dates[:len(revenue_data)],
        'Amount (INR)': revenue_data
    })
    
    print("📊 Extreme Data Created:")
    print(f"Total transactions: {len(df)}")
    print(f"Revenue range: ₹{df['Amount (INR)'].min():,.0f} to ₹{df['Amount (INR)'].max():,.0f}")
    
    # Simulate the exact calculation
    amount_column = 'Amount (INR)'
    revenue_data = df[df[amount_column] > 0].copy()
    
    # Calculate monthly revenue
    revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])
    monthly_revenue = revenue_data.groupby([revenue_data['Date'].dt.year, revenue_data['Date'].dt.month])[amount_column].sum()
    
    print(f"\n📊 Monthly Revenue: {monthly_revenue.tolist()}")
    print(f"Monthly mean: {monthly_revenue.mean():,.0f}")
    print(f"Monthly std: {monthly_revenue.std():,.0f}")
    
    # Calculate revenue consistency
    revenue_consistency = monthly_revenue.std() / monthly_revenue.mean() if monthly_revenue.mean() > 0 else 0
    print(f"\n📊 Revenue Consistency: {revenue_consistency:.6f}")
    
    # Calculate collection probability
    collection_probability = max(0.5, 1 - revenue_consistency)
    print(f"Original Collection Probability: {collection_probability:.6f}")
    
    # Apply the fix
    collection_probability_fixed = min(collection_probability, 1.0)
    print(f"Fixed Collection Probability: {collection_probability_fixed:.6f}")
    
    # Convert to percentage
    original_percentage = collection_probability * 100
    fixed_percentage = collection_probability_fixed * 100
    
    print(f"\n📊 Results:")
    print(f"Original Collection Probability (%): {original_percentage:.1f}%")
    print(f"Fixed Collection Probability (%): {fixed_percentage:.1f}%")
    
    if original_percentage > 100:
        print(f"❌ This extreme scenario produces {original_percentage:.1f}%")
        print(f"✅ But the fix caps it at {fixed_percentage:.1f}%")
    else:
        print(f"✅ This scenario is normal")
    
    return original_percentage, fixed_percentage

def test_5000_percent_scenario():
    """Test the exact scenario that could cause 5000%"""
    print("\n🔍 TESTING 5000% SCENARIO")
    print("=" * 60)
    
    # Create data that could cause exactly 5000%
    # This would require revenue_consistency = -49 (which is impossible)
    # Let me test what could cause such extreme values
    
    # Test with zero mean (which could cause division by zero issues)
    monthly_revenue = pd.Series([1000000, 1, 1000000, 1, 1000000, 1])
    
    print(f"📊 Test Data: {monthly_revenue.tolist()}")
    print(f"Mean: {monthly_revenue.mean():,.0f}")
    print(f"Std: {monthly_revenue.std():,.0f}")
    
    # Calculate revenue consistency
    revenue_consistency = monthly_revenue.std() / monthly_revenue.mean() if monthly_revenue.mean() > 0 else 0
    print(f"Revenue Consistency: {revenue_consistency:.6f}")
    
    # Calculate collection probability
    collection_probability = max(0.5, 1 - revenue_consistency)
    print(f"Collection Probability: {collection_probability:.6f}")
    
    # Apply fix
    collection_probability_fixed = min(collection_probability, 1.0)
    print(f"Fixed Collection Probability: {collection_probability_fixed:.6f}")
    
    # Convert to percentage
    original_percentage = collection_probability * 100
    fixed_percentage = collection_probability_fixed * 100
    
    print(f"\n📊 Results:")
    print(f"Original: {original_percentage:.1f}%")
    print(f"Fixed: {fixed_percentage:.1f}%")
    
    return original_percentage, fixed_percentage

def check_all_calculation_paths():
    """Check all possible calculation paths"""
    print("\n🔍 CHECKING ALL CALCULATION PATHS")
    print("=" * 60)
    
    # Test different scenarios
    scenarios = [
        ("Normal", [1000, 1100, 1200, 1300, 1400, 1500]),
        ("Volatile", [1000, 100, 1000, 100, 1000, 100]),
        ("Extreme", [1000000, 1, 1000000, 1, 1000000, 1]),
        ("Zero Mean", [0, 1000, 0, 1000, 0, 1000]),
        ("Negative", [-1000, 1000, -1000, 1000, -1000, 1000])
    ]
    
    for scenario_name, data in scenarios:
        print(f"\n📊 {scenario_name}:")
        monthly_revenue = pd.Series(data)
        
        # Calculate as in the system
        revenue_mean = monthly_revenue.mean()
        revenue_std = monthly_revenue.std()
        revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
        collection_probability = max(0.5, 1 - revenue_consistency)
        collection_probability_fixed = min(collection_probability, 1.0)
        
        # Convert to percentage
        original_percentage = collection_probability * 100
        fixed_percentage = collection_probability_fixed * 100
        
        print(f"  Data: {data}")
        print(f"  Mean: {revenue_mean:,.0f}")
        print(f"  Std: {revenue_std:,.0f}")
        print(f"  Consistency: {revenue_consistency:.6f}")
        print(f"  Original: {original_percentage:.1f}%")
        print(f"  Fixed: {fixed_percentage:.1f}%")
        
        if original_percentage > 100:
            print(f"  ❌ This could cause the issue!")
        else:
            print(f"  ✅ This is normal")

def main():
    """Main debug function"""
    print("🚀 DEBUGGING ACTUAL CALCULATION")
    print("=" * 60)
    
    # Test extreme scenario
    test_extreme_scenario()
    
    # Test 5000% scenario
    test_5000_percent_scenario()
    
    # Check all calculation paths
    check_all_calculation_paths()
    
    print("\n🎯 CONCLUSION:")
    print("The 5000% issue might be caused by:")
    print("1. Division by zero or very small numbers")
    print("2. Data with extreme volatility")
    print("3. A different calculation path being used")
    print("4. The fix not being applied to the correct function")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Check if there's a different function being called")
    print("2. Add more robust error handling")
    print("3. Force the collection probability to be capped at 100%")

if __name__ == "__main__":
    main() 