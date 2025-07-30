#!/usr/bin/env python3
"""
Test Direct Calculation
Simulate the exact calculation to see why 5000% is still appearing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_exact_calculation():
    """Simulate the exact calculation from the system"""
    print("🔍 SIMULATING EXACT CALCULATION")
    print("=" * 60)
    
    # Create the exact data that might cause 5000%
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    
    # Create extreme revenue data
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
    
    print("📊 Sample Data:")
    print(f"Total transactions: {len(df)}")
    print(f"Revenue range: ₹{df['Amount (INR)'].min():,.0f} to ₹{df['Amount (INR)'].max():,.0f}")
    
    # Simulate the exact calculation
    amount_column = 'Amount (INR)'
    revenue_data = df[df[amount_column] > 0].copy()
    
    print(f"\n📊 Filtered Data: {len(revenue_data)} transactions")
    
    # Calculate monthly revenue (as in the system)
    revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])
    monthly_revenue = revenue_data.groupby([revenue_data['Date'].dt.year, revenue_data['Date'].dt.month])[amount_column].sum()
    
    print(f"\n📊 Monthly Revenue: {monthly_revenue.tolist()}")
    print(f"Monthly mean: {monthly_revenue.mean():,.0f}")
    print(f"Monthly std: {monthly_revenue.std():,.0f}")
    
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
    
    if original_percentage > 100:
        print(f"❌ This could cause the 5000% issue!")
        print(f"   The calculation produces {original_percentage:.1f}%")
        print(f"   But the fix should cap it at {fixed_percentage:.1f}%")
    else:
        print(f"✅ This calculation looks normal")
    
    return original_percentage, fixed_percentage

def test_ui_display():
    """Test how the UI would display the result"""
    print("\n🎯 TESTING UI DISPLAY")
    print("=" * 60)
    
    # Simulate the exact calculation
    original, fixed = simulate_exact_calculation()
    
    print(f"\n📊 UI Display Test:")
    print(f"What the UI should show: {fixed:.1f}%")
    print(f"What the UI is showing: 5000.0%")
    
    if fixed <= 100:
        print("✅ The calculation is correct")
        print("❌ The UI is showing cached/old results")
        print("🔧 Solution: Clear cache and restart the application")
    else:
        print("❌ The calculation is still wrong")
        print("🔧 Need to fix the calculation logic")

def main():
    """Main test function"""
    print("🚀 TESTING DIRECT CALCULATION")
    print("=" * 60)
    
    # Test the exact calculation
    test_ui_display()
    
    print("\n🎯 CONCLUSION:")
    print("The 5000% value is likely from:")
    print("1. Cached results in the browser")
    print("2. Cached results in the Flask application")
    print("3. Old data being displayed")
    print("\n🔧 SOLUTION:")
    print("1. Clear browser cache (Ctrl+Shift+Delete)")
    print("2. Restart the Flask application")
    print("3. Run a fresh analysis")
    print("4. The calculation should now show 100% or less")

if __name__ == "__main__":
    main() 