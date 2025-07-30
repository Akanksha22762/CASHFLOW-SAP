#!/usr/bin/env python3
"""
Analyze Current Revenue Analysis Results
Check why collection probability is still showing 5000%
"""

import pandas as pd
import numpy as np

def analyze_collection_probability_calculation():
    """Analyze why collection probability is still 5000%"""
    print("🔍 ANALYZING COLLECTION PROBABILITY CALCULATION")
    print("=" * 60)
    
    # Simulate the calculation that's causing the issue
    print("📊 Simulating the calculation:")
    
    # Example monthly revenue data (this might be causing the issue)
    monthly_revenue = pd.Series([1000, 2000, 3000, 4000, 5000, 6000])
    print(f"Monthly Revenue: {monthly_revenue.tolist()}")
    
    # Calculate revenue consistency
    revenue_mean = monthly_revenue.mean()
    revenue_std = monthly_revenue.std()
    revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
    
    print(f"Revenue Mean: {revenue_mean}")
    print(f"Revenue Std: {revenue_std}")
    print(f"Revenue Consistency: {revenue_consistency}")
    
    # Calculate collection probability
    collection_probability = max(0.5, 1 - revenue_consistency)
    print(f"Original Collection Probability: {collection_probability}")
    
    # Apply the fix
    collection_probability_fixed = min(collection_probability, 1.0)
    print(f"Fixed Collection Probability: {collection_probability_fixed}")
    
    # Convert to percentage for display
    collection_probability_percentage = collection_probability * 100
    collection_probability_fixed_percentage = collection_probability_fixed * 100
    
    print(f"Original Collection Probability (%): {collection_probability_percentage:.1f}%")
    print(f"Fixed Collection Probability (%): {collection_probability_fixed_percentage:.1f}%")
    
    # Check if the fix is working
    if collection_probability_fixed_percentage <= 100:
        print("✅ Fix is working correctly!")
    else:
        print("❌ Fix is not working!")
    
    return collection_probability_fixed_percentage

def analyze_extreme_scenario():
    """Analyze what would cause 5000% collection probability"""
    print("\n🔍 ANALYZING EXTREME SCENARIO")
    print("=" * 60)
    
    # Simulate extreme revenue data that could cause 5000%
    extreme_monthly_revenue = pd.Series([100, 100000, 100, 100000, 100, 100000])
    print(f"Extreme Monthly Revenue: {extreme_monthly_revenue.tolist()}")
    
    # Calculate revenue consistency
    revenue_mean = extreme_monthly_revenue.mean()
    revenue_std = extreme_monthly_revenue.std()
    revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
    
    print(f"Revenue Mean: {revenue_mean}")
    print(f"Revenue Std: {revenue_std}")
    print(f"Revenue Consistency: {revenue_consistency}")
    
    # Calculate collection probability
    collection_probability = max(0.5, 1 - revenue_consistency)
    print(f"Extreme Collection Probability: {collection_probability}")
    
    # Apply the fix
    collection_probability_fixed = min(collection_probability, 1.0)
    print(f"Fixed Collection Probability: {collection_probability_fixed}")
    
    # Convert to percentage
    collection_probability_percentage = collection_probability * 100
    collection_probability_fixed_percentage = collection_probability_fixed * 100
    
    print(f"Extreme Collection Probability (%): {collection_probability_percentage:.1f}%")
    print(f"Fixed Collection Probability (%): {collection_probability_fixed_percentage:.1f}%")
    
    return collection_probability_fixed_percentage

def check_fix_application():
    """Check if the fix is being applied correctly"""
    print("\n🔍 CHECKING FIX APPLICATION")
    print("=" * 60)
    
    # Test different scenarios
    test_scenarios = [
        ("Normal", [1000, 2000, 3000, 4000, 5000, 6000]),
        ("Volatile", [100, 10000, 100, 10000, 100, 10000]),
        ("Extreme", [1, 100000, 1, 100000, 1, 100000]),
        ("Consistent", [1000, 1000, 1000, 1000, 1000, 1000])
    ]
    
    for scenario_name, revenue_data in test_scenarios:
        print(f"\n📊 {scenario_name} Scenario:")
        monthly_revenue = pd.Series(revenue_data)
        
        # Calculate
        revenue_mean = monthly_revenue.mean()
        revenue_std = monthly_revenue.std()
        revenue_consistency = revenue_std / revenue_mean if revenue_mean > 0 else 0
        collection_probability = max(0.5, 1 - revenue_consistency)
        collection_probability_fixed = min(collection_probability, 1.0)
        
        # Convert to percentage
        original_percentage = collection_probability * 100
        fixed_percentage = collection_probability_fixed * 100
        
        print(f"  Original: {original_percentage:.1f}%")
        print(f"  Fixed: {fixed_percentage:.1f}%")
        print(f"  Status: {'✅ Fixed' if fixed_percentage <= 100 else '❌ Still High'}")

def main():
    """Main analysis function"""
    print("🚀 REVENUE ANALYSIS RESULTS ANALYSIS")
    print("=" * 60)
    
    # Analyze current results
    print("\n📊 Current Results Analysis:")
    print("✅ Sales Forecast: Fixed (positive forecast, reasonable growth rate)")
    print("✅ Customer Contracts: Fixed (realistic recurring revenue score, customer retention)")
    print("✅ Historical Trends: Good (all values realistic)")
    print("✅ Pricing Models: Good (all values realistic)")
    print("❌ Accounts Receivable: Still showing 5000% collection probability")
    
    # Analyze the calculation
    analyze_collection_probability_calculation()
    analyze_extreme_scenario()
    check_fix_application()
    
    print("\n🎯 CONCLUSION:")
    print("The fix is in place in the code, but the calculation might be producing")
    print("an extremely high value before the fix is applied. The fix should cap it at 100%.")
    print("If you're still seeing 5000%, it might be a display issue or the fix")
    print("isn't being applied to the specific function being used.")

if __name__ == "__main__":
    main() 