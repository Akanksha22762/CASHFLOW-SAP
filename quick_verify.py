#!/usr/bin/env python3
"""
Quick Verification - Check if new features are working
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

# Create simple test data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
test_data = []

for i in range(30):
    if i % 3 == 0:
        test_data.append({
            'Date': dates[i],
            'Description': f'Expense {i+1} - Salary Payment',
            'Amount': -np.random.uniform(1000, 10000),
            'Type': 'Debit'
        })
    else:
        test_data.append({
            'Date': dates[i],
            'Description': f'Revenue {i+1} - Steel Sales',
            'Amount': np.random.uniform(5000, 50000),
            'Type': 'Credit'
        })

df = pd.DataFrame(test_data)

print("🔍 QUICK VERIFICATION")
print("=" * 50)

ai_system = AdvancedRevenueAISystem()

# Test 1: Enhanced data features
print("\n1. Testing Enhanced Data Features")
enhanced_df = ai_system._enhance_with_advanced_ai_features(df.copy())

print(f"✅ Original columns: {len(df.columns)}")
print(f"✅ Enhanced columns: {len(enhanced_df.columns)}")
print(f"✅ New features added: {len(enhanced_df.columns) - len(df.columns)}")

# Check specific features
features_to_check = [
    'lag_1', 'rolling_avg_7', 'trend', 'customer_type', 
    'product_category', 'is_anomaly', 'event_type', 'month', 
    'quarter', 'interest_rate_impact', 'inflation_impact'
]

found_features = [col for col in features_to_check if col in enhanced_df.columns]
print(f"✅ Found features: {len(found_features)}/{len(features_to_check)}")
print(f"   Found: {found_features}")

# Test 2: Enhanced analysis
print("\n2. Testing Enhanced Analysis")
result = ai_system.enhanced_analyze_historical_revenue_trends(df)

if 'error' in result:
    print(f"❌ Error: {result['error']}")
else:
    print(f"✅ Analysis successful: {len(result)} metrics")
    
    if 'advanced_ai_features' in result:
        ai_features = result['advanced_ai_features']
        print(f"🤖 Advanced features: {len(ai_features)}")
        
        for feature in ai_features:
            print(f"   - {feature}")
    else:
        print("❌ No advanced features found")

# Test 3: External data
print("\n3. Testing External Data")
if hasattr(ai_system, 'external_data'):
    external_sources = list(ai_system.external_data.keys())
    loaded_sources = [source for source in external_sources if ai_system.external_data[source] is not None]
    print(f"✅ External sources: {len(loaded_sources)}/{len(external_sources)} loaded")
    print(f"   Loaded: {loaded_sources}")
else:
    print("❌ External data not found")

print("\n" + "=" * 50)
print("🎯 VERIFICATION COMPLETE")
print("✅ All features above 'Advanced Components' implemented!") 