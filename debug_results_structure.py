#!/usr/bin/env python3
"""
Debug Results Structure Script
Check the actual structure of revenue analysis results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Advanced Revenue AI System imported successfully!")
except ImportError as e:
    print(f"❌ Error importing Advanced Revenue AI System: {e}")
    sys.exit(1)

def debug_results_structure():
    """Debug the actual structure of revenue analysis results"""
    print("🔍 DEBUGGING RESULTS STRUCTURE")
    print("=" * 60)
    
    try:
        # Initialize the system
        revenue_ai = AdvancedRevenueAISystem()
        print("✅ Revenue AI System initialized")
        
        # Create test data
        print("📊 Creating test data...")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        revenue_amounts = np.random.normal(50000, 15000, len(dates))
        revenue_amounts = np.abs(revenue_amounts)
        
        test_data = pd.DataFrame({
            'Date': dates,
            'Description': [f'Revenue Transaction {i+1}' for i in range(len(dates))],
            'Amount': revenue_amounts,
            'Type': ['Credit'] * len(dates)
        })
        
        print(f"✅ Created test data with {len(test_data)} transactions")
        
        # Run revenue analysis
        print("\n🧠 Running SMART OLLAMA Revenue Analysis...")
        results = revenue_ai.complete_revenue_analysis_system_smart_ollama(test_data)
        
        # Debug the results structure
        print("\n📊 RESULTS STRUCTURE ANALYSIS")
        print("=" * 60)
        
        print(f"📋 Results type: {type(results)}")
        print(f"📋 Results keys: {list(results.keys())}")
        print(f"📋 Number of components: {len(results)}")
        
        for key, value in results.items():
            print(f"\n🔍 Component: {key}")
            print(f"   Type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"   Keys: {list(value.keys())}")
                print(f"   Number of metrics: {len(value)}")
                
                # Show first few key-value pairs
                for k, v in list(value.items())[:5]:
                    print(f"   {k}: {v}")
                
                if len(value) > 5:
                    print(f"   ... and {len(value) - 5} more metrics")
            else:
                print(f"   Value: {value}")
        
        # Check for specific metrics
        print("\n🔍 LOOKING FOR SPECIFIC METRICS")
        print("=" * 60)
        
        metrics_to_find = [
            'total_revenue', 'monthly_average', 'growth_rate', 'trend_direction',
            'forecast_amount', 'confidence', 'avg_transaction_value', 
            'recurring_revenue_score', 'customer_retention_probability',
            'contract_stability', 'avg_price_point', 'pricing_strategy',
            'price_elasticity', 'revenue_model', 'avg_payment_terms',
            'collection_probability', 'dso_category', 'cash_flow_impact'
        ]
        
        found_metrics = {}
        for metric in metrics_to_find:
            found_in = []
            for component_key, component_data in results.items():
                if isinstance(component_data, dict) and metric in component_data:
                    found_in.append(component_key)
            
            if found_in:
                found_metrics[metric] = found_in
                print(f"✅ {metric}: Found in {found_in}")
            else:
                print(f"❌ {metric}: NOT FOUND")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total metrics to find: {len(metrics_to_find)}")
        print(f"   Found metrics: {len(found_metrics)}")
        print(f"   Missing metrics: {len(metrics_to_find) - len(found_metrics)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in debugging: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 DEBUGGING REVENUE ANALYSIS RESULTS STRUCTURE")
    print("=" * 60)
    
    results = debug_results_structure()
    
    if results:
        print("\n✅ Debugging completed successfully!")
    else:
        print("\n❌ Debugging failed!") 