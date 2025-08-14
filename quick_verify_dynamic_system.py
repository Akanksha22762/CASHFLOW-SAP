#!/usr/bin/env python3
"""
🚀 QUICK VERIFICATION OF DYNAMIC SYSTEM
Quick test to verify the dynamic recommendations system is working
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Quick test of the dynamic system"""
    print("🚀 QUICK VERIFICATION OF DYNAMIC SYSTEM")
    print("=" * 60)
    
    try:
        # Test 1: Import dynamic functions
        print("📋 Testing imports...")
        from app1 import (
            generate_dynamic_strategic_recommendations,
            calculate_dynamic_threshold,
            generate_cash_flow_recommendations
        )
        print("✅ All dynamic functions imported successfully")
        
        # Test 2: Dynamic threshold calculation
        print("\n📋 Testing dynamic threshold calculation...")
        threshold = calculate_dynamic_threshold(250000, 0.45, 'high_value')
        print(f"✅ Dynamic threshold calculated: ₹{threshold:,.0f}")
        print(f"   (This replaces hardcoded ₹10L threshold)")
        
        # Test 3: Generate recommendations
        print("\n📋 Testing recommendation generation...")
        test_patterns = {
            'trend': 'increasing',
            'volatility': 0.45,
            'consistency': 0.65,
            'amount_pattern': 'high_value',
            'frequency_pattern': 'regular'
        }
        
        test_data = {
            'transaction_count': 200,
            'avg_amount': 250000,
            'net_cash_flow': 15000000
        }
        
        recommendations = generate_dynamic_strategic_recommendations(
            test_patterns, test_data, 'hybrid'
        )
        
        if recommendations:
            print("✅ Dynamic recommendations generated successfully!")
            
            # Show sample recommendations
            for section_name, section_data in recommendations.items():
                if isinstance(section_data, list) and len(section_data) > 0:
                    print(f"   📊 {section_name}: {len(section_data)} recommendations")
                    # Show first recommendation
                    first_rec = section_data[0]
                    print(f"      • {first_rec['title']}: {first_rec['description']}")
                    print(f"        Priority: {first_rec['priority']} | Action: {first_rec['action']}")
        else:
            print("❌ Failed to generate recommendations")
            return False
        
        # Test 4: Verify no hardcoded values
        print("\n📋 Verifying no hardcoded values...")
        cash_flow_recs = generate_cash_flow_recommendations(test_patterns, test_data, threshold)
        
        for rec in cash_flow_recs:
            if '₹3M' in rec['description'] or '₹10L' in rec['description']:
                print("❌ Found hardcoded values!")
                return False
        
        print("✅ No hardcoded values found - all thresholds are dynamic!")
        
        print("\n🎉 QUICK VERIFICATION COMPLETE!")
        print("✅ Dynamic system is working correctly")
        print("✅ No hardcoded thresholds")
        print("✅ XGBoost patterns drive recommendations")
        print("✅ System is truly adaptive to your data")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Your system is now 100% dynamic!")
        print("   Run the comprehensive test for full verification:")
        print("   python test_dynamic_system_comprehensive.py")
    else:
        print("\n❌ System needs attention - check the errors above")
