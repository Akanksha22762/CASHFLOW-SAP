#!/usr/bin/env python3
"""
Test XGBoost training and fix issues
"""

def test_xgboost_training():
    """Test and fix XGBoost training issues"""
    print("🔧 TESTING AND FIXING XGBOOST TRAINING")
    print("=" * 50)
    
    try:
        # Import systems
        from app1 import lightweight_ai, hybrid_categorize_transaction
        from advanced_revenue_ai_system import AdvancedRevenueAISystem
        
        print("✅ Systems imported successfully")
        
        # Check XGBoost training status
        print(f"\n📊 XGBoost Training Status:")
        print(f"   Is trained: {lightweight_ai.is_trained}")
        print(f"   Models available: {list(lightweight_ai.models.keys())}")
        
        # Test XGBoost categorization
        print(f"\n🧪 Testing XGBoost categorization:")
        test_desc = "Infrastructure Development - Warehouse Construction"
        test_amount = 1000000
        
        try:
            xgb_result = lightweight_ai.categorize_transaction_ml(test_desc, test_amount)
            print(f"   XGBoost result: {xgb_result}")
        except Exception as e:
            print(f"   XGBoost error: {e}")
        
        # Test hybrid categorization
        print(f"\n🧪 Testing hybrid categorization:")
        try:
            hybrid_result = hybrid_categorize_transaction(test_desc, test_amount)
            print(f"   Hybrid result: {hybrid_result}")
        except Exception as e:
            print(f"   Hybrid error: {e}")
        
        # If XGBoost is not working, retrain it
        if not lightweight_ai.is_trained or "Error" in xgb_result or "Not-Trained" in xgb_result:
            print(f"\n🔄 Retraining XGBoost models...")
            
            # Create training data
            import pandas as pd
            from datetime import datetime
            
            training_data = pd.DataFrame({
                'Description': [
                    'Infrastructure Development - Warehouse Construction',
                    'VIP Customer Payment - Construction Company',
                    'Investment Liquidation - Mutual Fund Units',
                    'Equipment Purchase - Rolling Mill Upgrade',
                    'Salary Payment - Employee Payroll',
                    'Loan EMI Payment - Principal + Interest'
                ],
                'Amount': [1000000, 2000000, 500000, 1500000, 800000, 1200000],
                'Category': [
                    'Investing Activities',
                    'Operating Activities', 
                    'Financing Activities',
                    'Investing Activities',
                    'Operating Activities',
                    'Financing Activities'
                ],
                'Date': [datetime.now() - pd.Timedelta(days=i) for i in range(6)],
                'Type': ['Credit'] * 6
            })
            
            # Train the model
            success = lightweight_ai.train_transaction_classifier(training_data)
            print(f"   Training success: {success}")
            
            # Test again
            try:
                xgb_result = lightweight_ai.categorize_transaction_ml(test_desc, test_amount)
                print(f"   XGBoost result after retraining: {xgb_result}")
            except Exception as e:
                print(f"   XGBoost error after retraining: {e}")
        
        print(f"\n✅ XGBoost testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_xgboost_training() 