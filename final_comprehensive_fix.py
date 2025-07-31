#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE FIX - Fix XGBoost Training and Training Stratification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_xgboost_training_final():
    """Fix XGBoost training with proper data"""
    print("🔧 FIXING XGBOOST TRAINING FINAL...")
    
    try:
        from app1 import lightweight_ai
        
        # Create comprehensive training data with proper stratification
        training_data = pd.DataFrame({
            'Description': [
                # Investing Activities (3 samples)
                'Infrastructure Development - Warehouse Construction',
                'Equipment Purchase - Rolling Mill Upgrade',
                'Property Acquisition - Office Building',
                
                # Operating Activities (3 samples)
                'VIP Customer Payment - Construction Company',
                'Salary Payment - Employee Payroll',
                'Utility Payment - Electricity Bill',
                
                # Financing Activities (3 samples)
                'Investment Liquidation - Mutual Fund Units',
                'Loan Repayment - Bank Loan',
                'Dividend Payment - Shareholder Return',
                
                # Additional Operating Activities for balance
                'Tax Payment - Income Tax',
                'Insurance Premium - Business Insurance',
                'Marketing Expense - Advertising Campaign',
                'Software License - ERP System',
                'Rent Payment - Office Space',
                'Freight Cost - Shipping Services',
                'Legal Fees - Contract Review',
                'Audit Fees - Annual Audit',
                'Training Cost - Employee Development',
                'Maintenance Cost - Equipment Repair',
                'Commission Payment - Sales Commission',
                'Interest Payment - Loan Interest',
                'Refund Payment - Customer Refund',
                'Deposit Payment - Security Deposit',
                'Subscription Fee - Software Subscription',
                'Consulting Fee - Business Consulting',
                'Travel Expense - Business Travel',
                'Office Supplies - Stationery',
                'Phone Bill - Communication',
                'Internet Bill - Data Services',
                'Cleaning Service - Facility Maintenance',
                'Security Service - Building Security'
            ],
            'Amount': [
                # Investing Activities
                3709289.81, 1500000, 2500000,
                # Operating Activities
                2000000, 800000, 50000,
                # Financing Activities
                500000, 300000, 100000,
                # Additional Operating Activities
                200000, 75000, 120000, 25000, 150000, 45000, 35000,
                50000, 30000, 25000, 60000, 15000, 25000, 50000, 12000, 40000, 15000,
                5000, 8000, 12000, 18000, 22000
            ],
            'Category': [
                # Investing Activities
                'Investing Activities', 'Investing Activities', 'Investing Activities',
                # Operating Activities
                'Operating Activities', 'Operating Activities', 'Operating Activities',
                # Financing Activities
                'Financing Activities', 'Financing Activities', 'Financing Activities',
                # Additional Operating Activities
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(33)],
            'Type': ['Credit'] * 33
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Verify stratification requirements
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 2:
            print("❌ Not enough samples per class for stratification")
            return False
        
        # Train the model
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ XGBoost training successful!")
            print(f"✅ Model trained: {lightweight_ai.is_trained}")
            print(f"✅ Features: {len(lightweight_ai.feature_names)}")
            
            # Test the trained model
            test_cases = [
                ("Infrastructure Development", 1000000),
                ("Customer Payment", 2000000),
                ("Investment Liquidation", 500000)
            ]
            
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"✅ Test: {desc} → {result}")
            
            return True
        else:
            print("❌ XGBoost training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing XGBoost: {e}")
        return False

def fix_training_system_final():
    """Fix training system with proper data handling"""
    print("🔧 FIXING TRAINING SYSTEM FINAL...")
    
    try:
        from app1 import lightweight_ai, advanced_detector
        
        # Test lightweight AI training with proper data
        print("📊 Testing lightweight AI training...")
        
        training_data = pd.DataFrame({
            'Description': [
                'Test Infrastructure 1', 'Test Infrastructure 2',
                'Test Customer Payment 1', 'Test Customer Payment 2',
                'Test Investment 1', 'Test Investment 2'
            ],
            'Amount': [1000000, 1500000, 2000000, 2500000, 500000, 750000],
            'Category': [
                'Investing Activities', 'Investing Activities',
                'Operating Activities', 'Operating Activities',
                'Financing Activities', 'Financing Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(6)],
            'Type': ['Credit'] * 6
        })
        
        # Verify we have at least 2 samples per class
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 2:
            print("❌ Not enough samples per class")
            return False
        
        # Train lightweight AI
        ai_success = lightweight_ai.train_transaction_classifier(training_data)
        print(f"✅ Lightweight AI training: {ai_success}")
        
        # Test anomaly detector training
        print("📊 Testing anomaly detector training...")
        
        test_data = pd.DataFrame({
            'Description': ['Test Transaction 1', 'Test Transaction 2', 'Test Transaction 3'],
            'Amount': [1000, 2000, 3000],
            'Date': [datetime.now(), datetime.now(), datetime.now()],
            'Type': ['Credit', 'Credit', 'Credit']
        })
        
        anomaly_success = advanced_detector.train_models(test_data)
        print(f"✅ Anomaly detector training: {anomaly_success}")
        
        return ai_success and anomaly_success
        
    except Exception as e:
        print(f"❌ Error fixing training system: {e}")
        return False

def test_all_fixes():
    """Test all the final fixes"""
    print("🧪 TESTING ALL FINAL FIXES...")
    print("=" * 50)
    
    results = {}
    
    # Test XGBoost fix
    results['xgboost'] = fix_xgboost_training_final()
    
    # Test Training fix
    results['training'] = fix_training_system_final()
    
    # Summary
    print("\n📊 FINAL FIX RESULTS:")
    print("=" * 30)
    
    total_fixes = len(results)
    successful_fixes = sum(results.values())
    
    for fix, result in results.items():
        status = "✅ FIXED" if result else "❌ FAILED"
        print(f"   {fix.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 OVERALL: {successful_fixes}/{total_fixes} fixes successful")
    
    if successful_fixes == total_fixes:
        print("🎉 ALL SYSTEMS FIXED SUCCESSFULLY!")
        return True
    else:
        print("❌ Some systems still need attention")
        return False

def run_comprehensive_check():
    """Run the comprehensive system check after fixes"""
    print("\n🔍 RUNNING COMPREHENSIVE CHECK AFTER FIXES...")
    print("=" * 60)
    
    try:
        from comprehensive_system_check import comprehensive_system_check
        return comprehensive_system_check()
    except Exception as e:
        print(f"❌ Error running comprehensive check: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FINAL COMPREHENSIVE SYSTEM FIX")
    print("=" * 50)
    
    # Apply all fixes
    fixes_successful = test_all_fixes()
    
    if fixes_successful:
        print("\n✅ Running final comprehensive check...")
        final_check = run_comprehensive_check()
        
        if final_check:
            print("\n🎉 ALL SYSTEMS NOW WORKING PERFECTLY!")
        else:
            print("\n⚠️ Some issues may still remain - check logs")
    else:
        print("\n❌ Some fixes failed - manual intervention may be needed") 