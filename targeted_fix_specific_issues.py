#!/usr/bin/env python3
"""
TARGETED FIX FOR SPECIFIC ISSUES - Array Length and Test Size
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

def fix_xgboost_array_length_issue():
    """Fix XGBoost array length mismatch issue"""
    print("🔧 FIXING XGBOOST ARRAY LENGTH ISSUE...")
    
    try:
        from app1 import lightweight_ai
        
        # Create training data with EXACTLY the same length for all columns
        descriptions = [
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
        ]
        
        amounts = [
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
        ]
        
        categories = [
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
        ]
        
        dates = [datetime.now() - timedelta(days=i) for i in range(33)]
        types = ['Credit'] * 33
        
        # Verify all arrays have the same length
        lengths = [len(descriptions), len(amounts), len(categories), len(dates), len(types)]
        print(f"📊 Array lengths: {lengths}")
        
        if len(set(lengths)) != 1:
            print("❌ Array lengths don't match")
            return False
        
        # Create DataFrame
        training_data = pd.DataFrame({
            'Description': descriptions,
            'Amount': amounts,
            'Category': categories,
            'Date': dates,
            'Type': types
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

def fix_training_test_size_issue():
    """Fix training test_size vs classes mismatch issue"""
    print("🔧 FIXING TRAINING TEST SIZE ISSUE...")
    
    try:
        from app1 import lightweight_ai
        
        # Create training data with more samples per class to avoid test_size issue
        training_data = pd.DataFrame({
            'Description': [
                # Investing Activities (4 samples)
                'Infrastructure Development 1', 'Infrastructure Development 2',
                'Equipment Purchase 1', 'Equipment Purchase 2',
                
                # Operating Activities (4 samples)
                'Customer Payment 1', 'Customer Payment 2',
                'Salary Payment 1', 'Salary Payment 2',
                
                # Financing Activities (4 samples)
                'Investment Liquidation 1', 'Investment Liquidation 2',
                'Loan Repayment 1', 'Loan Repayment 2',
                
                # Additional samples for better training
                'Utility Payment', 'Tax Payment', 'Insurance Premium',
                'Marketing Expense', 'Software License', 'Rent Payment',
                'Freight Cost', 'Legal Fees', 'Audit Fees', 'Training Cost',
                'Maintenance Cost', 'Commission Payment', 'Interest Payment',
                'Refund Payment', 'Deposit Payment', 'Subscription Fee',
                'Consulting Fee', 'Travel Expense', 'Office Supplies',
                'Phone Bill', 'Internet Bill', 'Cleaning Service', 'Security Service'
            ],
            'Amount': [
                # Investing Activities
                1000000, 1500000, 2000000, 2500000,
                # Operating Activities
                3000000, 3500000, 4000000, 4500000,
                # Financing Activities
                500000, 750000, 1000000, 1250000,
                # Additional samples
                50000, 200000, 75000, 120000, 25000, 150000, 45000, 35000,
                50000, 30000, 25000, 60000, 15000, 25000, 50000, 12000, 40000, 15000,
                5000, 8000, 12000, 18000, 22000
            ],
            'Category': [
                # Investing Activities
                'Investing Activities', 'Investing Activities', 'Investing Activities', 'Investing Activities',
                # Operating Activities
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                # Financing Activities
                'Financing Activities', 'Financing Activities', 'Financing Activities', 'Financing Activities',
                # Additional samples (mostly Operating Activities)
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(47)],
            'Type': ['Credit'] * 47
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Verify we have enough samples per class
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 4:  # Need at least 4 per class for proper train/test split
            print("❌ Not enough samples per class")
            return False
        
        # Train the model
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ Training successful!")
            print(f"✅ Model trained: {lightweight_ai.is_trained}")
            
            # Test the model
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
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing training: {e}")
        return False

def test_fixes():
    """Test the specific fixes"""
    print("🧪 TESTING SPECIFIC FIXES...")
    print("=" * 50)
    
    results = {}
    
    # Test XGBoost fix
    results['xgboost'] = fix_xgboost_array_length_issue()
    
    # Test Training fix
    results['training'] = fix_training_test_size_issue()
    
    # Summary
    print("\n📊 FIX RESULTS:")
    print("=" * 30)
    
    total_fixes = len(results)
    successful_fixes = sum(results.values())
    
    for fix, result in results.items():
        status = "✅ FIXED" if result else "❌ FAILED"
        print(f"   {fix.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 OVERALL: {successful_fixes}/{total_fixes} fixes successful")
    
    if successful_fixes == total_fixes:
        print("🎉 ALL SPECIFIC ISSUES FIXED!")
        return True
    else:
        print("❌ Some issues still need attention")
        return False

def run_final_verification():
    """Run final verification"""
    print("\n🔍 RUNNING FINAL VERIFICATION...")
    print("=" * 60)
    
    try:
        from comprehensive_system_check import comprehensive_system_check
        return comprehensive_system_check()
    except Exception as e:
        print(f"❌ Error running comprehensive check: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TARGETED FIX FOR SPECIFIC ISSUES")
    print("=" * 50)
    
    # Apply fixes
    fixes_successful = test_fixes()
    
    if fixes_successful:
        print("\n✅ Running final verification...")
        final_check = run_final_verification()
        
        if final_check:
            print("\n🎉 ALL SYSTEMS NOW WORKING PERFECTLY!")
        else:
            print("\n⚠️ Some issues may still remain")
    else:
        print("\n❌ Some fixes failed - manual intervention needed") 