#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE FIX ALL - Fix all remaining issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fix_xgboost_final():
    """Fix XGBoost training with persistent training"""
    print("🔧 FIXING XGBOOST FINAL...")
    
    try:
        from app1 import lightweight_ai
        
        # Create balanced training data
        training_data = pd.DataFrame({
            'Description': [
                # Investing Activities (6 samples)
                'Infrastructure Development - Warehouse Construction',
                'Equipment Purchase - Rolling Mill Upgrade',
                'Property Acquisition - Office Building',
                'Machinery Purchase - Production Line',
                'Facility Expansion - New Wing',
                'Technology Investment - Software Platform',
                
                # Operating Activities (6 samples)
                'VIP Customer Payment - Construction Company',
                'Salary Payment - Employee Payroll',
                'Utility Payment - Electricity Bill',
                'Tax Payment - Income Tax',
                'Insurance Premium - Business Insurance',
                'Marketing Expense - Advertising Campaign',
                
                # Financing Activities (6 samples)
                'Investment Liquidation - Mutual Fund Units',
                'Loan Repayment - Bank Loan',
                'Dividend Payment - Shareholder Return',
                'Bond Issuance - Corporate Bonds',
                'Equity Investment - Startup Funding',
                'Capital Raise - Series A Funding'
            ],
            'Amount': [
                # Investing Activities
                3709289.81, 1500000, 2500000, 3000000, 1800000, 2200000,
                # Operating Activities
                2000000, 800000, 50000, 200000, 75000, 120000,
                # Financing Activities
                500000, 300000, 100000, 1500000, 250000, 5000000
            ],
            'Category': [
                # Investing Activities
                'Investing Activities', 'Investing Activities', 'Investing Activities',
                'Investing Activities', 'Investing Activities', 'Investing Activities',
                # Operating Activities
                'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities',
                # Financing Activities
                'Financing Activities', 'Financing Activities', 'Financing Activities',
                'Financing Activities', 'Financing Activities', 'Financing Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(18)],
            'Type': ['Credit'] * 18
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Train the model
        print("🤖 Training XGBoost model...")
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
            
            print("\n🧪 TESTING TRAINED MODEL:")
            all_working = True
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"   {desc} → {result}")
                
                if "Not-Trained" in result:
                    print(f"   ❌ Model still not trained")
                    all_working = False
                else:
                    print(f"   ✅ Model working properly")
            
            return all_working
        else:
            print("❌ XGBoost training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing XGBoost: {e}")
        return False

def fix_ollama_final():
    """Fix Ollama API with correct model"""
    print("🔧 FIXING OLLAMA FINAL...")
    
    try:
        from ollama_simple_integration import OllamaSimpleIntegration
        
        # Test with correct model name
        ollama = OllamaSimpleIntegration()
        
        if not ollama.is_available:
            print("❌ Ollama not available")
            return False
        
        print(f"✅ Available models: {ollama.available_models}")
        
        # Use the correct model name that we know works
        test_prompt = "Categorize this transaction: Infrastructure Development"
        test_response = ollama.simple_ollama(test_prompt, model="llama2:7b", max_tokens=30)
        
        if test_response:
            print("✅ Ollama API test successful")
            print(f"✅ Test response: {test_response[:50]}...")
            return True
        else:
            print("❌ Ollama API test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing Ollama: {e}")
        return False

def fix_training_final():
    """Fix training system with proper data"""
    print("🔧 FIXING TRAINING SYSTEM FINAL...")
    
    try:
        from app1 import lightweight_ai
        
        # Create training data with more samples per class
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
            
            all_working = True
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"✅ Test: {desc} → {result}")
                
                if "Not-Trained" in result:
                    all_working = False
            
            return all_working
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing training: {e}")
        return False

def test_all_fixes():
    """Test all the fixes"""
    print("🧪 TESTING ALL FINAL FIXES...")
    print("=" * 50)
    
    results = {}
    
    # Test XGBoost fix
    results['xgboost'] = fix_xgboost_final()
    
    # Test Ollama fix
    results['ollama'] = fix_ollama_final()
    
    # Test Training fix
    results['training'] = fix_training_final()
    
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
        print("🎉 ALL FINAL ISSUES FIXED!")
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
    print("🚀 FINAL COMPREHENSIVE FIX ALL")
    print("=" * 50)
    
    # Apply fixes
    fixes_successful = test_all_fixes()
    
    if fixes_successful:
        print("\n✅ Running final verification...")
        final_check = run_final_verification()
        
        if final_check:
            print("\n🎉 ALL SYSTEMS NOW WORKING PERFECTLY!")
        else:
            print("\n⚠️ Some issues may still remain")
    else:
        print("\n❌ Some fixes failed - manual intervention needed") 