#!/usr/bin/env python3
"""
Test script for category-based transaction analysis
Tests the new /get-transaction-details endpoint with category filtering
"""

import requests
import json

def test_category_analysis():
    """Test the category-based transaction analysis"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Category-Based Transaction Analysis")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        "Investing Activities",
        "Operating Activities", 
        "Financing Activities"
    ]
    
    for category in test_categories:
        print(f"\n📊 Testing Category: {category}")
        print("-" * 30)
        
        # Test the transaction details endpoint
        try:
            response = requests.post(
                f"{base_url}/get-transaction-details",
                json={
                    "category_type": category,
                    "analysis_type": "category_analysis"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    transactions = data.get('transactions', [])
                    print(f"✅ Success! Found {len(transactions)} transactions")
                    
                    # Show first few transactions
                    for i, t in enumerate(transactions[:3]):
                        print(f"  {i+1}. {t.get('date')} - {t.get('description')[:50]}...")
                        print(f"     Amount: ₹{t.get('amount'):,.2f}")
                        print(f"     Category: {t.get('category')}")
                        print(f"     Vendor: {t.get('vendor')}")
                    
                    if len(transactions) > 3:
                        print(f"  ... and {len(transactions) - 3} more transactions")
                        
                else:
                    print(f"❌ API returned success: false")
                    print(f"   Error: {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Category Analysis Testing Complete!")

if __name__ == "__main__":
    test_category_analysis()
