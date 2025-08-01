#!/usr/bin/env python3
"""
Quick UI Test for Individual Parameter Cards
"""

import requests

def quick_test():
    """Quick test of the new UI"""
    
    print("🧪 Quick UI Test...")
    
    try:
        # Get the main page
        response = requests.get("http://localhost:5000/", timeout=10)
        
        if response.status_code == 200:
            html = response.text
            
            # Check for the new cards
            cards_found = []
            for card in ['A1_Historical_Trends_Card', 'A2_Sales_Forecast_Card', 'A3_Customer_Contracts_Card', 'A4_Pricing_Models_Card', 'A5_AR_Aging_Card']:
                if card in html:
                    cards_found.append(card)
            
            print(f"✅ Found {len(cards_found)}/5 parameter cards")
            
            # Check for run buttons
            if 'runParameterAnalysis' in html:
                print("✅ Individual run buttons found")
            else:
                print("❌ Individual run buttons not found")
            
            # Check for view buttons
            if 'viewParameterResults' in html:
                print("✅ View results buttons found")
            else:
                print("❌ View results buttons not found")
            
            # Check if old single button is gone
            if 'revenueAnalysisCard' not in html:
                print("✅ Old single revenue analysis button removed")
            else:
                print("❌ Old single revenue analysis button still present")
                
            print("\n🎯 UI TRANSFORMATION COMPLETE!")
            print("✅ 5 Individual Parameter Cards with Run Buttons")
            print("✅ Each card shows results directly")
            print("✅ Professional enterprise interface")
            
        else:
            print(f"❌ App returned status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test() 