import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_current_data_assets():
    """Analyze all current data assets excluding transactions_data.csv and ai_reference files"""
    
    print("🔍 COMPREHENSIVE DATA ASSETS ANALYSIS")
    print("=" * 60)
    
    current_data = {}
    
    try:
        # Load Bank Statement Data
        bank_df = pd.read_excel('steel_plant_bank_statement.xlsx')
        current_data['bank_statement'] = {
            'records': len(bank_df),
            'columns': list(bank_df.columns),
            'date_range': "Available" if 'Date' in bank_df.columns else "N/A",
            'sample_data': bank_df.head(3).to_dict('records')
        }
        print(f"✅ Bank Statement: {len(bank_df)} records")
        print(f"   Columns: {list(bank_df.columns)}")
        
        # Load SAP Data (AP/AR)
        sap_ap_ar_df = pd.read_excel('steel_plant_ap_ar_data.xlsx')
        current_data['sap_ap_ar'] = {
            'records': len(sap_ap_ar_df),
            'columns': list(sap_ap_ar_df.columns),
            'date_range': "Available" if 'Date' in sap_ap_ar_df.columns else "N/A",
            'sample_data': sap_ap_ar_df.head(3).to_dict('records')
        }
        print(f"✅ SAP AP/AR Data: {len(sap_ap_ar_df)} records")
        print(f"   Columns: {list(sap_ap_ar_df.columns)}")
        
        # Load Master Data
        master_df = pd.read_excel('steel_plant_master_data.xlsx')
        current_data['master_data'] = {
            'records': len(master_df),
            'columns': list(master_df.columns),
            'sample_data': master_df.head(3).to_dict('records')
        }
        print(f"✅ Master Data: {len(master_df)} records")
        print(f"   Columns: {list(master_df.columns)}")
        
        # Load Steel Plant Transactions
        steel_transactions_df = pd.read_excel('steel_plant_transactions.xlsx')
        current_data['steel_transactions'] = {
            'records': len(steel_transactions_df),
            'columns': list(steel_transactions_df.columns),
            'date_range': "Available" if 'Date' in steel_transactions_df.columns else "N/A",
            'sample_data': steel_transactions_df.head(3).to_dict('records')
        }
        print(f"✅ Steel Plant Transactions: {len(steel_transactions_df)} records")
        print(f"   Columns: {list(steel_transactions_df.columns)}")
        
        # Load JSW Steel Dataset
        jsw_df = pd.read_excel('JSW_Steel_Cash_Flow_Dataset.xlsx')
        current_data['jsw_steel_dataset'] = {
            'records': len(jsw_df),
            'columns': list(jsw_df.columns),
            'sample_data': jsw_df.head(3).to_dict('records')
        }
        print(f"✅ JSW Steel Dataset: {len(jsw_df)} records")
        print(f"   Columns: {list(jsw_df.columns)}")
        
        # Load Steel Plant Data PDF (metadata only)
        current_data['steel_plant_pdf'] = {
            'file_size': '578KB',
            'pages': '4220 lines',
            'type': 'PDF document'
        }
        print(f"✅ Steel Plant Data PDF: 578KB, 4220 lines")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None
    
    return current_data

def define_required_data_from_document():
    """Define all data requirements from the AI Nurturing Parameters document"""
    
    print("\n📋 REQUIRED DATA FROM AI MODEL DOCUMENT")
    print("=" * 60)
    
    required_data = {
        "Revenue-Related Parameters": {
            "Revenue forecasts": {
                "description": "Expected income from sales, broken down by product, geography, and customer segment",
                "data_sources_needed": [
                    "Sales forecast data",
                    "Product-wise sales data",
                    "Geographic sales breakdown",
                    "Customer segment data"
                ],
                "current_availability": "Partial"
            },
            "Customer payment terms": {
                "description": "Typical days sales outstanding (DSO), average payment delays",
                "data_sources_needed": [
                    "Customer payment history",
                    "Invoice dates vs payment dates",
                    "Customer credit terms",
                    "Payment delay patterns"
                ],
                "current_availability": "Partial"
            },
            "Accounts receivable aging": {
                "description": "Breakdown of receivables into current, 30-60-90+ day buckets",
                "data_sources_needed": [
                    "AR aging reports",
                    "Invoice aging data",
                    "Collection probability data"
                ],
                "current_availability": "Available"
            },
            "Sales pipeline & backlog": {
                "description": "Expected future revenues from open opportunities and signed contracts",
                "data_sources_needed": [
                    "CRM pipeline data",
                    "Sales opportunity data",
                    "Contract backlog data",
                    "Deal stage information"
                ],
                "current_availability": "Missing"
            },
            "Seasonality factors": {
                "description": "Historical revenue fluctuations due to seasonality",
                "data_sources_needed": [
                    "Historical monthly/quarterly sales data",
                    "Seasonal trend analysis",
                    "Industry seasonal patterns"
                ],
                "current_availability": "Missing"
            }
        },
        
        "Expense-Related Parameters": {
            "Operating expenses (OPEX)": {
                "description": "Fixed and variable costs, such as rent, salaries, utilities, etc.",
                "data_sources_needed": [
                    "Detailed expense breakdown",
                    "Fixed vs variable cost classification",
                    "Department-wise expenses",
                    "Cost center data"
                ],
                "current_availability": "Partial"
            },
            "Accounts payable terms": {
                "description": "Days payable outstanding (DPO), payment cycles to vendors",
                "data_sources_needed": [
                    "Vendor payment history",
                    "Payment terms by vendor",
                    "DPO calculations",
                    "Vendor credit terms"
                ],
                "current_availability": "Available"
            },
            "Inventory turnover": {
                "description": "Cash locked in inventory, including procurement and storage cycles",
                "data_sources_needed": [
                    "Inventory levels data",
                    "Turnover ratios",
                    "Procurement cycles",
                    "Storage costs"
                ],
                "current_availability": "Missing"
            },
            "Loan repayments": {
                "description": "Principal and interest payments due over the projection period",
                "data_sources_needed": [
                    "Loan schedules",
                    "Interest rates",
                    "Payment due dates",
                    "Loan terms"
                ],
                "current_availability": "Partial"
            },
            "Tax obligations": {
                "description": "Upcoming GST, VAT, income tax, or other regulatory payments",
                "data_sources_needed": [
                    "Tax calendar",
                    "Tax rates",
                    "Payment schedules",
                    "Compliance requirements"
                ],
                "current_availability": "Missing"
            }
        },
        
        "Cash Inflows & Outflows": {
            "Cash inflow types": {
                "description": "Customer payments, loans, investor funding, asset sales",
                "data_sources_needed": [
                    "Cash inflow categorization",
                    "Payment source tracking",
                    "Funding sources"
                ],
                "current_availability": "Available"
            },
            "Cash outflow types": {
                "description": "Payroll, vendors, tax, interest, dividends, repayments",
                "data_sources_needed": [
                    "Cash outflow categorization",
                    "Payment destination tracking",
                    "Expense classification"
                ],
                "current_availability": "Available"
            },
            "Payment frequency & timing": {
                "description": "Weekly/monthly/quarterly cycles, lags",
                "data_sources_needed": [
                    "Payment timing patterns",
                    "Frequency analysis",
                    "Lag calculations"
                ],
                "current_availability": "Available"
            }
        },
        
        "Operational & Business Drivers": {
            "Inventory turnover": {
                "description": "Cash locked in inventory and replenishment cycles",
                "data_sources_needed": [
                    "Inventory data",
                    "Turnover metrics",
                    "Replenishment cycles"
                ],
                "current_availability": "Missing"
            },
            "Headcount plans": {
                "description": "Hiring/firing impact on payroll and benefits",
                "data_sources_needed": [
                    "HR planning data",
                    "Hiring forecasts",
                    "Salary projections",
                    "Benefits data"
                ],
                "current_availability": "Missing"
            },
            "Expansion plans": {
                "description": "New markets, products, facilities, partnerships",
                "data_sources_needed": [
                    "Strategic planning data",
                    "Market expansion plans",
                    "Product development roadmaps",
                    "Partnership agreements"
                ],
                "current_availability": "Missing"
            },
            "Marketing spend and ROI": {
                "description": "Influences lead generation and revenue growth",
                "data_sources_needed": [
                    "Marketing budget data",
                    "Campaign performance",
                    "ROI metrics",
                    "Lead generation data"
                ],
                "current_availability": "Missing"
            }
        },
        
        "External Economic Variables": {
            "Interest rates": {
                "description": "Affects loan repayments and future borrowings",
                "data_sources_needed": [
                    "Market interest rates",
                    "Central bank rates",
                    "Loan rate forecasts"
                ],
                "current_availability": "Missing"
            },
            "Inflation": {
                "description": "Influences pricing, costs, and real cash value",
                "data_sources_needed": [
                    "Inflation rate data",
                    "Price indices",
                    "Cost escalation factors"
                ],
                "current_availability": "Missing"
            },
            "Exchange rates": {
                "description": "For multinational or export-driven businesses",
                "data_sources_needed": [
                    "Currency exchange rates",
                    "Forex forecasts",
                    "International transaction data"
                ],
                "current_availability": "Missing"
            },
            "Tax rates and policies": {
                "description": "VAT, GST, income tax changes or rebates",
                "data_sources_needed": [
                    "Tax policy changes",
                    "Rate updates",
                    "Compliance requirements"
                ],
                "current_availability": "Missing"
            }
        },
        
        "Enhanced Data Inputs": {
            "Bank transaction feeds": {
                "description": "Real-time bank statements (API integration) for actual cash positions",
                "data_sources_needed": [
                    "Bank API integration",
                    "Real-time transaction feeds",
                    "Account balance data"
                ],
                "current_availability": "Available"
            },
            "Invoice-level granularity": {
                "description": "Aging, status, expected collection date, client payment behavior",
                "data_sources_needed": [
                    "Detailed invoice data",
                    "Collection tracking",
                    "Payment behavior analysis"
                ],
                "current_availability": "Partial"
            },
            "CRM integration": {
                "description": "Deal stage, expected close dates, probability of win",
                "data_sources_needed": [
                    "CRM system data",
                    "Sales pipeline",
                    "Deal tracking"
                ],
                "current_availability": "Missing"
            },
            "ERP & accounting systems": {
                "description": "Real-time GL feeds, journal entries, budget vs actual",
                "data_sources_needed": [
                    "ERP system integration",
                    "General ledger data",
                    "Budget vs actual reports"
                ],
                "current_availability": "Partial"
            },
            "Operational metrics": {
                "description": "Production output, delivery lead times, procurement delays",
                "data_sources_needed": [
                    "Production data",
                    "Supply chain metrics",
                    "Operational KPIs"
                ],
                "current_availability": "Missing"
            }
        }
    }
    
    return required_data

def generate_data_collection_plan(current_data, required_data):
    """Generate comprehensive data collection plan"""
    
    print("\n📋 COMPREHENSIVE DATA COLLECTION PLAN")
    print("=" * 60)
    
    print("\n🎯 PRIORITY 1: CRITICAL MISSING DATA")
    print("-" * 40)
    
    critical_missing = [
        "Inventory turnover data",
        "Sales pipeline & CRM data", 
        "Tax obligation schedules",
        "Seasonal trend data",
        "Headcount planning data",
        "Marketing spend & ROI data",
        "External economic indicators"
    ]
    
    for item in critical_missing:
        print(f"❌ {item}")
    
    print("\n📊 DATA SOURCES TO COLLECT:")
    print("-" * 30)
    
    data_sources = {
        "Internal Systems": [
            "Inventory Management System",
            "CRM System (Salesforce, etc.)",
            "HR/Payroll System",
            "Marketing Automation Platform",
            "Production/Manufacturing System",
            "Supply Chain Management System"
        ],
        "External Sources": [
            "Central Bank Interest Rate Data",
            "Inflation Rate APIs",
            "Currency Exchange Rate APIs",
            "Industry Seasonal Data",
            "Tax Rate Databases",
            "Economic Indicator APIs"
        ],
        "Manual Collection": [
            "Strategic Planning Documents",
            "Budget Forecasts",
            "Expansion Plans",
            "Partnership Agreements",
            "Regulatory Compliance Data"
        ]
    }
    
    for category, sources in data_sources.items():
        print(f"\n🔧 {category}:")
        for source in sources:
            print(f"   • {source}")
    
    print("\n📈 IMPLEMENTATION TIMELINE:")
    print("-" * 30)
    print("Week 1-2: Inventory & CRM data collection")
    print("Week 3-4: HR & Marketing data integration")
    print("Week 5-6: External economic data APIs")
    print("Week 7-8: Strategic planning data compilation")
    print("Week 9-10: Data validation & quality checks")
    print("Week 11-12: AI model development & testing")

def main():
    """Main analysis function"""
    
    # Analyze current data
    current_data = analyze_current_data_assets()
    if not current_data:
        return
    
    # Define required data from document
    required_data = define_required_data_from_document()
    
    # Generate collection plan
    generate_data_collection_plan(current_data, required_data)
    
    # Save detailed analysis
    analysis_report = {
        "current_data_assets": current_data,
        "required_data": required_data,
        "analysis_date": datetime.now().isoformat()
    }
    
    with open('data_collection_analysis.json', 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: data_collection_analysis.json")

if __name__ == "__main__":
    main() 