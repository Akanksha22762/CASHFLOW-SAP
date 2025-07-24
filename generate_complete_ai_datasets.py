import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

def generate_crm_sales_pipeline_data():
    """Generate CRM and Sales Pipeline Data"""
    print("Generating CRM Sales Pipeline Data...")
    
    # Sample data for sales pipeline
    deal_stages = ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
    customer_tiers = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
    products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India']
    
    data = []
    for i in range(450):
        deal_value = random.uniform(50000, 5000000)
        probability = random.choice([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        stage = random.choice(deal_stages)
        
        # Expected close date based on stage
        if stage in ['Prospecting', 'Qualification']:
            close_date = datetime.now() + timedelta(days=random.randint(30, 180))
        elif stage in ['Proposal', 'Negotiation']:
            close_date = datetime.now() + timedelta(days=random.randint(7, 60))
        else:
            close_date = datetime.now() + timedelta(days=random.randint(-30, 30))
        
        data.append({
            'Deal_ID': f'DEAL-{i+1:04d}',
            'Customer_Name': f'Customer_{i+1}',
            'Customer_Tier': random.choice(customer_tiers),
            'Product_Category': random.choice(products),
            'Region': random.choice(regions),
            'Deal_Stage': stage,
            'Deal_Value': round(deal_value, 2),
            'Probability': probability,
            'Expected_Close_Date': close_date.strftime('%Y-%m-%d'),
            'Sales_Rep': f'Sales_Rep_{random.randint(1, 20)}',
            'Created_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Last_Activity': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'Lead_Source': random.choice(['Website', 'Referral', 'Cold Call', 'Trade Show', 'Social Media']),
            'Industry': random.choice(['Manufacturing', 'Construction', 'Automotive', 'Infrastructure', 'Energy'])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('crm_sales_pipeline_data.xlsx', index=False)
    print(f"✅ Generated CRM Sales Pipeline Data: {len(df)} records")
    return df

def generate_inventory_management_data():
    """Generate Inventory Management Data"""
    print("Generating Inventory Management Data...")
    
    products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    warehouses = ['Warehouse A', 'Warehouse B', 'Warehouse C', 'Warehouse D']
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
    
    data = []
    for i in range(480):
        product = random.choice(products)
        current_stock = random.randint(100, 10000)
        reorder_point = random.randint(50, 500)
        max_stock = random.randint(500, 20000)
        
        data.append({
            'Inventory_ID': f'INV-{i+1:04d}',
            'Product_Name': product,
            'Product_Category': product,
            'SKU': f'SKU-{i+1:04d}',
            'Warehouse': random.choice(warehouses),
            'Current_Stock': current_stock,
            'Reorder_Point': reorder_point,
            'Max_Stock': max_stock,
            'Unit_Cost': round(random.uniform(50, 500), 2),
            'Total_Value': round(current_stock * random.uniform(50, 500), 2),
            'Supplier': random.choice(suppliers),
            'Lead_Time_Days': random.randint(7, 45),
            'Turnover_Rate': round(random.uniform(2, 12), 2),
            'Last_Updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'Expiry_Date': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
            'Storage_Cost_Per_Unit': round(random.uniform(1, 10), 2),
            'Quality_Grade': random.choice(['A', 'B', 'C']),
            'Location_Zone': random.choice(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4'])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('inventory_management_data.xlsx', index=False)
    print(f"✅ Generated Inventory Management Data: {len(df)} records")
    return df

def generate_hr_payroll_planning_data():
    """Generate HR and Payroll Planning Data"""
    print("Generating HR Payroll Planning Data...")
    
    departments = ['Production', 'Sales', 'Finance', 'HR', 'IT', 'Marketing', 'Operations', 'Quality']
    job_titles = ['Manager', 'Senior Executive', 'Executive', 'Assistant', 'Specialist', 'Analyst']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune']
    
    data = []
    for i in range(420):
        base_salary = random.uniform(30000, 200000)
        experience_years = random.randint(0, 20)
        
        data.append({
            'Employee_ID': f'EMP-{i+1:04d}',
            'Employee_Name': f'Employee_{i+1}',
            'Department': random.choice(departments),
            'Job_Title': random.choice(job_titles),
            'Location': random.choice(locations),
            'Base_Salary': round(base_salary, 2),
            'Experience_Years': experience_years,
            'Hire_Date': (datetime.now() - timedelta(days=random.randint(30, 3650))).strftime('%Y-%m-%d'),
            'Performance_Rating': round(random.uniform(1, 5), 1),
            'Bonus_Percentage': round(random.uniform(0, 30), 2),
            'Benefits_Cost': round(base_salary * random.uniform(0.1, 0.3), 2),
            'Overtime_Hours': random.randint(0, 20),
            'Overtime_Rate': round(base_salary / 160 * 1.5, 2),
            'Leave_Balance': random.randint(0, 30),
            'Training_Hours': random.randint(0, 40),
            'Promotion_Eligible': random.choice([True, False]),
            'Attrition_Risk': random.choice(['Low', 'Medium', 'High']),
            'Succession_Plan': random.choice(['Yes', 'No']),
            'Skills': random.choice(['Technical', 'Management', 'Sales', 'Analytical', 'Creative'])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('hr_payroll_planning_data.xlsx', index=False)
    print(f"✅ Generated HR Payroll Planning Data: {len(df)} records")
    return df

def generate_tax_obligation_schedules():
    """Generate Tax Obligation Schedules"""
    print("Generating Tax Obligation Schedules...")
    
    tax_types = ['GST', 'Income Tax', 'TDS', 'Professional Tax', 'ESIC', 'PF', 'Customs Duty']
    payment_frequencies = ['Monthly', 'Quarterly', 'Annually', 'Bi-annually']
    
    data = []
    for i in range(460):
        tax_amount = random.uniform(10000, 1000000)
        due_date = datetime.now() + timedelta(days=random.randint(1, 365))
        
        data.append({
            'Tax_ID': f'TAX-{i+1:04d}',
            'Tax_Type': random.choice(tax_types),
            'Tax_Period': f"{random.randint(2023, 2025)}-{random.randint(1, 12):02d}",
            'Due_Date': due_date.strftime('%Y-%m-%d'),
            'Amount_Due': round(tax_amount, 2),
            'Payment_Frequency': random.choice(payment_frequencies),
            'Status': random.choice(['Pending', 'Paid', 'Overdue', 'Partially Paid']),
            'Payment_Date': (due_date - timedelta(days=random.randint(-30, 30))).strftime('%Y-%m-%d') if random.random() > 0.3 else None,
            'Late_Fees': round(tax_amount * random.uniform(0, 0.1), 2) if random.random() > 0.7 else 0,
            'Tax_Rate': round(random.uniform(5, 30), 2),
            'Filing_Status': random.choice(['Filed', 'Not Filed', 'Extension Requested']),
            'Assessed_Value': round(tax_amount / random.uniform(0.05, 0.3), 2),
            'Compliance_Score': random.randint(70, 100),
            'Audit_Required': random.choice([True, False]),
            'Documentation_Complete': random.choice([True, False]),
            'Responsible_Person': f'Tax_Officer_{random.randint(1, 10)}'
        })
    
    df = pd.DataFrame(data)
    df.to_excel('tax_obligation_schedules.xlsx', index=False)
    print(f"✅ Generated Tax Obligation Schedules: {len(df)} records")
    return df

def generate_seasonal_trend_data():
    """Generate Seasonal Trend Data"""
    print("Generating Seasonal Trend Data...")
    
    products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India']
    
    data = []
    for year in range(2020, 2025):
        for month in range(1, 13):
            for product in products:
                for region in regions:
                    # Base sales with seasonal variation
                    base_sales = random.uniform(100000, 1000000)
                    
                    # Seasonal factors
                    if month in [3, 4, 5]:  # Spring - high demand
                        seasonal_factor = random.uniform(1.2, 1.5)
                    elif month in [6, 7, 8]:  # Monsoon - moderate demand
                        seasonal_factor = random.uniform(0.8, 1.1)
                    elif month in [9, 10, 11]:  # Autumn - high demand
                        seasonal_factor = random.uniform(1.1, 1.4)
                    else:  # Winter - lower demand
                        seasonal_factor = random.uniform(0.7, 1.0)
                    
                    actual_sales = base_sales * seasonal_factor
                    
                    data.append({
                        'Year': year,
                        'Month': month,
                        'Product_Category': product,
                        'Region': region,
                        'Base_Sales': round(base_sales, 2),
                        'Seasonal_Factor': round(seasonal_factor, 2),
                        'Actual_Sales': round(actual_sales, 2),
                        'Growth_Rate': round(random.uniform(-0.1, 0.3), 3),
                        'Market_Share': round(random.uniform(0.05, 0.25), 3),
                        'Customer_Count': random.randint(50, 500),
                        'Average_Order_Value': round(actual_sales / random.randint(50, 500), 2),
                        'Season': 'Spring' if month in [3,4,5] else 'Monsoon' if month in [6,7,8] else 'Autumn' if month in [9,10,11] else 'Winter',
                        'Holiday_Impact': random.choice(['High', 'Medium', 'Low']),
                        'Weather_Factor': round(random.uniform(0.8, 1.2), 2)
                    })
    
    df = pd.DataFrame(data)
    df.to_excel('seasonal_trend_data.xlsx', index=False)
    print(f"✅ Generated Seasonal Trend Data: {len(df)} records")
    return df

def generate_marketing_spend_roi_data():
    """Generate Marketing Spend and ROI Data"""
    print("Generating Marketing Spend and ROI Data...")
    
    channels = ['Digital Marketing', 'Trade Shows', 'Print Media', 'TV/Radio', 'Social Media', 'Email Marketing']
    campaigns = ['Brand Awareness', 'Lead Generation', 'Product Launch', 'Seasonal Promotion', 'Customer Retention']
    
    data = []
    for i in range(440):
        spend_amount = random.uniform(10000, 500000)
        roi = random.uniform(0.5, 5.0)
        revenue_generated = spend_amount * roi
        
        data.append({
            'Campaign_ID': f'CAM-{i+1:04d}',
            'Campaign_Name': f'Campaign_{i+1}',
            'Marketing_Channel': random.choice(channels),
            'Campaign_Type': random.choice(campaigns),
            'Start_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'End_Date': (datetime.now() + timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d'),
            'Budget_Allocated': round(spend_amount, 2),
            'Actual_Spend': round(spend_amount * random.uniform(0.8, 1.2), 2),
            'Revenue_Generated': round(revenue_generated, 2),
            'ROI': round(roi, 2),
            'Leads_Generated': random.randint(10, 1000),
            'Conversion_Rate': round(random.uniform(0.01, 0.15), 3),
            'Cost_Per_Lead': round(spend_amount / random.randint(10, 1000), 2),
            'Customer_Acquisition_Cost': round(spend_amount / random.randint(5, 100), 2),
            'Target_Audience': random.choice(['B2B', 'B2C', 'Both']),
            'Geographic_Scope': random.choice(['Local', 'Regional', 'National', 'International']),
            'Campaign_Status': random.choice(['Active', 'Completed', 'Paused', 'Planned']),
            'Performance_Rating': random.choice(['Excellent', 'Good', 'Average', 'Poor']),
            'Agency_Partner': f'Agency_{random.randint(1, 10)}',
            'Creative_Assets': random.randint(5, 50)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('marketing_spend_roi_data.xlsx', index=False)
    print(f"✅ Generated Marketing Spend ROI Data: {len(df)} records")
    return df

def generate_economic_indicators_data():
    """Generate Economic Indicators Data"""
    print("Generating Economic Indicators Data...")
    
    data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(500):
        current_date = base_date + timedelta(days=i*7)  # Weekly data
        
        # Realistic economic trends
        base_interest_rate = 6.5 + (i * 0.01) + random.uniform(-0.1, 0.1)
        inflation_rate = 4.0 + (i * 0.005) + random.uniform(-0.2, 0.2)
        gdp_growth = 7.0 + (i * 0.002) + random.uniform(-0.5, 0.5)
        
        data.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'Repo_Rate': round(base_interest_rate, 2),
            'Reverse_Repo_Rate': round(base_interest_rate - 0.25, 2),
            'Inflation_Rate': round(inflation_rate, 2),
            'GDP_Growth_Rate': round(gdp_growth, 2),
            'USD_INR_Rate': round(75 + (i * 0.01) + random.uniform(-1, 1), 2),
            'EUR_INR_Rate': round(85 + (i * 0.01) + random.uniform(-1, 1), 2),
            'Crude_Oil_Price': round(70 + (i * 0.02) + random.uniform(-5, 5), 2),
            'Steel_Price_Index': round(100 + (i * 0.1) + random.uniform(-5, 5), 2),
            'Industrial_Production_Index': round(120 + (i * 0.05) + random.uniform(-3, 3), 2),
            'Consumer_Price_Index': round(150 + (i * 0.02) + random.uniform(-2, 2), 2),
            'Wholesale_Price_Index': round(140 + (i * 0.02) + random.uniform(-2, 2), 2),
            'Unemployment_Rate': round(6.0 + random.uniform(-1, 1), 2),
            'Manufacturing_PMI': round(50 + random.uniform(-5, 5), 1),
            'Services_PMI': round(52 + random.uniform(-5, 5), 1),
            'FDI_Inflow': round(5000 + random.uniform(-500, 500), 2),
            'Export_Growth': round(10 + random.uniform(-5, 5), 2),
            'Import_Growth': round(12 + random.uniform(-5, 5), 2),
            'Fiscal_Deficit': round(6.5 + random.uniform(-1, 1), 2),
            'Current_Account_Deficit': round(2.0 + random.uniform(-1, 1), 2),
            'Foreign_Exchange_Reserves': round(600000 + random.uniform(-50000, 50000), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('economic_indicators_data.xlsx', index=False)
    print(f"✅ Generated Economic Indicators Data: {len(df)} records")
    return df

def generate_strategic_planning_data():
    """Generate Strategic Planning Data"""
    print("Generating Strategic Planning Data...")
    
    project_types = ['Market Expansion', 'Product Development', 'Technology Upgrade', 'Capacity Expansion', 'Acquisition']
    statuses = ['Planning', 'In Progress', 'Completed', 'On Hold', 'Cancelled']
    priorities = ['High', 'Medium', 'Low']
    
    data = []
    for i in range(400):
        budget = random.uniform(1000000, 100000000)
        duration_months = random.randint(6, 36)
        
        data.append({
            'Project_ID': f'PROJ-{i+1:04d}',
            'Project_Name': f'Strategic_Project_{i+1}',
            'Project_Type': random.choice(project_types),
            'Priority': random.choice(priorities),
            'Start_Date': (datetime.now() + timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Expected_Completion': (datetime.now() + timedelta(days=random.randint(365, 1095))).strftime('%Y-%m-%d'),
            'Budget': round(budget, 2),
            'Actual_Spent': round(budget * random.uniform(0, 0.8), 2),
            'Duration_Months': duration_months,
            'Status': random.choice(statuses),
            'ROI_Expected': round(random.uniform(0.1, 0.5), 3),
            'Risk_Level': random.choice(['Low', 'Medium', 'High']),
            'Department_Responsible': random.choice(['Operations', 'Sales', 'Finance', 'IT', 'HR', 'Marketing']),
            'Project_Manager': f'PM_{random.randint(1, 20)}',
            'Stakeholders': random.randint(3, 15),
            'Milestones_Completed': random.randint(0, 10),
            'Total_Milestones': random.randint(5, 20),
            'Resource_Allocation': round(random.uniform(0.1, 1.0), 2),
            'Success_Probability': round(random.uniform(0.5, 0.95), 2),
            'Market_Impact': random.choice(['High', 'Medium', 'Low']),
            'Competitive_Advantage': random.choice(['Yes', 'No']),
            'Regulatory_Approval_Required': random.choice([True, False])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('strategic_planning_data.xlsx', index=False)
    print(f"✅ Generated Strategic Planning Data: {len(df)} records")
    return df

def generate_production_operational_data():
    """Generate Production and Operational Data"""
    print("Generating Production Operational Data...")
    
    production_lines = ['Line A', 'Line B', 'Line C', 'Line D', 'Line E']
    products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars']
    shifts = ['Morning', 'Afternoon', 'Night']
    
    data = []
    for i in range(480):
        capacity = random.uniform(1000, 10000)
        actual_output = capacity * random.uniform(0.7, 1.0)
        efficiency = actual_output / capacity
        
        data.append({
            'Production_ID': f'PROD-{i+1:04d}',
            'Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Production_Line': random.choice(production_lines),
            'Product_Type': random.choice(products),
            'Shift': random.choice(shifts),
            'Planned_Capacity': round(capacity, 2),
            'Actual_Output': round(actual_output, 2),
            'Efficiency_Rate': round(efficiency, 3),
            'Quality_Score': round(random.uniform(0.85, 0.99), 3),
            'Downtime_Hours': round(random.uniform(0, 8), 2),
            'Maintenance_Hours': round(random.uniform(0, 4), 2),
            'Raw_Material_Consumption': round(actual_output * random.uniform(1.1, 1.3), 2),
            'Energy_Consumption': round(actual_output * random.uniform(50, 100), 2),
            'Labor_Hours': round(actual_output * random.uniform(0.1, 0.3), 2),
            'Cost_Per_Unit': round(random.uniform(50, 200), 2),
            'Waste_Percentage': round(random.uniform(0.01, 0.05), 3),
            'Safety_Incidents': random.randint(0, 3),
            'Equipment_Utilization': round(random.uniform(0.7, 0.95), 3),
            'Inventory_Turnover': round(random.uniform(2, 12), 2),
            'Lead_Time_Days': random.randint(1, 30),
            'Supplier_Delivery_Performance': round(random.uniform(0.8, 0.98), 3),
            'Customer_Satisfaction_Score': round(random.uniform(3.5, 5.0), 1),
            'Environmental_Compliance_Score': round(random.uniform(0.9, 1.0), 3)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('production_operational_data.xlsx', index=False)
    print(f"✅ Generated Production Operational Data: {len(df)} records")
    return df

def generate_customer_segmentation_data():
    """Generate Customer Segmentation Data"""
    print("Generating Customer Segmentation Data...")
    
    segments = ['Premium', 'Gold', 'Silver', 'Bronze']
    industries = ['Manufacturing', 'Construction', 'Automotive', 'Infrastructure', 'Energy', 'Aerospace']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India']
    
    data = []
    for i in range(450):
        annual_revenue = random.uniform(100000, 10000000)
        credit_limit = annual_revenue * random.uniform(0.1, 0.5)
        
        data.append({
            'Customer_ID': f'CUST-{i+1:04d}',
            'Customer_Name': f'Customer_{i+1}',
            'Segment': random.choice(segments),
            'Industry': random.choice(industries),
            'Region': random.choice(regions),
            'Annual_Revenue': round(annual_revenue, 2),
            'Credit_Limit': round(credit_limit, 2),
            'Payment_Terms_Days': random.choice([30, 45, 60, 90]),
            'Average_Order_Value': round(annual_revenue / random.randint(10, 100), 2),
            'Order_Frequency': random.randint(1, 50),
            'Customer_Since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Last_Order_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Total_Orders': random.randint(1, 200),
            'Payment_History_Score': round(random.uniform(0.5, 1.0), 3),
            'Days_Sales_Outstanding': random.randint(15, 90),
            'Customer_Satisfaction_Score': round(random.uniform(3.0, 5.0), 1),
            'Lifetime_Value': round(annual_revenue * random.uniform(1, 5), 2),
            'Churn_Risk': random.choice(['Low', 'Medium', 'High']),
            'Preferred_Products': random.choice(['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets']),
            'Contact_Person': f'Contact_{i+1}',
            'Phone': f'+91-{random.randint(7000000000, 9999999999)}',
            'Email': f'customer{i+1}@company.com',
            'Address': f'Address_{i+1}, {random.choice(regions)}',
            'Account_Manager': f'AM_{random.randint(1, 15)}'
        })
    
    df = pd.DataFrame(data)
    df.to_excel('customer_segmentation_data.xlsx', index=False)
    print(f"✅ Generated Customer Segmentation Data: {len(df)} records")
    return df

def main():
    """Generate all required datasets"""
    print("🚀 GENERATING COMPLETE AI CASH FLOW MODEL DATASETS")
    print("=" * 60)
    
    # Generate all datasets
    datasets = {}
    
    datasets['crm_sales'] = generate_crm_sales_pipeline_data()
    datasets['inventory'] = generate_inventory_management_data()
    datasets['hr_payroll'] = generate_hr_payroll_planning_data()
    datasets['tax_obligations'] = generate_tax_obligation_schedules()
    datasets['seasonal_trends'] = generate_seasonal_trend_data()
    datasets['marketing_roi'] = generate_marketing_spend_roi_data()
    datasets['economic_indicators'] = generate_economic_indicators_data()
    datasets['strategic_planning'] = generate_strategic_planning_data()
    datasets['production_ops'] = generate_production_operational_data()
    datasets['customer_segments'] = generate_customer_segmentation_data()
    
    # Create summary report
    summary = {
        'total_datasets': len(datasets),
        'total_records': sum(len(df) for df in datasets.values()),
        'datasets': {name: len(df) for name, df in datasets.items()},
        'generated_date': datetime.now().isoformat()
    }
    
    with open('ai_datasets_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n📊 DATASET GENERATION SUMMARY")
    print("=" * 40)
    print(f"Total Datasets Generated: {summary['total_datasets']}")
    print(f"Total Records Generated: {summary['total_records']:,}")
    print("\nIndividual Dataset Records:")
    for name, count in summary['datasets'].items():
        print(f"  • {name.replace('_', ' ').title()}: {count:,} records")
    
    print(f"\n💾 Summary saved to: ai_datasets_summary.json")
    print("\n✅ All datasets generated successfully!")

if __name__ == "__main__":
    main() 