# COMPLETE STEEL PLANT DATASETS CATALOG

## Overview
This catalog describes all datasets generated for the steel plant AI cash flow model, including comprehensive data for Operating, Investing, and Financing activities.

## Datasets

### 1. steel_plant_sap_data.xlsx
**Purpose**: SAP AP/AR transactions with all cash flow categories
**Records**: 400+ transactions
**Key Features**:
- Operating Activities (60%): Revenue and expenses
- Investing Activities (25%): Asset sales and capital expenditure
- Financing Activities (15%): Borrowing and repayments
- Steel-specific descriptions and categories

### 2. steel_plant_bank_data.xlsx
**Purpose**: Bank statement data with all cash flow categories
**Records**: 450+ transactions
**Key Features**:
- Operating Activities (65%): Customer payments and supplier payments
- Investing Activities (20%): Equipment sales and purchases
- Financing Activities (15%): Loan disbursements and repayments
- Detailed transaction categorization

### 3. steel_production_data.xlsx
**Purpose**: Production metrics and efficiency data
**Records**: 600+ production records
**Key Features**:
- Multiple production lines (Blast Furnace, Rolling Mills, etc.)
- Efficiency metrics and quality scores
- Energy consumption and labor hours
- Cost and revenue per tonne

### 4. steel_inventory_data.xlsx
**Purpose**: Inventory management for all materials
**Records**: 650+ inventory records
**Key Features**:
- Raw materials, finished goods, and spare parts
- Stock levels and movements
- Reorder points and turnover ratios
- Storage locations and suppliers

### 5. steel_customer_data.xlsx
**Purpose**: Customer profiles and relationship data
**Records**: 700+ customer records
**Key Features**:
- Industry-specific customer profiles
- Credit limits and payment terms
- Order history and revenue data
- Regional and product preferences

### 6. steel_supplier_data.xlsx
**Purpose**: Supplier profiles and performance data
**Records**: 600+ supplier records
**Key Features**:
- Supplier categories and supplied items
- Performance ratings and delivery times
- Contract terms and payment methods
- Quality and price ratings

### 7. steel_operational_data.xlsx
**Purpose**: Department-level operational metrics
**Records**: 650+ operational records
**Key Features**:
- Department-specific efficiency metrics
- Energy consumption and environmental data
- Safety incidents and maintenance hours
- Cost and revenue per tonne by department

### 8. steel_financial_data.xlsx
**Purpose**: Comprehensive financial statements
**Records**: 700+ financial records
**Key Features**:
- Complete cash flow statements (Operating, Investing, Financing)
- Financial ratios and working capital
- EBITDA, depreciation, and tax data
- Balance sheet components

## Cash Flow Categories Covered

### Operating Activities
- Revenue from steel sales
- Raw material purchases
- Operating expenses
- Customer and supplier payments

### Investing Activities
- Capital expenditure (new equipment)
- Asset sales (old machinery, scrap)
- Plant expansion and upgrades
- Technology investments

### Financing Activities
- Bank loans and borrowings
- Loan repayments and interest
- Working capital financing
- Equipment financing

## AI Model Integration
All datasets are designed to work together for comprehensive cash flow analysis:
- SAP/Bank data for transaction-level analysis
- Production data for operational efficiency impact
- Inventory data for working capital optimization
- Customer/Supplier data for relationship insights
- Operational data for cost optimization
- Financial data for comprehensive reporting

## Usage Instructions
1. Upload SAP and Bank data through the Flask interface
2. AI model automatically integrates with all steel plant datasets
3. Enhanced analysis considers all cash flow categories
4. Comprehensive predictions and insights generated
