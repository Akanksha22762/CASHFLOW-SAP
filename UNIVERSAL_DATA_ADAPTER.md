# Universal Data Adapter for Cash Flow Analysis System

## Overview

The Universal Data Adapter is a powerful component that enables the Cash Flow Analysis System to work with any dataset format. It automatically adapts input data to the standardized format required by the system, making it possible to analyze financial data from any source without manual preprocessing.

## Key Features

1. **Automatic Column Mapping**
   - Intelligently maps columns from any dataset to the standard format
   - Uses pattern matching, heuristics, and data profiling to identify relevant columns
   - Handles different naming conventions across various financial systems

2. **Data Type Standardization**
   - Converts date strings to proper datetime objects
   - Normalizes numeric values, handling currency symbols and formatting
   - Standardizes transaction types to consistent INWARD/OUTWARD classification

3. **Data Cleaning and Validation**
   - Handles missing values with intelligent defaults
   - Cleans and normalizes text fields
   - Validates data integrity and provides feedback on data quality

4. **Format Detection**
   - Automatically detects file formats (CSV, Excel)
   - Tries multiple encodings and separators for CSV files
   - Adapts to various banking and accounting system export formats

5. **Derived Column Generation**
   - Adds useful derived columns like Year, Month, Day
   - Calculates absolute amounts for easier analysis
   - Infers transaction types when not explicitly provided

## How It Works

The Universal Data Adapter follows a systematic process:

1. **Dataset Profiling**: Analyzes the input data to understand its structure, column types, and characteristics
2. **Column Mapping**: Maps columns to the standard format using pattern matching and heuristics
3. **Data Cleaning**: Cleans and normalizes the data, handling missing values and inconsistencies
4. **Type Standardization**: Converts data to appropriate types for analysis
5. **Derived Column Generation**: Adds useful derived columns for enhanced analysis

## Supported Data Formats

The adapter can handle a wide variety of data formats, including:

- Standard bank statements (CSV, Excel)
- SAP financial exports
- General ledger data
- Accounting system exports
- Custom financial datasets
- Various date and amount formats

## Usage

The Universal Data Adapter is automatically used when uploading files through the web interface. It can also be used programmatically:

```python
from universal_data_adapter import UniversalDataAdapter
from data_adapter_integration import preprocess_for_analysis

# Load and adapt a file
adapted_data = UniversalDataAdapter.load_and_adapt("financial_data.csv")

# Or adapt an existing DataFrame
import pandas as pd
data = pd.read_csv("financial_data.csv")
adapted_data = preprocess_for_analysis(data, "financial_data")
```

## Benefits

- **Universal Compatibility**: Works with data from any financial system
- **Time Savings**: Eliminates manual data preprocessing steps
- **Consistency**: Ensures all data follows the same format for reliable analysis
- **Error Reduction**: Reduces human error in data preparation
- **Flexibility**: Easily incorporate new data sources without code changes

## Technical Implementation

The adapter is implemented as a Python module with the following components:

- `universal_data_adapter.py`: Core adapter functionality
- `data_adapter_integration.py`: Integration with the main application
- `test_universal_adapter.py`: Test suite for the adapter

The system uses pattern matching, heuristics, and data profiling techniques to intelligently adapt any dataset to the required format, making the Cash Flow Analysis System truly universal in its application.