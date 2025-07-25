# PowerShell script to run Flask app with enhanced console output

Write-Host "🚀 Starting Cash Flow SAP Bank System with Enhanced Console Output..." -ForegroundColor Green
Write-Host ""

# Set environment variables for better console output
$env:PYTHONUNBUFFERED = "1"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "- PYTHONUNBUFFERED=1" -ForegroundColor Cyan
Write-Host "- FLASK_ENV=development" -ForegroundColor Cyan
Write-Host "- FLASK_DEBUG=1" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting Flask application..." -ForegroundColor Green
Write-Host "You should see console output immediately when clicking buttons." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Run the application
try {
    python run_app_with_debug.py
}
catch {
    Write-Host "Error running the application: $_" -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 