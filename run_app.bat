@echo off
echo Starting Cash Flow SAP Bank System with Enhanced Console Output...
echo.

REM Set environment variables for better console output
set PYTHONUNBUFFERED=1
set FLASK_ENV=development
set FLASK_DEBUG=1

echo Environment variables set:
echo - PYTHONUNBUFFERED=1
echo - FLASK_ENV=development
echo - FLASK_DEBUG=1
echo.

echo Starting Flask application...
echo You should see console output immediately when clicking buttons.
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python run_app_with_debug.py

pause 