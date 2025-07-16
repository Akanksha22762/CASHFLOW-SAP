import subprocess
import time
import requests

# Step 1: Start the Flask server in the background
print("Starting Flask app...")
flask_process = subprocess.Popen(["python", "app.py"])

# Step 2: Wait a few seconds for the server to start up
time.sleep(3)  # You can increase this if needed

# Step 3: Send request to your endpoint
try:
    file_key = "transactions"  # or 'cashflow', 'originmap'
    url = f"http://127.0.0.1:5000/load-file?file={file_key}"

    print(f"Sending request to {url} ...")
    response = requests.get(url)

    if response.ok:
        print("✅ Preview Data Received:")
        print(response.json())
    else:
        print("❌ Error:", response.status_code, response.text)

except Exception as e:
    print("⚠️ Exception during request:", str(e))

# Step 4: Optionally stop Flask after test
finally:
    print("Shutting down Flask server...")
    flask_process.terminate()
    flask_process.wait()
    print("✅ Flask stopped.")
