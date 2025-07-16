import time
import os
import subprocess

WATCH_FOLDER = r"C:\Users\akank\Documents\CASHFLOW"
CHECK_INTERVAL = 5  # seconds

def get_excel_files_mtime(folder):
    excel_files = [f for f in os.listdir(folder) if f.endswith(('.xlsx', '.xls'))]
    mtimes = {}
    for f in excel_files:
        path = os.path.join(folder, f)
        mtimes[f] = os.path.getmtime(path)
    return mtimes

def main():
    last_mtimes = get_excel_files_mtime(WATCH_FOLDER)
    print("Watching folder for changes...")

    while True:
        time.sleep(CHECK_INTERVAL)
        current_mtimes = get_excel_files_mtime(WATCH_FOLDER)

        # Check if any file has changed or new file appeared
        changed_files = [f for f in current_mtimes if f not in last_mtimes or current_mtimes[f] != last_mtimes[f]]

        if changed_files:
            print(f"Change detected in files: {changed_files}")
            print("Running automated_cashflow.py...")
            subprocess.run(["python", "cashflow_sap.py"], cwd=WATCH_FOLDER)
            last_mtimes = current_mtimes.copy()

if __name__ == "__main__":
    main()
