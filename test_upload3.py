import subprocess, sys, time, os, traceback
import requests
import pandas as pd
from io import BytesIO

print("Starting...", flush=True)

# Kill any process on port 8000
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 8000))
if result == 0:
    os.system(f'taskkill /F /PID {os.popen("netstat -ano | findstr :8000").read().strip().split()[-1]} >nul 2>&1')
    time.sleep(3)
sock.close()

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000'],
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
)
time.sleep(10)
print("Server started", flush=True)

try:
    f = open('sample_for_upload.xlsx', 'rb')
    files = {'file': ('sample.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}

    print("Sending request...", flush=True)
    resp = requests.post('http://127.0.0.1:8000/api/predict/upload', files=files, timeout=60)
    f.close()
    print(f"Response: {resp.status_code}", flush=True)

    if resp.status_code == 200:
        df = pd.read_excel(BytesIO(resp.content))
        print(f'SUCCESS: {len(df)} rows, cols: {list(df.columns)}', flush=True)
    else:
        print(f'ERROR: {resp.text[:500]}', flush=True)
        stderr = proc.stderr.read()
        if stderr:
            txt = stderr.decode('utf-8', errors='replace')[-3000:]
            print(f'SERVER: {txt}', flush=True)
except Exception as e:
    print(f'Exception: {e}', flush=True)
    traceback.print_exc()
finally:
    proc.terminate()
    print("Done", flush=True)
