import subprocess, sys, time, os, json, traceback
import urllib.request, urllib.error
import pandas as pd
from io import BytesIO
from pathlib import Path

# Start fresh server
for proc in subprocess.check_output('tasklist /FI "IMAGENAME eq python.exe" /FO CSV', shell=True).decode().split('\n'):
    if 'python.exe' in proc and 'test_upload' not in proc:
        pid = proc.split(',')[1].strip('" ')
        if pid.isdigit():
            os.system(f'taskkill /PID {pid} /F >nul 2>&1')

time.sleep(2)

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000', '--log-level', 'debug'],
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
)
time.sleep(8)

# Read sample file
with open('sample_for_upload.xlsx', 'rb') as f:
    file_data = f.read()

boundary = '----TestBoundary789'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="sample_for_upload.xlsx"\r\n'
    f'Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n\r\n'
).encode('utf-8') + file_data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/predict/upload',
    data=body,
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
    }
)

try:
    resp = urllib.request.urlopen(req, timeout=30)
    print(f'Upload OK: status={resp.status}')
    result_data = resp.read()
    df = pd.read_excel(BytesIO(result_data), engine='openpyxl')
    print(f'Result: {len(df)} rows, cols: {list(df.columns)}')
    print(df.head(3).to_string())
except urllib.error.HTTPError as e:
    error_body = e.read().decode('utf-8')
    print(f'HTTP Error {e.code}: {error_body}')
    # Get server stderr
    time.sleep(1)
    stderr = proc.stderr.read()
    if stderr:
        print(f'SERVER STDERR (last 2KB): {stderr[-2000:].decode("utf-8", errors="replace")}')
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()

proc.terminate()
time.sleep(1)
