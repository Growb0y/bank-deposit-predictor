import subprocess, sys, time, os, json
import urllib.request, urllib.error
import pandas as pd
from io import BytesIO

# Start server
proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(8)

# Check health
resp = urllib.request.urlopen('http://127.0.0.1:8000/api/health')
print('Health:', resp.read().decode())

# Read sample file
with open('sample_for_upload.xlsx', 'rb') as f:
    file_data = f.read()

# Build multipart request manually
boundary = '----TestBoundary123'
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
        'Content-Length': str(len(body)),
    }
)

try:
    resp = urllib.request.urlopen(req, timeout=30)
    print(f'Upload OK: status={resp.status}')
    result_data = resp.read()
    with open('test_result.xlsx', 'wb') as f:
        f.write(result_data)

    # Verify
    df = pd.read_excel(BytesIO(result_data), engine='openpyxl')
    print(f'Result columns: {list(df.columns)}')
    print(f'Result rows: {len(df)}')
    print('Top 3:')
    print(df.head(3).to_string())
    os.remove('test_result.xlsx')
    print('\nUpload test: SUCCESS')
except urllib.error.HTTPError as e:
    print(f'HTTP Error {e.code}: {e.read().decode("utf-8")}')
except Exception as e:
    print(f'Upload error: {e}')
    import traceback
    traceback.print_exc()

proc.terminate()
time.sleep(1)
