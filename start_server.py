import subprocess, time, urllib.request, json, sys

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

time.sleep(8)

# Verify
for i in range(5):
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)
        status = json.loads(resp.read())
        print(f'Server OK: {status}')
        break
    except Exception as e:
        if i < 4:
            time.sleep(3)
        else:
            print(f'Server failed to start: {e}')
            proc.terminate()
            sys.exit(1)

# Test strategy
data = json.dumps({'cost_per_call': 5, 'profit_per_deposit': 50}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/api/strategy', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req, timeout=30)
r = json.loads(resp.read())
print(f'Strategy OK: {r["clients_to_call"]} calls, profit={r["expected_profit"]}')

print('Server running at http://127.0.0.1:8000')
print('Press Ctrl+C to stop')
proc.wait()
