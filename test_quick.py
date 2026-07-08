import json, urllib.request

# Test strategy
data = json.dumps({'cost_per_call': 5, 'profit_per_deposit': 50}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/api/strategy', data=data, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    r = json.loads(resp.read())
    print(f'Strategy OK: {r["clients_to_call"]} calls, profit={r["expected_profit"]}')
except Exception as e:
    print(f'Server not running: {e}')

# Test health
try:
    resp = urllib.request.urlopen('http://127.0.0.1:8000/api/health')
    print(f'Health: {resp.read().decode()}')
except Exception as e:
    print(f'Health error: {e}')
