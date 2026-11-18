"""Exercise real API inference on every configured evaluator trip."""
import json
from fastapi.testclient import TestClient
from api.main import app
from src.config import RESULTS_DIR

client = TestClient(app)
assert client.get('/health').status_code == 200
ids = client.get('/trips').json()['trips']
assert len(ids) == 10
assert client.post('/predict', json={'trip_id':'../bad'}).status_code == 422
assert client.post('/predict', json={'trip_id':'99999999'}).status_code == 404
results = []
for trip in ids:
    response = client.post('/predict', json={'trip_id':trip})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert 'reference' not in json.dumps(payload).lower().replace('reference-to-gps', '')
    assert 'openstreetmap' in payload['map_html'].lower()
    assert 0 <= payload['metrics']['top1'] <= 1
    assert client.get('/map/' + trip).status_code == 200
    assert client.post('/evaluate', json={'trip_id':trip}).json() == payload['metrics']
    results.append(payload['metrics'])
    print(trip, payload['metrics']['top1'], flush=True)
(RESULTS_DIR / 'api_integration.json').write_text(json.dumps(results, indent=2) + '\n')
print('All 10 real inference requests and map endpoints passed.', flush=True)
