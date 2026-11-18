"""Run with: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from api.service import process, trip_ids

app = FastAPI(title='PathSnap inference server', version='1.0')

class TripRequest(BaseModel):
    trip_id: str = Field(pattern=r'^\d{8}$')

def run(trip_id):
    try:
        return process(trip_id)
    except KeyError:
        raise HTTPException(404, 'Trip not found')
    except Exception:
        logging.getLogger('uvicorn.error').exception('Trip processing failed: %s', trip_id)
        raise HTTPException(500, 'Trip processing failed. Check server logs.')

@app.get('/health')
def health():
    return {'status':'ok', 'trip_count':len(trip_ids())}

@app.get('/trips')
def trips():
    return {'trips':trip_ids()}

@app.post('/predict')
def predict(request: TripRequest):
    metrics, html = run(request.trip_id)
    return {'metrics':metrics, 'map_html':html}

@app.post('/evaluate')
def evaluate(request: TripRequest):
    return run(request.trip_id)[0]

@app.get('/map/{trip_id}', response_class=HTMLResponse)
def trip_map(trip_id: str):
    return run(trip_id)[1]
