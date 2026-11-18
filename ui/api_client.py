
import os
import requests

API_BASE_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

TIMEOUT = 60


def check_backend():
    """Check whether the FastAPI server is running."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/health",
            timeout=5
        )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def get_evaluation_trips():
    """Retrieve permitted evaluation trips from backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/trips",
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            return data.get("trips", [])

        return data

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not load evaluation trips: {exc}"
        ) from exc


def evaluate_trip(trip_id):
    """Send selected trip to the map-matching backend."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/evaluate",
            json={"trip_id": trip_id},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Trip evaluation failed: {exc}"
        ) from exc
