
import streamlit as st

from api_client import (
    check_backend,
    get_evaluation_trips,
    evaluate_trip
)

from components import (
    show_metric_cards,
    show_trip_details,
    show_route_map
)


st.set_page_config(
    page_title="PathSnap | Evaluation Dashboard",
    page_icon="📍",
    layout="wide"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>
.main {
    background-color: #f4f7fa;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #123044;
    font-weight: 750;
}

h2, h3 {
    color: #173e52;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e3eaf0;
}

section[data-testid="stSidebar"] {
    background-color: #142d3d;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.stButton > button {
    background-color: #119e9b;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #087d7a;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("📍 PathSnap")
    st.caption("MAP MATCHING SYSTEM")

    st.divider()

    st.markdown("### Workspace")
    st.markdown("Evaluation Dashboard")

    st.divider()

    st.markdown("### About")
    st.write(
        "Select a permitted GPS evaluation trip "
        "to inspect map matching and route reconstruction."
    )


# -----------------------------
# MAIN HEADER
# -----------------------------

st.markdown(
    "DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION"
)

st.title("Evaluation Dashboard")

st.write(
    "Inspect GPS trajectories and evaluate reconstructed routes."
)


# -----------------------------
# BACKEND STATUS
# -----------------------------

connected, health_info = check_backend()

if connected:
    st.success("Backend connected")
else:
    st.error("Backend disconnected")

    st.caption(
        "Check the FastAPI server and PATHSNAP_API_URL."
    )


# -----------------------------
# TRIP SELECTION
# -----------------------------

st.divider()

st.subheader("01 | Select Evaluation Trip")

trips = []

if connected:
    try:
        trips = get_evaluation_trips()
    except RuntimeError as exc:
        st.error(str(exc))


def get_trip_id(trip):
    if isinstance(trip, dict):
        return str(trip["trip_id"])
    return str(trip)


if trips:
    trip_ids = [get_trip_id(trip) for trip in trips]

    selected_trip = st.selectbox(
        "Evaluation Trip",
        options=trip_ids
    )

    if st.button(
        "▶ Run Map Matching",
        use_container_width=True
    ):

        with st.spinner(
            "Processing GPS trajectory and reconstructing route..."
        ):
            try:
                result = evaluate_trip(selected_trip)

                if result.get("status") != "success":
                    st.error(
                        result.get(
                            "message",
                            "Evaluation was unsuccessful."
                        )
                    )
                else:
                    st.session_state["evaluation_result"] = result
                    st.session_state["selected_trip"] = selected_trip

                    st.success("Evaluation completed!")

            except RuntimeError as exc:
                st.error(str(exc))

else:
    st.info(
        "Evaluation trips will appear here when the backend is connected."
    )


# -----------------------------
# EVALUATION RESULTS
# -----------------------------

result = st.session_state.get("evaluation_result")

st.divider()

st.subheader("02 | Evaluation Results")

if result:

    st.caption(
        f"Results for trip: {st.session_state.get('selected_trip')}"
    )

    metrics = result.get("metrics", {})

    show_metric_cards(metrics)

    st.divider()

    details = result.get("trip_details", {})

    show_trip_details(details)

    st.divider()

    show_route_map(result.get("map_html"))

else:
    st.info(
        "No trip evaluated yet. Select an evaluation trip "
        "and click Run Map Matching."
    )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "PathSnap | GPS Map Matching and Route Reconstruction"
)
