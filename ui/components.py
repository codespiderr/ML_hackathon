
import streamlit as st


def show_metric_cards(metrics):
    """Display evaluation metrics."""

    columns = st.columns(4)

    metric_config = [
        ("Top-1 Accuracy", "top1_accuracy", True),
        ("Macro F1-score", "macro_f1", True),
        ("Route Continuity", "route_continuity", True),
        ("Inference Time", "inference_time_ms", False),
    ]

    for column, (label, key, percentage) in zip(
        columns, metric_config
    ):
        value = metrics.get(key)

        if value is None:
            display_value = "—"
        elif percentage:
            display_value = f"{float(value) * 100:.2f}%"
        else:
            display_value = f"{float(value):.2f} ms"

        column.metric(label, display_value)


def show_trip_details(details):
    """Display selected trip information."""

    st.subheader("Trip Details")

    columns = st.columns(3)

    columns[0].metric(
        "GPS Observations",
        details.get("gps_observations", "—")
    )

    columns[1].metric(
        "Candidate Segments",
        details.get("candidate_count", "—")
    )

    columns[2].metric(
        "Reconstruction Method",
        details.get("reconstruction_method", "—")
    )


def show_route_map(map_html):
    """Display Folium HTML returned by the backend."""

    import streamlit.components.v1 as components

    st.subheader("Reconstructed Route")

    if map_html:
        components.html(
            map_html,
            height=600,
            scrolling=True
        )
    else:
        st.info("No route map was returned by the backend.")
