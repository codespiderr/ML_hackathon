"""Thin client: requires only Streamlit and requests; all ML stays on the server."""
import os
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='PathSnap', page_icon='🧭', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1400px}
div[data-testid="stMetric"]{background:#f3f7f8;border-radius:12px;padding:16px}
</style>''', unsafe_allow_html=True)
st.title('PathSnap')
st.caption('GPS observations → road predictions. Explore a trip on the map and inspect its evaluation.')
with st.sidebar:
    st.subheader('Connect to server')
    base = st.text_input('Server URL', os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000')).rstrip('/')
    st.caption('On another device, enter the server laptop’s LAN IP, for example http://192.168.1.20:8000.')
    st.divider()
    st.caption('Tuned Random Forest · HMM / Viterbi')
try:
    response = requests.get(base + '/trips', timeout=10)
    response.raise_for_status()
    trips = response.json()['trips']
except (requests.RequestException, ValueError, KeyError):
    st.error('Cannot connect to the API. Check the server URL, Wi-Fi connection and server terminal.')
    st.stop()
if not trips:
    st.warning('No evaluator trips found. Configure PATHSNAP_EVALUATOR_DIR on the server.')
    st.stop()
left, right = st.columns([3, 1], vertical_alignment='bottom')
trip = left.selectbox(f'Evaluator trip · {len(trips)} available', trips)
run = right.button('Run map matching', type='primary', use_container_width=True)
if st.session_state.get('source') != (base, trip):
    st.session_state.pop('result', None)
if run:
    try:
        with st.spinner('Server is matching GPS points and evaluating the route…'):
            response = requests.post(base + '/predict', json={'trip_id':trip}, timeout=300)
            response.raise_for_status()
            st.session_state.result = response.json()
            st.session_state.source = (base, trip)
    except (requests.RequestException, ValueError):
        st.error('The request failed or timed out. Check the API terminal, then retry.')
if 'result' in st.session_state:
    data = st.session_state.result
    m = data['metrics']
    cols = st.columns(4)
    cols[0].metric('Top-1 accuracy', f"{m['top1']:.2%}", f"{(m['top1']-m['classifier_top1'])*100:+.2f} pp vs forest")
    cols[1].metric('Direct continuity', f"{m['continuity']:.2%}")
    cols[2].metric('GPS observations', f"{m['observations']:,}")
    cols[3].metric('Inference time', f"{m['inference_seconds']:.2f} s")
    st.caption('Teal: predicted road segments · Orange: GPS observations. Use the map controls to toggle roads and classifier predictions. Map tiles need internet access.')
    components.html(data['map_html'], height=620, scrolling=False)
    with st.expander('Evaluation details'):
        st.write({'Candidate recall':f"{m['candidate_recall']:.2%}", 'Candidate macro F1':round(m['candidate_macro_f1'],4), 'Candidate log loss':round(m['candidate_log_loss'],4), 'Alignment distances above 50 m':m['alignment_over_50m']})
        st.caption(m['label_note'])
        st.download_button('Download metrics', data=__import__('json').dumps(m, indent=2), file_name=f'{trip}_metrics.json', mime='application/json')
else:
    st.info('Select a trip and run map matching to view its route and accuracy.')
