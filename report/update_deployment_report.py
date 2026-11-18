"""Update the existing technical report from actual API/evaluator results."""
import json
from pathlib import Path
from docx import Document
from docx.shared import RGBColor

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'report/PathSnap_Technical_Report.docx'
doc = Document(path)
# Replace the generated appendix on repeated runs instead of duplicating it.
for p in doc.paragraphs:
    if p.text == '7.1 API and client verification':
        element = p._element
        body = element.getparent()
        for child in list(body)[list(body).index(element):]:
            if not child.tag.endswith('sectPr'):
                body.remove(child)
        break
for p in doc.paragraphs:
    if p.text.startswith('The planned demonstration assigns FastAPI'):
        p.text = ('FastAPI is implemented in api/main.py and api/service.py. Laptop 1 loads the tuned '
                  'Random Forest and decoder configuration. The Streamlit client in ui/app.py runs on '
                  'Laptop 2, obtains the available evaluator trip IDs from GET /trips, and sends the selected '
                  'ID to POST /predict. The response contains measured metrics and a Folium map. '
                  'Reference labels are loaded only after prediction for internal scoring and are never returned to the UI.')
    if p.text.startswith('AI assistance was used'):
        p.text = ('AI assistance was used to draft and format this report from recorded code and experiment artifacts. '
                  'Team members should verify the descriptions against their runs. Separate-device connectivity and '
                  'genuine UI and receiving-terminal screenshots remain required before submission.')

doc.add_heading('7.1 API and client verification', level=2)
doc.add_paragraph('The integration check sent real inference requests for all ten evaluator trips, verified the '
                  'interactive map endpoints, and checked malformed and unknown trip IDs. All checks passed. '
                  'The Streamlit interaction test selected trip 00000023 and displayed four metrics returned '
                  'by the running API. These are local tests; a physical two-laptop demonstration has not yet been verified.')
doc.add_paragraph('The map uses actual trip coordinates over OpenStreetMap, with toggles for GPS fixes, local '
                  'roads, classifier-selected segments and Viterbi-selected segments. Start and end markers identify '
                  'the trip extent. Reference route geometry is kept off the evaluator UI. Selected segments are '
                  'drawn independently; missing connections are not fabricated as straight lines. Map assets and tiles require internet access.')
doc.add_paragraph('Available endpoints are GET /health, GET /trips, POST /predict, POST /evaluate, '
                  'GET /map/{trip_id} and GET /docs. Server logs record requests and inference metrics. '
                  'Repeated requests reuse an in-memory result; restart the API after changing data or model assets.')
doc.add_heading('7.2 Separate device setup', level=2)
doc.add_paragraph('On Laptop 1, install requirements.txt, point PATHSNAP_EVALUATOR_DIR at the ten raw trip '
                  'directories and run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000. '
                  'On Laptop 2, copy ui/app.py and ui/requirements.txt, install those client dependencies and '
                  'run python -m streamlit run app.py --server.port 8501. Enter the server laptop LAN address '
                  'in the client sidebar. Both devices must share a reachable network; allow TCP 8000 on the '
                  'server private network when necessary. Full instructions are in docs/deployment.md.')
doc.add_heading('8 Separate evaluator results', level=1)
r = json.loads((ROOT / 'results/evaluator_test/results.json').read_text())
doc.add_paragraph(f"The frozen tuned forest and decoder were evaluated on {r['trip_count']} separate trips, "
                  f"containing {r['observations']:,} GPS fixes. No retraining was performed on these trips. "
                  'The deployed model remains the previous Random Forest; the later exploratory boosted model was not adopted.')
t = doc.add_table(rows=1, cols=3)
t.style = 'Table Grid'
for c, s in zip(t.rows[0].cells, ['Metric', 'Random Forest', 'With Viterbi']): c.text = s
for name, a, b in [('Top-1 accuracy',r['raw_top1'],r['decoded_top1']),('Direct continuity',r['raw_continuity'],r['decoded_continuity'])]:
    cells=t.add_row().cells
    for c,s in zip(cells,[name,f'{a:.2%}',f'{b:.2%}']): c.text=s
doc.add_paragraph(f"Candidate recall was {r['candidate_recall']:.2%}. Trip 00000015 contains 6,854 fixes and "
                  'achieved 57.12% decoded accuracy, heavily influencing the overall result. Viterbi improved '
                  'overall Top-1 by 0.21 percentage points and reduced it on four trips. Scores use approximate '
                  f"monotonic reference alignment; {r['alignment_over_50m']} fixes aligned more than 50 m from their reference segment. "
                  'These results do not establish independent per-fix ground truth.')
doc.add_paragraph('Reproduce the evaluator run with python evaluate_external.py and verify API integration '
                  'with python -m api.check_integration. Results are in results/evaluator_test/results.json and '
                  'results/api_integration.json. The external archive and extracted reference data are local server inputs '
                  'and are excluded from this source-code push.')
for style_name in ('Title', 'Heading 1', 'Heading 2', 'Heading 3'):
    doc.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
doc.save(path)
print(path)
