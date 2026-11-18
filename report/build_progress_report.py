"""Build a results-only PathSnap report from recorded experiment outputs."""

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "report" / "PathSnap_Progress_Report.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#155e75"), spaceAfter=12))
styles.add(ParagraphStyle(name="SectionBlue", parent=styles["Heading2"], textColor=colors.HexColor("#155e75"), spaceBefore=13))
styles["BodyText"].leading = 14


def paragraph(text):
    return Paragraph(text, styles["BodyText"])


def heading(text):
    return Paragraph(text, styles["SectionBlue"])


def table(rows, widths=None):
    item = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#155e75")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef5f7")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bdd1d8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return item


def figure(path, width=170 * mm):
    from reportlab.lib.utils import ImageReader

    image_width, image_height = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * image_height / image_width)


def footer(canvas, document):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(20 * mm, 15 * mm, "PathSnap | Progress report | 18 November 2026")
    canvas.drawRightString(190 * mm, 15 * mm, f"Page {document.page}")


def main():
    with (RESULTS / "model_comparison" / "comparison_table.csv").open(newline="") as stream:
        comparison = list(csv.DictReader(stream))
    tuning = json.loads((RESULTS / "random_forest_tuning" / "summary.json").read_text())
    decoding = json.loads((RESULTS / "random_forest_tuning" / "decoder_results.json").read_text())
    sweep = json.loads((RESULTS / "baseline" / "preprocessing" / "candidate_sweep.json").read_text())

    story = [Paragraph("PathSnap | Technical progress report", styles["TitleBlue"]),
             paragraph("Results recorded through 18 November 2026. This report covers completed data preparation, "
                       "five-model comparison, Random Forest tuning, calibration assessment, and route decoding."),
             heading("1. Dataset and preprocessing"),
             paragraph("The supplied development data contains 80 training trips with 208,417 GPS observations "
                       "and 10 student-test trips with 19,160 observations. Each trip has a GPS track, road nodes, "
                       "directed arcs, and an ordered reference route. Input validation checks finite coordinates, "
                       "valid node and arc IDs, and strictly increasing timestamps. Gaps over 30 seconds start "
                       "new GPS blocks."),
             paragraph("Each trip is converted to a local metre grid. A spatial grid indexes directed road segments. "
                       "The pipeline retrieves the nearest 12 candidate arcs per GPS fix and calculates eight "
                       "distance, heading, speed, and geometry features per candidate. Arc IDs are identifiers "
                       "and are excluded from model inputs. Dynamic programming aligns GPS fixes monotonically "
                       "to the ordered reference route to produce approximate pointwise labels."),
             table([["Split", "Trips", "GPS fixes", "Candidate recall", "Nearest-road Top-1"],
                    ["Train", "80", "208,417", "92.80%", "49.42%"],
                    ["Student test", "10", "19,160", "99.88%", "53.48%"]],
                   [29*mm, 19*mm, 30*mm, 43*mm, 45*mm]),
             Spacer(1, 5*mm),
             paragraph("A training-only sweep found 96.06% candidate recall at K=32 and 98.45% at K=64. "
                       "The model results below use the published K=12 archives. The route alignment is an "
                       "estimated label source and limits the certainty of pointwise metrics."),
             figure(RESULTS / "baseline" / "preprocessing" / "candidate_sweep.png"),
             PageBreak(),
             heading("2. Five-model comparison"),
             paragraph("The repo records five trip-grouped cross-validation folds and a separate student-test "
                       "evaluation. The same candidate features and scoring functions were used for each model. "
                       "Training caps differ by algorithm and are recorded in the experiment CSVs."),
             table([["Model", "CV Top-1", "CV macro F1", "Student Top-1", "Student macro F1"]]
                   + [[row["model"].upper(), f'{float(row["top1_mean"]):.3f}',
                       f'{float(row["macro_f1_mean"]):.3f}', f'{float(row["student_top1"]):.3f}',
                       f'{float(row["student_macro_f1"]):.3f}'] for row in comparison],
                   [39*mm, 30*mm, 34*mm, 34*mm, 39*mm]),
             Spacer(1, 6*mm),
             figure(RESULTS / "plots" / "cv_model_comparison.png"),
             paragraph("The team selected Random Forest for the tuning and route-decoding experiment. "
                       "Its recorded CV Top-1 was 0.7443 and student-test Top-1 was 0.8472 in the initial "
                       "five-model comparison. The tuning comparison below retrains the original and tuned "
                       "configurations with the same 60,000-fix cap."),
             PageBreak(),
             heading("3. Bayesian optimisation and calibration"),
             paragraph("Optuna ran 30 TPE trials on training trips only. Each trial maximised mean Top-1 "
                       "accuracy over the five saved trip folds, with at most 12,000 training fixes per fold. "
                       "The best search objective was 0.74564. The selected Random Forest uses 200 trees, "
                       "maximum depth 10, minimum leaf size 6, maximum feature fraction 0.8144, and "
                       "bootstrap sample fraction 0.6502."),
             table([["Student-test metric", "Original RF", "Tuned RF"],
                    ["Top-1 accuracy", "0.84870", "0.85063"],
                    ["Candidate-row macro F1", "0.91307", "0.91544"],
                    ["Candidate-row log loss", "0.06547", "0.06408"],
                    ["Direct continuity", "0.92203", "0.92239"],
                    ["Inference on 10 trips", "0.17 s", "0.29 s"]],
                   [73*mm, 47*mm, 47*mm]),
             Spacer(1, 4*mm),
             paragraph("At the same 60,000-fix training cap, mean five-fold Top-1 was 0.74446 for the "
                       "original configuration and 0.74622 for the tuned configuration. Isotonic calibration "
                       "was fitted using a held-out training fold. It improved log loss on that same fold, "
                       "but student-test log loss rose to 0.08265. The saved final model therefore uses "
                       "the tuned forest's raw probabilities."),
             figure(RESULTS / "random_forest_tuning" / "history.png", 108*mm),
             figure(RESULTS / "random_forest_tuning" / "reliability.png", 76*mm),
             PageBreak(),
             heading("4. HMM/Viterbi route reconstruction"),
             paragraph("The hidden state at each GPS fix is a candidate directed arc. Emission scores are "
                       "classifier probabilities normalised over that fix's candidates. Transition scores "
                       "compare directed network travel distance with GPS displacement. Unreachable pairs "
                       "receive a finite penalty, and decoding restarts after GPS gaps. Four decoder settings "
                       "were compared on two held-out training trips. The selected distance scale was 50 m "
                       "and the disconnected-pair penalty was 4."),
             table([["Student-test metric", "Classifier", "Classifier + Viterbi"],
                    ["Pointwise Top-1", "0.85063", "0.86608"],
                    ["Direct continuity", "0.92239", "0.96642"],
                    ["Mean trip transition F1", "0.64373", "0.74757"]],
                   [65*mm, 49*mm, 53*mm]),
             Spacer(1, 6*mm),
             paragraph("Viterbi lowered pointwise accuracy on three of ten student-test trips. On trip "
                       "00000065 it fell by 0.00651. Per-trip results, trial settings, the Optuna study, "
                       "trial CSV, and model artifacts are retained in the repository."),
             heading("5. Remaining submission components"),
             paragraph("The evaluator-facing FastAPI service, two-laptop UI demonstration, Folium route "
                       "comparison, and their genuine screenshots are still to be produced. The supplied "
                       "evaluator-test partition has not been used for model development. This progress "
                       "report is a draft of the required consolidated final report."),
             heading("Reproduction"),
             paragraph("Run <font name='Courier'>python -m src.data_processing.prepare</font> to rebuild the "
                       "K=12 archives, <font name='Courier'>python -m src.models.bayesian_opt --trials 30 "
                       "--search-cap 12000 --final-cap 60000</font> for tuning, and "
                       "<font name='Courier'>python -m src.route_reconstruction.evaluate_decoder</font> "
                       "for decoding. Recorded data and code are under results/, saved_models/, and src/.")]

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=23*mm,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
