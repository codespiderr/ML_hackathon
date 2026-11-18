import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parent
comparison = pd.read_csv(root / 'model_comparison' / 'comparison_table.csv')
decoded = json.loads((root / 'random_forest_tuning' / 'decoder_results.json').read_text())['student_test']
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
ax = axes[0]
labels = {'mlp':'Neural network', 'rf':'Random forest', 'hgb':'Gradient boosting', 'logreg':'Logistic regression', 'knn':'k-nearest neighbors'}
names = [labels[x] for x in comparison.model]
y = list(range(len(names)))
ax.barh(y, comparison.top1_mean * 100, color='#3e79b7')
ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel('Mean Top-1 across five trip folds (%)')
ax.set_title('Initial model comparison')
for i, v in enumerate(comparison.top1_mean * 100): ax.text(v + 1, i, f'{v:.2f}%', va='center')
ax = axes[1]
rows = ['Top-1 accuracy', 'Direct continuity']
raw = [decoded['raw_top1']*100, decoded['raw_continuity']*100]
after = [decoded['decoded_top1']*100, decoded['decoded_continuity']*100]
for i, (a, b) in enumerate(zip(raw, after)):
    ax.barh(i-.18, a, height=.32, color='#8296aa', label='Tuned forest' if i == 0 else None)
    ax.barh(i+.18, b, height=.32, color='#267a64', label='Forest + decoder' if i == 0 else None)
    ax.text(a + 0.6, i-.18, f'{a:.2f}%', va='center')
    ax.text(b + 0.6, i+.18, f'{b:.2f}%', va='center')
ax.set_yticks([0,1], rows); ax.invert_yaxis(); ax.set_xlim(0, 110)
ax.set_xlabel('Student-test observations / pairs (%)'); ax.set_title('Effect of sequence decoding')
ax.legend(loc='lower right', frameon=False)
fig.suptitle('PathSnap development results', fontsize=15)
fig.savefig(root / 'model_explanation.png', dpi=180)
