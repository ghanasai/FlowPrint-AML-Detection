"""FlowPrint faculty demonstration dashboard.
Run from your FlowPrint project directory: python flowprint_demo.py
Dependencies: pandas, matplotlib, networkx (optional for graph rendering).
Reads saved experiment outputs only; does not retrain or modify research results.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'


def load(filename, required=None):
    path = RESULTS / filename
    if not path.exists():
        print(f'  [Not available] {filename}')
        return None
    try:
        frame = pd.read_csv(path)
        missing = set(required or []) - set(frame.columns)
        if missing:
            print(f'  [Missing columns] {filename}: {sorted(missing)}')
            return None
        return frame
    except Exception as exc:
        print(f'  [Could not read] {filename}: {exc}')
        return None


def section(name):
    print('\n' + '=' * 76)
    print(name)
    print('=' * 76)


def table(df, columns=None, digits=4):
    if df is None or df.empty:
        print('  Results unavailable.')
        return
    display = df[columns].copy() if columns else df.copy()
    print(display.round(digits).to_string(index=False))


def plot_comparison(df):
    if df is None or not {'Model', 'PR-AUC', 'ROC-AUC', 'F1'}.issubset(df.columns):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ['PR-AUC', 'ROC-AUC', 'F1']):
        ax.bar(df['Model'], df[metric])
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelrotation=15)
        for i, v in enumerate(df[metric]):
            ax.text(i, min(float(v) + .025, .96), f'{v:.3f}', ha='center')
    fig.suptitle('FlowPrint: chronological test results')
    fig.tight_layout()
    return fig


def plot_ablation(df):
    if df is None or not {'Model', 'PR-AUC'}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(df['Model'], df['PR-AUC'])
    ax.invert_yaxis()
    ax.set_xlabel('Validation PR-AUC')
    ax.set_xlim(0, max(.75, df['PR-AUC'].max() * 1.15))
    for bar, val in zip(bars, df['PR-AUC']):
        ax.text(bar.get_width() + .012, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
    ax.set_title('Controlled feature ablation (validation)')
    fig.tight_layout()
    return fig


def network_metrics():
    pred = load('final_test_predictions.csv', ['nameOrig', 'nameDest', 'isMoneyLaundering', 'predicted_suspicious'])
    if pred is None:
        return
    try:
        import networkx as nx
    except ImportError:
        print('  Install networkx to calculate test network metrics.')
        return
    true = pred[pred['isMoneyLaundering'] == 1]
    flagged = pred[pred['predicted_suspicious'] == 1]
    detected = flagged[flagged['isMoneyLaundering'] == 1]
    true_accounts = set(true['nameOrig']) | set(true['nameDest'])
    flagged_accounts = set(flagged['nameOrig']) | set(flagged['nameDest'])
    true_pairs = set(zip(true['nameOrig'], true['nameDest']))
    detected_pairs = set(zip(detected['nameOrig'], detected['nameDest']))
    g_true, g_pred = nx.Graph(), nx.Graph()
    g_true.add_edges_from(true_pairs)
    g_pred.add_edges_from(zip(flagged['nameOrig'], flagged['nameDest']))
    truth_components = list(nx.connected_components(g_true))
    pred_components = list(nx.connected_components(g_pred))
    lookup = {node: i for i, comp in enumerate(pred_components) for node in comp}
    coverage = []
    for comp in truth_components:
        overlap = {}
        for node in comp:
            if node in lookup:
                idx = lookup[node]
                overlap[idx] = overlap.get(idx, 0) + 1
        coverage.append(max(overlap.values(), default=0) / len(comp))
    print(f'  True AML accounts: {len(true_accounts)}')
    print(f'  AML account coverage: {len(true_accounts & flagged_accounts) / len(true_accounts):.2%}' if true_accounts else '  No true AML accounts')
    print(f'  AML transaction recall: {len(detected) / len(true):.2%}' if len(true) else '  No true AML transactions')
    print(f'  Directed pair recall: {len(detected_pairs) / len(true_pairs):.2%}' if true_pairs else '  No true AML pairs')
    print(f'  True components: {len(truth_components)}; candidate networks: {len(pred_components)}')
    print(f'  Mean largest-component account coverage: {sum(coverage)/len(coverage):.2%}' if coverage else '  No true components')
    print(f'  Fully covered true components: {sum(v == 1 for v in coverage)}/{len(coverage)}')
    print(f'  Transaction-weighted candidate purity: {len(detected)/len(flagged):.2%}' if len(flagged) else '  No flagged transactions')


def main():
    section('FLOWPRINT |RESEARCH DEMONSTRATION')
    print('Open-world temporal graph intelligence for AML detection,')
    print('candidate-network reconstruction, and topology-based roles.')
    print('This dashboard reads saved results only; NO model retraining.')
    print('Validation and test findings are labeled separately.')

    section('1 | CHRONOLOGICAL TEST: DETECTION PERFORMANCE')
    final = load('final_test_comparison.csv', ['Model', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1'])
    table(final)
    if final is not None:
        plot_comparison(final)

    section('2 | CONTROLLED FEATURE ABLATION (VALIDATION)')
    ablation = load('controlled_ablation_results.csv', ['Model', 'PR-AUC'])
    table(ablation)
    if ablation is not None:
        plot_ablation(ablation)
    print('  Note: historical features outperform the full model on validation.')

    section('3 | FINAL TEST: NETWORK RECONSTRUCTION')
    network_metrics()
    component = load('final_test_component_recovery.csv')
    purity = load('final_test_network_purity.csv')
    if component is not None:
        print(f'  Component recovery rows saved: {len(component)}')
    if purity is not None and 'purity' in purity:
        print(f'  Mean candidate-network purity: {purity["purity"].mean():.2%}')

    section('4 | STRUCTURAL ROLES (VALIDATION CANDIDATE NETWORKS)')
    roles = load('improved_account_roles.csv')
    if roles is not None:
        if 'roles' in roles:
            for role in ['Source-like', 'Destination-like', 'Splitter-like', 'Convergence-like', 'Intermediary-like']:
                count = roles['roles'].fillna('').str.split(',').apply(lambda values: role in [v.strip() for v in values]).sum()
                print(f'  {role}: {count}')
        else:
            print('  Role file available; columns:', ', '.join(roles.columns))
    print('  Roles are structural descriptors, not verified laundering roles.')
    picture = RESULTS / 'largest_candidate_network.png'
    if picture.exists():
        try:
            img = plt.imread(picture)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title('Example candidate network (VALIDATION)')
            fig.tight_layout()
        except Exception as exc:
            print(f'  Network image not displayable: {exc}')
    else:
        print('  Optional validation network image not found.')

    section('5 | TYPOLOGY RESULTS')
    print('  Test-set known-typology detection:')
    table(load('final_test_typology.csv'))
    print('\n  Held-out typology experiments (VALIDATION):')
    table(load('open_world_results.csv'))

    section('6 | RESEARCH INTERPRETATION')
    print('  * Strong evidence: historical account features improve test PR-AUC.')
    print('  * Candidate network recovery is partial, not full chain tracing.')
    print('  * Integration detection is weak; open-world evidence is limited.')
    print('  * Feature availability and dataset-generation assumptions matter.')
    print('  * The baseline and full model use different decision thresholds;')
    print('    PR-AUC is the cleaner threshold-independent comparison.')

    figures = plt.get_fignums()
    if figures:
        print('\nOpening figures. Close chart windows to finish the demo.')
        plt.show()
    print('\nFaculty demonstration completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('\nDemo stopped.')
