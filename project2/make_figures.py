"""Aggregate results (mean +- std over seeds), compute two-stream fusion, and
produce the tables/figures used in the report (results/figures, results/tables.tex)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)
DATA = os.path.dirname(HERE)
NOLEAK_ROOT = os.path.join(DATA, 'datasets', 'ucf101_noleakage')   # kept torch-free (OpenMP clash with matplotlib)

# reference palette (light): categorical slots 1-2, chrome, sequential blue ramp
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 7, 'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})

LABELS = {
    'perframe': 'Per-frame (avg.)', 'late': 'Late fusion', 'early': 'Early fusion',
    '3d_scratch': '3D CNN (scratch)', '3d_inflated': '3D CNN (inflated)',
    'flow': 'Flow stream', 'twostream': 'Two-stream (fused)',
}
ORDER = ['perframe', 'late', 'early', '3d_scratch', '3d_inflated']


def run_dir(dataset, model, seed):
    return os.path.join(RES, 'runs', f'{dataset}_{model}_s{seed}')


def two_stream_rows(df):
    """Late fusion of class scores: average spatial (per-frame RGB) and temporal (flow) softmax."""
    rows = []
    seeds = sorted(set(df[(df.dataset == 'noleak') & (df.model == 'flow')].seed)
                   & set(df[(df.dataset == 'noleak') & (df.model == 'perframe')].seed))
    for s in seeds:
        sp = np.load(os.path.join(run_dir('noleak', 'perframe', s), 'test_probs.npy'))
        tp = np.load(os.path.join(run_dir('noleak', 'flow', s), 'test_probs.npy'))
        y = np.load(os.path.join(run_dir('noleak', 'flow', s), 'test_labels.npy'))
        spv = np.load(os.path.join(run_dir('noleak', 'perframe', s), 'val_probs.npy'))
        tpv = np.load(os.path.join(run_dir('noleak', 'flow', s), 'val_probs.npy'))
        rows.append(dict(dataset='noleak', model='twostream', seed=s,
                         test_acc=((sp + tp) / 2).argmax(1).__eq__(y).mean(),
                         val_acc=np.nan, probs=(sp + tp) / 2, y=y))
    return rows


def fmt(m, s):
    return f'{100 * m:.1f} $\\pm$ {100 * s:.1f}'


def main():
    df = pd.read_csv(os.path.join(RES, 'results.csv'))
    df = df.drop_duplicates(['dataset', 'model', 'seed'], keep='last')
    ts = two_stream_rows(df)
    full = pd.concat([df, pd.DataFrame([{k: v for k, v in r.items() if k not in ('probs', 'y')}
                                        for r in ts])], ignore_index=True)

    agg = full.groupby(['dataset', 'model']).agg(
        test_mean=('test_acc', 'mean'), test_std=('test_acc', 'std'),
        val_mean=('val_acc', 'mean'), n=('seed', 'count'),
        frame_mean=('test_frame_acc', 'mean'), params_M=('params_M', 'first'),
        time_s=('train_time_s', 'mean')).reset_index()
    agg['test_std'] = agg.test_std.fillna(0)
    agg.to_csv(os.path.join(RES, 'summary.csv'), index=False)
    print(agg.to_string())

    def get(d, m):
        r = agg[(agg.dataset == d) & (agg.model == m)]
        return (r.test_mean.item(), r.test_std.item()) if len(r) else (np.nan, 0)

    # ---------------------------------------------------------------- tables
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             r'Model & Params & Leaky test acc. & No-leak test acc. \\', r'\midrule']
    for m in ORDER:
        p = agg[(agg.model == m)].params_M.iloc[0]
        lines.append(f'{LABELS[m]} & {p:.1f}M & {fmt(*get("leaky", m))} & {fmt(*get("noleak", m))} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    lines += [r'\begin{tabular}{lc}', r'\toprule', r'Stream & No-leak test acc. \\', r'\midrule']
    for m, name in [('perframe', 'Spatial (RGB, per-frame avg.)'), ('flow', 'Temporal (stacked flow)'),
                    ('twostream', 'Two-stream (score averaging)')]:
        lines.append(f'{name} & {fmt(*get("noleak", m))} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    pf = agg[(agg.model == 'perframe')].set_index('dataset').frame_mean
    lines += ['', f'% single-frame test acc (no aggregation): leaky {100 * pf.get("leaky", np.nan):.1f}, '
                  f'noleak {100 * pf.get("noleak", np.nan):.1f}']
    with open(os.path.join(RES, 'tables.tex'), 'w') as f:
        f.write('\n'.join(lines))

    # ---------------------------------------------------------------- fig: leaky vs no-leak
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    x = np.arange(len(ORDER))
    w = 0.38
    for k, (d, c, lab) in enumerate([('leaky', BLUE, 'Kaggle splits (leakage)'),
                                     ('noleak', ORANGE, 'Subject-disjoint splits')]):
        ms = [get(d, m) for m in ORDER]
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, [100 * a for a, _ in ms], w, color=c, label=lab, linewidth=0)
        ax.errorbar(xs, [100 * a for a, _ in ms], [100 * s for _, s in ms],
                    fmt='none', ecolor=INK2, elinewidth=0.6, capsize=1.5)
        for xi, (a, s) in zip(xs, ms):
            ax.text(xi, 100 * (a + s) + 1.5, f'{100 * a:.0f}', ha='center', va='bottom',
                    fontsize=5.5, color=INK2)
    ax.set_xticks(x, [LABELS[m].replace(' (', '\n(').replace(' fusion', '\nfusion') for m in ORDER],
                  fontsize=6)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6)
    fig.savefig(os.path.join(FIG, 'accuracy_leakage.pdf'))
    fig.savefig(os.path.join(FIG, 'accuracy_leakage.png'))
    plt.close(fig)

    # ---------------------------------------------------------------- fig: confusion (two-stream)
    meta = pd.read_csv(os.path.join(DATA, 'datasets', 'ufc101', 'metadata', 'train.csv'))
    names = list(meta.drop_duplicates('label').sort_values('label').action)
    if ts:
        cm = np.zeros((10, 10))
        for r in ts:
            for t, p in zip(r['y'], r['probs'].argmax(1)):
                cm[t, p] += 1
        cm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('seq', SEQ), vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f'{100 * cm[i, j]:.0f}', ha='center', va='center', fontsize=5,
                            color='white' if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(10), names, rotation=60, ha='right', fontsize=5.5)
        ax.set_yticks(range(10), names, fontsize=5.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.savefig(os.path.join(FIG, 'confusion_twostream.pdf'))
        fig.savefig(os.path.join(FIG, 'confusion_twostream.png'))
        plt.close(fig)

    # ---------------------------------------------------------------- fig: example frames + flow
    root = NOLEAK_ROOT
    if os.path.isdir(root):
        plot_examples(root)
    else:
        print(f'skipping examples figure: {root} not found')
    plot_curves()


def plot_examples(root):
    meta = pd.read_csv(os.path.join(root, 'metadata', 'train.csv'))
    picks = [meta[meta.action == a].iloc[0] for a in ['JumpingJack', 'PullUps']]
    fig, axes = plt.subplots(2 * len(picks), 5, figsize=(3.3, 2.75))
    for r, row in enumerate(picks):
        rel = os.path.splitext(row.video_path)[0]
        for c, i in enumerate([1, 3, 5, 7, 9]):
            axes[2 * r, c].imshow(Image.open(os.path.join(root, 'frames', 'train', rel, f'frame_{i}.jpg')))
            axes[2 * r + 1, c].imshow(Image.open(os.path.join(root, 'flows_png', 'train', rel,
                                                              f'flow_{i}_{i + 1}.png')))
        axes[2 * r, 0].set_ylabel(row.action, fontsize=6)
        axes[2 * r + 1, 0].set_ylabel('flow', fontsize=6)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(FIG, 'examples.pdf'))
    fig.savefig(os.path.join(FIG, 'examples.png'))
    plt.close(fig)


def plot_curves():
    """Learning curves for seed 0."""
    models = ORDER + ['flow']
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.0), sharey=True)
    for a, m in zip(axes.flat, models):
        for d, c in [('leaky', BLUE), ('noleak', ORANGE)]:
            p = os.path.join(run_dir(d, m, 0), 'history.json')
            if not os.path.exists(p):
                continue
            h = json.load(open(p))['history']
            ep = [e['epoch'] for e in h]
            a.plot(ep, [100 * e['train_acc'] for e in h], color=c, lw=1.0, ls='--')
            a.plot(ep, [100 * e['val_acc'] for e in h], color=c, lw=1.4)
        a.set_title(LABELS[m], fontsize=7, color=INK)
        a.set_ylim(0, 102)
    for a in axes[1]:
        a.set_xlabel('Epoch')
    for a in axes[:, 0]:
        a.set_ylabel('Accuracy (%)')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=ORANGE, lw=1.4),
                Line2D([], [], color=INK2, lw=1.0, ls='--'), Line2D([], [], color=INK2, lw=1.4)],
               ['Leaky splits', 'No-leak splits', 'Train', 'Validation'],
               loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'curves.pdf'))
    fig.savefig(os.path.join(FIG, 'curves.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
