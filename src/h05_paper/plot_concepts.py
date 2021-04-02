import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser
from util import util

seed = 7
idx_concepts = 'Parameter_ID'
idx_lang = 'Language_ID'
num_tests = 100000
per_word = False
per_word_str = '-per_word' if per_word else ''

util.config(seed)

aspect = {
    'height': 6.5,
    'font_scale': 1.8,
    'labels': False,
    'ratio': 1.625,
}
aspect = {
    'height': 6.5,
    'font_scale': 1.8,
    'labels': True,
    'name_suffix': '',
    'ratio': 2.125,
}
sns.set_palette("muted")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
# mpl.rcParams['text.usetex'] = True


def label_points(x, y, val, should_plot, significant, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val, 'should_plot': should_plot, 'significant': significant}, axis=1)
    texts = []
    color = 'C6'
    for i, point in a.iterrows():
        if point['should_plot']:
            texts += [ax.text(point['x'], point['y'], str(point['val']), color=color, size=20, weight='bold')]

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color=color, lw=0.5))


def plot_with_names(df, alpha, folder):
    colors = {'$p \geq %.2f$' % alpha: "C0", '$p < %.2f$' % alpha: "C2"}
    # df['phoneme_len'] = df['phoneme_len']
    fig = sns.lmplot(
        x="length", y="nMI (%)", data=df, hue='Significant',
        fit_reg=False, height=aspect['height'], aspect=aspect['ratio'], legend_out=False, truncate=False,
        palette=colors)
    ax = plt.gca()
    label_points(df['length'], df['nMI (%)'], df['concept_name'], df['should_plot'], df['Significant'], ax)

    plt.xlabel('Average Wordform Length (# tokens)')
    plt.ylabel('Uncertainty Coefficient (%)')

    fname = 'plot_concepts.pdf'
    if folder:
        util.mkdir(folder)
        fname = os.path.join(folder, fname)
    fig.savefig(fname, bbox_inches='tight')
    plt.close()


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'
    alpha = 0.01

    fname = os.path.join(args.rfolder_base, 'concepts_results--corrected.tsv')
    df = pd.read_csv(fname, sep='\t', index_col=0)

    assert not df.isna().any().any(), 'No nan values in data'

    df['should_plot'] = df['significant-%.2f' % alpha]
    df['Significant'] = '$p \geq %.2f$' % alpha
    df.loc[df['should_plot'], 'Significant'] = '$p < %.2f$' % alpha

    df['nMI (%)'] = df['unc-' + context] * 100

    # import ipdb; ipdb.set_trace()

    df.sort_values('Significant', inplace=True, ascending=False)

    folder = os.path.join(args.rfolder_base, 'plots')
    plot_with_names(df, alpha, folder)

    print('# Significants (%.2f): %d' % (alpha, df['significant-%.2f' % alpha].sum()))


if __name__ == '__main__':
    main()
