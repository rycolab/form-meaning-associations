import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

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
    'height': 7,
    'font_scale': 1.8,
    'labels': True,
    'name_suffix': '',
    'ratio': 2.125,
}
sns.set_palette("muted")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def plot_languages(df, folder=None):
    # Set the dimension of the figure
    my_dpi = 96
    plt.figure(figsize=(2600 / my_dpi, 1800 / my_dpi), dpi=my_dpi)

    # Make the background map
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)
    m.drawmapboundary(fill_color='#FFFFFF', linewidth=0)
    m.fillcontinents(color='#72A0C1', zorder=1, alpha=0.5)
    m.drawcoastlines(linewidth=0.1, color="black")

    # Plot uncertainty coefficients
    colours = ['C0', 'C8', 'C1', 'C2',]
    ax = sns.scatterplot(
        x="Longitude", y="Latitude", size="size", hue="Significant", palette=colours,
        data=df, sizes=(20, 800), zorder=2, legend='brief', edgecolor='black', linewidth=0)

    h, l = ax.get_legend_handles_labels()
    col_lgd = plt.legend(
        h[:5], l[:5], loc='lower left')
        # h[:4], l[:4], loc='lower left')

    fname = 'plot_languages.pdf'
    if folder:
        util.mkdir(folder)
        fname = os.path.join(folder, fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'

    fname = os.path.join(args.rfolder_base, 'languages_results--corrected.tsv')
    df = pd.read_csv(fname, sep='\t', index_col=0, keep_default_na=False)

    df['nMI (%)'] = df['unc-' + context] * 100

    df['Significant'] = '$p \geq 0.1$'
    df['significant_order'] = 0
    df.loc[df['significant-0.10'], 'Significant'] = '$p<0.1$'
    df.loc[df['significant-0.10'], 'significant_order'] = 1
    df.loc[df['significant-0.05'], 'Significant'] = '$p<0.05$'
    df.loc[df['significant-0.05'], 'significant_order'] = 2
    df.loc[df['significant-0.01'], 'Significant'] = '$p<0.01$'
    df.loc[df['significant-0.01'], 'significant_order'] = 3
    df.sort_values('significant_order', inplace=True, ascending=True)

    df['size'] = df['nMI (%)']
    df.loc[df[df['nMI (%)'] < 0].index, 'size'] = 0

    folder = os.path.join(args.rfolder_base, 'plots')
    plot_languages(df, folder)

    print('Count significance')
    for significance in df.Significant.unique():
        print('%s: %d' % (significance, (df.Significant == significance).sum()))

    df_temp = df.groupby('iso_code').agg('sum')
    print('Significance 0.01: %d' % (df_temp['significant-0.01'] > 0).sum())
    print('Significance 0.05: %d' % (df_temp['significant-0.05'] > 0).sum())
    print('Significance 0.10: %d' % (df_temp['significant-0.10'] > 0).sum())
    print('Non Significant: %d' % (df_temp['significant-0.10'] <= 0).sum())
    print('Total: %d' % df_temp.shape[0])

    print('Total doculets: %d' % df.shape[0])


if __name__ == '__main__':
    main()
