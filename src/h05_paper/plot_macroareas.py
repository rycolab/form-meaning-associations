import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.asjp import AsjpInfo
from util import argparser
from util import util


aspect = {
    'height': 7,
    'font_scale': 7.5,
    'labels': True,
    'name_suffix': '',
    'ratio': 2.125,
}
sns.set_palette("muted")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def legend(fig, ax, x0=1,y0=1, direction = "v", padpoints = 3,**kwargs):
    otrans = ax.figure.transFigure
    h, l = ax.get_legend_handles_labels()

    t = ax.legend(h[:4], l[:4], ncol=4, bbox_to_anchor=(x0,y0), loc=1, bbox_transform=otrans, handletextpad=-0.5, columnspacing=1.3, frameon=False, borderpad=0, **kwargs)
    for i in range(4):
        t.legendHandles[i]._sizes = [500]

    plt.tight_layout(pad=0)
    ax.figure.canvas.draw()
    plt.tight_layout(pad=0)
    ppar = [0, -padpoints / 72.] if direction == "v" else [-padpoints / 72., 0]
    trans2 = mpl.transforms.ScaledTranslation(ppar[0], ppar[1], fig.dpi_scale_trans) + \
        ax.figure.transFigure.inverted()
    tbox = t.get_window_extent().transformed(trans2)
    bbox = ax.get_position()
    if direction == "v":
        diff = (bbox.width - tbox.width) / 2
        ax.set_position([tbox.x0, bbox.y1 - .05, tbox.x1, tbox.y1 + bbox.y0])
    else:
        ax.set_position([bbox.x0, bbox.y0, tbox.x0 - bbox.x0, bbox.height])


def plot_languages(df, family_column, folder=None):
    colors = {
        'Americas': 'C4',
        'Africa': 'C0',
        'Eurasia': 'C2',
        'Pacific': 'C3',
    }

    # Set the dimension of the figure
    my_dpi = 96
    fig, ax = plt.subplots(figsize=(2600 / my_dpi, 1800 / my_dpi), dpi=my_dpi)

    # Make the background map
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)
    m.drawmapboundary(fill_color='#FFFFFF', linewidth=0)
    m.fillcontinents(color='#72A0C1', zorder=1, alpha=0.5)
    m.drawcoastlines(linewidth=0.1, color="black")

    # Plot languages
    df.sort_values('order', ascending=True, inplace=True)
    ax = sns.scatterplot(
        x="Longitude", y="Latitude", hue=family_column, palette=colors,
        data=df, s=70, zorder=2, legend='brief', linewidth=0)

    legend(fig, ax, borderaxespad=0.2, direction='v')

    fname = 'plot_languages-%s.pdf' % family_column
    if folder:
        util.mkdir(folder)
        fname = os.path.join(folder, fname)
    plt.savefig(fname, bbox_inches='tight')


def main():
    args = argparser.parse_args(csv_folder='cv')
    df = AsjpInfo.get_df(args.ffolder)

    df = df[['language_id', 'Latitude', 'Longitude', 'macroarea', 'macroarea_orig']]
    df.drop_duplicates(['language_id', 'macroarea', 'macroarea_orig'], inplace=True)

    order = {
        'Americas': 0,
        'Africa': 1,
        'Eurasia': 2,
        'Pacific': 3,
    }

    df['order'] = df['macroarea'].apply(lambda x: order[x])
    df['Macro Area'] = df['macroarea']
    df['Macro Area Original'] = df['macroarea_orig']
    folder = os.path.join(args.rfolder_base, 'plots')

    plot_languages(df, family_column='macroarea', folder=folder)


if __name__ == '__main__':
    main()
