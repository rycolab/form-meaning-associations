import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser


def format_float(value, digits, decimals, ever_negative):
    res = ''
    if ever_negative:
        digits -= 1

    real_digits = 1
    while abs(value) >= 10**real_digits:
        real_digits += 1

    res += '\\phantom{0}' * (digits - real_digits)

    if ever_negative:
        if value > 0 and ((digits - real_digits) >= 0 or digits < 1):
            res += '\\phantom{-}'

    res += '%%.%df' % (decimals)
    return res % value


def format_int(number, max_digits):
    real_digits = 1
    while abs(number) >= 10**real_digits:
        real_digits += 1
    n_zeros = max_digits - real_digits
    l_zeros = '\\phantom{0}' * n_zeros

    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return l_zeros + s + '{,}'.join(reversed(groups))


def format_significance(string, p_value):
    if p_value < 0.01:
        sig_str = '$^\ddagger$'
    elif p_value < 0.05:
        sig_str = '$^\dagger$'
    elif p_value < 0.1:
        sig_str = '$^{*}$'
    else:
        sig_str = '\phantom{$^\dagger$}'

    return string + sig_str


def main():
    args = argparser.parse_args(csv_folder='cv')

    fname = os.path.join(args.rfolder_base, 'macroarea_results.tsv')
    df = pd.read_csv(fname, sep='\t')
    fname = os.path.join(args.rfolder_base, 'overall_results.tsv')
    df_mean = pd.read_csv(fname, sep='\t')

    print(df)

    context = 'onehot'
    base = 'none'

    base_str = '%s & %s & %s & %.3f & %s & %.3f\\%% \\\\'
    cols = ['train', 'dev', 'test', base, 'mi-' + context, 'unc-' + context]

    df['mi-' + context] = df['mi-' + context].apply(lambda x: '%.3f' % x)
    df['mi-' + context] = df.apply(lambda x: format_significance(x['mi-' + context], x['p_value-' + context]), axis=1)
    df['unc-' + context] = df['unc-' + context] * 100
    # print(df)


    for i, x in df[cols].iterrows():
        print(base_str % tuple(x.values))
    print('\\midrule')


    df_mean['mi-' + context] = df_mean['mi-' + context].apply(lambda x: '%.3f' % x)
    df_mean['mi-' + context] = df_mean.apply(lambda x: format_significance(x['mi-' + context], x['p_value-' + context]), axis=1)
    df_mean['unc-' + context] = df_mean['unc-' + context] * 100

    cols = [base, 'mi-' + context, 'unc-' + context]
    print('\multicolumn{3}{l}{Average} & %.3f & %s & %.3f\\%% \\\\' % tuple(df_mean[cols].values[0]))


if __name__ == '__main__':
    main()
