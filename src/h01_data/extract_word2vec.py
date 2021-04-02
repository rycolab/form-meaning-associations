import os
import sys
import pickle
import gensim

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.asjp import AsjpInfo
import util.argparser as parser


def get_word2vec(language, data_path):
    if language in 'eng':
        # Load Google's pre-trained Word2Vec model.
        path = '%s/word2vec/GoogleNews-vectors-negative300.bin.gz' % (data_path)
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        raise ValueError('Invalid word2vec language name: %s' % language)


def extract_asjp(args):
    df = AsjpInfo().get_df(args.ffolder)
    base_df = df[['concept_id_train', 'concept_name']].drop_duplicates()

    model = get_word2vec('eng', args.data_path)

    model_filtered = {
        row['concept_id_train']: model[row['concept_name'].lower()] if row['concept_name'].lower() in model else None
        for _, row in base_df.iterrows()
    }

    print('Gotten:', sum([1 for x in model_filtered.values() if x is not None]))
    print('Unknown:', sum([1 for x in model_filtered.values() if x is None]))
    print('Total:', df['concept_id_train'].unique().shape[0])

    fname = '%s/filtered-word2vec.pckl' % (args.ffolder)
    with open(fname, 'wb') as f:
        pickle.dump(model_filtered, f, protocol=-1)


def main():
    args = parser.parse_args()

    if args.data == 'asjp':
        extract_asjp(args)
    else:
        raise ValueError('Invalid data name: %s' % args.data)


if __name__ == '__main__':
    main()
