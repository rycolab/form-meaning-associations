import pandas as pd


def _get_opt_params(fname, delimiter='\t'):
    results = pd.read_csv(fname, delimiter=delimiter)
    instance = results.iloc[0]

    embedding_size = int(instance['embedding_size'])
    hidden_size = int(instance['hidden_size'])
    concept_size = int(instance['concept_size'])
    nlayers = int(instance['nlayers'])
    dropout = instance['dropout']

    return embedding_size, hidden_size, concept_size, nlayers, dropout


def get_opt_params(args):
    # context = args.context if 'shuffle' not in args.context else args.context[:-8]

    rfolder = args.rfolder
    rfolder = rfolder.replace('/cv/', '/bayes-opt/')
    rfolder = rfolder.replace('/normal/', '/bayes-opt/')
    rfolder = rfolder.replace('/opt/', '/bayes-opt/')

    fname = '%s/%s__%s__opt-results.csv' \
        % (rfolder, args.model, args.context)
    return _get_opt_params(fname, delimiter=',')
