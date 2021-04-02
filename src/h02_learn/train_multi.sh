DATA_SPLITS=macroarea

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for context in 'onehot' 'none'
    do
        echo ${context} - ${seed}
        make CONTEXT=${context} SEED=${seed} DATASET=asjp SPLITS=${DATA_SPLITS}
    done
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for context in 'onehot-shuffle'
    do
        echo ${context} - ${seed}
        make CONTEXT=${context} SEED=${seed} DATASET=asjp SPLITS=${DATA_SPLITS}
    done
done
