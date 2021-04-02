# form-meaning-associations

Code accompanying the paper "Finding Concept-specific Biases in Form--Meaning Associations" accepted at NAACL 2021.


## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
```

## Parse data

To pre-process the data run:
```bash
$ make process_data SPLITS=<split-used>
```
As detailed in the paper, `split-used` can be either `macroarea` or `family`.


## Train models

To train the models run:
```bash
$ make train SPLITS=<split-used> CONTEXT=<context>
```
Or train all seeds in sequence with:
```bash
$ python src/h02_learn/train_multi.sh
```

Context can be:
* none: No context used
* onehot: OneHot context used (this was the model used in the paper)
* word2vec: Word2Vec context used
* onehot-shuffle: OneHot context with shuffled meaning ids


## Analyse the results

To produce the agregate analysis files, run:
```bash
$ make analysis
$ make get_seed_results SPLITS=macroarea
$ make get_seed_results SPLITS=family
```
This will produce all the result files for macroarea, and a per seed result for the family splits.
With these at hand you can run the scripts in `src/h05_paper/` directly to get the paper plots and tables.

The family vs macroarea analysis in the paper was made manually using the results files


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing the paper:
```bash
@inproceedings{pimentel-etal-2021-finding,
    title = "Finding Concept-specific Biases in Form--Meaning Associations",
    author = "Pimentel, Tiago  and
      Roark, Brian  and
      Wichmann, S{\o}ren  and
      Cotterell, Ryan  and
      Bl\'{a}si, Damian",
    booktitle = "Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2021",
    address = "Virtual",
    publisher = "Association for Computational Linguistics",
}
```

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/form-meaning-associations/issues).
