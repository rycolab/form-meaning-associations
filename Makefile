CONTEXT := none
SEED := 7

DATASET := asjp
SPLITS := macroarea

DATA_DIR_BASE := ./datasets
DATA_DIR := $(DATA_DIR_BASE)/$(DATASET)
PROCESSED_DATA_DIR := $(DATA_DIR)/preprocess-$(SPLITS)

CHECKPOINT_DIR_BASE := ./checkpoints
CHECKPOINT_DIR := $(CHECKPOINT_DIR_BASE)/$(DATASET)/$(SPLITS)
CHECKPOINT_DIR_BAYES := $(CHECKPOINT_DIR)/bayes-opt/seed_00${SEED}
CHECKPOINT_DIR_CV := $(CHECKPOINT_DIR)/cv/seed_00${SEED}

RESULTS_DIR_BASE := ./results
RESULTS_DIR := $(RESULTS_DIR_BASE)/$(DATASET)/$(SPLITS)
RESULTS_DIR_BAYES := $(RESULTS_DIR)/bayes-opt/seed_00${SEED}
RESULTS_DIR_CV := $(RESULTS_DIR)/cv/seed_00${SEED}

DATA_FILE_EXTRACTED := $(DATA_DIR)/extracted.tsv
WORD2VEC_EXTRACTED := $(DATA_DIR)/filtered-word2vec.pckl
DATA_FILE_PROCESSED := $(PROCESSED_DATA_DIR)/info.pckl
BAYESIAN_OPT_FILE := $(RESULTS_DIR_BAYES)/lstm__${CONTEXT}__opt-results.csv
CV_MODEL_FILE := $(CHECKPOINT_DIR_CV)/lstm__${CONTEXT}__fold-3.tch
LOSSES_FILE := $(CHECKPOINT_DIR_CV)/losses__lstm__${CONTEXT}.pckl
LOSSES_FILE_TRAIN := $(CHECKPOINT_DIR_CV)/results-avg__lstm__${CONTEXT}.csv

ASJP_EXTRACTED := $(DATA_DIR_BASE)/asjp/extracted.tsv
ASJP_RAW := $(DATA_DIR)/listss19_formatted.tsv

RESULTS_OVERALL := $(RESULTS_DIR)/overall_results.tsv
RESULTS_SEED := $(RESULTS_DIR)/seed_results.tsv
RESULTS_AVG_SEED := $(RESULTS_DIR)/avg_seed_results.tsv
RESULTS_AVG_SEED_PER_POS := $(RESULTS_DIR)/avg_seed_results_per_pos.tsv
RESULTS_CONCEPTS := $(RESULTS_DIR)/concepts_results.tsv
RESULTS_LANGUAGES := $(RESULTS_DIR)/languages_results.tsv
RESULTS_TOKENS := $(RESULTS_DIR)/tokens_results.tsv
RESULTS_TOKENS_CORRECTIONS := $(RESULTS_DIR)/tokens_results--corrected.tsv

all: process_data train_bayesian train_cv eval

analysis: $(RESULTS_OVERALL) $(RESULTS_AVG_SEED) $(RESULTS_AVG_SEED_PER_POS) $(RESULTS_CONCEPTS) $(RESULTS_LANGUAGES) $(RESULTS_TOKENS) $(RESULTS_TOKENS_CORRECTIONS)

get_seed_results: $(RESULTS_SEED)

eval: $(LOSSES_FILE)

eval_train: $(LOSSES_FILE_TRAIN)

train: train_bayesian train_cv

train_cv: $(CV_MODEL_FILE)

train_bayesian: $(BAYESIAN_OPT_FILE)

process_data: $(DATA_FILE_PROCESSED) $(WORD2VEC_EXTRACTED)

$(RESULTS_SEED):
	python src/h04_analysis/get_results_per_seed.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_TOKENS_CORRECTIONS):
	python src/h04_analysis/add_corrections.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_TOKENS):
	python src/h04_analysis/get_results_per_token.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_LANGUAGES):
	python src/h04_analysis/get_results_per_language.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_CONCEPTS):
	python src/h04_analysis/get_results_per_concept.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_AVG_SEED_PER_POS):
	python src/h04_analysis/avg_seed_per_pos.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_AVG_SEED):
	python src/h04_analysis/avg_seed_results.py --data $(DATASET) --data-split $(SPLITS)

$(RESULTS_OVERALL):
	python src/h04_analysis/get_results_per_macroarea.py --data $(DATASET) --data-split $(SPLITS)

$(LOSSES_FILE_TRAIN): | $(CV_MODEL_FILE)
	echo "Eval cv models" $(LOSSES_FILE_TRAIN)
	python src/h03_eval/eval_train.py --data $(DATASET) --data-split $(SPLITS) --context $(CONTEXT) --seed $(SEED) --opt

$(LOSSES_FILE): | $(CV_MODEL_FILE)
	echo "Eval cv models" $(LOSSES_FILE)
	python src/h03_eval/eval.py --data $(DATASET) --data-split $(SPLITS) --context $(CONTEXT) --seed $(SEED) --opt

$(CV_MODEL_FILE): | $(BAYESIAN_OPT_FILE)
	echo "Train cv models" $(CV_MODEL_FILE)
	python src/h02_learn/train_cv.py --data $(DATASET) --data-split $(SPLITS) --context $(CONTEXT) --seed $(SEED) --opt

# Train types Model
$(BAYESIAN_OPT_FILE): | $(DATA_FILE_PROCESSED)
	echo "Train bayesian models" $(BAYESIAN_OPT_FILE)
	python src/h02_learn/train_bayesian.py --data $(DATASET) --data-split $(SPLITS) --context $(CONTEXT) --seed $(SEED)

$(DATA_FILE_PROCESSED): | $(DATA_FILE_EXTRACTED) $(WORD2VEC_EXTRACTED)
	python src/h01_data/parse.py --data $(DATASET) --data-split $(SPLITS)

$(WORD2VEC_EXTRACTED): | $(DATA_FILE_EXTRACTED)
	python src/h01_data/extract_word2vec.py --data $(DATASET)

$(ASJP_EXTRACTED): | $(ASJP_RAW)
	python src/h01_data/extract_asjp.py --data $(DATASET)
