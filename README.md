Neural Network Model for Predicting Monoclonal Antibody Binding Affinity

**FINAL MODEL**

final_model_script.py - runs saved model as well and calls upon additional script (esm_embed_seq_finalmodel.py) to generate embeddings. Plots all the results.

esm_embed_seq_finalmodel.py - ran inside the esm3 directory (copied off github) in order to generate single embedding 

**TRAINING**

Dataset Creation Workflow

Getting SILCS training Dataset (env: dataset_creation)

1. copy_clean.py - based on sheet.csv that I-En updates - copied mAb folders into my work area
2. convert_all_maps - changes all maps files to dx files using
   map2dx.py
3. convert_all_mabs - makes all residue level data for training 
   using the convert_residue_level.py script

Making embeddings (env: esm3)

1. Collect fab sequences (fab_sequences.xlsx)
2. esm_embed_seq.py - this is a folder I run out of the ESM3 directory that was copied off of github.

Make full Dataset (env: dataset_creation)

1. build_merged_dataset.py - combines all silcs and embeddings data in one .csv; splits into test, validation, and training .csvs
2. alignment.py - adds spacing rows to dataset to make all proteins same length; the model has not yet been trained on this kind of data, however it would work by just ignoring the empty rows.

Model (env: silcs_NN)

1. train_model.py - uses created dataset and splits it into 3 groups. Training (70%), Validation (15%), Test (15%). A csv for the test results is created showing all the rows of residues, the embeddings, the predictions for each affinity groups, and which antibody they belong to.
2. ANN_stats.py - creates graphing data and csv with table results



