Neural Network Model for Predicting Monoclonal Antibody Binding Affinity

**FINAL MODEL**

final_model_script.py - runs saved model as well and calls upon additional script (esm_embed_seq_finalmodel.py) to generate embeddings. Plots all the results.
esm_embed_seq_finalmodel.py - ran inside the esm3 folder (copied off github) in order to generate single embedding 

**TRAINING**

Dataset Creation Workflow

Getting SILCS training Dataset
1. copy_clean.py - based on sheet.csv that I-En updates - copied mAb folders into my work area
2. convert_all_maps - changes all maps files to dx files using
   map2dx.py
3. convert_all_mabs - makes all residue level data for training 
   using the convert_residue_level.py script

Making embeddings

1. Collect fab sequences (fab_sequences.xlsx)
2. esm_embed_seq.py

Make full Dataset

1. build_merged_dataset.py - combines all silcs and embeddings data in one .csv; splits into test, validation, and training .csvs
2. alignment.py - adds spacing rows to dataset to make all proteins same length

Model

1. train_model_[#].py
2. ANN_stats.py - creates graphing data



