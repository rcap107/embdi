EmbDI
=======
EmbDI is a Python library developed to perform Entity Resolution (ER) and Schema Matching (SM) tasks by employing 
word embeddings. 

The repository includes a `main` script and a collection of additional scripts used to generate the files needed to 
perform the tasks specified by the user.  
 * `main.py` is used to train the embeddings and/or perform postprocessing (ER, SM, EQ) on them. Execution may be done 
 in batch by reading all files in a directory or by passing a single configuration file on the command line. 
 Configuration files are explained in a separate section.
 * `generate_tests.py` produces the files needed to test embedding quality (`MA`, `MR`, `MC`).
 
 
#### Directory tree
Due to the different files required to run the code, both input and output files are saved in a directory
labeled `pipeline`. All data input/output will assume the presence of a directory with this name. 

The expected directory tree is the following:
```
root
  ├─EmbDI
  | 
  ├─pipeline
        ├── config_files
        ├── datasets
        ├── dump
        ├── embeddings
        ├── experiments
        ├── generated-matches
        ├── info
        ├── matches
        ├── replacements
        ├── sim_files
        ├── test_dir
        └── walks
```

Additional directories are present in the project directory: they contain additional (not required) code or are needed
for preprocessing or analysis. 

main.py
---
`main.py` allows to execute the full pipeline (graph and corpus generation, embeddings training and matching), or a subset
of the pipeline steps depending on the parameters found in the `task` field present in the configuration file. 

The script requires one of two command line parameters:
* `-d`, `--config-dir` require the path to a directory that should contain configuration files. All valid files will be 
executed alphabetically. All files present in the provided directory that do not start with `default` and that are not 
directories themselves will be considered as "valid".
```
python main.py -d path/to/config/dir/
```

* `-f`, `--config-file` should be used instead to run only the provided configuration file.

```
python main.py -f path/to/config/file
```

Configuration files are described in their own section. The `task` field is mandatory, as it specifies what operations
should be performed.
The possible tasks are the following:
* `train`: train new embeddings using the parameters supplied in the configuration file and save them on the given
output file. No further operations are performed on the output file. 
* `test`: given a complete embeddings file (in text form), perform the tests (either EQ, ER, SM) specified in the 
configuration file.
* `match`: given a complete embeddings file (in text form), execute ER and SM and save on a file the matches, without
testing their correctness. These matches can then be used for other tasks. 
* `refinement`: perform rotation on the supplied datasets.
* `train-test`: train new embeddings and immediately run tests on them.
* `train-match`: train new embeddings and immediately produce matches. 

Configuration files
---
Some example configuration files are stored in the folder `pipeline/config_files/demo`. An example of generic 
configuration file is reported below.
Please note that the order of lines is not important, but the first field (before the `:`) has to be written how it is 
shown here. 
All lines starting with `#` will be treated as comments and will be ignored. 
In the example below, lines starting with `*` are optional and, if missing, will be assigned a default value.

```
# Input configuration:
task: {training, testing, matching, refinement, train-test, train-match}
input_file: path/to/dataset/dataset_name.csv
output_file: name_of_output_file
dataset_info: path/to/dataset/info_file.txt
experiment_type: {EQ, ER, SM}
match_file: path/to/match/file.txt [required for SM, ER]
test_dir: path/to/EQ/tests/ [required for EQ]
similarity_file: path/to/similarity/file.txt [Needed to apply replacement, ignored otherwise]
embeddings_file: path/to/embeddings/file.emb [This field will be ignored if the task requires training]
walks_file: path/to/walks/file/file.walks [A valid walks file will bypass the graph and walks generation]
# Walks configuration:
*sentence_length: length_of_sentences
*n_sentences: {number_of_sentences, default}
*walks_strategy: {basic, replacement}
*follow_sub: {true, false}
*smoothing_method: {no, smooth, inverse_smooth, log, piecewise} [Refer to the section on smoothing for more details]
*backtrack: {True, False} 
*write_walks: {True, False} 
# Embeddings configuration:
*learning_method: {CBOW, skipgram}
*window_size: w_size
*n_dimensions: n_dim
# Test configuration:
*ntop: n_closest_neighbors_to_study
*ncand: n_candidates_from_ntop
*max_rank: cutoff_rank_for_SM

```

#### Default values
Default values are hardcoded in `EmbDI.utils.py`. Their values are the following:
```
    'with_rid' = WITH_RID_FIRST
    'with_cid' = WITH_CID_ALL
    'ntop' = 10
    'ncand' = 1
    'max_rank' = 3
    'walks_file' = ''
    'follow_sub' = False
    'smoothing_method' = 'no'
    'graph_file' = ''
    'backtrack' = True
    'training_algorithm' = 'word2vec'
    'write_walks' = True
    'flatten' = True
    'indexing' = 'basic'
    'epsilon' = 0.1
    'num_trees' = 250
    'compression' = True
    'n_sentences' = 'default'
    'walks_strategy' = 'basic'
    'learning_method' = 'skipgram'
    'window_size' = 3
    'n_dimensions' = 300
```

#### Smoothing parameters
Experiments have shown that weighted random walks may perform better than regular random walks. Multiple weighing functions are proposed and can be accessed by changing the `smoothing_parameter` in the configuration file. 

The `smoothing_parameter` field should be a string that contains the smoothing method and (if needed) its parameters. If only the name of the smoothing method is provided, default parameters will be used. The optional parameters will be shown in `[]`.

##### Implemented smoothing methods
 -  `no`: default strategy, all nodes will have the same weight and will therefore be chosen with the same likelyhood.
 -  `smooth,[k=0.2, target=200]`: inverse exponential function. The slope of the exponential can be tweaked by changing the value of `target` (larger targed = lower slope). The exponential function will also approximate the value `k`, rather than 0. This is done to prevent situations where some nodes will never be reached. This function will penalize frequent nodes, assigning them a weight of `k`.
 -  `inverse_smooth, [s=0.1]`: a flipped exponential function that will approximate weight 1 as the frequency of a node increases. In this case, the minimum weight that can be assigned to a node is `0.5` for nodes with frequency 1. We observed that this function behaves better than `smooth` when applied to some datasets. The slope of the function can be set with the parameter `s`, a smaller value of `s` will reduce the slope. In this case, test cases with `s` = 0.1, 0.01... as the function increases very quickly. 

Example of tuple:

```smoothing_method: smooth,0.5,200```

This will use the `smooth` function with `k=0.5` and `target=200` .

Experiment logging
---
The system employs the `mlflow` library to log results and parameters. At the moment, the library is required by the
code (a version of the system that does not rely on it will be implemented later).

All results are accessible by using the mlflow library in the root directory: 

```$ mlflow ui```

By default, the UI will be available on ```localhost:5000``` on the local machine. 

Additional files
----

#### Dataset info
The **dataset info file** is used to perform some operations during the execution, which rely on knowing the boundaries 
of each dataset. Each dataset info file should contain a line for each dataset that was used to build the input dataset, 
in the order of concatenation.

Some example Jupyter notebooks that present the operations made to prepare input datasets are reported in the `notebooks/` directory.

*Example*:

Dataset 1: imdb.csv

Dataset 2: movielens.csv

Input dataset = concatenate (dataset 1, dataset 2)

The info file for this case will have the following structure:

```
path/to/dataset/1/imdb.csv,n_lines_dataset_1
path/to/dataset/2/movielens.csv,n_lines_dataset_2
```

#### Similarity file 
The **similarity file** contains candidates for replacement with the structure, optionally with their similarity in the 
range [0,1]. If no distance is supplied, it is assumed to be 1 (word1 == word2).

```
...
word1_dataset1,word1_dataset2,similarity1
word2_dataset1,word2_dataset2,similarity2
...
```

Example:

```
...
english,en, 0.8
french,fr, 0.8
italian,it, 0.8
...
```

#### Matches file
The **matches file** is the ground truth to be used when performing ER or SM tasks. 
The system will expect a known format for the ground truth. 

Below is shown an example for the ER case. Each line should contain a match between a tuple in one 
dataset and one tuple in the other dataset written as "idx_[line_id]". The line_id value refers 
to the line number of the tuple in the dataset passed as input to the training algorithm, 
after the concatenation of the original datasets. 

Note: the indices start from idx_0 (rather than idx_1), so in a typical spreadsheet application 
idx_0 will correspond to line 2 because of the header and the fact that row indices start from 1, 
rather than from 0. 

Multiple matches are allowed, but the current code implementation will only find the closest one. 


```
...
idx_1007,idx_13670
idx_1007,idx_20696
idx_1009,idx_3334
idx_1010,idx_17557
idx_1011,idx_15967
idx_101,idx_5147
idx_1014,idx_9582
idx_102,idx_14313
...
```

The matches file for the SM case looks like this: 
```
dblp,scholar
1,1
2,2
3,3
4,4
5,5
6,6
7,7
```
SM tasks take as inputs datasets whose columns were renamed to "label_dataset_n_[column_id]". The first line of the matches file should contain the respective label (e.g. in the case above, "label_dataset_1" would be dblp). 
All following lines should contain the ground truth matches. If a column can be matched arbitrarily to multiple columns, multiple matches are allowed. As in the ER case, however, only the closest ones will be considered. This behavior will change in a future version of the code. 

Generation of tests
---
EQ tests can be generated by using the `generate_test.py` script. In its current implementation, parameters must be 
changed in the script itself. Later versions will read parameters from an input file.

Three different tests are currently provided:
* nmr (no-match-rows) tests observe whether the provided embeddings are able to recognize what value within a tuple was 
exchanged with a value that does not belong to that tuple. 
* nmc (no-match-column) tests observe whether the embeddings can find what value does not belong to  a given attribute
* nmcon (no-match-concept) tests observe if a semantic link between values is stronger than the link between values 
belonging to the same attribute (e.g. given director Steven Spielberg and target values Jurassic Park, Saving Private 
Ryan, Jaws and Star Wars: A new Hope, nmcon tests expect Star Wars as the answer, since it was not directed 
by Steven Spielberg).

#### How tests are generated

For NMR tests, `n_sentences` tests will be generated for all attributes in the list supplied to `nmr_attrs_to_change`. For 
all such values, `n_sentences` random tuples will be chosen (with replacement) and the value corresponding to the 
target attribute is substituted by a different value. The test is successful if the new value is selected. 

In NMC tests, `n_sentences` tests will be prepared for all combinations of attributes found in the list 
`nmc_col_combinations`. For each pair of attributes, a random selection of unique values coming from the first attribute is 
extracted, and then a random value from the second attribute is selected. The test is passed if the testing algorithm 
chooses the value from the second attribute as its answer.

In NMCON tests, pairs supplied in the list `fd` will be tested as by finding those values in the first attribute that 
have more than `test_length` distinct values in the second column (e.g. given pair `director, movie_title` and 
`test_length=5`, only those directors found in 5 or more tuples with distinct movie titles will be chosen). Then, 
tests are generated by picking a random eligible value, `test_length-1` related values and one unrelated value. Finally,
the test is passed if the unrelated value is chosen by the EQ algorithm. 


Data Preprocessing
---

The Data Preprocessing module can be used to process the datasets to be tested. It includes the following features: 
 * Data exploration by column
 * Removal of missing values (the user can provide additional examples)
 * Multiple strategies to handle missing values
 * Automatic rounding of columns
 * Automatic merging of similar words


##### ```check_info(df, columns)```
A simple exploration per column:
* no of not null values
* no of null values
* no of unique values
* no of duplicated values
* no of duplicated instances

##### ```data_preprocessing(dfs, params)```
The main data process function. It takes as input a list of datasets to preprocess and returns the concatenated dataset with the application of all preprocessing steps. 
The supported preprocessings steps are the following: 
* normalization of  missing values: the given strings will be treated as missing values. 
* normalization of text values: strings are set to lowercase, trailing spaces are removed, some special characters are removed, words are concatenated into one single string. 
* rounding of numerical fields: numerical fields (user-supplied, if needed) are rounded to the given number of digits after the decimal point
* concatenation of datasets: the input datasets are concatenated using one of the given methods: `{horizon,inner,outer}`
* token merging: similar tokens (according to multiple similarity measures) are merged into one to increase the amount of overlap between the two tables. MinHash LSH merging is supported. Tokens that satisfy the LSH threshold and the distance threshold are merged into a new one.
* replacement of missing values: three strategies are provided to replace missing values `{'separated_null', 'one_null', 'ignore'}`.

Default params:
```
parameters = {
        'missing_value': 'nan,ukn,none,unknown,',
        'missing_value_strategy': '',
        'round_number': -1,
        'round_columns': '',
        'concatenate': '',
        'auto_merge': False,
        'mh_k_shingles': 3,
        'mh_threshold': .5,
        'mh_perm': 128,
        'distance': 'normalized_edit_distance',
        'distance_threshold': .20,
        'merge_columns': ''
}
```

Example of use:
```
parameters = {
        'round_number': 0,
        'round_columns': 'price',
        'auto_merge': False
}
df_c = data_preprocessing([df1, df2], parameters)
```

##### LSHMerge

A class used to handle token merging. It is used as a first blocking step to find merge candidates. 
Multiple user-defined parameters are supported.  
The merge process is done in two steps:
* `LSH MinHash`: Neighbors are found and placed in buckets. Params:
    * `k-shingles`: group characters by k-shingles or by word. Shingles separate words in substrings and index them independently. 
    * `minhash threshold`: the similarity threshold to decide whether two items are similar or not. 
    * `perm`: number of MinHash permutations.
* `Distance metric`: Once the buckets are ready, the distance metric is applied to all points inside the same bucket. For the moment, two distance metrics are provided. 
    * `edit_distance`: the distance will measure the number of characters to edit, d >= 0
    * `normalized_distance`: ```edit_distance/len(str)```, d in [0, 1]
    * `none`: do not use a distance function, rely only on LSH buckets. 

We suggest not to rely on the default values and to observe the replacement candidates before replacing them: the edit distance does not take into consideration semantic closeness, so it may happen that unrelated terms will be merged in the same token. A conservative distance value should be employed. 

Default params:
```
k_shingles=3
mh_threshold=.5
mh_num_perm=128
delimiter='_'
```

Example of use:
```
uniq_values = set(['white', 'whitte', 'wwhite'])
lsh = LSHMerge(uniq_values, 4, .5, 128)
# find neighbors of 'white'
neighbors = lsh.get_similarities('white')
# list out 5 random bins
lsh.get_sample_blocks(5)
# get the final replacement list to do the merge
replacement = lsh.get_replacement('normalized_edit_distance', .4)
```


