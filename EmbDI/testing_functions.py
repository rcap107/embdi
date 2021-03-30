import os

from EmbDI.embeddings_quality import embeddings_quality
from EmbDI.entity_resolution import entity_resolution
from EmbDI.logging import *
from EmbDI.schema_matching import schema_matching
from EmbDI.utils import remove_prefixes


def test_driver(embeddings_file, configuration=None):
    test_type = configuration['experiment_type']
    info_file = configuration['dataset_info']
    if test_type == 'EQ':
        # print('#'*80)
        print('# EMBEDDINGS QUALITY')
        if configuration['training_algorithm'] == 'fasttext':
            newf = embeddings_file
            mem_results.res_dict = embeddings_quality(newf, configuration)
        else:
            newf = remove_prefixes(configuration['input_file'], embeddings_file)
            mem_results.res_dict = embeddings_quality(newf, configuration)
            os.remove(newf)
    elif test_type == 'ER':
        print('# ENTITY RESOLUTION')

        mem_results.res_dict = entity_resolution(embeddings_file, configuration, info_file=info_file)
    elif test_type == 'SM':
        print('# SCHEMA MATCHING')
        mem_results.res_dict = schema_matching(embeddings_file, configuration)
    else:
        raise ValueError('Unknown test type.')


def match_driver(embeddings_file, df, configuration):
    test_type = configuration['experiment_type']
    info_file = configuration['dataset_info']
    print('Extracting matched tuples')
    m_tuples = entity_resolution(embeddings_file, configuration, info_file=info_file,
                                 task='match')
    # print('Extracting matched columns')
    # m_columns = match_columns(df, embeddings_file)

    return m_tuples, []
