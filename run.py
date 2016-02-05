__author__ = 'david_torrejon'


from extract_sentences import give_vocabulary, read_json_file, create_sentence_ds
from model_simple_rnn import paper_model
from shuffle_data import get_test_train_sets


#split parameters for dev set
percentage_split_ds = 0.85

df_data = read_json_file()

dataset = create_sentence_ds(df_data)
print dataset[:1]

vocabulary, size_vocabulary = give_vocabulary(df_data)

train_set, test_set = get_test_train_sets(dataset, percentage_split_ds)
# need to embed sentences and labels like [100][010][001]


snli_model = paper_model(3, size_vocabulary)
snli_model.build_model()
