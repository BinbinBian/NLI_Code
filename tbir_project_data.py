__author__ = 'david_torrejon'
from extract_sentences import label_output_data

def load_data(file_test='./premise_hypo.txt'):
    test_data = []
    with open(file_test, 'r') as f:
        for line in f:
            t_data = line.rstrip().split("#&#")
            test_data.append([t_data[1], t_data[2],t_data[3], t_data[0], t_data[4]]) #sentence 1, sentence 2, id_test, id_train
    return test_data
