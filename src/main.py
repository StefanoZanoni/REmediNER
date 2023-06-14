from src.data_utilities import load_data, split_train_test, get_input_output, pre_processing,\
    compute_iob, pre_process_texts

bert_model = 'xxx'


def main():
    data = load_data()
    pre_process_texts(data)
    compute_iob(data)
    train_data, test_data = split_train_test(data)
    train_x, train_y = get_input_output(train_data)
    test_x, test_y = get_input_output(test_data)
    encoded_train_x = pre_processing(train_x, bert_model)
    encoded_test_x = pre_processing(test_x, bert_model)


if __name__ == '__main__':
    main()
