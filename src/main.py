from src.data_utilities import load_data, split_train_test, tokenize_text, \
    compute_iob, pre_process_texts, get_labels_id, get_bert_inputs, get_bert_outputs
from src.model import create_model
from src.training import training
from transformers import BertTokenizer

bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)


def main():
    data = load_data()
    pre_process_texts(data)
    compute_iob(data)
    id_label, label_id, len_labels = get_labels_id(data)
    tokenized_texts, tokenized_labels = tokenize_text(data, tokenizer)
    max_len = data['num_tokens_text'].max()
    inputs = get_bert_inputs(tokenized_texts, tokenized_labels, max_len, tokenizer, label_id)
    outputs = get_bert_outputs(data)
    train_in, test_in, train_out, test_out = split_train_test(inputs, outputs)
    model = create_model(bert_model, len_labels, id_label, label_id, max_len)
    training(train_in, train_out, model)


if __name__ == '__main__':
    main()
