import spacy
import torch


def compute_pos(batch_output_ner):
    nlp = spacy.load("en_core_web_sm")
    batch_pos = []
    for text in batch_output_ner:
        doc = nlp(text)
        pos = [token.pos_ for token in doc]
        batch_pos.append(pos)

    return batch_pos


class FinalModel(torch.nn.Module):

    def __init__(self, ner, re, tokenizer, id_label, re_input_length):
        super(FinalModel, self).__init__()

        self.ner = ner
        self.re = re
        self.tokenizer = tokenizer
        self.id_label = id_label
        self.re_input_length = re_input_length

    def forward(self, ids, mask, labels):
        device = ids.device
        ner_output = self.ner(ids, mask, None)
        ner_logits = ner_output['logits']
        entities = torch.argmax(ner_logits, dim=-1)
        entities = entities.cpu()
        entities = entities.tolist()
        ids = ids.cpu()
        ids = ids.tolist()
        output_ner = self.__convert_output_to_masked_text(entities, ids)
        ids, mask, pos = self.__prepare_re_inputs(output_ner)
        ids = ids.to(device)
        mask = mask.to(device)
        pos = pos.to(device)
        re_output = self.re(ids, mask, pos, labels)
        re_logits = re_output['logits']

        return {'logits': re_logits, 'loss': torch.zeros(1)}

    def __convert_output_to_masked_text(self, batch_entities, ids):
        new_entities = ['O', 'DRUG', 'EFFECT']
        new_label_id = {label: i for i, label in enumerate(new_entities)}
        new_id_label = {i: label for i, label in enumerate(new_entities)}
        tokenizer = self.tokenizer
        id_label = self.id_label

        batch_new_entities = []
        for entities in batch_entities:
            new_entities = []
            for el in entities:
                if id_label[el] == 'O':
                    new_entities.append(new_label_id['O'])
                elif id_label[el] == 'B-Drug':
                    new_entities.append(new_label_id['DRUG'])
                elif id_label[el] == 'I-Drug':
                    new_entities.append(new_label_id['DRUG'])
                elif id_label[el] == 'B-Effect':
                    new_entities.append(new_label_id['EFFECT'])
                elif id_label[el] == 'I-Effect':
                    new_entities.append(new_label_id['EFFECT'])
            batch_new_entities.append(new_entities)

        batch_tokens = []
        for l in ids:
            batch_tokens.append(tokenizer.convert_ids_to_tokens(l))

        for batch, tokens in enumerate(batch_tokens):
            indexes_to_remove = []
            for i, token in enumerate(tokens):
                if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                    indexes_to_remove.append(i)
            for i, index in enumerate(indexes_to_remove):
                del batch_tokens[batch][index - i]
                del batch_new_entities[batch][index - i]

        for batch, tokens in enumerate(batch_tokens):
            for i, token in enumerate(tokens):
                entity = batch_new_entities[batch][i]
                if new_id_label[entity] == 'DRUG':
                    batch_tokens[batch][i] = 'DRUG'
                elif new_id_label[entity] == 'EFFECT':
                    batch_tokens[batch][i] = 'EFFECT'

        batch_texts = []
        for batch, tokens in enumerate(batch_tokens):
            text = []
            new_token = ''
            de_append = False
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    new_token += token.replace('##', '')
                elif token != 'DRUG' and token != 'EFFECT':
                    if new_token != '':
                        text.append(new_token)
                        new_token = token
                        de_append = False
                    else:
                        text.append(token)
                        de_append = False
                else:
                    if not de_append:
                        text.append(token)
                        de_append = True

            batch_texts.append(' '.join(text))

        return batch_texts

    def __prepare_re_inputs(self, batch_output_ner):
        batch_pos = compute_pos(batch_output_ner)
        batch_tokenized_texts, batch_tokenized_pos = self.__tokenize_inputs_re(batch_output_ner, batch_pos)
        pos = self.__compute_pos_indexes(batch_tokenized_pos)
        ids, masks = self.__get_re_inputs(batch_tokenized_texts)

        return ids, masks, pos

    def __get_re_inputs(self, batch_tokenized_texts):
        bert_ids = []
        bert_masks = []
        max_len = self.re_input_length

        for text in batch_tokenized_texts:
            tokenized_text = ["[CLS]"] + text + ["[SEP]"]

            # truncation
            if len(tokenized_text) > max_len:
                tokenized_text = tokenized_text[:max_len]
            # padding
            if len(tokenized_text) < max_len:
                tokenized_text = tokenized_text + ['[PAD]' for _ in range(max_len - len(tokenized_text))]

            attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_text]
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            bert_ids.append(ids)
            bert_masks.append(attention_mask)

        re_ids = torch.tensor(bert_ids, dtype=torch.long)
        re_masks = torch.tensor(bert_masks, dtype=torch.long)

        return re_ids, re_masks

    def __compute_pos_indexes(self, batch_tokenized_pos):
        input_length = self.re_input_length

        max_number_pos = set()
        for l in batch_tokenized_pos:
            for pos in l:
                max_number_pos.add(pos)

        pos_indexes = {pos: i for i, pos in enumerate(max_number_pos, start=1)}

        indexes_global = []
        for l in batch_tokenized_pos:
            indexes_local = [0]
            # CLS
            for pos in l:
                indexes_local.append(pos_indexes[pos])
            # SEP
            indexes_local.append(0)
            # padding
            if len(indexes_local) < input_length:
                for i in range(input_length - len(indexes_local)):
                    indexes_local.append(0)
            # truncation
            if len(indexes_local) > input_length:
                indexes_local = indexes_local[:input_length]

            indexes_global.append(indexes_local)

        indexes_global = torch.tensor(indexes_global, dtype=torch.long)

        return indexes_global

    def __tokenize_inputs_re(self, batch_texts, batch_pos):
        temp_tokenized_texts = []
        temp_tokenized_tags = []

        for text, pos in zip(batch_texts, batch_pos):
            tokenized_text = []
            tokenized_pos = []
            for word, pos_tag in zip(text.split(), pos):
                tokenized_word = self.tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)
                tokenized_text.append(tokenized_word)
                tokenized_pos.append([pos_tag] * n_subwords)

            temp_tokenized_texts.append(tokenized_text)
            temp_tokenized_tags.append(tokenized_pos)

        tokenized_texts = []
        tokenized_pos = []

        for text in temp_tokenized_texts:
            tokenized_text = []
            for word in text:
                for token in word:
                    tokenized_text.append(token)
            tokenized_texts.append(tokenized_text)

        for pos_tags in temp_tokenized_tags:
            tokenized_tag = []
            for pos_tag in pos_tags:
                for token in pos_tag:
                    tokenized_tag.append(token)
            tokenized_pos.append(tokenized_tag)

        return tokenized_texts, tokenized_pos
