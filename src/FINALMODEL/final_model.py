import spacy
import torch


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
        ids, mask = self.__prepare_re_inputs(output_ner)
        ids = ids.to(device)
        mask = mask.to(device)
        re_output = self.re(ids, mask, labels)
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
                if not token.startswith('##') and new_token != '':
                    if new_token != 'DRUG' and new_token != 'EFFECT':
                        text.append(new_token.lower())
                        de_append = False
                    else:
                        if not de_append:
                            text.append(new_token)
                            de_append = True
                    new_token = ''
                if token == 'DRUG' or token == 'EFFECT':
                    new_token = token
                else:
                    new_token += token.replace('##', '')

            batch_texts.append(' '.join(text))

        return batch_texts

    def __prepare_re_inputs(self, batch_output_ner):
        batch_tokenized_texts = self.__tokenize_inputs_re(batch_output_ner)
        ids, masks = self.__get_re_inputs(batch_tokenized_texts)

        return ids, masks

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

    def __tokenize_inputs_re(self, batch_texts):
        temp_tokenized_texts = []

        for text in batch_texts:
            tokenized_text = []
            for word in text.split():
                tokenized_word = self.tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)
                tokenized_text.append(tokenized_word)

            temp_tokenized_texts.append(tokenized_text)

        tokenized_texts = []

        for text in temp_tokenized_texts:
            tokenized_text = []
            for word in text:
                for token in word:
                    tokenized_text.append(token)
            tokenized_texts.append(tokenized_text)

        return tokenized_texts
