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

        # NER computation
        ner_output = self.ner(ids, mask, None)
        ner_logits = ner_output['logits']
        entities = torch.argmax(ner_logits, dim=-1)
        entities = entities.cpu()
        entities = entities.tolist()
        ids = ids.cpu()
        ids = ids.tolist()
        output_ner = self.__convert_output_to_masked_text(entities, ids)

        # RE computation
        ids, mask = self.__prepare_re_inputs(output_ner)
        ids = ids.to(device)
        mask = mask.to(device)
        re_output = self.re(ids, mask, labels)
        re_logits = re_output['logits']

        return {'logits': re_logits, 'loss': torch.zeros(1).to(device)}

    def __convert_output_to_masked_text(self, batch_entities, ids):
        new_entities = ['O', 'DRUG', 'EFFECT']
        new_label_id = {label: i for i, label in enumerate(new_entities)}
        new_id_label = {i: label for i, label in enumerate(new_entities)}
        tokenizer = self.tokenizer
        id_label = self.id_label

        # Replace the IOB-tags ids with O|DRUG|EFFECT ids
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

        # Replace the BERT ids into tokens.
        batch_tokens = []
        for l in ids:
            batch_tokens.append(tokenizer.convert_ids_to_tokens(l))

        # Discard special BERT tokens (CLS, SEP, PAD).
        for batch, tokens in enumerate(batch_tokens):
            indexes_to_remove = []
            for i, token in enumerate(tokens):
                if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                    indexes_to_remove.append(i)
            for i, index in enumerate(indexes_to_remove):
                del batch_tokens[batch][index - i]
                del batch_new_entities[batch][index - i]

        # Replace the IOB-tags with O|DRUG|EFFECT
        for batch, tokens in enumerate(batch_tokens):
            for i, token in enumerate(tokens):
                entity = batch_new_entities[batch][i]
                if new_id_label[entity] == 'DRUG':
                    batch_tokens[batch][i] = 'DRUG'
                elif new_id_label[entity] == 'EFFECT':
                    batch_tokens[batch][i] = 'EFFECT'

        # Reconstruction of the entire word from subwords.
        batch_texts = []
        for batch, tokens in enumerate(batch_tokens):
            text = []
            new_token = ''
            de_append = False
            for i, token in enumerate(tokens):
                # if the token is not a sub-token and the new token is not empty
                # (means this is the first new word I'm analyzing)
                if not token.startswith('##') and new_token != '':
                    # if the new token is not a masked token, I append it in lower case
                    if new_token != 'DRUG' and new_token != 'EFFECT':
                        text.append(new_token.lower())
                        de_append = False
                    # else a DRUG or an EFFECT will be appended as they were
                    else:
                        # This check prevents multiple append of DRUG or EFFECT
                        # if a drug or an effect were composed by multiple sub-tokens
                        if not de_append:
                            text.append(new_token)
                            de_append = True
                    new_token = ''
                # if the current token is a drug or an effect, it remains unchanged
                if token == 'DRUG' or token == 'EFFECT':
                    new_token = token
                # It was found a sub-token. It will be concatenated to the new token
                else:
                    new_token += token.replace('##', '')

            text.append(new_token)

            batch_texts.append(' '.join(text))

        return batch_texts

    def __prepare_re_inputs(self, batch_output_ner):
        batch_tokenized_texts = self.__tokenize_inputs_re(batch_output_ner)
        ids, masks = self.__get_re_inputs(batch_tokenized_texts)

        return ids, masks

    # this function is used to prepare input in BERT format
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

        # words tokenization
        for text in batch_texts:
            tokenized_text = []
            for word in text.split():
                tokenized_word = self.tokenizer.tokenize(word)
                tokenized_text.append(tokenized_word)

            temp_tokenized_texts.append(tokenized_text)

        tokenized_texts = []

        # transform the list of lists of tokens into a list of tokens
        for text in temp_tokenized_texts:
            tokenized_text = []
            for word in text:
                for token in word:
                    tokenized_text.append(token)
            tokenized_texts.append(tokenized_text)

        return tokenized_texts
