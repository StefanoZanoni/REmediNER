import torch
import numpy as np

from itertools import combinations

def compute_all_entities_pairs(batch_entities):

    batch_entities_pairs = []

    for batch in batch_entities:
        batch_entities_pairs.append({comb for comb in combinations(batch, 2)})

    return batch_entities_pairs


def get_entities_and_context_span(entities_vector, context_vector, label_id):

    batch_entities = []
    batch_contexts_spans = []

    batch = 0
    for list in entities_vector.tolist():
        entities_spans = []
        start = None
        end = None
        i = 0
        # o b b b i i o o b o
        for el in list:
            if el == label_id['B-Drug'] or el == label_id['B-Effect']:
                if el == label_id['B-Drug']:
                    e_type = 0
                else:
                    e_type = 1
                if start is None:
                    start = i
                    end = i
                else:
                    end = i
            elif el == label_id['I-Drug'] or el == label_id['I-Effect']:
                end = i
            elif el == label_id['O'] and end is not None:
                entities_spans.append((start, end, e_type))
                start = None
                end = None
            i += 1

        if end is not None:
            entities_spans.append((start, end, e_type))

        if entities_spans:
            context_start = entities_spans[0][0]
            context_end = entities_spans[len(entities_spans) - 1][1]
            context_span = context_vector[batch, context_start:context_end, :]
        else:
            context_span = torch.empty(size=(0, 0, 0), dtype=torch.float32)

        entities = []
        for span in entities_spans:
            entity_start = span[0]
            entity_end = span[1]
            if entity_start != entity_end:
                test = context_vector[batch, entity_start:entity_end, :]
                entity = \
                    (torch.sum(context_vector[batch, entity_start:entity_end, :], dim=0), entity_start, entity_end, span[2])
            else:
                entity = \
                    (context_vector[batch, entity_start, :], entity_start, entity_end, span[2])
            entities.append(entity)

        batch_entities.append(entities)
        batch_contexts_spans.append(context_span)

        batch += 1

    return batch_entities, batch_contexts_spans


class ReModel(torch.nn.Module):
    def __init__(self, context_mean_length, entity_embeddings_length):
        super(ReModel, self).__init__()

        context_pool_dim = (int(context_mean_length), int(np.ceil(entity_embeddings_length * 4 / context_mean_length)))
        linear_input_dim = (entity_embeddings_length * 4 * 2) + (context_pool_dim[0] * context_pool_dim[1])

        self.avg_pooling = torch.nn.AdaptiveAvgPool2d(context_pool_dim)  # we chose Average pooling - check max-pooling
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=linear_input_dim, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(ReModel.parameters(self),
                                      lr=1e-5,  # args.learning_rate - default is 5e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        return optimizer

    def forward(self, entities_vector, context_vector, label_id):

        batch_entities, batch_context_span = \
            get_entities_and_context_span(entities_vector, context_vector, label_id)
        batch_entities_pairs = compute_all_entities_pairs(batch_entities)

        outputs = []
        for i in range(len(batch_entities_pairs)):
            context_span = batch_context_span[i]
            entities_pairs = batch_entities_pairs[i]
            shape = list(context_span.size())
            context_span = torch.reshape(context_span, shape=(1, shape[0], shape[1]))
            context_pooling = self.avg_pooling(context_span)
            context_flattened = self.flatten(context_pooling)
            context_flattened = torch.flatten(context_flattened)
            for pair in entities_pairs:
                entity1 = pair[0][0]
                entity2 = pair[1][0]
                linear_in = torch.cat([entity1, context_flattened, entity2])
                output_linear = self.linear(linear_in)
                out_re = self.sigmoid(output_linear)
                output = (entity1, entity2, out_re)

        return outputs
