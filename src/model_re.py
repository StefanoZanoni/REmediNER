import torch


def compute_all_entities_pairs(entities):
    entities_pairs = []
    for entity in entities:
        for other in entities:
            # if they are not pairs of drugs or effects
            if entity[2] != other[2]:
                entities_pairs.append((entity, other))

    return entities_pairs


def get_entities_and_context_span(entities_vector, context_vector, label_id):
    entities_spans = []
    end = None
    for i in range(entities_vector):
        if entities_vector[i] == label_id['B-Drug'] or entities_vector[i] == label_id['B-Effect']:
            if end is not None:
                entities_spans.append((start, end, e_type))
                end = None
            if entities_vector[i] == label_id['B-Drug']:
                e_type = 0
            else:
                e_type = 1
            start = i
        elif entities_vector[i] == label_id['I-Drug'] or entities_vector[i] == label_id['I-Effect']:
            end = i
        elif entities_vector[i] == label_id['O'] and end is not None:
            entities_spans.append((start, end, e_type))
            end = None

    if end is not None:
        entities_spans.append((start, end, e_type))

    context_start = entities_spans[0][0]
    context_end = entities_spans[len(entities_spans) - 1][1]
    context_span = context_vector[context_start:context_end + 1]

    entities = []
    for span in entities_spans:
        entity_start = span[0]
        entity_end = span[1]
        entity = (torch.sum(context_vector[entity_start:entity_end], dim=-1), entity_start, entity_end, span[2])
        entities.append(entity)

    return entities, context_span


class ReModel(torch.nn.Module):
    def __init__(self, context_mean_length, entity_embeddings_length):
        super(ReModel, self).__init__()

        context_pool_dim = (context_mean_length, 768 * 4 / context_mean_length)
        linear_input_dim = (entity_embeddings_length * 2) + (context_pool_dim[0] * context_pool_dim[1])

        self.avg_pooling = torch.nn.AdaptiveAvgPool2d(context_pool_dim) # we chose Average pooling - check maxpooling
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
        entities, context_span = \
            get_entities_and_context_span(entities_vector, context_vector, label_id)
        entities_pairs = compute_all_entities_pairs(entities)
        outputs = []

        context_pooling = self.avg_pooling(context_span)
        context_flattened = self.flatten(context_pooling)
        for pair in entities_pairs:
            linear_in = torch.cat([pair[0], context_flattened, pair[1]])
            output_linear = self.linear(linear_in)
            out_re = self.sigmoid(output_linear)
            outputs.append((pair[0], pair[1], out_re))

        return outputs

#  o o o o d d o o e o e e
