import os
import torch
import numpy as np

from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score

from src.FINALMODEL.final_model import FinalModel
from src.FINALMODEL.final_dataset import FinalDataset


# Custom compute metrics function. Used metrics are precision, recall and F1 score.
def compute_metrics(p: 'EvalPrediction'):
    # Flatten the predictions and labels to 1D arrays
    preds_flat = np.argmax(p.predictions, axis=-1).flatten()
    labels_flat = p.label_ids.flatten()

    # Filter out any labels with value -100, since these should be ignored
    mask = labels_flat != -100
    preds_flat = preds_flat[mask].tolist()
    labels_flat = labels_flat[mask].tolist()

    # Compute the metrics
    precision = precision_score(labels_flat, preds_flat, average='macro')
    recall = recall_score(labels_flat, preds_flat, average='macro')
    f1 = f1_score(labels_flat, preds_flat, average='macro')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def test_final(ner_model, re_model, ner_ids, ner_masks, re_annotations,
               tokenizer_ner, id_label, re_input_length, batch_size):

    test_dataset = FinalDataset(ner_ids, ner_masks, re_annotations)
    model = FinalModel(ner_model, re_model, tokenizer_ner, id_label, re_input_length)

    if os.path.exists('../models/FINAL_model'):
        model.load_state_dict(torch.load('../models/FINAL_model'))

    if not os.path.exists('../models'):
        os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/FINAL_model')

    training_args = TrainingArguments(
        output_dir="../results/FINAL_results",
        per_device_eval_batch_size=batch_size,
        seed=0,
        data_seed=0,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        compute_metrics=compute_metrics,
    )

    # show predictions
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        ner_ids = [ner_ids.to(f"cuda:{i}") for i in range(num_gpus)]
        ner_masks = [ner_masks.to(f"cuda:{i}") for i in range(num_gpus)]
        re_annotations = [re_annotations.to(f"cuda:{i}") for i in range(num_gpus)]
    else:
        ner_ids = ner_ids.to("cuda:0")
        ner_masks = ner_masks.to("cuda:0")
        re_annotations = re_annotations.to("cuda:0")

    final_logits = model(ner_ids, ner_masks, re_annotations)['logits']
    final_outputs = torch.argmax(final_logits, dim=-1).tolist()
    if not os.path.exists('../final predictions'):
        os.makedirs('../final predictions', exist_ok=True)
    write_list_to_file('../final predictions/final_predictions.txt', final_outputs)


def write_list_to_file(filename, data_list):
    with open(filename, 'w') as file:
        for sublist in data_list:
            line = ' '.join(map(str, sublist))  # Join sublist elements into a space-separated string
            file.write(line + '\n')
