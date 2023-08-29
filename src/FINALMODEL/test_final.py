import torch

from transformers import Trainer, TrainingArguments
from src.FINALMODEL.final_model import FinalModel

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


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


def test_final(ner_model, re_model, test_dataset, tokenizer_ner, id_label, re_input_length, batch_size):
    model = FinalModel(ner_model, re_model, tokenizer_ner, id_label, re_input_length)

    training_args = TrainingArguments(
        output_dir="./FINALMODEL/results",
        per_device_eval_batch_size=batch_size,
        seed=0,
        data_seed=0,
    )

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # test performance
    results = trainer.evaluate(test_dataset)
    test_precision = results['eval_precision']
    test_recall = results['eval_recall']
    test_f1 = results['eval_f1']

    print(f"Final Precision: {test_precision}")
    print(f"Final Recall: {test_recall}")
    print(f"Final F1 Score: {test_f1}")

    return model
