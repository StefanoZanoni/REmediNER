import logging
import torch

from transformers import Trainer, TrainingArguments
from src.NER.model_ner import NerModel

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

from src.plot import plot_heat_map


class NERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_weights = model.loss_weights.to(0)

        loss_fun = torch.nn.CrossEntropyLoss(weight=loss_weights, reduction='none')

        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss_masked = loss_fun(logits, labels)
        pad = -100
        loss_mask = labels != pad
        loss = loss_masked.sum() / loss_mask.sum()

        return (loss, outputs) if return_outputs else loss


def compute_metrics(p: 'EvalPrediction'):
    # Flatten the predictions and labels to 1D arrays
    preds_flat = np.argmax(p.predictions, axis=-1).flatten()
    labels_flat = p.label_ids.flatten()

    # Filter out any labels with value -100, since these should be ignored
    mask = labels_flat != -100
    preds_flat = preds_flat[mask]
    labels_flat = labels_flat[mask]

    # Compute the metrics
    precision = precision_score(labels_flat, preds_flat, average='macro')
    recall = recall_score(labels_flat, preds_flat, average='macro')
    f1 = f1_score(labels_flat, preds_flat, average='macro')
    conf_matrix = confusion_matrix(labels_flat, preds_flat, normalize='true')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': preds_flat,
        'true_labels': labels_flat
    }


def train_test_ner(bert_model, train_dataset, validation_dataset, input_size, batch_size, epochs, loss_weights):
    model_name = bert_model['bert_model']
    id_label = bert_model['id_label']
    label_id = bert_model['label_id']
    model = NerModel(model_name, input_size, id_label, label_id, loss_weights)

    # Define training arguments

    training_args = TrainingArguments(
        output_dir="./NER/results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="epoch",
        logging_dir="./NER/logs",
        logging_first_step=True,
        push_to_hub=False,
        log_level='error'
    )

    # Initialize the Trainer

    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=validation_dataset,
    )

    # Train the model
    trainer.train()

    # training performance
    results = trainer.evaluate(train_dataset)
    train_precision = results['eval_precision']
    train_recall = results['eval_recall']
    train_f1 = results['eval_f1']
    train_confusion_matrix = np.array(results['eval_confusion_matrix'])

    # validation performance
    results = trainer.evaluate(validation_dataset)
    val_precision = results['eval_precision']
    val_recall = results['eval_recall']
    val_f1 = results['eval_f1']
    val_confusion_matrix = np.array(results['eval_confusion_matrix'])

    # Return or print the metrics as desired
    print(f"Train Precision: {train_precision}")
    print(f"Train Recall: {train_recall}")
    print(f"Train F1 Score: {train_f1}")
    plot_heat_map(train_confusion_matrix, 'Training confusion matrix')

    # Return or print the metrics as desired
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1 Score: {val_f1}")
    plot_heat_map(val_confusion_matrix, 'Validation confusion matrix')

    return model