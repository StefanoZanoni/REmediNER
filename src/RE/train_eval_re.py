import logging
import torch

from transformers import Trainer, TrainingArguments
from src.RE.model_re import ReModel

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

from src.plot import plot_heat_map


class RETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        annotations = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_weights = model.module.loss_weights.to(inputs['ids'].device) if hasattr(model, 'module') \
            else model.loss_weights.to(inputs['ids'].device)

        loss_fun = torch.nn.CrossEntropyLoss(weight=loss_weights, reduction='none')

        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss_masked = loss_fun(logits, annotations)
        pad = -100
        loss_mask = annotations != pad
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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': preds_flat,
        'true_labels': labels_flat
    }


def train_test_re(model_name, train_dataset, validation_dataset, input_size, batch_size, epochs,
                  max_number_pos, loss_weights_train, loss_weights_val):

    embedding = torch.nn.Embedding(max_number_pos, 768, padding_idx=0)
    model = ReModel(model_name, input_size, embedding, loss_weights_train)

    # Define training arguments

    training_args = TrainingArguments(
        output_dir="./RE/results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-4,
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        logging_dir="./RE/logs",
        logging_first_step=True,
        push_to_hub=False,
        seed=0,
        data_seed=0,
    )

    # Initialize the Trainer

    trainer = RETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # validation performance
    results = trainer.evaluate(validation_dataset)
    val_precision = results['eval_precision']
    val_recall = results['eval_recall']
    val_f1 = results['eval_f1']

    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1 Score: {val_f1}")

    # re-train on the whole dataset
    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, validation_dataset])
    n = len(train_val_dataset)
    train_len = len(train_dataset)
    val_len = len(validation_dataset)
    loss_weights_train = loss_weights_train * train_len / n
    loss_weights_val = loss_weights_val * val_len / n
    loss_weights = loss_weights_train + loss_weights_val
    model = ReModel(model_name, input_size, embedding, loss_weights)
    trainer = RETrainer(
        model=model,
        args=training_args,
        train_dataset=train_val_dataset,
    )
    # Train the model
    trainer.train()

    return model
