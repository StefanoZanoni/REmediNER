from transformers import Trainer, TrainingArguments
from src.NER.model_ner import NerModel

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np


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
    conf_matrix = confusion_matrix(labels_flat, preds_flat)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': preds_flat,
        'true_labels': labels_flat
    }


def train_test_ner(bert_model, train_dataset, validation_dataset, input_size, batch_size, epochs):
    model_name = bert_model['bert_model']
    id_label = bert_model['id_label']
    label_id = bert_model['label_id']
    model = NerModel(model_name, input_size, id_label, label_id)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="steps",
        logging_dir="./logs",
        logging_first_step=True,
        push_to_hub=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # training performance
    results = trainer.evaluate(train_dataset)
    train_precision = results['eval_precision']
    train_recall = results['eval_recall']
    train_f1 = results['eval_f1']
    train_confusion_matrix = results['eval_confusion_matrix']

    # validation performance
    results = trainer.evaluate(validation_dataset)
    val_precision = results['eval_precision']
    val_recall = results['eval_recall']
    val_f1 = results['eval_f1']
    val_confusion_matrix = results['eval_confusion_matrix']

    # Return or print the metrics as desired
    print(f"Train Precision: {train_precision}")
    print(f"Train Recall: {train_recall}")
    print(f"Train F1 Score: {train_f1}")
    print(f"Train Confusion Matrix: \n{train_confusion_matrix}")

    # Return or print the metrics as desired
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1 Score: {val_f1}")
    print(f"Validation Confusion Matrix: \n{val_confusion_matrix}")

    return model
