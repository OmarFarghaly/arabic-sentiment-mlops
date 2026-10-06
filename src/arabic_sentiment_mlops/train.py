from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_TO_ID = {"Negative": 0, "Positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def compute_metrics(prediction):
    predicted_labels = prediction.predictions.argmax(axis=-1)
    true_labels = prediction.label_ids

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "f1_macro": f1_score(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
    }


def main():
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    config = params["train"]
    seed = params["base"]["seed"]

    # Seed before loading the model: its classification head starts randomly.
    set_seed(seed)

    data_path = Path(params["prepare"]["output_dir"]) / "train.csv"
    frame = pd.read_csv(data_path, encoding="utf-8-sig")

    if frame.empty:
        raise ValueError("Training data is empty.")

    if frame[["text", "label"]].isna().any().any():
        raise ValueError("Training data contains missing texts or labels.")

    if frame["text"].str.strip().eq("").any():
        raise ValueError("Training data contains blank texts.")

    if set(frame["label"]) != set(LABEL_TO_ID):
        raise ValueError("Expected both Negative and Positive labels.")

    frame["labels"] = frame["label"].map(LABEL_TO_ID)
    frame = frame[["text", "labels"]]

    training_frame, validation_frame = train_test_split(
        frame,
        test_size=config["validation_size"],
        random_state=seed,
        stratify=frame["labels"],
    )

    print(
        f"Training rows: {len(training_frame)}; "
        f"validation rows: {len(validation_frame)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config["max_length"],
        )

    training_dataset = Dataset.from_pandas(
        training_frame,
        preserve_index=False,
    ).map(tokenize, batched=True, remove_columns=["text"])

    validation_dataset = Dataset.from_pandas(
        validation_frame,
        preserve_index=False,
    ).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    output_dir = Path(config["model_output_dir"])

    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        use_cpu=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Training restores the checkpoint with the highest validation macro F1.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    validation_metrics = trainer.evaluate()
    trainer.save_metrics("validation", validation_metrics)

    print(f"Saved model and tokenizer to {output_dir}")
    print(validation_metrics)


if __name__ == "__main__":
    main()