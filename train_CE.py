import argparse
import math
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator


def load_examples(pkl_path):
    with open(pkl_path, "rb") as f:
        q_map_dic = pickle.load(f)

    examples = []
    for key in q_map_dic:
        qtext = q_map_dic[key]["qtext"]
        doctext = q_map_dic[key]["doc_text"]
        performance = q_map_dic[key]["performance"]
        examples.append(InputExample(texts=[qtext, doctext], label=performance))
    return examples


def train_with_epoch_checkpoints(model, train_dataloader, epochs, warmup_steps, output_path,
                                  evaluator=None, weight_decay=0.01, max_grad_norm=1,
                                  lr=2e-5, show_progress_bar=True):
    """Trains a CrossEncoder with one continuous optimizer/LR-schedule across all
    epochs (matching what a single CrossEncoder.fit(epochs=N) call does internally),
    saving a checkpoint to f"{output_path}_epoch{N}" after every epoch.
    """
    train_dataloader.collate_fn = model.smart_batching_collate
    model.model.to(model._target_device)

    num_train_steps = int(len(train_dataloader) * epochs)

    param_optimizer = list(model.model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    loss_fct = nn.BCEWithLogitsLoss() if model.config.num_labels == 1 else nn.CrossEntropyLoss()

    for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        model.model.zero_grad()
        model.model.train()

        for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
            logits = model.model(**features, return_dict=True).logits
            if model.config.num_labels == 1:
                logits = logits.view(-1)
            loss_value = loss_fct(logits, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_num = epoch + 1
        epoch_output_path = f"{output_path}_epoch{epoch_num}"
        model.save(epoch_output_path)
        print(f"[INFO] Saved epoch {epoch_num} checkpoint to {epoch_output_path}")

        if evaluator is not None:
            score = evaluator(model, output_path=epoch_output_path, epoch=epoch_num, steps=-1)
            print(f"[INFO] Epoch {epoch_num} evaluator score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a BERT-QPP cross-encoder on query performance labels.")
    parser.add_argument("--train-pkl", type=str, required=True, help="Training pickle from create_train_pkl_file.py")
    parser.add_argument("--val-pkl", type=str, default=None, help="Optional validation pickle for CECorrelationEvaluator")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="Starting model checkpoint (default: bert-base-uncased)")
    parser.add_argument("--num-labels", type=int, default=1, help="Number of output labels (default: 1, regression)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=str, required=True, help="Output path prefix; each epoch is saved to '<output>_epoch<N>'")

    args = parser.parse_args()

    train_set = load_examples(args.train_pkl)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data for warm-up

    model = CrossEncoder(args.checkpoint, num_labels=args.num_labels)

    evaluator = None
    if args.val_pkl:
        val_set = load_examples(args.val_pkl)
        evaluator = CECorrelationEvaluator.from_input_examples(val_set, name="dev")

    train_with_epoch_checkpoints(
        model=model,
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output,
        evaluator=evaluator,
    )

    print(f"[INFO] Finished training. Per-epoch checkpoints saved as {args.output}_epoch1 .. {args.output}_epoch{args.epochs}")
