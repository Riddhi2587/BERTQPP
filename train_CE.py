import argparse
import math
import pickle

from torch.utils.data import DataLoader
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a BERT-QPP cross-encoder on query performance labels.")
    parser.add_argument("--train-pkl", type=str, required=True, help="Training pickle from create_train_pkl_file.py")
    parser.add_argument("--val-pkl", type=str, default=None, help="Optional validation pickle for CECorrelationEvaluator")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="Starting model checkpoint (default: bert-base-uncased)")
    parser.add_argument("--num-labels", type=int, default=1, help="Number of output labels (default: 1, regression)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--evaluation-steps", type=int, default=0, help="Evaluate every N steps when --val-pkl is set (0 = once per epoch)")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save the trained model")

    args = parser.parse_args()

    train_set = load_examples(args.train_pkl)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data for warm-up

    model = CrossEncoder(args.checkpoint, num_labels=args.num_labels)

    evaluator = None
    if args.val_pkl:
        val_set = load_examples(args.val_pkl)
        evaluator = CECorrelationEvaluator.from_input_examples(val_set, name="dev")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        output_path=args.output,
        save_best_model=evaluator is not None,
    )

    if evaluator is None:
        model.save(args.output)

    print(f"[INFO] Saved trained model to {args.output}")
