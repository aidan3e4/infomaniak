import argparse
import random


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into train and test sets.")
    parser.add_argument("input", help="Path to the input JSONL file")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train-out", default=None, help="Output path for train set")
    parser.add_argument("--test-out", default=None, help="Output path for test set")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * args.ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    base = args.input.rsplit(".", 1)[0]
    train_path = args.train_out or f"{base}_train.jsonl"
    test_path = args.test_out or f"{base}_test.jsonl"

    with open(train_path, "w") as f:
        f.writelines(train_lines)

    with open(test_path, "w") as f:
        f.writelines(test_lines)

    print(f"Total: {len(lines)} | Train: {len(train_lines)} | Test: {len(test_lines)}")
    print(f"Written to: {train_path}, {test_path}")


if __name__ == "__main__":
    main()
