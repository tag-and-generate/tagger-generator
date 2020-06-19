import sys
import sentencepiece as sp
import argparse


def train(args):
    arg_string = "".join(
        arg + ("=" if arg.startswith("--") else " ")
        for arg in args
    ).strip()
    sp.SentencePieceTrainer.Train(arg_string)


def load(model_path):
    model = sp.SentencePieceProcessor()
    model.Load(model_path)
    return model


def desegment(tokens):
    return ("".join(tokens)).replace("▁",  " ").strip()


def get_args():
    parser = argparse.ArgumentParser("Subword training/segmentation")
    subparsers = parser.add_subparsers(help="Actions")
    # Training
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(which="train")
    train_parser.add_argument("--input", required=True, type=str)
    train_parser.add_argument("--model_prefix", required=True, type=str)
    train_parser.add_argument("--vocab_size", required=True, type=int)
    train_parser.add_argument("--model_type", required=True, type=str)
    # Segmentation
    segment_parser = subparsers.add_parser("segment")
    segment_parser.set_defaults(which="segment")
    segment_parser.add_argument("--model", required=True, type=str)
    # De-segmentation
    segment_parser = subparsers.add_parser("desegment")
    segment_parser.set_defaults(which="desegment")
    # Parse
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.which == "train":
        train(sys.argv[2:])
    elif args.which == "segment":
        model = load(args.model)
        for line in sys.stdin:
            print(" ".join(model.EncodeAsPieces(line)))
    elif args.which == "desegment":
        for line in sys.stdin:
            print(desegment(line.strip().split()))


if __name__ == "__main__":
    main()
