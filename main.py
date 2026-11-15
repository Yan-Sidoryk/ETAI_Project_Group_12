"""
Entry point. Run from the repository root:

    python main.py train --task classification
    python main.py train --task regression
    python main.py predict
    python main.py client --input examples/new_customer.json
"""
import argparse


def train(args):
    raise SystemExit("train is not built yet")


def predict(args):
    raise SystemExit("predict is not built yet")


def client(args):
    raise SystemExit("client is not built yet")


def main():
    parser = argparse.ArgumentParser(description="Repeat-purchase and 90-day spend pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", help="train one task from the raw data and save the model")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.set_defaults(run=train)

    p = commands.add_parser("predict", help="predict test.csv with the saved models")
    p.set_defaults(run=predict)

    p = commands.add_parser("client", help="predict one new purchase")
    p.add_argument("--input", required=True, help="JSON file with one purchase")
    p.set_defaults(run=client)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
