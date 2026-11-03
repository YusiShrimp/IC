"""Command-line interface: ``python -m mycoolproject NAME [--age AGE]``."""
import argparse

from .cool_module import func1


def main():
    """Parse the command-line arguments, and say hello."""
    parser = argparse.ArgumentParser(prog="python -m mycoolproject",
                                     description="A minimal package script.")
    parser.add_argument("name", type=str, help="Your name")
    parser.add_argument("-a", "--age", type=int, default=0, help="Your age")
    args = parser.parse_args()

    print(f"Hello, {args.name}!")
    if args.age:
        print(f"You are {args.age} years old.")
    func1()


if __name__ == "__main__":
    main()
