import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    first_subparser = parser.add_subparsers(dest="command")

    verify = first_subparser.add_parser("verify")

    embed_text = first_subparser.add_parser("embed_text")

    embed_text.add_argument("query")

    verify_embeddings = first_subparser.add_parser('verify_embedding')


    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)
