import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    first_subparser = parser.add_subparsers(dest="command")

    verify = first_subparser.add_parser("verify")

    embed_text = first_subparser.add_parser("embed_text")

    embed_text.add_argument("query")

    verify_embeddings = first_subparser.add_parser("verify_embeddings")

    embed_query = first_subparser.add_parser("embed_query")
    embed_query.add_argument("query")

    search = first_subparser.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    chunk = first_subparser.add_parser("chunk")
    chunk.add_argument("query")
    chunk.add_argument("--chunk-size", type=int, default=200)
    chunk.add_argument("--overlap", type=int, default=0)

    semantic_chunk = first_subparser.add_parser("semantic_chunk")
    semantic_chunk.add_argument("query")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk.add_argument("--overlap", type=int, default=0)

    first_subparser.add_parser("embed_chunks")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)
