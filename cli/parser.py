import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    parser.add_argument('--data', default='data/movies.json')
    first_subparser = parser.add_subparsers(dest="command")

    search =  first_subparser.add_parser('search')
    search.add_argument('query')

    tf = first_subparser.add_parser('tf')
    tf.add_argument('id', type=int)
    tf.add_argument('term')

    idf = first_subparser.add_parser('idf')
    idf.add_argument('term')


    tfidf = first_subparser.add_parser('tfidf')
    tfidf.add_argument('id', type=int)
    tfidf.add_argument('term')

    bm25_idf_parser = first_subparser.add_parser(
    "bm25idf", help="Get BM25 IDF score for a given term")

    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    from keyword_search_cli import BM25_K1, BM25_B
    bm25_tf_parser = first_subparser.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
    "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = first_subparser.add_parser(
    "bm25search", help="Search movies using full BM25 scoring"
)
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5)


    first_subparser.add_parser('build')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)
