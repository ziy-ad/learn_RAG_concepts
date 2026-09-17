import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    parser.add_argument('--data', default='data/movies.json')
    first_subparser = parser.add_subparsers(dest="command")

    search =  first_subparser.add_parser('search')
    search.add_argument('query')

    tf = first_subparser.add_parser('tf')
    tf.add_argument('id')
    tf.add_argument('term')

    idf = first_subparser.add_parser('idf')
    idf.add_argument('term')


    tfidf = first_subparser.add_parser('tfidf')
    tfidf.add_argument('id')
    tfidf.add_argument('term')


    first_subparser.add_parser('build')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)
