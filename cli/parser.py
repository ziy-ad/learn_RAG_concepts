import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    parser.add_argument('--data', default='data/movies.json')
    first_subparser = parser.add_subparsers(dest="command")

    search =  first_subparser.add_parser('search')
    search.add_argument('query')

    build = first_subparser.add_parser('build')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)
    else:
        return (parser, args)
