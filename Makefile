install:
	uv sync

build:
	uv run python3 cli/keyword_search_cli.py build
