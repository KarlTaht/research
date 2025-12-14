#!/usr/bin/env python3
"""Unified CLI for data operations.

Usage:
    python -m common.cli.data <command> [options]

Commands:
    download dataset   Download HF dataset
    download model     Download HF model
    analyze            Token distribution analysis
    pretokenize        Pre-tokenize dataset for training
    fineweb sample     Download FineWeb sample
    fineweb index      Build domain index
    fineweb query      Query domain index
    fineweb extract    Extract domain corpus

Examples:
    python -m common.cli.data download dataset --name squad
    python -m common.cli.data download model --repo-id gpt2
    python -m common.cli.data analyze --dataset tinystories
    python -m common.cli.data pretokenize --dataset tinystories --tokenizer gpt2
    python -m common.cli.data fineweb sample --tokens 10000000
    python -m common.cli.data fineweb index --status
    python -m common.cli.data fineweb query --top-domains 50
    python -m common.cli.data fineweb extract --corpus automotive
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "download":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.data download <dataset|model> [options]")
            print("  dataset  Download a HuggingFace dataset")
            print("  model    Download a HuggingFace model")
            sys.exit(1)

        subcommand = sys.argv[2]
        # Remove 'download' and subcommand from argv
        sys.argv = [sys.argv[0]] + sys.argv[3:]

        if subcommand == "dataset":
            from common.cli._internal.download_dataset import main as download_dataset_main
            download_dataset_main()
        elif subcommand == "model":
            from common.cli._internal.download_model import main as download_model_main
            download_model_main()
        else:
            print(f"Unknown download subcommand: {subcommand}")
            print("Available: dataset, model")
            sys.exit(1)

    elif command == "analyze":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from common.cli._internal.analyze_tokens import main as analyze_main
        sys.exit(analyze_main())

    elif command == "pretokenize":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from common.cli._internal.pretokenize import main as pretokenize_main
        pretokenize_main()

    elif command == "fineweb":
        if len(sys.argv) < 3:
            print("Usage: python -m common.cli.data fineweb <sample|index|query|extract> [options]")
            print("  sample   Download FineWeb sample")
            print("  index    Build domain index")
            print("  query    Query domain index")
            print("  extract  Extract domain corpus")
            sys.exit(1)

        subcommand = sys.argv[2]
        sys.argv = [sys.argv[0]] + sys.argv[3:]

        if subcommand == "sample":
            from common.cli._internal.download_fineweb import main as fineweb_sample_main
            fineweb_sample_main()
        elif subcommand == "index":
            from common.cli._internal.build_domain_index import main as fineweb_index_main
            fineweb_index_main()
        elif subcommand == "query":
            from common.cli._internal.query_domain_index import main as fineweb_query_main
            fineweb_query_main()
        elif subcommand == "extract":
            from common.cli._internal.extract_corpus import main as fineweb_extract_main
            fineweb_extract_main()
        else:
            print(f"Unknown fineweb subcommand: {subcommand}")
            print("Available: sample, index, query, extract")
            sys.exit(1)

    elif command in ("-h", "--help", "help"):
        print(__doc__)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
