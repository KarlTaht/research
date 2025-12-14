#!/usr/bin/env python3
"""Unified CLI for infrastructure operations.

Usage:
    python -m common.cli.infra <command> [options]

Commands:
    env       Test environment setup
    lambda    Check Lambda Labs availability

Examples:
    python -m common.cli.infra env
    python -m common.cli.infra lambda
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "env":
        from common.cli._internal.test_env import main as env_main
        sys.exit(env_main())
    elif command == "lambda":
        from common.cli._internal.lambda_availability import main as lambda_main
        sys.exit(lambda_main())
    elif command in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
