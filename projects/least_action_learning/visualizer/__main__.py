"""CLI entry point for the routing visualizer."""

import argparse
from pathlib import Path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Routing pattern visualizer for Least Action Learning experiments"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to search for experiments (default: project outputs/)",
    )
    args = parser.parse_args()

    # Import here to avoid slow startup
    from .app import launch

    project_root = Path(__file__).parent.parent
    print("=" * 60)
    print("Least Action Learning: Routing Visualizer")
    print("=" * 60)
    print(f"Looking for experiments in: {project_root / 'outputs'}")
    print(f"Server port: {args.port}")
    if args.share:
        print("Creating public shareable link...")
    print("=" * 60)

    launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
