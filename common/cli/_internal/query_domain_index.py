#!/usr/bin/env python3
"""
Query the FineWeb domain index.

Usage:
    # Show summary stats
    python -m common.cli._internal.query_domain_index

    # Top domains by document count
    python -m common.cli._internal.query_domain_index --top-domains 50

    # Top TLDs
    python -m common.cli._internal.query_domain_index --top-tlds 20

    # Filter by TLD
    python -m common.cli._internal.query_domain_index --tld .edu
    python -m common.cli._internal.query_domain_index --tld .gov

    # Filter by domain pattern (substring match)
    python -m common.cli._internal.query_domain_index --domain-contains wiki

    # Filter by URL path pattern
    python -m common.cli._internal.query_domain_index --path-contains /wiki/BMW
    python -m common.cli._internal.query_domain_index --path-contains /wiki/Honda

    # Custom SQL query
    python -m common.cli._internal.query_domain_index --sql "SELECT tld, SUM(token_count) as tokens FROM idx GROUP BY tld ORDER BY tokens DESC LIMIT 10"
"""

import argparse
import sys

import duckdb

from common.data import get_datasets_dir

INDEX_PATH = get_datasets_dir() / "HuggingFaceFW/fineweb/domain_index.parquet"


def get_connection():
    """Get DuckDB connection with index loaded."""
    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH}")
        print("Run 'python -m common.cli.data fineweb index' first to create the index.")
        sys.exit(1)

    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW idx AS SELECT * FROM read_parquet('{INDEX_PATH}')")
    return con


def show_summary(con):
    """Show summary statistics."""
    print("=" * 60)
    print("FineWeb Domain Index Summary")
    print("=" * 60)

    # Basic counts
    result = con.execute("""
        SELECT
            COUNT(*) as total_docs,
            COUNT(DISTINCT etld_plus_one) as unique_domains,
            COUNT(DISTINCT tld) as unique_tlds,
            SUM(token_count) as total_tokens,
            AVG(token_count) as avg_tokens_per_doc
        FROM idx
    """).fetchone()

    print(f"Total documents: {result[0]:,}")
    print(f"Unique eTLD+1 domains: {result[1]:,}")
    print(f"Unique TLDs: {result[2]:,}")
    print(f"Total tokens: {result[3]:,}")
    print(f"Avg tokens/doc: {result[4]:.0f}")
    print()


def top_domains(con, n: int = 30):
    """Show top domains by document count."""
    print(f"Top {n} Domains (by document count)")
    print("-" * 60)

    result = con.execute(f"""
        SELECT
            etld_plus_one,
            COUNT(*) as docs,
            SUM(token_count) as tokens,
            AVG(token_count) as avg_tokens
        FROM idx
        GROUP BY etld_plus_one
        ORDER BY docs DESC
        LIMIT {n}
    """).fetchall()

    print(f"{'Domain':<35} {'Docs':>10} {'Tokens':>12} {'Avg':>8}")
    print("-" * 65)
    for row in result:
        print(f"{row[0]:<35} {row[1]:>10,} {row[2]:>12,} {row[3]:>8.0f}")
    print()


def top_tlds(con, n: int = 20):
    """Show top TLDs by document count."""
    print(f"Top {n} TLDs (by document count)")
    print("-" * 60)

    result = con.execute(f"""
        SELECT
            tld,
            COUNT(*) as docs,
            SUM(token_count) as tokens,
            COUNT(DISTINCT etld_plus_one) as unique_domains
        FROM idx
        GROUP BY tld
        ORDER BY docs DESC
        LIMIT {n}
    """).fetchall()

    print(f"{'TLD':<12} {'Docs':>12} {'Tokens':>14} {'Unique Domains':>16}")
    print("-" * 55)
    for row in result:
        print(f"{row[0]:<12} {row[1]:>12,} {row[2]:>14,} {row[3]:>16,}")
    print()


def filter_by_tld(con, tld: str, top_n: int = 30):
    """Show top domains for a specific TLD."""
    # Ensure TLD starts with dot
    if not tld.startswith("."):
        tld = f".{tld}"

    print(f"Top {top_n} Domains for TLD: {tld}")
    print("-" * 60)

    # Summary for this TLD
    summary = con.execute(f"""
        SELECT COUNT(*) as docs, SUM(token_count) as tokens
        FROM idx WHERE tld = '{tld}'
    """).fetchone()

    print(f"Total docs: {summary[0]:,}, Total tokens: {summary[1]:,}")
    print()

    result = con.execute(f"""
        SELECT
            etld_plus_one,
            COUNT(*) as docs,
            SUM(token_count) as tokens
        FROM idx
        WHERE tld = '{tld}'
        GROUP BY etld_plus_one
        ORDER BY docs DESC
        LIMIT {top_n}
    """).fetchall()

    print(f"{'Domain':<40} {'Docs':>10} {'Tokens':>12}")
    print("-" * 62)
    for row in result:
        print(f"{row[0]:<40} {row[1]:>10,} {row[2]:>12,}")
    print()


def filter_by_domain_pattern(con, pattern: str, top_n: int = 30):
    """Show domains matching a pattern."""
    print(f"Domains containing '{pattern}' (top {top_n})")
    print("-" * 60)

    result = con.execute(f"""
        SELECT
            etld_plus_one,
            COUNT(*) as docs,
            SUM(token_count) as tokens
        FROM idx
        WHERE etld_plus_one LIKE '%{pattern}%'
        GROUP BY etld_plus_one
        ORDER BY docs DESC
        LIMIT {top_n}
    """).fetchall()

    print(f"{'Domain':<40} {'Docs':>10} {'Tokens':>12}")
    print("-" * 62)
    for row in result:
        print(f"{row[0]:<40} {row[1]:>10,} {row[2]:>12,}")
    print()


def filter_by_path_pattern(con, pattern: str, top_n: int = 50):
    """Show documents matching a URL path pattern."""
    print(f"Documents with URL path containing '{pattern}' (top {top_n})")
    print("-" * 80)

    # Summary stats
    summary = con.execute(f"""
        SELECT COUNT(*) as docs, SUM(token_count) as tokens
        FROM idx WHERE url_path LIKE '%{pattern}%'
    """).fetchone()

    print(f"Total matches: {summary[0]:,} docs, {summary[1]:,} tokens")
    print()

    # Top paths matching pattern
    result = con.execute(f"""
        SELECT
            url_path,
            etld_plus_one,
            COUNT(*) as docs,
            SUM(token_count) as tokens
        FROM idx
        WHERE url_path LIKE '%{pattern}%'
        GROUP BY url_path, etld_plus_one
        ORDER BY docs DESC
        LIMIT {top_n}
    """).fetchall()

    print(f"{'URL Path':<50} {'Domain':<20} {'Docs':>8} {'Tokens':>10}")
    print("-" * 90)
    for row in result:
        path_display = row[0][:47] + "..." if len(row[0]) > 50 else row[0]
        domain_display = row[1][:17] + "..." if len(row[1]) > 20 else row[1]
        print(f"{path_display:<50} {domain_display:<20} {row[2]:>8,} {row[3]:>10,}")
    print()


def run_sql(con, sql: str):
    """Run custom SQL query."""
    print("Custom SQL Query")
    print("-" * 60)
    print(f"Query: {sql}")
    print()

    result = con.execute(sql).fetchall()
    columns = [desc[0] for desc in con.description]

    # Print header
    print(" | ".join(f"{col:<15}" for col in columns))
    print("-" * (17 * len(columns)))

    # Print rows
    for row in result:
        print(" | ".join(f"{str(val):<15}" for val in row))
    print()


def main():
    parser = argparse.ArgumentParser(description="Query FineWeb domain index")
    parser.add_argument("--top-domains", type=int, metavar="N", help="Show top N domains")
    parser.add_argument("--top-tlds", type=int, metavar="N", help="Show top N TLDs")
    parser.add_argument("--tld", type=str, help="Filter by TLD (e.g., .edu, .gov)")
    parser.add_argument("--domain-contains", type=str, help="Filter domains containing pattern")
    parser.add_argument("--path-contains", type=str, help="Filter by URL path pattern (e.g., /wiki/BMW)")
    parser.add_argument("--sql", type=str, help="Run custom SQL (table name: idx)")

    args = parser.parse_args()

    con = get_connection()

    # If no specific query, show summary
    if not any([args.top_domains, args.top_tlds, args.tld, args.domain_contains, args.path_contains, args.sql]):
        show_summary(con)
        top_tlds(con, 15)
        top_domains(con, 30)
        return

    if args.top_domains:
        top_domains(con, args.top_domains)

    if args.top_tlds:
        top_tlds(con, args.top_tlds)

    if args.tld:
        filter_by_tld(con, args.tld)

    if args.domain_contains:
        filter_by_domain_pattern(con, args.domain_contains)

    if args.path_contains:
        filter_by_path_pattern(con, args.path_contains)

    if args.sql:
        run_sql(con, args.sql)


if __name__ == "__main__":
    main()
