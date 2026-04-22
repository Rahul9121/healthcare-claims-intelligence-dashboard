from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load members and claims CSVs to PostgreSQL.")
    parser.add_argument(
        "--members-path", type=Path, default=Path("data/raw/members.csv"), help="Members CSV path."
    )
    parser.add_argument(
        "--claims-path", type=Path, default=Path("data/raw/claims.csv"), help="Claims CSV path."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", ""),
        help="SQLAlchemy PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="public",
        help="Destination PostgreSQL schema name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db_url:
        raise ValueError("Database URL not found. Pass --db-url or set DATABASE_URL.")

    members = pd.read_csv(args.members_path)
    claims = pd.read_csv(args.claims_path)
    engine = create_engine(args.db_url)

    with engine.begin() as connection:
        members.to_sql(
            "members",
            con=connection,
            schema=args.schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        claims.to_sql(
            "claims",
            con=connection,
            schema=args.schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
    print(f"Loaded members ({len(members)}) and claims ({len(claims)}) into schema '{args.schema}'.")


if __name__ == "__main__":
    main()
