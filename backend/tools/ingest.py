"""Ingest clinical trials into the vector store.

Ingestion is a batch job, not an API: it loads the embedding model, walks a
597k-row parquet, and runs for hours. Run it from the backend root:

    .venv/Scripts/python.exe tools/ingest.py bulk --limit 1000
    .venv/Scripts/python.exe tools/ingest.py single NCT05129748
    .venv/Scripts/python.exe tools/ingest.py sync --since 2026-10-01
    .venv/Scripts/python.exe tools/ingest.py delete NCT05129748 NCT05893030
"""

import argparse
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.controller.ingestion import get_ingestion_controller  # noqa: E402
from src.models.ingestion import IngestionResult  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def report(result: IngestionResult) -> int:
    """Log the outcome and return the exit code: non-zero if anything failed."""
    logger.info(
        "processed=%s succeeded=%s failed=%s",
        result.processed,
        result.succeeded,
        result.failed,
    )
    for nct_id, error in result.errors.items():
        logger.error("  %s: %s", nct_id, error)
    if getattr(result, "checked", None) is not None:
        logger.info("checked=%s unchanged=%s", result.checked, result.unchanged)
    return 1 if result.failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Ingest one trial, fetched from the API.")
    p_single.add_argument("nct_id")

    p_bulk = sub.add_parser("bulk", help="Ingest a range of the local parquet corpus.")
    p_bulk.add_argument(
        "--start",
        type=int,
        default=0,
        help="First raw file row to read. Default: 0.",
    )
    p_bulk.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "How many raw file rows to read from --start. Counts rows read, not "
            "trials ingested: most rows are filtered out. Default: to the end."
        ),
    )
    p_bulk.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Ingest every row in the range. By default only interventional "
            "trials that are recruiting, active-not-recruiting or enrolling by "
            "invitation are embedded -- roughly 85%% of the corpus is skipped."
        ),
    )

    p_update = sub.add_parser("update", help="Re-fetch and re-embed trials by id.")
    p_update.add_argument("nct_ids", nargs="+")

    p_delete = sub.add_parser("delete", help="Remove trials from the vector store.")
    p_delete.add_argument("nct_ids", nargs="+")

    p_sync = sub.add_parser(
        "sync", help="Re-embed trials the registry changed since a date."
    )
    p_sync.add_argument(
        "--since",
        default=None,
        help="ISO date (YYYY-MM-DD). Default: 7 days ago.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controller = get_ingestion_controller()

    if args.command == "single":
        return report(controller.ingest_single(args.nct_id))

    if args.command == "bulk":
        result = controller.ingest_bulk_from_parquet(
            limit=args.limit,
            start=args.start,
            apply_filter=not args.no_filter,
        )
        # State the range explicitly so the next run can start where this ended.
        logger.info(
            "rows %s-%s read=%s skipped=%s",
            result.row_start,
            result.row_end - 1,
            result.row_end - result.row_start,
            result.skipped,
        )
        logger.info("resume the next chunk with --start %s", result.row_end)
        return report(result)

    if args.command == "update":
        return report(controller.update_bulk(args.nct_ids))

    if args.command == "delete":
        controller.delete_bulk(args.nct_ids)
        logger.info("deleted=%s", len(args.nct_ids))
        return 0

    return report(controller.sync(args.since))


if __name__ == "__main__":
    raise SystemExit(main())
