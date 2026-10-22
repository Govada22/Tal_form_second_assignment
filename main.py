"""Financial Sales Pipeline QA - orchestrator.

Usage:
    python -m src.main --demo         # offline run against seeded fixtures
    python -m src.main                 # live run against MySQL / API / dashboards

The live path reads db_config.yaml, api_config.yaml and validation_rules.yaml,
connects with the connectors, and feeds the resulting DataFrames into the same
validators used by the demo path, so the validation logic is identical.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# allow "python src/main.py" as well as "python -m src.main"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config, config_dir          # noqa: E402
from src.utils.logger import get_logger                              # noqa: E402
from src.reports.report_writer import write_reports                  # noqa: E402
from src.validators import (                                         # noqa: E402
    count_validator as cnt,
    duplicate_validator as dup,
    null_validator as nul,
    comparison_validator as cmp,
    transformation_validator as tr,
    aggregate_validator as agg,
    report_validator as rpt,
)

LOG = get_logger()


# ---------------------------------------------------------------------------
def run_all(datasets: dict, rules: dict) -> list:
    """Execute every validation layer against the supplied datasets."""
    results = []
    tol = rules["tolerances"]["amount_abs"]
    filt = rules["load_filters"]

    retail_src = datasets["retail_src"]
    retail_stg = datasets["retail_stg"]
    dist_src = datasets["dist_src"]
    dist_stg = datasets["dist_stg"]
    online_src = datasets["online_src"]
    online_stg = datasets["online_stg"]
    pmaster = datasets["product_master"]
    rmap = datasets["region_mapping"]
    cmaster = datasets["customer_master"]
    dm = datasets["datamart"]
    summaries = datasets["summaries"]
    dashboard = datasets["dashboard"]

    LOG.info("Layer 1: Source-to-Staging")
    # ---- RETAIL ----
    results.append(cnt.count_check(
        "RET-V01", "Source-to-Staging", "Retail CSV (COMPLETED) count vs staging count",
        "retail_sales_raw.csv", "stg_retail_sales",
        cnt.filtered_source_count(retail_src, "status", filt["retail_status"]),
        len(retail_stg)))
    results.append(dup.duplicate_check(
        "RET-V02", "Source-to-Staging", "Retail duplicate transaction_id",
        "retail_sales_raw.csv", retail_src.rename(columns={"retail_txn_id": "transaction_id"}),
        "transaction_id"))
    results.append(cmp.rename_fidelity_check(
        "RET-V03", "Source-to-Staging", "retail_txn_id -> transaction_id rename fidelity",
        "retail_sales_raw.csv", "stg_retail_sales",
        retail_src, retail_stg, "retail_txn_id", "transaction_id"))
    results.append(nul.null_check(
        "RET-V04", "Source-to-Staging", "Retail mandatory customer_id not null",
        "stg_retail_sales", retail_stg, ["customer_id"]))
    results.append(tr.excluded_rows_check(
        "RET-V05", "Source-to-Staging", "Retail CANCELLED excluded from staging",
        "stg_retail_sales", retail_stg.get("status", pd.Series(dtype=str)).eq("CANCELLED"),
        "CANCELLED"))
    results.append(cmp.reference_integrity_check(
        "RET-V07", "Source-to-Staging", "Retail branch_code exists in region_mapping",
        "stg_retail_sales", "region_mapping", retail_stg, "branch_code",
        set(rmap["branch_code"])))
    results.append(cmp.reference_integrity_check(
        "RET-V08", "Source-to-Staging", "Retail product_code exists in product_master",
        "stg_retail_sales", "product_master_raw", retail_stg, "product_code",
        set(pmaster["product_code"])))

    # ---- DISTRIBUTOR ----
    results.append(cnt.count_check(
        "DIST-V01", "Source-to-Staging", "Distributor (APPROVED) count vs staging count",
        "distributor_sales_raw", "stg_distributor_sales",
        cnt.filtered_source_count(dist_src, "status", filt["distributor_status"]),
        len(dist_stg)))
    results.append(tr.excluded_rows_check(
        "DIST-V02", "Source-to-Staging", "Only APPROVED rows in distributor staging",
        "stg_distributor_sales",
        ~dist_stg.get("status", pd.Series(dtype=str)).eq("APPROVED"),
        "non-APPROVED"))
    results.append(cmp.rename_fidelity_check(
        "DIST-V03", "Source-to-Staging", "distributor_txn_id -> transaction_id rename fidelity",
        "distributor_sales_raw", "stg_distributor_sales",
        dist_src, dist_stg, "distributor_txn_id", "transaction_id"))
    results.append(nul.null_check(
        "DIST-V04", "Source-to-Staging", "Distributor null customer_id flagged",
        "stg_distributor_sales", dist_stg, ["customer_id"], severity="Medium"))
    # commission must be <= gross
    comm_bad = (pd.to_numeric(dist_stg["commission_amount"], errors="coerce")
                > pd.to_numeric(dist_stg["gross_amount"], errors="coerce")).fillna(False)
    results.append(tr.excluded_rows_check(
        "DIST-V06", "Data Quality", "Distributor commission_amount <= gross_amount",
        "stg_distributor_sales", comm_bad, "commission > gross"))

    # ---- ONLINE ----
    results.append(cnt.count_check(
        "ONL-V01", "Source-to-Staging", "Online API (COMPLETED) count vs staging count",
        "/api/online-sales", "stg_online_sales",
        cnt.filtered_source_count(online_src, "status", filt["online_status"]),
        len(online_stg)))
    results.append(tr.excluded_rows_check(
        "ONL-V03", "Source-to-Staging", "Only COMPLETED rows in online staging",
        "stg_online_sales",
        ~online_stg.get("status", pd.Series(dtype=str)).eq("COMPLETED"),
        "non-COMPLETED"))
    results.append(dup.duplicate_check(
        "ONL-V05", "Source-to-Staging", "Online transaction_id unique across pages",
        "/api/online-sales", online_stg, "transaction_id"))

    LOG.info("Layer 2: Data Quality")
    mand_ids = {"stg_retail_sales": "DQ-V01", "stg_distributor_sales": "DQ-V02",
                "stg_online_sales": "DQ-V03"}
    for tbl, cols in rules["mandatory_fields"].items():
        df = {"stg_retail_sales": retail_stg, "stg_distributor_sales": dist_stg,
              "stg_online_sales": online_stg}[tbl]
        results.append(nul.null_check(
            mand_ids[tbl], "Data Quality", f"Mandatory fields populated - {tbl}",
            tbl, df, cols))
    neg_amt = (pd.to_numeric(dm["gross_amount"], errors="coerce") <= 0).fillna(True)
    results.append(tr.excluded_rows_check(
        "DQ-V04", "Data Quality", "No non-positive gross_amount in data mart",
        "dm_sales_transaction", neg_amt, "non-positive amount"))
    # orphan customers & rejected KYC
    rejected = set(cmaster.loc[cmaster["kyc_status"] == "REJECTED", "customer_id"])
    kyc_bad = dm["customer_id"].isin(rejected)
    results.append(tr.excluded_rows_check(
        "DQ-V06", "Data Quality", "REJECTED-KYC customers not contributing to net sales",
        "dm_sales_transaction", kyc_bad, "REJECTED-KYC contributor"))
    # inactive product sold
    inactive = set(pmaster.loc[pmaster["active_flag"] == "N", "product_code"])
    results.append(tr.excluded_rows_check(
        "DQ-V05", "Data Quality", "No inactive (active_flag=N) products in current sales",
        "dm_sales_transaction", dm["product_code"].isin(inactive), "inactive-product sale"))

    LOG.info("Layer 3: Staging-to-Data Mart")
    results.append(tr.product_type_harmonisation_check(
        "DM-V02", "Staging-to-Data Mart", "Product type harmonisation vs master",
        "stg_*", "dm_sales_transaction", dm, "product_type_raw", "standard_product_type",
        rules["product_type_harmonisation"]))
    results.append(tr.ulip_insurance_check(
        "DM-V03", "Staging-to-Data Mart", "ULIP maps to INSURANCE",
        "dm_sales_transaction", dm, "product_type_raw", "standard_product_type"))
    results.append(tr.product_name_harmonisation_check(
        "DM-V04", "Staging-to-Data Mart", "Data mart stores standard_product_name",
        "product_master_raw", "dm_sales_transaction", dm, "product_code",
        "standard_product_name", pmaster))
    results.append(tr.channel_mapping_check(
        "DM-V07", "Staging-to-Data Mart", "Channel mapping to Retail/Distributor/Online",
        "dm_sales_transaction", dm, "channel_code_raw", "source_channel",
        rules["channel_harmonisation"], rules["domains"]["channels"]))
    results.append(tr.region_domain_check(
        "DM-V08", "Staging-to-Data Mart", "Region values within valid domain",
        "dm_sales_transaction", dm, "region_name", rules["domains"]["regions"]))
    results.append(tr.customer_cleansing_check(
        "DM-V05", "Staging-to-Data Mart", "Customer names trimmed and title-cased",
        "dm_sales_transaction", dm, "customer_name"))
    results.append(tr.net_amount_check(
        "DM-V10", "Staging-to-Data Mart", "net_sales_amount = gross - discount - reversal",
        "dm_sales_transaction", dm, abs_tol=tol))

    LOG.info("Layer 4: Aggregate Models")
    ex = summaries["executive"]
    results.append(agg.sum_check(
        "AGG-V01", "Aggregate Model", "Total net sales = sum of dm transactions",
        "dm_sales_transaction", "dm_executive_sales_summary",
        float(dm["net_sales_amount"].sum()), ex["total_net_sales"], abs_tol=tol))
    results.append(agg.sum_check(
        "AGG-V03", "Aggregate Model", "Insurance premium = INSURANCE net sales",
        "dm_sales_transaction", "dm_executive_sales_summary",
        float(dm.loc[dm["standard_product_type"] == "INSURANCE", "net_sales_amount"].sum()),
        ex["insurance_premium"], abs_tol=tol))
    results.append(agg.sum_check(
        "AGG-V04", "Aggregate Model", "Mutual fund sales = MUTUAL_FUND net sales",
        "dm_sales_transaction", "dm_executive_sales_summary",
        float(dm.loc[dm["standard_product_type"] == "MUTUAL_FUND", "net_sales_amount"].sum()),
        ex["mutual_fund_sales"], abs_tol=tol))
    results.append(agg.average_ticket_check(
        "AGG-V05", "Aggregate Model", "Average ticket size = net / count",
        "dm_executive_sales_summary", ex["total_net_sales"], ex["transaction_count"],
        ex["avg_ticket_size"], abs_tol=tol))
    results.append(agg.grouping_crossfoot_check(
        "AGG-V06", "Aggregate Model", "Region summary cross-foots to total",
        "dm_sales_region_summary", "dm_sales_transaction",
        float(dm["net_sales_amount"].sum()),
        float(summaries["region_summary"]["net_sales_amount"].sum()), abs_tol=tol))
    results.append(agg.grouping_crossfoot_check(
        "AGG-V07", "Aggregate Model", "Channel summary cross-foots to total",
        "dm_sales_channel_summary", "dm_sales_transaction",
        float(dm["net_sales_amount"].sum()),
        float(summaries["channel_summary"]["net_sales_amount"].sum()), abs_tol=tol))

    LOG.info("Layer 5: Data Mart-to-Reporting")
    results.append(rpt.dashboard_value_check(
        "RPT-V01", "Data Mart-to-Reporting", "Executive dashboard total vs data mart",
        "dm_executive_sales_summary", "Executive Dashboard",
        ex["total_net_sales"], dashboard["executive_total"], abs_tol=tol))
    results.append(rpt.dashboard_value_check(
        "RPT-V02", "Data Mart-to-Reporting", "Insurance premium tile vs data mart",
        "dm_executive_sales_summary", "Executive Dashboard",
        ex["insurance_premium"], dashboard["insurance_premium"], abs_tol=tol))
    ch = summaries["channel_summary"]
    online_dm = float(ch.loc[ch["source_channel"] == "Online", "net_sales_amount"].sum())
    results.append(rpt.filter_check(
        "RPT-V08", "Data Mart-to-Reporting", "Channel filter = Online reproduces figure",
        "dm_sales_channel_summary", "Channel Performance Dashboard",
        online_dm, dashboard["channel_online"], "channel=Online", abs_tol=tol))

    return results


# ---------------------------------------------------------------------------
def build_demo_datasets(rules) -> dict:
    from src.utils import demo_data as d
    pmaster = d.product_master()
    rmap = d.region_mapping()
    cmaster = d.customer_master()
    retail_src = d.retail_source()
    retail_stg = d.retail_staging(retail_src)
    dist_src = d.distributor_source()
    dist_stg = d.distributor_staging(dist_src)
    online_src = d.online_source()
    online_stg = d.online_staging(online_src)
    dm = d.build_datamart(retail_stg, dist_stg, online_stg, pmaster, rmap, cmaster)
    summaries = d.build_summaries(dm)
    dashboard = d.build_dashboard(summaries)
    return dict(retail_src=retail_src, retail_stg=retail_stg,
                dist_src=dist_src, dist_stg=dist_stg,
                online_src=online_src, online_stg=online_stg,
                product_master=pmaster, region_mapping=rmap, customer_master=cmaster,
                datamart=dm, summaries=summaries, dashboard=dashboard)


# ---------------------------------------------------------------------------
# Live-mode helpers
# ---------------------------------------------------------------------------
def _load_live_configs() -> tuple[dict, dict]:
    """Load db_config.yaml and api_config.yaml with ${ENV:-default} resolved."""
    cfg = config_dir()
    db_cfg = load_config(cfg / "db_config.yaml")
    api_cfg = load_config(cfg / "api_config.yaml")
    return db_cfg, api_cfg


def _read_table(db_cfg: dict, logical_db: str, sql: str) -> pd.DataFrame:
    """Run a read-only SELECT against a logical database and return a DataFrame."""
    from src.connectors.mysql_connector import MySQLConnector
    with MySQLConnector(db_cfg, logical_db) as conn:
        return conn.query_df(sql)


def _executive_summary_row(df: pd.DataFrame) -> dict:
    """Collapse the single-row executive summary view into the dict the
    aggregate validators expect. Coerces to native numeric types so the
    tolerance comparisons in aggregate_validator behave identically to demo.
    """
    if df.empty:
        raise RuntimeError("dm_executive_sales_summary returned no rows.")
    row = df.iloc[0]
    return {
        "total_net_sales": float(row["total_net_sales"]),
        "insurance_premium": float(row["insurance_premium"]),
        "mutual_fund_sales": float(row["mutual_fund_sales"]),
        "transaction_count": int(row["transaction_count"]),
        "avg_ticket_size": float(row["avg_ticket_size"]),
    }


def _dashboard_figures(api_cfg: dict) -> dict:
    """Read the Flask reporting dashboards and extract the three figures the
    report layer reconciles against the data mart.

    The dashboards render labelled metric tables (metric | value). We match on
    the metric *label* rather than a fixed cell position, so a column re-order
    on the page won't silently break extraction. If the live dashboards use
    different wording, adjust the label strings passed to ``_figure`` below.
    Values are read from the live pages on purpose (not the data mart) so a
    genuine dashboard/data-mart mismatch is actually caught rather than masked.
    """
    try:
        import requests
    except ImportError as exc:                      # pragma: no cover
        raise RuntimeError(
            "requests is not installed; needed for live dashboard checks. "
            "Install it or run with --demo."
        ) from exc
    from src.connectors.file_connector import FileConnector

    app = api_cfg["reporting_app"]
    base = app["base_url"].rstrip("/")

    def _figure(page_path: str, label: str) -> float:
        url = f"{base}{page_path}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for table in FileConnector.read_report_html(resp.text):
            cols = [str(c).strip().lower() for c in table.columns]
            table = table.copy()
            table.columns = cols
            label_col = next((c for c in cols
                              if c in ("metric", "label", "kpi", "measure")), cols[0])
            value_col = next((c for c in cols
                              if c in ("value", "amount", "figure", "total")), cols[-1])
            hit = table[table[label_col].astype(str).str.strip().str.lower()
                        == label.strip().lower()]
            if not hit.empty:
                raw = str(hit.iloc[0][value_col])
                cleaned = re.sub(r"[^0-9.\-]", "", raw)
                if cleaned in ("", "-", "."):
                    raise RuntimeError(f"Non-numeric value for '{label}' on {url}: {raw!r}")
                return float(cleaned)
        raise RuntimeError(
            f"Could not locate metric '{label}' on {url}. "
            "Check the dashboard markup / label wording in _dashboard_figures()."
        )

    pages = app["pages"]
    return {
        "executive_total": _figure(pages["executive"], "Total Net Sales"),
        "insurance_premium": _figure(pages["executive"], "Insurance Premium"),
        "channel_online": _figure(pages["channel"], "Online"),
    }


def build_live_datasets(rules) -> dict:
    """Assemble datasets from the live GCP environment via the connectors.

    Returns the identical dict shape as build_demo_datasets(), so run_all() is
    completely agnostic to where the data came from. Requires the environment
    variables documented in the README (MYSQL_*, ONLINE_API_*, REPORT_APP_BASE_URL,
    RETAIL_CSV_PATH). If you don't have environment access, run with --demo.

    Wiring map (README section 4):
      retail source  -> FileConnector.read_csv(RETAIL_CSV_PATH)
      staging tables -> MySQLConnector('staging').query_df(...)
      online source  -> APIConnector.fetch_all(...)  (paginated, COMPLETED)
      source masters -> MySQLConnector('source_master' / 'source_distributor')
      data mart      -> MySQLConnector('datamart').query_df(...)
      summary views  -> MySQLConnector('datamart').query_df(...)
      dashboards     -> FileConnector.read_report_html(<flask page html>)
    """
    from src.connectors.file_connector import FileConnector
    from src.connectors.api_connector import APIConnector

    db_cfg, api_cfg = _load_live_configs()
    src = rules["sources"]
    business_date = src["business_date"]

    # ---- Retail source: daily CSV ----
    LOG.info("Live: reading retail CSV source (%s)", src["retail_csv_path"])
    retail_src = FileConnector.read_csv(src["retail_csv_path"])

    # ---- Source-system tables (MySQL) ----
    LOG.info("Live: reading distributor source and master tables from MySQL")
    dist_src = _read_table(db_cfg, "source_distributor",
                           "SELECT * FROM distributor_sales_raw")
    pmaster = _read_table(db_cfg, "source_master", "SELECT * FROM product_master_raw")
    rmap = _read_table(db_cfg, "source_master", "SELECT * FROM region_mapping")
    cmaster = _read_table(db_cfg, "source_master", "SELECT * FROM customer_master")

    # ---- Staging tables (MySQL) ----
    LOG.info("Live: reading staging tables from MySQL")
    retail_stg = _read_table(db_cfg, "staging", "SELECT * FROM stg_retail_sales")
    dist_stg = _read_table(db_cfg, "staging", "SELECT * FROM stg_distributor_sales")
    online_stg = _read_table(db_cfg, "staging", "SELECT * FROM stg_online_sales")

    # ---- Online source: REST API (paginated, COMPLETED only) ----
    LOG.info("Live: fetching online sales from API for business date %s", business_date)
    online_src = APIConnector(api_cfg).fetch_all(
        from_date=business_date, to_date=business_date,
        status=api_cfg["online_sales_api"]["params"]["default_status"])

    # ---- Data mart (MySQL) ----
    LOG.info("Live: reading data mart from MySQL")
    dm = _read_table(db_cfg, "datamart", "SELECT * FROM dm_sales_transaction")

    # ---- Aggregate summary views (MySQL) ----
    LOG.info("Live: reading aggregate summary views from MySQL")
    executive_df = _read_table(db_cfg, "datamart",
                               "SELECT * FROM dm_executive_sales_summary")
    region_summary = _read_table(db_cfg, "datamart",
                                 "SELECT * FROM dm_sales_region_summary")
    channel_summary = _read_table(db_cfg, "datamart",
                                  "SELECT * FROM dm_sales_channel_summary")
    summaries = {
        "executive": _executive_summary_row(executive_df),
        "region_summary": region_summary,
        "channel_summary": channel_summary,
    }

    # ---- Reporting dashboards (Flask HTML) ----
    LOG.info("Live: reading reporting dashboards for report-level reconciliation")
    dashboard = _dashboard_figures(api_cfg)

    return dict(retail_src=retail_src, retail_stg=retail_stg,
                dist_src=dist_src, dist_stg=dist_stg,
                online_src=online_src, online_stg=online_stg,
                product_master=pmaster, region_mapping=rmap, customer_master=cmaster,
                datamart=dm, summaries=summaries, dashboard=dashboard)


def main() -> int:
    parser = argparse.ArgumentParser(description="Financial Sales Pipeline QA suite")
    parser.add_argument("--demo", action="store_true",
                        help="Run offline against seeded fixtures.")
    args = parser.parse_args()

    cfg = config_dir()
    rules = load_config(cfg / "validation_rules.yaml")
    LOG.info("Configuration loaded from %s", cfg)

    if args.demo:
        LOG.info("Running in DEMO mode (offline fixtures with seeded defects).")
        datasets = build_demo_datasets(rules)
    else:
        LOG.info("Running in LIVE mode.")
        try:
            datasets = build_live_datasets(rules)
        except Exception as exc:                    # noqa: BLE001
            LOG.error("Live run could not assemble datasets: %s", exc)
            LOG.error("Verify the environment variables from the README "
                      "(MYSQL_*, ONLINE_API_*, REPORT_APP_BASE_URL, RETAIL_CSV_PATH) "
                      "and that the sources are reachable. To run offline, use --demo.")
            return 2

    results = run_all(datasets, rules)

    out_dir = Path(__file__).resolve().parents[1] / "output" / "validation_reports"
    paths = write_reports(results, out_dir)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    LOG.info("Execution complete: %d checks, %d PASS, %d FAIL (%.1f%% pass rate)",
             total, passed, failed, passed / total * 100 if total else 0)
    for fmt, p in paths.items():
        LOG.info("Report (%s): %s", fmt, p)

    print("\n" + "=" * 68)
    print(f" VALIDATION SUMMARY : {passed}/{total} passed, {failed} failed")
    print("=" * 68)
    for r in results:
        if r.status != "PASS":
            print(f" [{r.status}] {r.validation_id:<10} {r.severity:<6} {r.validation_name}")
            print(f"         expected={r.expected_value}  actual={r.actual_value}  -> {r.remarks}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
