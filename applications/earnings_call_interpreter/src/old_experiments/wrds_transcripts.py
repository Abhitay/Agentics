"""
WRDS → CIQ → Earnings call sections & full transcripts.

This is a refactor of `data_wrds.ipynb` into reusable functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import wrds


UNIVERSE_DIR = Path("data/universe")
CORPUS_DIR = Path("data/corpus")

UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

TOP_TECH_PARQUET = UNIVERSE_DIR / "tech_universe_top20.parquet"
TECH_CIQ_PARQUET = UNIVERSE_DIR / "tech_universe_top20_with_ciq.parquet"
CALL_SECTION_PARQUET = CORPUS_DIR / "tech_call_sections.parquet"
EARNINGS_CALL_PARQUET = CORPUS_DIR / "tech_earnings_calls.parquet"


def _connect_wrds() -> wrds.Connection:
    """
    Connect to WRDS using env variable WRDS_USERNAME or fallback string.
    """
    username = os.getenv("WRDS_USERNAME", "priyamvadadaga")
    db = wrds.Connection(wrds_username=username)
    return db


def build_tech_universe(db: wrds.Connection, top_n: int = 20) -> pd.DataFrame:
    """
    Build top-N tech universe (by market cap) and link to CIQ company IDs.
    Returns df with ciq_company_id, ticker, company_name, market_cap, etc.
    """
    # 1) Compustat names
    names_query = """
        SELECT DISTINCT gvkey, tic AS ticker, conm AS company_name, gind, gsubind, sic, naics
        FROM comp.names
        WHERE tic IS NOT NULL
    """
    names_df = db.raw_sql(names_query)

    # filter: GICS sector 45 (Information Technology)
    names_with_gind = names_df.dropna(subset=["gind"]).copy()
    names_with_gind.loc[:, "gind_str"] = names_with_gind["gind"].astype(str)

    tech_universe = (
        names_with_gind[names_with_gind["gind_str"].str.startswith("45")]
        .drop(columns=["gind_str"])
        .drop_duplicates("gvkey")
    )

    # 2) Market cap from funda
    mktcap_query = """
        WITH latest AS (
            SELECT gvkey, datadate, mkvalt, prcc_f, csho,
                ROW_NUMBER() OVER (
                    PARTITION BY gvkey
                    ORDER BY datadate DESC
                ) AS rn
            FROM comp.funda
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
        )
        SELECT gvkey, datadate, mkvalt, prcc_f, csho
        FROM latest
        WHERE rn = 1
    """
    mktcap_df = db.raw_sql(mktcap_query)

    mktcap_df.loc[:, "market_cap"] = mktcap_df["mkvalt"]
    missing_mask = mktcap_df["market_cap"].isna()
    mktcap_df.loc[missing_mask, "market_cap"] = (
        mktcap_df.loc[missing_mask, "prcc_f"] * mktcap_df.loc[missing_mask, "csho"]
    )
    mktcap_df = mktcap_df.dropna(subset=["market_cap"])

    tech_with_mktcap = tech_universe.merge(
        mktcap_df[["gvkey, market_cap".split()[0], "market_cap"]],
        on="gvkey",
        how="left",
    ).dropna(subset=["market_cap"])

    tech_sorted = tech_with_mktcap.sort_values("market_cap", ascending=False)
    top_tech = tech_sorted.head(top_n)

    top_tech.to_parquet(TOP_TECH_PARQUET, index=False)

    # 3) Link to CIQ company IDs
    link_query = """
        SELECT DISTINCT companyid, gvkey, ticker, companyname
        FROM ciq.wrds_ciqsymbol_primary
        WHERE gvkey IS NOT NULL AND companyid IS NOT NULL
    """
    ciq_links = db.raw_sql(link_query)

    ciq_links.loc[:, "gvkey"] = ciq_links["gvkey"].astype(str)
    top_tech.loc[:, "gvkey"] = top_tech["gvkey"].astype(str)

    top_tech_with_ciq = top_tech.merge(
        ciq_links.rename(
            columns={"companyid": "ciq_company_id", "companyname": "ciq_company_name"}
        ),
        on="gvkey",
        how="inner",
    )

    top_tech_with_ciq = (
        top_tech_with_ciq.dropna(subset=["ciq_company_id"])
        .drop_duplicates("ciq_company_id")
    )
    top_tech_with_ciq.loc[:, "ciq_company_id"] = top_tech_with_ciq[
        "ciq_company_id"
    ].astype(int)

    top_tech_with_ciq.to_parquet(TECH_CIQ_PARQUET, index=False)
    return top_tech_with_ciq


def build_call_sections(db: wrds.Connection, top_tech_with_ciq: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-statement call sections (clean_text) and save to corpus parquet.
    """
    company_ids = top_tech_with_ciq["ciq_company_id"].tolist()
    ids = ", ".join(str(cid) for cid in company_ids)

    sql_query = f"""
    SELECT 
        d.companyid, d.transcriptid, d.headline,
        d.mostimportantdateutc, d.mostimportanttimeutc,
        d.keydeveventtypeid, d.keydeveventtypename,
        d.companyname AS detail_companyname,
        d.transcriptcollectiontypeid, p.transcriptcomponenttypeid,
        p.transcriptcomponenttypename, p.transcriptpersonid,
        p.transcriptpersonname, p.proid, p.companyofperson,
        p.speakertypeid, p.speakertypename,
        p.componentorder,
        p.componenttextpreview,
        c.componenttext
    FROM ciq.wrds_transcript_detail d
    JOIN ciq.wrds_transcript_person p ON d.transcriptid = p.transcriptid
    JOIN ciq.ciqtranscriptcomponent c ON p.transcriptcomponentid = c.transcriptcomponentid
    WHERE d.companyid IN ({ids})
      AND d.mostimportantdateutc >= (CURRENT_DATE - INTERVAL '1 year')
      AND d.keydeveventtypename = 'Earnings Calls'
    ORDER BY d.companyid, d.transcriptid, p.componentorder;
    """

    raw_segments = db.raw_sql(sql_query)

    segments_enriched = raw_segments.merge(
        top_tech_with_ciq,
        left_on="companyid",
        right_on="ciq_company_id",
        how="left",
    )

    call_section = segments_enriched.loc[
        :,
        [
            "companyid",
            "company_name",
            "market_cap",
            "transcriptid",
            "headline",
            "mostimportantdateutc",
            "mostimportanttimeutc",
            "keydeveventtypeid",
            "keydeveventtypename",
            "transcriptcollectiontypeid",
            "transcriptcomponenttypename",
            "transcriptpersonname",
            "speakertypename",
            "componentorder",
            "componenttext",
        ],
    ].copy()

    call_section.loc[:, "segment_id"] = (
        call_section["transcriptid"].astype(str)
        + "_"
        + call_section["componentorder"].astype(str)
    )
    call_section = call_section.reset_index(drop=True)
    call_section["segment_idx"] = call_section.index

    call_section["mostimportantdateutc"] = pd.to_datetime(
        call_section["mostimportantdateutc"], errors="coerce"
    ).astype("datetime64[ns]")

    call_section.loc[:, "call_year"] = call_section["mostimportantdateutc"].dt.year
    call_section.loc[:, "call_quarter"] = (
        "Q" + call_section["mostimportantdateutc"].dt.quarter.astype(str)
    )
    call_section.loc[:, "call_period"] = (
        call_section["call_year"].astype(str)
        + " "
        + call_section["call_quarter"]
    )

    import re

    def clean_component_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.replace("\r", " ").strip()
        t = re.sub(r"\s+", " ", t)
        return t

    call_section.loc[:, "clean_text"] = call_section["componenttext"].apply(
        clean_component_text
    )

    call_section.to_parquet(CALL_SECTION_PARQUET, index=False)
    return call_section


def build_call_level(call_section: pd.DataFrame) -> pd.DataFrame:
    """
    Full transcript per call (concatenated componenttext).
    """
    call_section["call_datetime"] = pd.to_datetime(
        call_section["mostimportantdateutc"].astype(str)
        + " "
        + call_section["mostimportanttimeutc"].astype(str),
        errors="coerce",
    )

    call_level = (
        call_section.sort_values(
            ["companyid", "transcriptid", "componentorder"]
        )
        .groupby(
            [
                "companyid",
                "company_name",
                "market_cap",
                "transcriptid",
                "headline",
                "call_datetime",
            ],
            as_index=False,
        )
        .agg({"componenttext": lambda parts: "\n".join(parts)})
    )
    call_level = call_level.rename(
        columns={"componenttext": "full_transcript"}
    )

    call_level["call_date"] = pd.to_datetime(
        call_level["call_datetime"], errors="coerce"
    ).astype("datetime64[ns]")
    call_level["call_year"] = call_level["call_date"].dt.year
    call_level["call_quarter"] = "Q" + call_level["call_date"].dt.quarter.astype(str)
    call_level["call_period"] = (
        call_level["call_year"].astype(str)
        + " "
        + call_level["call_quarter"]
    )

    call_level.to_parquet(EARNINGS_CALL_PARQUET, index=False)
    return call_level


def run_full_wrds_pipeline(top_n: int = 20):
    """
    Entry point: builds universe + transcripts in parquet form.
    """
    db = _connect_wrds()
    top_tech_with_ciq = build_tech_universe(db, top_n=top_n)
    call_section = build_call_sections(db, top_tech_with_ciq)
    call_level = build_call_level(call_section)
    print("Universe + call corpus built.")
    return top_tech_with_ciq, call_section, call_level
