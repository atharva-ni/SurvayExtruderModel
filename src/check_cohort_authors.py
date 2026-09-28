"""
Author-record check for the cohort profiles (OpenAlex)
======================================================
For every author in data/cohort_openalex.json: the OpenAlex author ID used
(one author record per person, no records merged), the affiliations the
author lists on their own papers, the share of papers outside the field
(a sign of namesakes merged into the record), and, for Cohort B, how many of
the author's 20 most-cited Google Scholar papers the profile contains.
Writes reports/cohort/author_check.md.
"""

import os
import re
import sys
import json
from collections import Counter
from datetime import date

import pandas as pd
from tabulate import tabulate

import openalex

IN_FIELD = {"Computer Science", "Engineering", "Mathematics", "Physics and Astronomy", "Decision Sciences"}


def norm(title) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def own_affiliations(author_id: str) -> tuple:
    works = openalex.get_all("works", {"filter": f"author.id:{author_id}", "select": "authorships"})
    inst, none = Counter(), 0
    for w in works:
        mine = [a for a in w.get("authorships") or [] if openalex.short_id((a.get("author") or {}).get("id")) == author_id]
        names = {i.get("display_name") for a in mine for i in a.get("institutions") or [] if i.get("display_name")}
        none += not names
        inst.update(names)
    top = "; ".join(f"{n} {100 * c / len(works):.0f}%" for n, c in inst.most_common(3))
    return len(works), top, 100 * none / len(works)


def outside_field_share(author_id: str) -> float:
    groups = openalex.get("works", {"filter": f"author.id:{author_id}", "group_by": "primary_topic.field.id"})["group_by"]
    total = sum(g["count"] for g in groups)
    return 100 * sum(g["count"] for g in groups if g["key_display_name"] not in IN_FIELD) / max(1, total)


def scholar_coverage(profile_csv: str, top20: list) -> str:
    keys = set(pd.read_csv(profile_csv)["title"].map(norm))
    found = sum(any(k.startswith(norm(t)) for k in keys) if len(norm(t)) >= 20 else norm(t) in keys for t, _ in top20)
    return f"{found}/20"


def run(record_json="data/cohort_openalex.json", scholar_json="data/scholar_reference.json",
        profile_dir="data/cohort_openalex", report="reports/cohort/author_check.md"):
    rec = json.load(open(record_json, encoding="utf-8"))
    scholar = json.load(open(scholar_json, encoding="utf-8")) if os.path.exists(scholar_json) else {}
    rows = []
    for group, cohort in (("survey", "B"), ("comparison", "A")):
        for key, v in rec[group].items():
            aid = v["openalex"]
            name = openalex.get(f"authors/{aid}", {"select": "display_name"})["display_name"]
            n, top, no_aff = own_affiliations(aid)
            cover = scholar_coverage(os.path.join(profile_dir, group, f"{key}.csv"), scholar[key]["top20"]) \
                if key in scholar else "n/a"
            rows.append([cohort, name, aid, n, top, f"{no_aff:.0f}%", f"{outside_field_share(aid):.1f}%", cover])
            print(rows[-1])
    table = tabulate(rows, headers=["Cohort", "Author", "OpenAlex ID", "Works", "Own affiliations (share of works)",
                                    "No affiliation", "Outside field", "Scholar top-20 found"], tablefmt="github")
    md = [f"# Cohort author records ({date.today()})", "",
          "One OpenAlex author record per person, used as is (no records merged). 'Outside field' = share of works "
          f"whose primary field is not one of {sorted(IN_FIELD)}.", "", table, ""]
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"✅ Report saved to '{report}'")


if __name__ == "__main__":
    run()
