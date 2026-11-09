"""Build a normalized source catalog and conservative BibTeX from checked cards.

No network, experiments, historical repositories, or citation metadata guessing.
Raw track cards remain available; normalization decisions are recorded.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def load_track(track):
    obj = json.loads((ROOT / f"track_{track}_sources.json").read_text(encoding="utf-8"))
    entries = obj.get("sources", []) if isinstance(obj, dict) else obj
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"missing records in track {track}")
    return entries


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {as_text(v)}" for k, v in value.items())
    return str(value)


def tex(value):
    # Titles/notes are ordinary text; URL fields are handled separately.
    return as_text(value).replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def bib_authors(value):
    names = re.split(r"\s*;\s*|\s+and\s+|,\s*", as_text(value))
    return " and ".join(tex(name.strip()) for name in names if name.strip())


def main():
    records, adjustments = [], []
    for track in "abcd":
        for raw in load_track(track):
            row = dict(raw)
            row["track"] = track.upper()
            row["kind"] = "software" if row["id"] == "A33" else "paper"
            row["bibliography_key"] = row["id"]
            row["review_depth"] = (
                "selected_full_text_sections" if row.get("evidence_level", "").startswith("full_text")
                else "abstract_or_limited_primary_text"
            )
            if row["id"] == "B25":
                row["evidence_level_original"] = row["evidence_level"]
                row["evidence_level"] = "primary_text_excerpts"
                row["review_depth"] = "abstract_or_limited_primary_text"
                adjustments.append("B25: indexed primary-PDF excerpts classified as limited access, not full-text access.")
            row["claim_summary"] = as_text(row.get("specific_claims"))
            row["selection_mechanism"] = as_text(row.get("context_mechanism", row.get("mechanism")))
            row["budget_scope"] = as_text(row.get("label_compute_budgets", row.get("label_compute_budget")))
            row["reviewer_inference"] = as_text(row.get("novelty_relevance", row.get("relevance_novelty", row.get("relevance_to_novelty"))))
            row["authorship_status"] = "anonymous_primary_manuscript" if row["id"] == "A28" else "primary_metadata_recorded"
            records.append(row)

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source identifiers")
    normalized_titles = [re.sub(r"\W+", "", r["title"]).lower() for r in records]
    duplicates = [key for key, count in Counter(normalized_titles).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate titles need manual reconciliation: {duplicates}")
    for row in records:
        assert row.get("title") and row.get("authors") and isinstance(row.get("year"), int), row["id"]
        assert row["year"] <= 2026, row["id"]
        assert urlparse(row["url"]).scheme in ("http", "https"), row["id"]
        assert "\ufffd" not in json.dumps(row, ensure_ascii=False), row["id"]

    counts = {
        "sources": len(records),
        "papers": sum(r["kind"] == "paper" for r in records),
        "software": sum(r["kind"] == "software" for r in records),
        "selected_full_text_sections": sum(r["review_depth"] == "selected_full_text_sections" for r in records),
        "limited_primary_text": sum(r["review_depth"] == "abstract_or_limited_primary_text" for r in records),
        "source_records_with_code_link": sum(bool(r.get("code_url")) for r in records),
        "tracks": dict(Counter(r["track"] for r in records)),
    }
    # Keep detailed paraphrases in their original track cards once. The master
    # registry points to them rather than duplicating abstracts/annotations.
    metadata_keys = ("id", "track", "kind", "bibliography_key", "title", "authors", "authorship_status", "year", "url", "venue_status", "venue_url", "version", "version_date", "version_date_verified", "preprint_title", "full_text_url", "additional_urls", "doi", "code_url", "code_status", "code_revision_verified", "code_license", "evidence_level", "evidence_level_original", "review_depth", "evidence_access_note", "evidence_sections")
    registry = []
    for row in records:
        item = {key: row[key] for key in metadata_keys if key in row}
        item["annotation_reference"] = {"file": f"track_{row['track'].lower()}_sources.json", "source_id": row["id"]}
        registry.append(item)
    master = {
        "title": "Context selection for foundation models in uplift",
        "cutoff_inclusive": "2026-11-08",
        "review_type": "Broad targeted narrative review with logged searches and citation snowballing; not PRISMA or an exhaustive citation-database census.",
        "depth_definition": "Full text means relevant sections inspected, not proof verification or experimental reproduction.",
        "missing_code_definition": "A null URL means author code availability was not verified here.",
        "counts": counts,
        "normalization_decisions": adjustments,
        "sources": registry,
    }
    (ROOT / "sources.json").write_text(json.dumps(master, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    catalog = [
        "# Каталог первичных источников", "",
        f"Дата отсечения — 8 ноября 2026 года. {counts['papers']} научных работ и {counts['software']} официальный программный ресурс. Препринт и финальная версия одной работы не считаются двумя независимыми результатами.", "",
        "Это навигационный каталог. sources.json содержит общий реестр и ссылки на подробные карточки в track_*_sources.json: конкретные утверждения, бюджет, ограничения и вывод обзора для каждого ID. В REVIEW_RU.md изложен сравнительный разбор. Метаданные здесь не подменяют проверку методов.", "",
        "Уровень «разделы полного текста» означает прочтение релевантных частей, а не независимый аудит доказательств. «Ограниченный первичный текст» включает abstract-only и явно отмеченные выдержки. Отсутствие подтверждённой ссылки на код не означает отсутствия кода.", "",
    ]
    names = {"A": "Контекст и архитектура табличных FM", "B": "Причинные FM и приобретение данных", "C": "Matching retrieval и оценивание данных", "D": "Uplift policy learning и оценка"}
    for track in "ABCD":
        catalog += [f"## {names[track]}", ""]
        for row in records:
            if row["track"] != track:
                continue
            depth = "разделы полного текста" if row["review_depth"] == "selected_full_text_sections" else "ограниченный первичный текст"
            catalog += [f"### {row['id']} {row['title']}", "", f"{row['authors']}. **{row['year']}**. {row.get('venue_status', '')}.", "", f"[Первичный источник]({row['url']}). Уровень: {depth}.", ""]
            version = row.get("version") or row.get("version_date") or row.get("version_date_verified")
            if version:
                catalog += [f"Версия и дата: {as_text(version)}.", ""]
            if row.get("preprint_title"):
                catalog += [f"Прежнее название: {row['preprint_title']}.", ""]
            if row.get("code_url"):
                catalog += [f"[Подтверждённая ссылка на код]({row['code_url']}). Код в рамках этого обзора не запускался.", ""]
            if row.get("evidence_access_note"):
                catalog += [f"Доступ: {row['evidence_access_note']}", ""]
            if row["id"] == "A28":
                catalog += ["Авторы и решение о публикации не установлены по доступному анонимному первичному PDF; сведения агрегаторов не подставлялись.", ""]
    (ROOT / "SOURCES.md").write_text("\n".join(catalog), encoding="utf-8")

    bibliography = ["% Verified primary-source metadata as of 2026-11-08.", "% Conservative @misc entries preserve venue/status in howpublished.", "% Abstract-only sources are not represented as full-text proof audits.", "% Keys are stable source-card IDs used throughout this review.", ""]
    for row in records:
        fields = {"title": "{" + tex(row["title"]) + "}", "year": str(row["year"]), "url": row["url"], "howpublished": tex(row.get("venue_status", ""))}
        if row["authorship_status"] != "anonymous_primary_manuscript":
            authors = bib_authors(row["authors"])
            fields["author"] = "{" + authors + "}" if row["kind"] == "software" else authors
        note = "Primary metadata checked 2026-11-08; " + row["review_depth"].replace("_", " ")
        if row["id"] == "A28":
            note += "; authors unverified in anonymous manuscript"
        if row["kind"] == "software":
            note += "; software resource, not a research paper"
        fields["note"] = tex(note)
        if row.get("doi"):
            fields["doi"] = row["doi"]
        arxiv = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})", row["url"])
        if arxiv:
            fields.update(eprint=arxiv.group(1), archivePrefix="arXiv")
        bibliography += [f"@misc{{{row['id']},"]
        bibliography += [f"  {key} = {{{value}}}," for key, value in fields.items() if value]
        bibliography += ["}", ""]
    (ROOT / "references.bib").write_text("\n".join(bibliography), encoding="utf-8")

    review = ROOT / "REVIEW_RU.md"
    content = review.read_text(encoding="utf-8")
    substitutions = {"SOURCE_COUNT": counts["sources"], "PAPER_COUNT": counts["papers"], "SOFTWARE_COUNT": counts["software"], "FULL_TEXT_COUNT": counts["selected_full_text_sections"], "LIMITED_COUNT": counts["limited_primary_text"]}
    for key, value in substitutions.items():
        content = content.replace("{{" + key + "}}", str(value))
    content, count_updates = re.subn(
        r"В итоговый реестр включены \*\*\d+ источников: \d+ научных работ и \d+ официальный software resource\*\*\. Уровни доступа и прочтения отмечены отдельно: для \*\*\d+\*\* изучены релевантные разделы полного текста, для \*\*\d+\*\* доступны",
        f"В итоговый реестр включены **{counts['sources']} источников: {counts['papers']} научных работ и {counts['software']} официальный software resource**. Уровни доступа и прочтения отмечены отдельно: для **{counts['selected_full_text_sections']}** изучены релевантные разделы полного текста, для **{counts['limited_primary_text']}** доступны",
        content,
    )
    if count_updates != 1:
        raise ValueError("review count paragraph missing or duplicated")
    if re.search(r"\{\{[A-Z_]+\}\}", content):
        raise ValueError("unfilled review placeholders")
    review.write_text(content, encoding="utf-8")

    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "cutoff": "2026-11-08", "counts": counts, "checks": ["unique IDs and normalized titles", "required primary metadata", "nonfuture publication years", "valid URL schemes", "no Unicode replacement character", "no unfilled review placeholders"], "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("track_*_sources.json"))}}
    (ROOT / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
