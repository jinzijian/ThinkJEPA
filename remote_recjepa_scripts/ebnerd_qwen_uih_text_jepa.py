#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

from ebnerd_pipeline import article_lookup, cosine, ranking_metrics, write_json


@dataclass
class Sample:
    sample_id: int
    impression_id: int
    user_id: int
    time: pd.Timestamp
    source_split: str
    split: str
    current_inview: list[int]
    current_clicked: list[int]
    past_shown: list[int]
    past_clicked: list[int]
    past_not_clicked: list[int]
    future_shown: list[int]
    future_clicked: list[int]
    future_not_clicked: list[int]
    context: dict
    z_current: np.ndarray
    z_current_set: np.ndarray
    z_future_target: np.ndarray
    z_prior: np.ndarray | None = None
    delta_z: np.ndarray | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean(value: Any, limit: int = 280) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\n", " ").strip()[:limit]


def unit_rows(x: np.ndarray) -> np.ndarray:
    return normalize(x.astype(np.float32)).astype(np.float32)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        out = pd.isna(value)
        if isinstance(out, (bool, np.bool_)):
            return bool(out)
    except (TypeError, ValueError):
        pass
    return False


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def clean_id(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def ids(values: Iterable[Any]) -> list[int]:
    return [x for x in (clean_id(v) for v in as_list(values)) if x is not None]


def load_cached(path: Path) -> tuple[list[Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


def limit_samples(samples: list[Sample], max_per_split: int) -> list[Sample]:
    if max_per_split <= 0:
        out = list(samples)
    else:
        counts: dict[str, int] = defaultdict(int)
        out = []
        for sample in samples:
            if counts[sample.split] >= max_per_split:
                continue
            counts[sample.split] += 1
            out.append(sample)
    for idx, sample in enumerate(out):
        sample.sample_id = idx
    return out


def article_text_from_row(row: pd.Series, include_body: bool = False) -> str:
    topics = ", ".join(clean(x, 40) for x in as_list(row.get("topics"))[:5])
    parts = [
        f"title: {clean(row.get('title'), 220)}",
        f"subtitle: {clean(row.get('subtitle'), 320)}",
        f"category: {clean(row.get('category_str'), 80)}",
        f"topics: {topics}",
        f"type: {clean(row.get('article_type'), 60)}",
        f"sentiment: {clean(row.get('sentiment_label'), 40)}",
    ]
    if include_body:
        parts.append(f"body: {clean(row.get('body'), 900)}")
    return " | ".join(p for p in parts if not p.endswith(": "))


def rich_article_lookup(articles: pd.DataFrame, include_body: bool = False) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for _, row in articles.iterrows():
        aid = int(row["article_id"])
        out[aid] = {
            "text": article_text_from_row(row, include_body=include_body),
            "title": clean(row.get("title"), 220) or f"article {aid}",
            "subtitle": clean(row.get("subtitle"), 320),
            "body": clean(row.get("body"), 900) if include_body else "",
            "category": clean(row.get("category_str"), 80),
            "topics": ", ".join(clean(x, 40) for x in as_list(row.get("topics"))[:5]),
            "type": clean(row.get("article_type"), 60),
            "sentiment": clean(row.get("sentiment_label"), 40),
            "sentiment_score": clean_float(row.get("sentiment_score"), 0.0),
            "premium": bool(row.get("premium")) if not is_missing(row.get("premium")) else False,
            "image_count": len(as_list(row.get("image_ids"))),
            "total_inviews": clean_float(row.get("total_inviews"), 0.0),
            "total_pageviews": clean_float(row.get("total_pageviews"), 0.0),
            "total_read_time": clean_float(row.get("total_read_time"), 0.0),
        }
    return out


def compact_article(
    aid: int,
    articles: dict[int, dict[str, Any]],
    *,
    include_extra: bool = False,
    body_chars: int = 0,
) -> str:
    art = articles.get(aid)
    if art is None:
        return f"article_id={aid}"
    bits = [f"id={aid}", f"title={art['title']}"]
    if art.get("subtitle"):
        bits.append(f"subtitle={art['subtitle']}")
    if art.get("category"):
        bits.append(f"category={art['category']}")
    if include_extra and art.get("topics"):
        bits.append(f"topics={art['topics']}")
    if art.get("type"):
        bits.append(f"type={art['type']}")
    if art.get("sentiment"):
        bits.append(f"sentiment={art['sentiment']}")
    if include_extra:
        bits.append(f"sentiment_score={art.get('sentiment_score', 0.0):.2f}")
        bits.append(f"premium={int(bool(art.get('premium')))}")
        bits.append(f"image_count={int(art.get('image_count', 0))}")
        if art.get("total_inviews", 0.0) > 0:
            bits.append(f"log_total_inviews={math.log1p(float(art['total_inviews'])):.2f}")
        if art.get("total_pageviews", 0.0) > 0:
            bits.append(f"log_total_pageviews={math.log1p(float(art['total_pageviews'])):.2f}")
        if art.get("total_read_time", 0.0) > 0:
            bits.append(f"log_total_read_time={math.log1p(float(art['total_read_time'])):.2f}")
    if body_chars > 0 and art.get("body"):
        bits.append(f"body={art['body'][:body_chars]}")
    return " ; ".join(bits)


def add_events(lines: list[str], prefix: str, ids: Iterable[int], articles: dict[int, dict[str, Any]], limit: int) -> None:
    seen = 0
    for aid in list(ids)[-limit:]:
        lines.append(f"{prefix}: {compact_article(int(aid), articles)}")
        seen += 1
    if seen == 0:
        lines.append(f"{prefix}: none")


def build_source_text(sample: Sample, articles: dict[int, dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "Task: encode user interaction history for recommendation reranking.",
        f"User: {sample.user_id}",
        f"Current impression time: {sample.time}",
        "Past UIH before current impression:",
    ]
    add_events(lines, "past_clicked", sample.past_clicked, articles, args.max_past_clicked_text)
    add_events(lines, "past_not_clicked_exposure", sample.past_not_clicked, articles, args.max_past_not_clicked_text)
    lines.append("Current candidate set, feedback hidden:")
    for rank, aid in enumerate(sample.current_inview[: args.max_current_text], start=1):
        lines.append(f"candidate_{rank}: {compact_article(int(aid), articles)}")
    return "\n".join(lines)


def build_target_text(sample: Sample, articles: dict[int, dict[str, Any]], args: argparse.Namespace) -> str:
    clicked = set(sample.current_clicked)
    not_clicked_current = [aid for aid in sample.current_inview if aid not in clicked]
    lines = [
        "Task: encode post-current user interaction outcome for recommendation reranking.",
        f"User: {sample.user_id}",
        f"Current impression time: {sample.time}",
        "Current impression response:",
    ]
    add_events(lines, "current_clicked", sample.current_clicked, articles, args.max_current_clicked_text)
    add_events(lines, "current_not_clicked_exposure", not_clicked_current, articles, args.max_current_not_clicked_text)
    lines.append("Future UIH after current impression:")
    add_events(lines, "future_clicked", sample.future_clicked, articles, args.max_future_clicked_text)
    add_events(lines, "future_not_clicked_exposure", sample.future_not_clicked, articles, args.max_future_not_clicked_text)
    return "\n".join(lines)


def load_behavior_rows(data_dir: Path) -> dict[int, list[Any]]:
    frames = []
    for split in ["train", "validation"]:
        path = data_dir / split / "behaviors.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).copy()
        frame["source_split"] = split
        frames.append(frame)
    if not frames:
        return {}
    behaviors = pd.concat(frames, ignore_index=True)
    behaviors["impression_time"] = pd.to_datetime(behaviors["impression_time"])
    by_user: dict[int, list[Any]] = {}
    for user_id, group in behaviors.sort_values(["user_id", "impression_time", "impression_id"]).groupby("user_id", sort=False):
        by_user[int(user_id)] = list(group.itertuples(index=False))
    return by_user


def load_history_events(data_dir: Path) -> dict[int, list[dict[str, Any]]]:
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for split in ["train", "validation"]:
        path = data_dir / split / "history.parquet"
        if not path.exists():
            continue
        history = pd.read_parquet(path)
        for row in history.itertuples(index=False):
            user_id = clean_id(getattr(row, "user_id"))
            if user_id is None:
                continue
            times = as_list(getattr(row, "impression_time_fixed"))
            article_ids = as_list(getattr(row, "article_id_fixed"))
            read_times = as_list(getattr(row, "read_time_fixed"))
            scrolls = as_list(getattr(row, "scroll_percentage_fixed"))
            for idx, aid in enumerate(article_ids):
                article_id = clean_id(aid)
                if article_id is None:
                    continue
                event_time = pd.to_datetime(times[idx]) if idx < len(times) else pd.NaT
                by_user[user_id].append(
                    {
                        "time": event_time,
                        "article_id": article_id,
                        "read_time": clean_float(read_times[idx]) if idx < len(read_times) else 0.0,
                        "scroll": clean_float(scrolls[idx]) if idx < len(scrolls) else 0.0,
                    }
                )
    for user_id in list(by_user):
        by_user[user_id] = sorted(by_user[user_id], key=lambda ev: (ev["time"], ev["article_id"]))
    return by_user


class FullSignalIndex:
    def __init__(self, data_dir: Path) -> None:
        self.by_user = load_behavior_rows(data_dir)
        self.row_index_by_user = {
            user_id: {int(getattr(row, "impression_id")): idx for idx, row in enumerate(rows)}
            for user_id, rows in self.by_user.items()
        }
        self.histories = load_history_events(data_dir)
        self.history_times = {
            user_id: [pd.Timestamp(ev["time"]).value if not pd.isna(ev["time"]) else -1 for ev in events]
            for user_id, events in self.histories.items()
        }

    def current(self, sample: Sample) -> tuple[list[Any], int | None, Any | None]:
        rows = self.by_user.get(sample.user_id, [])
        idx = self.row_index_by_user.get(sample.user_id, {}).get(sample.impression_id)
        row = rows[idx] if idx is not None and 0 <= idx < len(rows) else None
        return rows, idx, row

    def history_before(self, user_id: int, current_time: pd.Timestamp, limit: int) -> list[dict[str, Any]]:
        events = self.histories.get(user_id, [])
        if not events:
            return []
        times = self.history_times.get(user_id, [])
        pos = bisect_left(times, pd.Timestamp(current_time).value)
        return events[max(0, pos - limit) : pos]


def fmt_num(value: float, digits: int = 1) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def signed_gap_hours(event_time: Any, current_time: pd.Timestamp) -> float:
    if event_time is None or pd.isna(event_time):
        return 0.0
    return (pd.Timestamp(event_time) - pd.Timestamp(current_time)).total_seconds() / 3600.0


def behavior_summary(row: Any, prefix: str, current_time: pd.Timestamp | None = None) -> str:
    parts = [prefix]
    event_time = pd.Timestamp(getattr(row, "impression_time"))
    if current_time is not None:
        parts.append(f"dt_h={signed_gap_hours(event_time, current_time):+.1f}")
    parts.append(f"time={event_time.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"session={clean_id(getattr(row, 'session_id', None)) or 0}")
    parts.append(f"device={clean_id(getattr(row, 'device_type', None)) or 0}")
    parts.append(f"candidate_count={len(ids(getattr(row, 'article_ids_inview')))}")
    parts.append(f"clicked_count={len(ids(getattr(row, 'article_ids_clicked')))}")
    return " ; ".join(parts)


def outcome_bits(row: Any, *, use_next: bool = True) -> list[str]:
    bits = [
        f"read_s={fmt_num(clean_float(getattr(row, 'read_time', 0.0)))}",
        f"scroll_pct={fmt_num(clean_float(getattr(row, 'scroll_percentage', 0.0)))}",
    ]
    if use_next:
        bits.extend(
            [
                f"next_read_s={fmt_num(clean_float(getattr(row, 'next_read_time', 0.0)))}",
                f"next_scroll_pct={fmt_num(clean_float(getattr(row, 'next_scroll_percentage', 0.0)))}",
            ]
        )
    return bits


def article_event_line(
    prefix: str,
    aid: int,
    articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    *,
    row: Any | None = None,
    current_time: pd.Timestamp | None = None,
    event_time: Any | None = None,
    read_time: float | None = None,
    scroll: float | None = None,
    use_next: bool = False,
) -> str:
    bits = [prefix]
    if current_time is not None and event_time is not None:
        bits.append(f"dt_h={signed_gap_hours(event_time, current_time):+.1f}")
    if row is not None:
        bits.append(f"session={clean_id(getattr(row, 'session_id', None)) or 0}")
    if read_time is not None:
        bits.append(f"read_s={fmt_num(max(0.0, read_time))}")
    if scroll is not None:
        bits.append(f"scroll_pct={fmt_num(max(0.0, scroll))}")
    if row is not None and use_next:
        bits.append(f"next_read_s={fmt_num(clean_float(getattr(row, 'next_read_time', 0.0)))}")
        bits.append(f"next_scroll_pct={fmt_num(clean_float(getattr(row, 'next_scroll_percentage', 0.0)))}")
    bits.append(compact_article(aid, articles, include_extra=True, body_chars=args.compact_body_chars))
    return " ; ".join(bits)


def future_horizon_summaries(rows: list[Any], current_time: pd.Timestamp, horizons: Iterable[float]) -> list[str]:
    lines = []
    for horizon in horizons:
        end = current_time + pd.Timedelta(hours=float(horizon))
        kept = [row for row in rows if pd.Timestamp(getattr(row, "impression_time")) <= end]
        shown = sum(len(ids(getattr(row, "article_ids_inview"))) for row in kept)
        clicked = sum(len(ids(getattr(row, "article_ids_clicked"))) for row in kept)
        total_read = sum(clean_float(getattr(row, "read_time", 0.0)) for row in kept)
        next_read = sum(clean_float(getattr(row, "next_read_time", 0.0)) for row in kept)
        scrolls = [clean_float(getattr(row, "scroll_percentage", 0.0)) for row in kept if clean_float(getattr(row, "scroll_percentage", -1.0), -1.0) >= 0]
        avg_scroll = sum(scrolls) / max(1, len(scrolls))
        lines.append(
            "future_summary_{}h: impressions={} ; shown={} ; clicked={} ; read_s={} ; next_read_s={} ; avg_scroll_pct={}".format(
                fmt_num(float(horizon), 0),
                len(kept),
                shown,
                clicked,
                fmt_num(total_read),
                fmt_num(next_read),
                fmt_num(avg_scroll),
            )
        )
    return lines


def build_full_signal_source_text(
    sample: Sample,
    articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    full_index: FullSignalIndex,
) -> str:
    rows, idx, current_row = full_index.current(sample)
    current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
    context = sample.context or {}
    lines = [
        "Task: encode full-signal source UIH for recommendation reranking.",
        f"User: {sample.user_id}",
        f"Current impression time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Current request context, feedback hidden: "
        + " ; ".join(
            [
                f"hour={current_time.hour}",
                f"weekday={current_time.dayofweek}",
                f"device={context.get('device_type', 0)}",
                f"is_sso={context.get('is_sso_user', 0)}",
                f"is_subscriber={context.get('is_subscriber', 0)}",
                f"age={context.get('age', -1.0)}",
                f"candidate_count={len(sample.current_inview)}",
            ]
        ),
        "Current candidate set, feedback hidden:",
    ]
    for aid in sample.current_inview[: args.max_current_text]:
        lines.append(article_event_line("current_candidate", int(aid), articles, args, row=current_row, current_time=current_time, event_time=current_time))

    lines.append("Past clicked reading history with recency/read/scroll:")
    history_events = full_index.history_before(sample.user_id, current_time, args.full_signal_max_history_events)
    if history_events:
        for ev in history_events:
            lines.append(
                article_event_line(
                    "history_clicked",
                    int(ev["article_id"]),
                    articles,
                    args,
                    current_time=current_time,
                    event_time=ev["time"],
                    read_time=float(ev.get("read_time", 0.0)),
                    scroll=float(ev.get("scroll", 0.0)),
                )
            )
    else:
        lines.append("history_clicked: none")

    lines.append("Past exposure impressions with clicked/not-clicked feedback:")
    if idx is None:
        lines.append("past_impression: missing_current_row")
    else:
        past_rows = rows[max(0, idx - args.full_signal_max_past_impressions) : idx]
        if not past_rows:
            lines.append("past_impression: none")
        for past in past_rows:
            lines.append(behavior_summary(past, "past_impression", current_time))
            clicked = set(ids(getattr(past, "article_ids_clicked")))
            for aid in ids(getattr(past, "article_ids_inview"))[: args.full_signal_max_events_per_impression]:
                if aid in clicked:
                    lines.append(
                        article_event_line(
                            "past_clicked",
                            aid,
                            articles,
                            args,
                            row=past,
                            current_time=current_time,
                            event_time=getattr(past, "impression_time"),
                            read_time=clean_float(getattr(past, "read_time", 0.0)),
                            scroll=clean_float(getattr(past, "scroll_percentage", 0.0)),
                            use_next=True,
                        )
                    )
                else:
                    lines.append(
                        article_event_line(
                            "past_not_clicked_exposure",
                            aid,
                            articles,
                            args,
                            row=past,
                            current_time=current_time,
                            event_time=getattr(past, "impression_time"),
                        )
                    )
    return "\n".join(lines)


def build_full_signal_target_text(
    sample: Sample,
    articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    full_index: FullSignalIndex,
) -> str:
    rows, idx, current_row = full_index.current(sample)
    current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
    lines = [
        "Task: encode full-signal post-current UIH target for recommendation reranking.",
        f"User: {sample.user_id}",
        f"Current impression time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Current impression response:",
    ]
    current_clicked = set(sample.current_clicked if current_row is None else ids(getattr(current_row, "article_ids_clicked")))
    current_inview = sample.current_inview if current_row is None else ids(getattr(current_row, "article_ids_inview"))
    for aid in current_inview[: args.max_current_text]:
        if aid in current_clicked:
            lines.append(
                article_event_line(
                    "current_clicked",
                    int(aid),
                    articles,
                    args,
                    row=current_row,
                    current_time=current_time,
                    event_time=current_time,
                    read_time=clean_float(getattr(current_row, "read_time", 0.0)) if current_row is not None else None,
                    scroll=clean_float(getattr(current_row, "scroll_percentage", 0.0)) if current_row is not None else None,
                    use_next=current_row is not None,
                )
            )
        else:
            lines.append(article_event_line("current_not_clicked_exposure", int(aid), articles, args, row=current_row, current_time=current_time, event_time=current_time))

    lines.append("Future multi-horizon UIH after current impression:")
    future_rows: list[Any] = []
    if idx is not None:
        horizon_end = current_time + pd.Timedelta(hours=args.full_signal_horizon_hours)
        for fut in rows[idx + 1 :]:
            fut_time = pd.Timestamp(getattr(fut, "impression_time"))
            if fut_time > horizon_end:
                break
            future_rows.append(fut)
            if len(future_rows) >= args.full_signal_max_future_impressions:
                break
    lines.extend(future_horizon_summaries(future_rows, current_time, [1.0, 6.0, args.full_signal_horizon_hours]))
    if not future_rows:
        lines.append("future_impression: none")
    for fut in future_rows:
        lines.append(behavior_summary(fut, "future_impression", current_time))
        clicked = set(ids(getattr(fut, "article_ids_clicked")))
        for aid in ids(getattr(fut, "article_ids_inview"))[: args.full_signal_max_events_per_impression]:
            if aid in clicked:
                lines.append(
                    article_event_line(
                        "future_clicked",
                        aid,
                        articles,
                        args,
                        row=fut,
                        current_time=current_time,
                        event_time=getattr(fut, "impression_time"),
                        read_time=clean_float(getattr(fut, "read_time", 0.0)),
                        scroll=clean_float(getattr(fut, "scroll_percentage", 0.0)),
                        use_next=True,
                    )
                )
            else:
                lines.append(
                    article_event_line(
                        "future_not_clicked_exposure",
                        aid,
                        articles,
                        args,
                        row=fut,
                        current_time=current_time,
                        event_time=getattr(fut, "impression_time"),
                    )
                )
    return "\n".join(lines)


def encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
    desc: str,
    pool: dict[str, Any] | None = None,
) -> np.ndarray:
    print(json.dumps({"encoding": desc, "num_texts": len(texts), "batch_size": batch_size}), flush=True)
    if pool is not None:
        return model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        ).astype(np.float32)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings, convert_to_numpy=True, show_progress_bar=True).astype(
        np.float32
    )


def transform_in_chunks(raw: np.ndarray, svd: TruncatedSVD, chunk_size: int) -> np.ndarray:
    outs = []
    for start in range(0, raw.shape[0], chunk_size):
        outs.append(svd.transform(raw[start : start + chunk_size]).astype(np.float32))
    return unit_rows(np.vstack(outs))


def build_or_load_qwen_cache(
    samples: list[Sample],
    articles_df: pd.DataFrame,
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    full_index: FullSignalIndex | None = None,
) -> tuple[dict[int, int], np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, str]]:
    cache_path = Path(args.qwen_uih_cache)
    text_example_path = cache_path.with_suffix(".examples.json")
    if cache_path.exists() and not args.rebuild_qwen_cache:
        packed = np.load(cache_path)
        meta = json.loads(cache_path.with_suffix(".json").read_text())
        article_ids = packed["article_ids"].astype(np.int64)
        article_emb = packed["article_emb"].astype(np.float32)
        source_emb = packed["source_emb"].astype(np.float32)
        target_emb = packed["target_emb"].astype(np.float32)
        examples = json.loads(text_example_path.read_text()) if text_example_path.exists() else {}
        return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, article_emb, source_emb, target_emb, meta, examples

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.qwen_model, device=args.qwen_device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    pool = None
    if args.qwen_multi_process_devices:
        devices = [device.strip() for device in args.qwen_multi_process_devices.split(",") if device.strip()]
        print(json.dumps({"qwen_multi_process_devices": devices}), flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)

    try:
        articles_df = articles_df.drop_duplicates("article_id").reset_index(drop=True)
        article_ids = articles_df["article_id"].astype("int64").to_numpy()
        article_texts = [article_text_from_row(row, include_body=args.include_body) for _, row in articles_df.iterrows()]
        article_raw = encode_texts(model, article_texts, args.qwen_batch_size, True, "articles", pool=pool)
        reduced_dim = min(args.reduced_dim, article_raw.shape[1], max(2, article_raw.shape[0] - 1))
        svd = TruncatedSVD(n_components=reduced_dim, random_state=args.seed)
        article_emb = unit_rows(svd.fit_transform(article_raw).astype(np.float32))

        def source_text(sample: Sample) -> str:
            if args.full_signal_uih:
                assert full_index is not None
                return build_full_signal_source_text(sample, rich_articles, args, full_index)
            return build_source_text(sample, rich_articles, args)

        def target_text(sample: Sample) -> str:
            if args.full_signal_uih:
                assert full_index is not None
                return build_full_signal_target_text(sample, rich_articles, args, full_index)
            return build_target_text(sample, rich_articles, args)

        source_parts = []
        target_parts = []
        examples = {
            "source_text": source_text(samples[0]) if samples else "",
            "target_text": target_text(samples[0]) if samples else "",
        }
        for start in range(0, len(samples), args.encode_text_chunk_size):
            end = min(len(samples), start + args.encode_text_chunk_size)
            chunk = samples[start:end]
            source_raw = encode_texts(
                model,
                [source_text(sample) for sample in chunk],
                args.qwen_batch_size,
                True,
                f"source_uih_{start}_{end}",
                pool=pool,
            )
            target_raw = encode_texts(
                model,
                [target_text(sample) for sample in chunk],
                args.qwen_batch_size,
                True,
                f"target_uih_{start}_{end}",
                pool=pool,
            )
            source_parts.append(transform_in_chunks(source_raw, svd, args.svd_transform_chunk_size))
            target_parts.append(transform_in_chunks(target_raw, svd, args.svd_transform_chunk_size))
        source_emb = np.vstack(source_parts).astype(np.float32)
        target_emb = np.vstack(target_parts).astype(np.float32)
    finally:
        if pool is not None:
            SentenceTransformer.stop_multi_process_pool(pool)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        article_ids=article_ids,
        article_emb=article_emb.astype(np.float16 if args.save_float16 else np.float32),
        source_emb=source_emb.astype(np.float16 if args.save_float16 else np.float32),
        target_emb=target_emb.astype(np.float16 if args.save_float16 else np.float32),
    )
    meta = {
        "model": args.qwen_model,
        "num_articles": int(len(article_ids)),
        "num_samples": int(len(samples)),
        "raw_dim": int(article_raw.shape[1]),
        "reduced_dim": int(article_emb.shape[1]),
        "svd_explained_variance_on_articles": float(svd.explained_variance_ratio_.sum()),
        "max_seq_length": int(args.max_seq_length),
        "include_body": bool(args.include_body),
        "compact_body_chars": int(args.compact_body_chars),
        "full_signal_uih": bool(args.full_signal_uih),
        "source_text_rule": "full signal past UIH plus current candidate set, no current feedback"
        if args.full_signal_uih
        else "past UIH plus current candidate set, no current feedback",
        "target_text_rule": "current response with read/scroll plus multi-horizon future UIH"
        if args.full_signal_uih
        else "current response plus future UIH",
        "full_signal": {
            "max_history_events": int(args.full_signal_max_history_events),
            "max_past_impressions": int(args.full_signal_max_past_impressions),
            "max_future_impressions": int(args.full_signal_max_future_impressions),
            "horizon_hours": float(args.full_signal_horizon_hours),
            "max_events_per_impression": int(args.full_signal_max_events_per_impression),
        }
        if args.full_signal_uih
        else None,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    text_example_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, article_emb, source_emb, target_emb, meta, examples


def article_dense_meta(articles_by_id: dict[int, dict], aid: int, sample: Sample, n: int) -> list[float]:
    art = articles_by_id.get(aid, {})
    published = art.get("published_time", pd.NaT)
    age_hours = 0.0
    if not pd.isna(published):
        age_hours = max(0.0, (sample.time - published).total_seconds() / 3600.0)
    hour = sample.time.hour + sample.time.minute / 60.0
    dow = sample.time.dayofweek
    past_shown = Counter(sample.past_shown)
    past_clicked = Counter(sample.past_clicked)
    return [
        math.log1p(n),
        math.log1p(len(sample.past_shown)),
        math.log1p(len(sample.past_clicked)),
        math.log1p(len(sample.past_not_clicked)),
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0),
        math.cos(2 * math.pi * dow / 7.0),
        float(sample.context.get("is_sso_user", 0)),
        float(sample.context.get("is_subscriber", 0)),
        float(sample.context.get("age", -1.0)) / 100.0,
        float(art.get("premium", 0)),
        math.log1p(age_hours),
        math.log1p(float(art.get("total_inviews", 0.0))),
        math.log1p(float(art.get("total_pageviews", 0.0))),
        math.log1p(float(art.get("total_read_time", 0.0))),
        float(art.get("sentiment_score", 0.0)),
        float(past_shown[aid] > 0),
        float(past_clicked[aid] > 0),
        math.log1p(past_shown[aid]),
        math.log1p(past_clicked[aid]),
    ]


class ImpressionDataset(Dataset):
    def __init__(self, samples: list[Sample], split: str):
        self.samples = [s for s in samples if s.split == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class SetBatchBuilder:
    scalar_dim = 25

    def __init__(
        self,
        article_to_idx: dict[int, int],
        item_emb: np.ndarray,
        source_emb: np.ndarray,
        post_emb: np.ndarray,
        articles_by_id: dict[int, dict],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        use_source_feature: bool = True,
        use_post_feature: bool = True,
        use_delta_feature: bool = True,
    ) -> None:
        self.article_to_idx = article_to_idx
        self.item_emb = item_emb
        self.source_emb = source_emb
        self.post_emb = post_emb
        self.articles_by_id = articles_by_id
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.use_source_feature = use_source_feature
        self.use_post_feature = use_post_feature
        self.use_delta_feature = use_delta_feature

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        bsz = len(samples)
        max_c = max(len(s.current_inview) for s in samples)
        d = self.item_emb.shape[1]
        item_emb = np.zeros((bsz, max_c, d), dtype=np.float32)
        scalar = np.zeros((bsz, max_c, self.scalar_dim), dtype=np.float32)
        labels = np.zeros((bsz, max_c), dtype=np.float32)
        mask = np.zeros((bsz, max_c), dtype=bool)
        item_idx = np.zeros((bsz, max_c), dtype=np.int64)
        user_idx = np.zeros((bsz,), dtype=np.int64)
        source = np.zeros((bsz, d), dtype=np.float32)
        candidate_set = np.zeros((bsz, d), dtype=np.float32)
        post = np.zeros((bsz, d), dtype=np.float32)
        groups = np.zeros((bsz, max_c), dtype=np.int64)
        sample_ids = np.zeros((bsz,), dtype=np.int64)
        for bi, sample in enumerate(samples):
            z_source = self.source_emb[sample.sample_id].astype(np.float32)
            z_post = self.post_emb[sample.sample_id].astype(np.float32)
            source[bi] = z_source
            post[bi] = z_post
            sample_ids[bi] = sample.sample_id
            user_idx[bi] = self.user_to_idx.get(sample.user_id, 0)
            clicked = set(sample.current_clicked)
            n = len(sample.current_inview)
            current_items = []
            for ci, aid in enumerate(sample.current_inview):
                idx = self.article_to_idx.get(aid)
                item = self.item_emb[idx].astype(np.float32) if idx is not None else np.zeros(d, dtype=np.float32)
                current_items.append(item)
                item_emb[bi, ci] = item
                z_delta = z_post - z_source
                source_cos = cosine(item, z_source) if self.use_source_feature else 0.0
                post_cos = cosine(item, z_post) if self.use_post_feature else 0.0
                delta_cos = cosine(item, z_delta) if self.use_delta_feature else 0.0
                scalar[bi, ci] = np.array(
                    [
                        source_cos,
                        post_cos,
                        delta_cos,
                        float(ci) / max(1.0, float(n - 1)),
                    ]
                    + article_dense_meta(self.articles_by_id, aid, sample, n),
                    dtype=np.float32,
                )
                labels[bi, ci] = float(aid in clicked)
                mask[bi, ci] = True
                item_idx[bi, ci] = self.item_to_idx.get(aid, 0)
                groups[bi, ci] = sample.impression_id
            if current_items:
                candidate_set[bi] = np.mean(current_items, axis=0).astype(np.float32)
        return {
            "item_emb": torch.from_numpy(item_emb),
            "scalar": torch.from_numpy(scalar),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "item_idx": torch.from_numpy(item_idx),
            "user_idx": torch.from_numpy(user_idx),
            "source": torch.from_numpy(source),
            "candidate_set": torch.from_numpy(candidate_set),
            "post": torch.from_numpy(post),
            "group_ids": groups,
            "sample_ids": sample_ids,
        }


class SetTransformerRanker(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_source_token: bool,
        use_post_token: bool,
        use_post_dot: bool,
        init_post_dot_scale: float,
    ) -> None:
        super().__init__()
        self.use_source_token = use_source_token
        self.use_post_token = use_post_token
        self.use_post_dot = use_post_dot
        self.post_dot_scale = nn.Parameter(torch.tensor(float(init_post_dot_scale), dtype=torch.float32))
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(5, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.scalar_head = nn.Sequential(nn.LayerNorm(scalar_dim), nn.Linear(scalar_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        tokens = [user_tok.unsqueeze(1)]
        if self.use_source_token:
            source_tok = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
            tokens.append(source_tok.unsqueeze(1))
        tokens.append(set_tok.unsqueeze(1))
        if self.use_post_token:
            tokens.append(self.vec_proj(batch["post"]).unsqueeze(1) + self.type_emb.weight[3])
        item_tok = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.item_id_emb(batch["item_idx"]) + self.type_emb.weight[4]
        tokens.append(item_tok)
        x = torch.cat(tokens, dim=1)
        ctx_len = 2 + int(self.use_source_token) + int(self.use_post_token)
        ctx_mask = torch.zeros((x.shape[0], ctx_len), dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([ctx_mask, ~batch["mask"]], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        item_out = out[:, ctx_len:]
        score = self.head(item_out).squeeze(-1) + self.scalar_head(batch["scalar"]).squeeze(-1)
        if self.use_post_dot:
            score = score + self.post_dot_scale * batch["scalar"][..., 1]
        return score


class QwenTextJepaPredictor(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_candidate_mix: bool,
    ) -> None:
        super().__init__()
        self.use_candidate_mix = use_candidate_mix
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(4, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.mix_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.mix_proj = nn.Linear(emb_dim, d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source_tok = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        item_tok = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.item_id_emb(batch["item_idx"]) + self.type_emb.weight[2]
        pred_tok = self.mask_token.repeat(item_tok.shape[0], 1, 1) + self.type_emb.weight[3]
        x = torch.cat([user_tok.unsqueeze(1), source_tok.unsqueeze(1), item_tok, pred_tok], dim=1)
        ctx_mask = torch.zeros((x.shape[0], 3), dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([ctx_mask[:, :2], ~batch["mask"], ctx_mask[:, :1]], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        pred = out[:, -1]
        if self.use_candidate_mix:
            item_out = out[:, 2:-1]
            mix_logits = self.mix_head(item_out).squeeze(-1).masked_fill(~batch["mask"], -1e4)
            mix_weight = F.softmax(mix_logits, dim=1)
            item_mix = torch.einsum("bc,bcd->bd", mix_weight, batch["item_emb"])
            pred = pred + self.mix_proj(item_mix)
        return self.out(pred)


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e4)
    pos = labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    target = labels / pos
    return (-(target * F.log_softmax(scores, dim=1)).sum(dim=1)).mean()


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_regularizer(z: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = F.normalize(z, dim=-1)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum() / z.shape[1]
    return var_loss, cov_loss


def candidate_distribution_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    item = F.normalize(item_emb, dim=-1)
    pred_u = F.normalize(pred, dim=-1)
    target_u = F.normalize(target, dim=-1)
    teacher = torch.einsum("bcd,bd->bc", item, target_u) / temperature
    student = torch.einsum("bcd,bd->bc", item, pred_u) / temperature
    teacher = teacher.masked_fill(~mask, -1e4)
    student = student.masked_fill(~mask, -1e4)
    teacher_prob = F.softmax(teacher, dim=1).detach()
    distill = -(teacher_prob * F.log_softmax(student, dim=1)).sum(dim=1).mean()
    label_rank = listwise_loss(student, labels, mask)
    return distill, label_rank


def inbatch_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def jepa_loss(pred: torch.Tensor, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["post"]
    pred_u = F.normalize(pred, dim=-1)
    target_u = F.normalize(target, dim=-1)
    latent = F.mse_loss(pred_u, target_u) + (1.0 - (pred_u * target_u).sum(dim=-1).mean())
    distill, label_rank = candidate_distribution_loss(pred, target, batch["item_emb"], batch["labels"], batch["mask"], args.candidate_temperature)
    nce = inbatch_nce(pred, target.detach(), args.inbatch_temperature)
    var_loss, cov_loss = vicreg_regularizer(pred, args.variance_gamma)
    total = (
        args.latent_weight * latent
        + args.candidate_kl_weight * distill
        + args.candidate_label_weight * label_rank
        + args.inbatch_nce_weight * nce
        + args.variance_weight * var_loss
        + args.covariance_weight * cov_loss
    )
    parts = {
        "latent": float(latent.detach().cpu()),
        "candidate_kl": float(distill.detach().cpu()),
        "candidate_label": float(label_rank.detach().cpu()),
        "inbatch_nce": float(nce.detach().cpu()),
        "variance": float(var_loss.detach().cpu()),
        "covariance": float(cov_loss.detach().cpu()),
    }
    return total, parts


@torch.no_grad()
def predict_ranker(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, scores, groups = [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(moved).detach().cpu().numpy()
        lab = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        scores.append(out[mask])
        groups.append(group[mask])
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(groups)


@torch.no_grad()
def predict_latents(model: nn.Module, loader: DataLoader, device: torch.device, n_samples: int, emb_dim: int) -> np.ndarray:
    model.eval()
    pred = np.zeros((n_samples, emb_dim), dtype=np.float32)
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = F.normalize(model(moved), dim=-1).detach().cpu().numpy().astype(np.float32)
        pred[batch["sample_ids"]] = out
    return pred


def candidate_dot_metrics(samples: list[Sample], split: str, article_to_idx: dict[int, int], item_emb: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    labels, scores, groups = [], [], []
    for sample in samples:
        if sample.split != split:
            continue
        z = vectors[sample.sample_id]
        clicked = set(sample.current_clicked)
        for aid in sample.current_inview:
            idx = article_to_idx.get(aid)
            score = cosine(item_emb[idx], z) if idx is not None else 0.0
            labels.append(int(aid in clicked))
            scores.append(score)
            groups.append(sample.impression_id)
    return ranking_metrics(np.asarray(labels), np.asarray(scores), np.asarray(groups))


def cosine_summary(name: str, pred: np.ndarray, target: np.ndarray, idxs: list[int]) -> dict[str, Any]:
    if not idxs:
        return {"name": name, "num_samples": 0}
    p = pred[idxs]
    t = target[idxs]
    pairwise_n = min(len(idxs), 5000)
    sub = p[:pairwise_n]
    pair = sub @ sub.T
    tri = pair[np.triu_indices(pairwise_n, k=1)] if pairwise_n > 1 else np.array([0.0])
    return {
        "name": name,
        "num_samples": int(len(idxs)),
        "cosine_to_target_text_uih": float(np.mean([cosine(a, b) for a, b in zip(p, t)])),
        "mse_to_target_text_uih": float(np.mean((p - t) ** 2)),
        "pred_mean_pairwise_cosine": float(np.mean(tri)),
    }


def make_loaders(
    samples: list[Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    post_emb: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
    splits: list[str],
    use_source_feature: bool = True,
    use_post_feature: bool = True,
    use_delta_feature: bool = True,
) -> dict[str, DataLoader]:
    builder = SetBatchBuilder(
        article_to_idx,
        item_emb,
        source_emb,
        post_emb,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        use_source_feature=use_source_feature,
        use_post_feature=use_post_feature,
        use_delta_feature=use_delta_feature,
    )
    return {
        split: DataLoader(
            ImpressionDataset(samples, split),
            batch_size=args.batch_size,
            shuffle=split in {"ranker_train", "jepa_train"},
            num_workers=args.num_workers,
            pin_memory=torch.device(args.device).type == "cuda",
            collate_fn=builder,
        )
        for split in splits
    }


def train_ranker(
    name: str,
    samples: list[Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    post_emb: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    use_source_token: bool,
    use_post_token: bool,
    use_source_feature: bool,
    use_post_feature: bool,
    use_delta_feature: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    loaders = make_loaders(
        samples,
        article_to_idx,
        item_emb,
        source_emb,
        post_emb,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        args,
        ["ranker_train", "ranker_val", "ranker_test"],
        use_source_feature=use_source_feature,
        use_post_feature=use_post_feature,
        use_delta_feature=use_delta_feature,
    )
    model = SetTransformerRanker(
        emb_dim=item_emb.shape[1],
        scalar_dim=SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_source_token=use_source_token,
        use_post_token=use_post_token,
        use_post_dot=use_post_feature,
        init_post_dot_scale=args.init_post_dot_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y0, s0, g0 = predict_ranker(model, loaders["ranker_val"], device)
    initial = ranking_metrics(y0, s0, g0)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metric = initial["ndcg@10"]
    history = [{"epoch": 0, "loss": None, "ranker_val": initial}]
    print(json.dumps({name: {"epoch": 0, "val_mrr": initial["mrr"], "val_ndcg10": initial["ndcg@10"]}}, ensure_ascii=False), flush=True)
    patience = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            score = model(moved)
            loss = listwise_loss(score, moved["labels"], moved["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y, s, g = predict_ranker(model, loaders["ranker_val"], device)
        val = ranking_metrics(y, s, g)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "ranker_val": val}
        history.append(row)
        print(json.dumps({name: {"epoch": epoch, "loss": row["loss"], "val_mrr": val["mrr"], "val_ndcg10": val["ndcg@10"]}}, ensure_ascii=False), flush=True)
        if val["ndcg@10"] > best_metric:
            best_metric = val["ndcg@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    out = {
        "history": history,
        "use_source_token": use_source_token,
        "use_post_token": use_post_token,
        "use_source_feature": use_source_feature,
        "use_post_feature": use_post_feature,
        "use_delta_feature": use_delta_feature,
    }
    for split, loader in loaders.items():
        y, s, g = predict_ranker(model, loader, device)
        out[split] = ranking_metrics(y, s, g)
    return out


def train_jepa(
    samples: list[Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    loaders = make_loaders(samples, article_to_idx, item_emb, source_emb, target_emb, articles_by_id, user_to_idx, item_to_idx, args, ["jepa_train", "jepa_val", "ranker_train", "ranker_val", "ranker_test"])
    model = QwenTextJepaPredictor(
        emb_dim=item_emb.shape[1],
        scalar_dim=SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.jepa_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_candidate_mix=args.jepa_candidate_mix,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.jepa_lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    history = []
    patience = args.patience
    for epoch in range(1, args.jepa_epochs + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for batch in loaders["jepa_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred = model(moved)
            loss, loss_parts = jepa_loss(pred, moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                parts[key].append(value)
        val_pred = predict_latents(model, loaders["jepa_val"], device, len(samples), item_emb.shape[1])
        val_idxs = [s.sample_id for s in samples if s.split == "jepa_val"]
        val_cos = cosine_summary("jepa_val", val_pred, target_emb, val_idxs)
        val_dot = candidate_dot_metrics(samples, "jepa_val", article_to_idx, item_emb, val_pred)
        metric = val_dot["ndcg@10"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_parts": {k: float(np.mean(v)) for k, v in parts.items()},
            "jepa_val": val_cos,
            "jepa_val_candidate_dot": val_dot,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "qwen_text_jepa": {
                        "epoch": epoch,
                        "loss": row["loss"],
                        "val_cos": val_cos["cosine_to_target_text_uih"],
                        "val_pairwise": val_cos["pred_mean_pairwise_cosine"],
                        "val_dot_mrr": val_dot["mrr"],
                        "val_dot_ndcg10": val_dot["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_all = np.zeros_like(target_emb, dtype=np.float32)
    for split, loader in loaders.items():
        split_pred = predict_latents(model, loader, device, len(samples), item_emb.shape[1])
        idxs = [s.sample_id for s in samples if s.split == split]
        pred_all[idxs] = split_pred[idxs]
    split_eval = {}
    for split in ["jepa_val", "ranker_train", "ranker_val", "ranker_test"]:
        idxs = [s.sample_id for s in samples if s.split == split]
        split_eval[split] = cosine_summary(split, pred_all, target_emb, idxs)
        split_eval[split]["candidate_dot"] = candidate_dot_metrics(samples, split, article_to_idx, item_emb, pred_all)
    return pred_all, {"history": history, "split_eval": split_eval}


def summarize_baseline(path: str) -> dict[str, Any]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if "models" not in raw:
        return raw.get("baseline_test", {})
    return {name: value["ranker_test"] for name, value in raw["models"].items() if isinstance(value, dict) and "ranker_test" in value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Qwen full-UIH text encoder + JEPA set reranker for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qwen-uih-cache", required=True)
    parser.add_argument("--qwen-model", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-batch-size", type=int, default=24)
    parser.add_argument("--qwen-multi-process-devices", default="")
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--reduced-dim", type=int, default=256)
    parser.add_argument("--encode-text-chunk-size", type=int, default=4096)
    parser.add_argument("--svd-transform-chunk-size", type=int, default=4096)
    parser.add_argument("--include-body", action="store_true")
    parser.add_argument("--compact-body-chars", type=int, default=0)
    parser.add_argument("--full-signal-uih", action="store_true")
    parser.add_argument("--full-signal-max-history-events", type=int, default=40)
    parser.add_argument("--full-signal-max-past-impressions", type=int, default=8)
    parser.add_argument("--full-signal-max-future-impressions", type=int, default=8)
    parser.add_argument("--full-signal-horizon-hours", type=float, default=24.0)
    parser.add_argument("--full-signal-max-events-per-impression", type=int, default=20)
    parser.add_argument("--save-float16", action="store_true")
    parser.add_argument("--rebuild-qwen-cache", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--max-past-clicked-text", type=int, default=30)
    parser.add_argument("--max-past-not-clicked-text", type=int, default=30)
    parser.add_argument("--max-current-text", type=int, default=40)
    parser.add_argument("--max-current-clicked-text", type=int, default=5)
    parser.add_argument("--max-current-not-clicked-text", type=int, default=40)
    parser.add_argument("--max-future-clicked-text", type=int, default=30)
    parser.add_argument("--max-future-not-clicked-text", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--jepa-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--jepa-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--jepa-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--init-post-dot-scale", type=float, default=5.0)
    parser.add_argument("--candidate-temperature", type=float, default=0.07)
    parser.add_argument("--inbatch-temperature", type=float, default=0.07)
    parser.add_argument("--variance-gamma", type=float, default=0.05)
    parser.add_argument("--latent-weight", type=float, default=0.5)
    parser.add_argument("--candidate-kl-weight", type=float, default=1.0)
    parser.add_argument("--candidate-label-weight", type=float, default=1.0)
    parser.add_argument("--inbatch-nce-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.5)
    parser.add_argument("--covariance-weight", type=float, default=0.02)
    parser.add_argument("--jepa-candidate-mix", action="store_true")
    parser.add_argument("--ranker-ablation-suite", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-predicted", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, prepare = load_cached(Path(args.samples_pkl))
    samples = limit_samples(samples, args.max_samples_per_split)
    data_dir = Path(args.data_dir)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    rich_articles = rich_article_lookup(articles_df, include_body=args.include_body)
    articles_by_id = article_lookup(articles_df)
    full_index = FullSignalIndex(data_dir) if args.full_signal_uih else None
    article_to_idx, item_emb, source_emb, target_emb, qwen_meta, text_examples = build_or_load_qwen_cache(
        samples, articles_df, rich_articles, args, full_index
    )
    users = sorted({s.user_id for s in samples})
    items = sorted({aid for s in samples for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}
    zeros = np.zeros_like(target_emb, dtype=np.float32)

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_uih": qwen_meta,
        "setting": {
            "task": "one impression is one listwise set-to-rank sample",
            "source_text": qwen_meta.get("source_text_rule", "past UIH plus current candidate set, current feedback hidden"),
            "target_text": qwen_meta.get("target_text_rule", "current response plus future UIH"),
            "full_signal_uih": bool(args.full_signal_uih),
            "qwen": "frozen Qwen encodes complete UIH text directly",
            "jepa": "trainable predictor maps source UIH text latent and candidate item tokens to target UIH text latent",
            "jepa_candidate_mix": bool(args.jepa_candidate_mix),
            "ranker_ablation_suite": bool(args.ranker_ablation_suite),
            "ranker": "set transformer with pure listwise softmax",
        },
        "text_example": text_examples,
        "baseline_test": summarize_baseline(args.baseline_results),
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_dot": {
            "oracle_qwen_target_uih": {
                split: candidate_dot_metrics(samples, split, article_to_idx, item_emb, target_emb)
                for split in ["jepa_val", "ranker_val", "ranker_test"]
            }
        },
        "models": {},
    }
    write_json(out_dir / "results.partial.json", results)

    def add_ranker_result(
        name: str,
        post_vectors: np.ndarray,
        *,
        use_source_token: bool,
        use_post_token: bool,
        use_source_feature: bool,
        use_post_feature: bool,
        use_delta_feature: bool,
    ) -> None:
        print(f"training {name}", flush=True)
        results["models"][name] = train_ranker(
            name,
            samples,
            article_to_idx,
            item_emb,
            source_emb,
            post_vectors,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            use_source_token,
            use_post_token,
            use_source_feature,
            use_post_feature,
            use_delta_feature,
            args,
        )
        print(json.dumps({name: results["models"][name]["ranker_test"]}, indent=2), flush=True)
        write_json(out_dir / "results.partial.json", results)

    if not args.skip_baseline:
        add_ranker_result(
            "S0_source_only",
            zeros,
            use_source_token=True,
            use_post_token=False,
            use_source_feature=True,
            use_post_feature=False,
            use_delta_feature=False,
        )

    if not args.skip_oracle:
        if args.ranker_ablation_suite:
            add_ranker_result(
                "O0_oracle_only",
                target_emb,
                use_source_token=False,
                use_post_token=True,
                use_source_feature=False,
                use_post_feature=True,
                use_delta_feature=False,
            )
            add_ranker_result(
                "O1_source_oracle",
                target_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=False,
            )
            add_ranker_result(
                "O2_source_oracle_delta",
                target_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=True,
            )
        else:
            add_ranker_result(
                "O2_source_oracle_delta",
                target_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=True,
            )

    if not args.skip_predicted:
        print("training Qwen full-UIH JEPA predictor", flush=True)
        pred_emb, jepa_metrics = train_jepa(
            samples,
            article_to_idx,
            item_emb,
            source_emb,
            target_emb,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            args,
        )
        results["qwen_text_jepa"] = jepa_metrics
        results["candidate_dot"]["predicted_qwen_target_uih"] = {
            split: candidate_dot_metrics(samples, split, article_to_idx, item_emb, pred_emb)
            for split in ["jepa_val", "ranker_val", "ranker_test"]
        }
        if args.ranker_ablation_suite:
            add_ranker_result(
                "P0_predicted_only",
                pred_emb,
                use_source_token=False,
                use_post_token=True,
                use_source_feature=False,
                use_post_feature=True,
                use_delta_feature=False,
            )
            add_ranker_result(
                "P1_source_predicted",
                pred_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=False,
            )
            add_ranker_result(
                "P2_source_predicted_delta",
                pred_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=True,
            )
        else:
            add_ranker_result(
                "P2_source_predicted_delta",
                pred_emb,
                use_source_token=True,
                use_post_token=True,
                use_source_feature=True,
                use_post_feature=True,
                use_delta_feature=True,
            )

    write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
