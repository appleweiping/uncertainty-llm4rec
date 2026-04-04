from __future__ import annotations

import ast
import re
from typing import Any, Iterable

import pandas as pd


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    """
    通用文本清洗：
    - 处理 None / NaN
    - 去除多余空白
    - 将非字符串安全转成字符串
    """
    if _is_missing(value):
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = _normalize_whitespace(text)
    return text


def _flatten_iterable(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for v in values:
        if isinstance(v, (list, tuple, set)):
            result.extend(_flatten_iterable(v))
        else:
            text = clean_text(v)
            if text:
                result.append(text)
    return result


def parse_categories(value: Any) -> str:
    """
    将 categories 字段统一转成可读字符串。
    支持：
    - list / nested list
    - 字符串化 list，如 "['Beauty', 'Hair Care']"
    - 普通字符串
    """
    if _is_missing(value):
        return ""

    if isinstance(value, (list, tuple, set)):
        parts = _flatten_iterable(value)
        return " > ".join(parts)

    text = clean_text(value)
    if not text:
        return ""

    # 尝试把形如 "['a', 'b']" 的字符串解析回来
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                parts = _flatten_iterable(parsed)
                return " > ".join(parts)
        except Exception:
            pass

    return text


def parse_description(value: Any, max_desc_len: int = 1000) -> str:
    """
    将 description 统一清洗成短文本，避免把超长脏文本直接塞进 candidate_text。
    """
    if _is_missing(value):
        return ""

    if isinstance(value, (list, tuple, set)):
        parts = _flatten_iterable(value)
        text = " ".join(parts)
    else:
        text = clean_text(value)

    if not text:
        return ""

    text = _normalize_whitespace(text)

    if max_desc_len > 0 and len(text) > max_desc_len:
        text = text[:max_desc_len].rstrip()

    return text


def build_candidate_text(
    row: pd.Series,
    strategy: str = "title_categories_description",
    max_desc_len: int = 1000,
    fallback_to_title_categories: bool = True,
) -> str:
    """
    根据 item 行构造 candidate_text。
    第一版默认策略：
    title + categories + description
    如果 description 为空，可退化为 title + categories
    """
    title = clean_text(row.get("title", ""))
    categories = parse_categories(row.get("categories", ""))
    description = parse_description(row.get("description", ""), max_desc_len=max_desc_len)

    parts: list[str] = []

    if strategy == "title_only":
        if title:
            parts.append(f"Title: {title}")

    elif strategy == "title_categories":
        if title:
            parts.append(f"Title: {title}")
        if categories:
            parts.append(f"Categories: {categories}")

    else:
        # 默认走 title_categories_description
        if title:
            parts.append(f"Title: {title}")
        if categories:
            parts.append(f"Categories: {categories}")
        if description:
            parts.append(f"Description: {description}")
        elif fallback_to_title_categories:
            # 没有 description 时不额外做事，直接保留 title/categories
            pass

    candidate_text = "\n".join([p for p in parts if p.strip()])
    candidate_text = _normalize_whitespace(candidate_text.replace("\n", " \n ").replace(" \n ", "\n"))

    return candidate_text.strip()


def attach_candidate_text(
    items_df: pd.DataFrame,
    strategy: str = "title_categories_description",
    max_desc_len: int = 1000,
    fallback_to_title_categories: bool = True,
) -> pd.DataFrame:
    """
    给 items_df 增加 candidate_text 列。
    期望至少有 item_id，建议有 title/categories/description。
    """
    if "item_id" not in items_df.columns:
        raise ValueError("items_df 缺少必要字段: item_id")

    df = items_df.copy()

    if "title" not in df.columns:
        df["title"] = ""
    if "categories" not in df.columns:
        df["categories"] = ""
    if "description" not in df.columns:
        df["description"] = ""

    df["title"] = df["title"].map(clean_text)
    df["categories"] = df["categories"].map(parse_categories)
    df["description"] = df["description"].map(lambda x: parse_description(x, max_desc_len=max_desc_len))

    df["candidate_text"] = df.apply(
        lambda row: build_candidate_text(
            row=row,
            strategy=strategy,
            max_desc_len=max_desc_len,
            fallback_to_title_categories=fallback_to_title_categories,
        ),
        axis=1,
    )

    return df