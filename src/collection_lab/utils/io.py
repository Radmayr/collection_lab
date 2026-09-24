"""Запись файлов, которые пользователь открывает в Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# UTF-8 с меткой BOM: Excel распознаёт кодировку и не превращает кириллицу в «РџСЂРѕ...».
# pandas и другие инструменты читают такие файлы как обычный UTF-8.
CSV_ENCODING = "utf-8-sig"


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs) -> Path:
    """Сохраняет DataFrame в CSV, который корректно открывается в Excel (кириллица)."""
    path = Path(path)
    df.to_csv(path, index=index, encoding=CSV_ENCODING, **kwargs)
    return path
