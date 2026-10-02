"""Выполняет reports/rtk_report.ipynb и сохраняет результат в reports/out/.

    python reports/run_report.py                      # настройки — из ноутбука и settings_local.py
    python reports/run_report.py --settings my.py     # другой файл настроек

Результат: out/rtk_report.html (без кода, для показа) и out/rtk_report_executed.ipynb.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", help="файл настроек вместо settings_local.py")
    parser.add_argument("--out", default=str(HERE / "out"), help="папка для результата")
    parser.add_argument("--name", default="rtk_report", help="имя файлов результата")
    args = parser.parse_args()
    if args.settings:
        os.environ["RTK_REPORT_SETTINGS"] = str(Path(args.settings).resolve())

    nb = nbformat.read(HERE / "rtk_report.ipynb", as_version=4)
    started = time.time()
    print("выполняю отчёт…")
    ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(HERE)}})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out / f"{args.name}_executed.ipynb")
    title = nb.metadata.get("title", args.name)
    html, _ = HTMLExporter(exclude_input=True).from_notebook_node(nb, {"metadata": {"name": title}})
    (out / f"{args.name}.html").write_text(html, encoding="utf-8")
    print(f"готово за {time.time() - started:.0f} с: {out / (args.name + '.html')}")


if __name__ == "__main__":
    main()
