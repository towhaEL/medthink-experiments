"""
Notebook discovery, inspection, slug generation, and kernel metadata builder.
Zero external dependencies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import config
from .models import Account, NotebookInfo, RunOptions


class NotebookManager:
    """Discovers, parses, and stages local notebooks for Kaggle execution."""

    def __init__(
        self,
        notebooks_dir: Path | None = None,
        staging_dir: Path | None = None,
    ) -> None:
        self.notebooks_dir = Path(notebooks_dir or config.DEFAULT_NOTEBOOKS_DIR).resolve()
        self.staging_dir = Path(staging_dir or config.STAGING_DIR).resolve()
        self.staging_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    @staticmethod
    def derive_slug(filename_or_title: str) -> str:
        """Derive a clean, valid Kaggle kernel slug (lowercase alphanumeric and hyphens)."""
        stem = Path(filename_or_title).stem
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
        if not cleaned:
            cleaned = "kaggle-notebook"
        # If slug starts with digits or is short, prefix with medthink-
        if cleaned[0].isdigit():
            cleaned = f"medthink-{cleaned}"
        elif len(cleaned) < 5:
            cleaned = f"run-{cleaned}"
        return cleaned[:68]

    @staticmethod
    def derive_title(filename: str, first_header: str | None = None) -> str:
        """Derive a human-readable title from markdown header or filename."""
        if first_header:
            return first_header.strip()
        stem = Path(filename).stem
        parts = stem.replace("_", " ").replace("-", " ").split()
        capitalized = " ".join(p.capitalize() if not p.isupper() else p for p in parts)
        if stem.startswith("0") or stem[0].isdigit():
            return f"MedThink {capitalized}"
        return capitalized

    def inspect_notebook(self, notebook_path: str | Path) -> tuple[NotebookInfo, dict[str, Any]]:
        """Inspect a local .ipynb notebook file without external libraries."""
        path = Path(notebook_path).resolve()
        if not path.is_file():
            # Try resolving relative to notebooks_dir
            candidate = (self.notebooks_dir / notebook_path).resolve()
            if candidate.is_file():
                path = candidate
            else:
                candidate = (self.notebooks_dir / Path(notebook_path).name).resolve()
                if not candidate.name.endswith(".ipynb"):
                    candidate = candidate.with_suffix(".ipynb")
                if candidate.is_file():
                    path = candidate
                else:
                    raise FileNotFoundError(
                        f"Notebook file not found at: {notebook_path} "
                        f"(checked {path} and {self.notebooks_dir})"
                    )

        try:
            with open(path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse notebook JSON at {path}: {e}")

        cells = nb_data.get("cells", [])
        total_cells = len(cells)
        code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
        markdown_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")

        # Search for first level-1 markdown header
        extracted_header = None
        for cell in cells:
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                lines = source if isinstance(source, list) else source.splitlines()
                for line in lines:
                    line_str = line.strip()
                    if line_str.startswith("# "):
                        extracted_header = line_str.lstrip("#").strip()
                        break
                if extracted_header:
                    break

        title = self.derive_title(path.name, extracted_header)
        slug = self.derive_slug(path.name)

        stat = path.stat()
        file_size = stat.st_size
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat()

        info = NotebookInfo(
            name=path.name,
            path=str(path),
            slug=slug,
            title=title,
            total_cells=total_cells,
            code_cells=code_cells,
            markdown_cells=markdown_cells,
            file_size_bytes=file_size,
            last_modified=mtime,
        )

        return info, nb_data

    def list_notebooks(self, directory: Path | None = None) -> list[NotebookInfo]:
        """Scan directory for *.ipynb notebooks and return sorted list of NotebookInfo."""
        target_dir = Path(directory or self.notebooks_dir).resolve()
        if not target_dir.exists():
            return []

        results: list[NotebookInfo] = []
        for file in sorted(target_dir.glob("*.ipynb")):
            if file.name.startswith(".") or ".ipynb_checkpoints" in str(file):
                continue
            try:
                info, _ = self.inspect_notebook(file)
                results.append(info)
            except Exception:
                continue
        return results

    def generate_kernel_metadata(
        self,
        options: RunOptions,
        account: Account,
        notebook_path: Path,
    ) -> dict[str, Any]:
        """Generate clean, official Kaggle kernel-metadata.json dictionary."""
        slug = options.slug or self.derive_slug(notebook_path.name)
        title = options.title or self.derive_title(notebook_path.name)
        kernel_id = f"{account.username}/{slug}"

        accelerator = config.normalize_accelerator(options.accelerator)
        is_tpu = options.enable_tpu or accelerator.startswith("Tpu")
        is_gpu = options.enable_gpu and not is_tpu and accelerator != "None"

        metadata: dict[str, Any] = {
            "id": kernel_id,
            "title": title,
            "code_file": notebook_path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true" if is_gpu else "false",
            "enable_tpu": "true" if is_tpu else "false",
            "enable_internet": "true" if options.enable_internet else "false",
            "dataset_sources": options.dataset_sources or [],
            "competition_sources": options.competition_sources or [],
            "kernel_sources": options.kernel_sources or [],
            "model_sources": options.model_sources or [],
        }

        return metadata

    def prepare_staging(
        self,
        job_id: str,
        notebook_path: str | Path,
        options: RunOptions,
        account: Account,
    ) -> Path:
        """Create staging folder, copy notebook, and generate kernel-metadata.json."""
        source_path = Path(notebook_path).resolve()
        if not source_path.exists():
            # Try finding in default notebooks directory
            fallback = self.notebooks_dir / source_path.name
            if fallback.exists():
                source_path = fallback
            else:
                raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

        staging_job_dir = self.staging_dir / job_id
        staging_job_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

        # Copy notebook file into staging
        dest_notebook = staging_job_dir / source_path.name
        shutil.copy2(source_path, dest_notebook)

        # Generate and save kernel-metadata.json
        metadata = self.generate_kernel_metadata(options, account, dest_notebook)
        meta_file = staging_job_dir / "kernel-metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return staging_job_dir
