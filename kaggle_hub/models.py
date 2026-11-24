"""
Data models and schemas for Kaggle Hub.
Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Returns current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class JobStatus:
    SUBMITTED = "SUBMITTED"
    STAGING = "STAGING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    DOWNLOADED = "DOWNLOADED"


@dataclass
class Account:
    id: str
    label: str
    username: str
    auth_type: str = "api_key"  # 'api_key' or 'oauth_token'
    credentials: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    last_used_at: str | None = None
    status: str = "valid"  # 'valid', 'invalid', 'untested'
    quota_notes: str = "30h weekly GPU"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Account:
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            username=data.get("username", ""),
            auth_type=data.get("auth_type", "api_key"),
            credentials=data.get("credentials", {}),
            created_at=data.get("created_at") or utc_now_iso(),
            last_used_at=data.get("last_used_at"),
            status=data.get("status", "valid"),
            quota_notes=data.get("quota_notes", "30h weekly GPU"),
            notes=data.get("notes", ""),
        )


@dataclass
class NotebookInfo:
    name: str
    path: str
    slug: str
    title: str
    total_cells: int = 0
    code_cells: int = 0
    markdown_cells: int = 0
    file_size_bytes: int = 0
    last_modified: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotebookInfo:
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            total_cells=data.get("total_cells", 0),
            code_cells=data.get("code_cells", 0),
            markdown_cells=data.get("markdown_cells", 0),
            file_size_bytes=data.get("file_size_bytes", 0),
            last_modified=data.get("last_modified", ""),
        )


@dataclass
class RunOptions:
    notebook: str
    account_id: str | None = None
    title: str | None = None
    slug: str | None = None
    accelerator: str = "NvidiaTeslaT4"
    enable_gpu: bool = True
    enable_tpu: bool = False
    enable_internet: bool = True
    dataset_sources: list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)
    auto_download: bool = True
    watch: bool = False
    timeout: int = 3600

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunOptions:
        return cls(
            notebook=data.get("notebook", ""),
            account_id=data.get("account_id"),
            title=data.get("title"),
            slug=data.get("slug"),
            accelerator=data.get("accelerator", "NvidiaTeslaT4"),
            enable_gpu=data.get("enable_gpu", True),
            enable_tpu=data.get("enable_tpu", False),
            enable_internet=data.get("enable_internet", True),
            dataset_sources=data.get("dataset_sources", []),
            competition_sources=data.get("competition_sources", []),
            kernel_sources=data.get("kernel_sources", []),
            model_sources=data.get("model_sources", []),
            auto_download=data.get("auto_download", True),
            watch=data.get("watch", False),
            timeout=data.get("timeout", 3600),
        )


@dataclass
class JobRecord:
    id: str
    account_id: str
    username: str
    kernel_slug: str
    kernel_id: str
    title: str
    notebook_path: str
    staging_path: str
    accelerator: str = "NvidiaTeslaT4"
    enable_gpu: bool = True
    enable_internet: bool = True
    dataset_sources: list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)
    status: str = JobStatus.SUBMITTED
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    output_dir: str | None = None
    exit_code: int | None = None
    error_message: str | None = None
    last_log_snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        return cls(
            id=data.get("id", ""),
            account_id=data.get("account_id", ""),
            username=data.get("username", ""),
            kernel_slug=data.get("kernel_slug", ""),
            kernel_id=data.get("kernel_id", ""),
            title=data.get("title", ""),
            notebook_path=data.get("notebook_path", ""),
            staging_path=data.get("staging_path", ""),
            accelerator=data.get("accelerator", "NvidiaTeslaT4"),
            enable_gpu=data.get("enable_gpu", True),
            enable_internet=data.get("enable_internet", True),
            dataset_sources=data.get("dataset_sources", []),
            competition_sources=data.get("competition_sources", []),
            kernel_sources=data.get("kernel_sources", []),
            model_sources=data.get("model_sources", []),
            status=data.get("status", JobStatus.SUBMITTED),
            created_at=data.get("created_at") or utc_now_iso(),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            output_dir=data.get("output_dir"),
            exit_code=data.get("exit_code"),
            error_message=data.get("error_message"),
            last_log_snippet=data.get("last_log_snippet"),
        )
