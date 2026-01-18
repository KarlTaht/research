"""Core data schemas for the research manager agent."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Project:
    """Represents a project or paper implementation in the monorepo."""

    name: str
    path: Path
    type: str  # "project" | "paper" | "archive"
    description: Optional[str] = None
    has_train_script: bool = False
    has_eval_script: bool = False
    config_files: list[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.type,
            "description": self.description,
            "has_train_script": self.has_train_script,
            "has_eval_script": self.has_eval_script,
            "config_files": self.config_files,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
        }


@dataclass
class Experiment:
    """Represents a tracked experiment run."""

    name: str
    project: str
    config_path: Optional[Path] = None
    metrics: dict = field(default_factory=dict)  # perplexity, loss, etc.
    timestamp: Optional[datetime] = None
    checkpoint_path: Optional[Path] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "project": self.project,
            "config_path": str(self.config_path) if self.config_path else None,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "notes": self.notes,
        }


@dataclass
class Script:
    """Represents a runnable script in the repo."""

    name: str
    path: Path
    purpose: str  # train, evaluate, download, etc.
    project: Optional[str] = None
    arguments: list[str] = field(default_factory=list)
    example_command: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "purpose": self.purpose,
            "project": self.project,
            "arguments": self.arguments,
            "example_command": self.example_command,
        }


@dataclass
class Config:
    """Represents a YAML configuration file."""

    path: Path
    project: str
    sections: dict  # parsed YAML content
    model_params: Optional[dict] = None
    training_params: Optional[dict] = None
    data_params: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "project": self.project,
            "sections": self.sections,
            "model_params": self.model_params,
            "training_params": self.training_params,
            "data_params": self.data_params,
        }
