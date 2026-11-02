"""Freeze the exact configured Markdown Skills used by study assessors."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agent.llm import SkillDocument, load_skills


def load_skill_snapshot(directory: str | Path) -> tuple[list[SkillDocument], dict[str, str]]:
    skills = load_skills(directory)
    fingerprints = {
        skill.name: hashlib.sha256((Path(directory) / skill.name).read_bytes()).hexdigest()
        for skill in skills
    }
    return skills, fingerprints


def verify_skill_snapshot(
    directory: str | Path, expected: dict[str, str] | None,
) -> list[SkillDocument]:
    skills, fingerprints = load_skill_snapshot(directory)
    if fingerprints != expected:
        raise ValueError("Skill snapshot missing or changed; prepare a new plan")
    return skills
