"""Versioned external-work character-set filtering."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelManifestError
from .recognition import LabelScore, top_k_scores


@dataclass(frozen=True, slots=True)
class Character:
    tag: str
    display_name: str
    aliases: tuple[str, ...] = ()
    force_review: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "force_review": self.force_review,
        }


@dataclass(frozen=True, slots=True)
class CharacterSet:
    schema_version: str
    set_id: str
    name: str
    characters: tuple[Character, ...]

    @property
    def by_tag(self) -> dict[str, Character]:
        return {character.tag: character for character in self.characters}

    @classmethod
    def from_json_file(cls, path: str | Path) -> CharacterSet:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError("角色集合无法读取", str(exc)) from exc
        if not isinstance(raw, dict):
            raise ModelManifestError("角色集合根节点必须是对象")
        schema_version = raw.get("schema_version")
        set_id = raw.get("set_id")
        name = raw.get("name")
        characters = raw.get("characters")
        if (
            not isinstance(schema_version, str)
            or not isinstance(set_id, str)
            or not isinstance(name, str)
        ):
            raise ModelManifestError("角色集合缺少 schema_version/set_id/name")
        if not isinstance(characters, list):
            raise ModelManifestError("角色集合 characters 必须是数组")
        parsed: list[Character] = []
        tags: set[str] = set()
        for index, item in enumerate(characters):
            if not isinstance(item, dict):
                raise ModelManifestError(f"角色集合第 {index + 1} 项不是对象")
            tag = item.get("tag")
            display_name = item.get("display_name")
            aliases = item.get("aliases", [])
            force_review = item.get("force_review", False)
            if (
                not isinstance(tag, str)
                or not tag
                or not isinstance(display_name, str)
                or not display_name
            ):
                raise ModelManifestError(f"角色集合第 {index + 1} 项缺少 tag/display_name")
            if tag in tags:
                raise ModelManifestError(f"角色集合存在重复标签: {tag}")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) for alias in aliases
            ):
                raise ModelManifestError(f"角色集合标签 {tag} 的 aliases 必须是字符串数组")
            if not isinstance(force_review, bool):
                raise ModelManifestError(f"角色集合标签 {tag} 的 force_review 必须是布尔值")
            tags.add(tag)
            parsed.append(Character(tag, display_name, tuple(aliases), force_review))
        if not parsed:
            raise ModelManifestError("角色集合不能为空")
        return cls(schema_version, set_id, name, tuple(parsed))

    def model_intersection(self, model_tags: Iterable[str]) -> tuple[set[str], set[str]]:
        """Return (supported whitelist tags, whitelist tags missing in model)."""

        model_set = set(model_tags)
        whitelist = set(self.by_tag)
        return whitelist & model_set, whitelist - model_set

    def filter_scores(
        self,
        scores: Iterable[LabelScore | Mapping[str, object]],
        model_tags: Iterable[str],
        *,
        k: int = 5,
    ) -> list[LabelScore]:
        supported, _ = self.model_intersection(model_tags)
        named = [
            LabelScore(item.tag, item.score, self.by_tag[item.tag].display_name, item.category)
            if isinstance(item, LabelScore)
            else item
            for item in scores
            if (
                item.tag
                if isinstance(item, LabelScore)
                else item.get("tag", item.get("character_tag"))
            )
            in supported
        ]
        return top_k_scores(named, k, allowed_tags=supported)

    def force_review_tags(self) -> frozenset[str]:
        return frozenset(character.tag for character in self.characters if character.force_review)


def load_character_set(path: str | Path) -> CharacterSet:
    """Load any versioned external-work character set."""

    return CharacterSet.from_json_file(path)


# Backward-compatible generic aliases for callers that used the longer names.
ExternalCharacter = Character
ExternalCharacterSet = CharacterSet
