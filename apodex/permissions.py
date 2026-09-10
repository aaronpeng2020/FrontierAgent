"""Persistent per-command allow/deny rules (Claude-Code parity).

A session-global "auto-approve everything" (the ``[a]`` key) trades all future
safety for convenience. This adds granular, persisted rules — "always allow
``npm test``", "never allow ``git push``" — matched by command prefix, so the
gate stays livable without being all-or-nothing.

Rules are strings: ``Bash(npm test)`` / ``Bash(git push)`` for shell (matched by
prefix across every ``&&``/``|``/``;`` segment, fail-safe), or a bare tool name
(``write_file``) for everything else.

Safety contract: this store only ever *downgrades a plain confirm to safe*, or
*forces a deny*. It is consulted in :func:`agent_tools.assess_tool_risk` AFTER
danger detection and the hard denylist — so a saved ``Bash(git)`` allow can
never green-light a dangerous ``git push --force``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field

_DEFAULT_PATH = os.path.expanduser("~/.config/apodex/permissions.json")
# Commands whose first token alone is too coarse — keep two words so
# "always allow npm test" doesn't also allow "npm publish".
_MULTI_VERB = frozenset({
    "git", "npm", "pnpm", "yarn", "uv", "pip", "pip3", "cargo", "go", "docker",
    "poetry", "conda", "make", "apt", "apt-get", "brew", "kubectl", "gh",
    # exec-capable / destructive programs: a one-word rule ("always allow
    # curl") would green-light exfiltration or deletion with any arguments.
    "python", "python3", "node", "bash", "sh", "zsh", "curl", "wget", "rm",
    "ssh", "scp", "rsync", "chmod", "chown", "kill", "pkill", "find", "xargs",
})
# ``\n``/``\r`` are separators bash honours; a rule must see every segment.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||\||;|\r\n|\n|\r")
# Segments skipped when matching. ``env``/``source``/``.``/``export`` are NOT
# helpers: ``env`` execs its arguments, ``source`` runs a file, ``export``
# rewrites PATH/LD_PRELOAD for the segment the rule does allow.
_HELPER_CMDS = frozenset({"cd", "pwd", "set", "echo", "clear", "true"})
# Constructs a saved prefix rule cannot vouch for: command substitution,
# process substitution, redirections and a background ``&`` all smuggle a
# second command past ``startswith(prefix)``. Any of them -> no match -> confirm.
_UNMATCHABLE = re.compile(r"\$\(|`|<\(|>>?|(?<![&>])&(?!&)|<<<")


def _extract_prefix_from_segment(seg: str) -> str:
    try:
        toks = shlex.split(seg)
    except Exception:
        toks = seg.split()
    if not toks:
        return ""
    if toks[0] in _MULTI_VERB and len(toks) > 1:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def _bash_prefix(cmd: str) -> str:
    """A reusable prefix for a bash command (``cd /app && npm test`` → ``npm test``)."""
    segs = [s.strip() for s in _SEGMENT_SPLIT.split(cmd or "") if s.strip()]
    if not segs:
        return ""
    # Prefer the first non-helper segment so 'cd /foo && python script.py' saves Bash(python)
    for seg in segs:
        p = _extract_prefix_from_segment(seg)
        if p and p.split()[0] not in _HELPER_CMDS:
            return p
    return _extract_prefix_from_segment(segs[0])


def rule_for(name: str, args: dict) -> str:
    """The allow-rule string an 'always allow' on this call would create.

    Empty for a bash call with no usable prefix: the old fallback ``"bash"``
    was a wildcard that allowed every future command.
    """
    if name == "bash":
        p = _bash_prefix(str(args.get("command", "")))
        return f"Bash({p})" if p else ""
    return name


def _is_dangerous_call(name: str, args: dict) -> bool:
    if name != "bash":
        return False
    try:
        from apodex.agent_tools import detect_danger
    except Exception:
        return True  # cannot assess -> refuse to persist
    return bool(detect_danger(str(args.get("command", ""))))


@dataclass
class PermissionStore:
    """Allow/deny rules, persisted to a JSON file."""

    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    path: str = _DEFAULT_PATH

    @classmethod
    def load(cls, path: str = _DEFAULT_PATH) -> PermissionStore:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(set(d.get("allow") or []), set(d.get("deny") or []), path)
        except Exception:
            return cls(path=path)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"allow": sorted(self.allow), "deny": sorted(self.deny)}, f, indent=2)
        except Exception:
            pass

    def allows(self, name: str, args: dict) -> bool:
        return self._matches(self.allow, name, args)

    def denies(self, name: str, args: dict) -> bool:
        return self._matches(self.deny, name, args)

    def add_allow(self, name: str, args: dict) -> str:
        """Persist 'always allow' for this call; returns the rule added, or
        ``""`` when nothing was saved: a dangerous command (``rm -rf``, force
        push, ...) must be typed-confirmed every time, and an empty prefix
        would be a wildcard."""
        rule = rule_for(name, args)
        if not rule or _is_dangerous_call(name, args):
            return ""
        self.allow.add(rule)
        self.save()
        return rule

    @staticmethod
    def _matches(rules: set[str], name: str, args: dict) -> bool:
        if name in rules:
            return True
        if name == "bash":
            if "bash" in rules or "Bash" in rules or "Bash(*)" in rules:
                return True
            prefixes = {r[5:-1] for r in rules if r.startswith("Bash(") and r.endswith(")")}
            prefixes.discard("")
            if not prefixes:
                return False
            command = str(args.get("command", ""))
            if _UNMATCHABLE.search(command):
                return False
            segs = [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]
            # Filter out helper segments (e.g. 'cd /foo') unless all segments are helpers
            non_helpers = [
                s for s in segs
                if _extract_prefix_from_segment(s).split()[0] not in _HELPER_CMDS
            ]
            check_segs = non_helpers if non_helpers else segs
            return bool(check_segs) and all(
                any(seg == p or seg.startswith(p + " ") for p in prefixes) for seg in check_segs
            )
        return False


__all__ = ["PermissionStore", "rule_for"]
