"""Launch isolated Pi planner sessions and parse tool access records."""

from __future__ import annotations

import json

import os

import shutil

from pathlib import Path

from typing import Any

from gemini_llm import DEFAULT_GEMINI_MODEL

PI_PROVIDER = os.environ.get("AUDIT_PI_PROVIDER", "google").strip()

PI_MODEL = os.environ.get("AUDIT_PI_MODEL", DEFAULT_GEMINI_MODEL).strip()

PI_REASONING = os.environ.get("AUDIT_PI_REASONING", "high").strip()

PI_TOOLS = "read,grep,compare_attempts"

PI_COMPARE_EXTENSION = Path(__file__).with_name("pi_compare_attempts_extension.ts")

PROVIDER_API_KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PI_API_KEY_ENV = os.environ.get(
    "AUDIT_PI_API_KEY_ENV",
    PROVIDER_API_KEY_ENVS.get(PI_PROVIDER, ""),
).strip()

if not PI_PROVIDER:
    raise ValueError("AUDIT_PI_PROVIDER cannot be empty")

if not PI_MODEL:
    raise ValueError("AUDIT_PI_MODEL cannot be empty")

if PI_REASONING not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
    raise ValueError(f"unsupported AUDIT_PI_REASONING: {PI_REASONING!r}")

if not PI_API_KEY_ENV:
    raise ValueError(
        f"no API-key environment variable is configured for provider {PI_PROVIDER!r}"
    )


def remove_python_bytecode(root: Path) -> None:
    """Remove generated Python bytecode without following evidence symlinks."""
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(names):
            path = base / name
            if name == "__pycache__":
                shutil.rmtree(path)
                names.remove(name)
            elif path.is_symlink():
                names.remove(name)
        for name in files:
            if name.endswith(".pyc"):
                (base / name).unlink()


def canonical_view_access_path(raw: str, view: Path) -> str:
    """Return one path relative to ``view`` or reject it fail-closed."""
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("evidence access path is invalid")
    path = Path(raw.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError(f"evidence access escapes the frozen view: {raw!r}")
    view_path = Path(os.path.abspath(view))
    candidate = path if path.is_absolute() else view_path / path
    candidate = Path(os.path.normpath(candidate))
    try:
        relative = candidate.relative_to(view_path)
    except ValueError as exc:
        raise ValueError(
            f"evidence access is outside the frozen view: {raw!r}"
        ) from exc
    return relative.as_posix() if relative.parts else "."


def canonical_view_access_paths(paths: list[str], view: Path) -> set[str]:
    """Normalize every recorded path; an unrecognized path is never dropped."""
    return {canonical_view_access_path(str(path), view) for path in paths}


def build_pi_subprocess_env(
    executable: str,
    *,
    home: Path,
    tmpdir: str | os.PathLike[str],
    extra: dict[str, str] | None = None,
    provider: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, str]:
    """Build the isolated Pi environment without losing its Node runtime."""
    effective_provider = provider or PI_PROVIDER
    effective_api_key_env = api_key_env or PI_API_KEY_ENV
    if not os.environ.get(effective_api_key_env):
        raise RuntimeError(f"{effective_api_key_env} is not set")
    executable_dir = str(Path(executable).parent)
    path = os.pathsep.join(
        dict.fromkeys(
            (
                executable_dir,
                str(Path(shutil.which("node") or "/usr/bin/node").parent),
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            )
        )
    )
    env = {
        effective_api_key_env: os.environ[effective_api_key_env],
        "PATH": path,
        "HOME": str(home),
        "TMPDIR": str(tmpdir),
        "CI": "1",
        "TERM": "dumb",
    }

    # Pi's built-in OpenAI catalog carries api.openai.com as the model URL;
    # an isolated HOME therefore needs an explicit provider override for MAST.
    base_url = (
        os.environ.get("OPENAI_BASE_URL") if effective_provider == "openai" else None
    )
    if base_url:
        agent_dir = home / ".pi" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "models.json").write_text(
            json.dumps({"providers": {"openai": {"baseUrl": base_url}}}) + "\n",
            encoding="utf-8",
        )
        env["OPENAI_BASE_URL"] = base_url

    if extra:
        env.update(extra)
    return env


def _plain_json(text: str) -> str:
    """Return the single complete JSON object, tolerating a prose/fence wrapper."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].lower() in ("```", "```json")
        and lines[-1] == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stripped[start:])
        except ValueError:
            continue
        suffix = stripped[start + end :].strip()
        if isinstance(value, dict) and suffix in {"", "```"}:
            return stripped[start : start + end]
    return stripped


DEFAULT_FINAL_MESSAGE = (
    "Return the audit JSON now. The entire response must begin with { and "
    "end with }; output no prose and no Markdown fence."
)


def build_pi_argv(
    model: str,
    prompt: str,
    executable: str = "pi",
    *,
    extra_extensions: tuple[str | Path, ...] = (),
    extra_tools: tuple[str, ...] = (),
    tools: str | None = None,
    session_id: str | None = None,
    user_message: str = DEFAULT_FINAL_MESSAGE,
    provider: str | None = None,
    reasoning: str | None = None,
) -> list[str]:
    argv = [
        executable,
        "--print",
        "--mode",
        "json",
        "--provider",
        provider or PI_PROVIDER,
        "--model",
        model,
        "--thinking",
        reasoning or PI_REASONING,
        "--system-prompt",
        prompt,
        "--extension",
        str(PI_COMPARE_EXTENSION),
    ]
    for extension in extra_extensions:
        argv += ["--extension", str(extension)]
    argv += ["--no-session"] if session_id is None else ["--session-id", session_id]
    argv += [
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--no-approve",
        "--tools",
        tools or ",".join((PI_TOOLS, *extra_tools)),
        user_message,
    ]
    return argv


def parse_pi_stream(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "final_text": None,
        "session_id": None,
        "tool_names": [],
        "successful_read_paths": [],
        "direct_read_paths": [],
        "tool_internal_read_paths": [],
        "read_events": [],
        "compare_events": [],
        "unfold_selection_events": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cli_cost_usd": 0.0,
        "malformed_lines": 0,
    }
    pending_tools: dict[str, tuple[str, dict[str, Any], int]] = {}
    event_index = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            result["malformed_lines"] += 1
            continue
        current_event_index = event_index
        event_index += 1
        if event.get("type") == "session":
            result["session_id"] = event.get("id")
        if event.get("type") == "tool_execution_start":
            tool_name = str(event.get("toolName") or "?")
            result["tool_names"].append(tool_name)
            args = event.get("args")
            pending_tools[str(event.get("toolCallId") or "")] = (
                tool_name,
                args if isinstance(args, dict) else {},
                current_event_index,
            )
        if event.get("type") == "tool_execution_end" and not event.get("isError"):
            tool_name, args, start_index = pending_tools.get(
                str(event.get("toolCallId") or ""), ("", {}, current_event_index)
            )
            path = args.get("path")
            if tool_name in {"read", "grep"} and isinstance(path, str):
                result["successful_read_paths"].append(path)
                result["direct_read_paths"].append(path)
                result["read_events"].append(
                    {
                        "tool_name": tool_name,
                        "path": path,
                        "start_event_index": start_index,
                        "end_event_index": current_event_index,
                    }
                )
            if tool_name == "compare_attempts":
                details = (event.get("result") or {}).get("details") or {}
                for read_path in details.get("readPaths") or []:
                    if isinstance(read_path, str):
                        result["successful_read_paths"].append(read_path)
                        result["tool_internal_read_paths"].append(read_path)
                for read_path in details.get("privateReadPaths") or []:
                    if isinstance(read_path, str):
                        result["tool_internal_read_paths"].append(read_path)
                result["compare_events"].append(
                    {
                        "left": args.get("left"),
                        "right": args.get("right"),
                        "start_event_index": start_index,
                        "end_event_index": current_event_index,
                    }
                )
            if tool_name == "read_registered_attempt":
                details = (event.get("result") or {}).get("details") or {}
                for read_path in details.get("readPaths") or []:
                    if not isinstance(read_path, str):
                        continue
                    result["successful_read_paths"].append(read_path)
                    result["tool_internal_read_paths"].append(read_path)
                    result["read_events"].append(
                        {
                            "tool_name": tool_name,
                            "path": read_path,
                            "start_event_index": start_index,
                            "end_event_index": current_event_index,
                        }
                    )
                for read_path in details.get("privateReadPaths") or []:
                    if isinstance(read_path, str):
                        result["tool_internal_read_paths"].append(read_path)
            if tool_name == "read_public_file":
                details = (event.get("result") or {}).get("details") or {}
                for read_path in details.get("readPaths") or []:
                    if not isinstance(read_path, str):
                        continue
                    result["successful_read_paths"].append(read_path)
                    result["direct_read_paths"].append(read_path)
                    result["read_events"].append(
                        {
                            "tool_name": tool_name,
                            "path": read_path,
                            "start_event_index": start_index,
                            "end_event_index": current_event_index,
                        }
                    )
            if tool_name == "register_unfold_selection":
                details = (event.get("result") or {}).get("details") or {}
                selections = details.get("selections", args.get("selections"))
                result["unfold_selection_events"].append(
                    {
                        "selections": selections
                        if isinstance(selections, list)
                        else [],
                        "start_event_index": start_index,
                        "end_event_index": current_event_index,
                    }
                )
        message = event.get("message") if event.get("type") == "message_end" else None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        pieces = [
            block["text"]
            for block in message.get("content", [])
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if pieces:
            result["final_text"] = _plain_json("".join(pieces))
        usage = message.get("usage") or {}
        result["input_tokens"] += int(usage.get("input") or 0)
        result["output_tokens"] += int(usage.get("output") or 0)
        result["cache_read_tokens"] += int(usage.get("cacheRead") or 0)
        result["cache_write_tokens"] += int(usage.get("cacheWrite") or 0)
        result["cli_cost_usd"] += float((usage.get("cost") or {}).get("total") or 0)
    result["tool_calls"] = len(result["tool_names"])
    return result
