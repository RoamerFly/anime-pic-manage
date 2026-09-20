#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow into an anime-pic-manage template.

ComfyUI stores workflows in its own UI format (``nodes`` + ``links``) and only
exports the API format from the browser (``File -> Export (API)``). This script
performs the same conversion offline so existing workflows can be used as
templates, and can queue the result against a running ComfyUI to verify it is
accepted.

Usage:
    python scripts/comfy_convert_workflow.py <workflow.json> [--name NAME]
        [--output DIR] [--server http://127.0.0.1:8188] [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
TEMPLATE_VERSION = 1


def fetch_object_info(server: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{server}/object_info", timeout=60) as response:
        return json.loads(response.read())


def widget_inputs(node_info: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Widget-like inputs of a node class, in the order the UI shows them."""

    inputs = node_info.get("input", {})
    ordered: list[tuple[str, dict[str, Any]]] = []
    for group in ("required", "optional"):
        for name, spec in (inputs.get(group) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            kind = spec[0]
            is_widget = isinstance(kind, list) or (
                isinstance(kind, str) and kind in WIDGET_SCALARS
            )
            if not is_widget:
                continue
            config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            ordered.append((name, config))
    return ordered


def link_map(workflow: dict[str, Any]) -> dict[int, tuple[int, int]]:
    links: dict[int, tuple[int, int]] = {}
    for entry in workflow.get("links") or []:
        if isinstance(entry, list) and len(entry) >= 5:
            links[int(entry[0])] = (int(entry[1]), int(entry[2]))
        elif isinstance(entry, dict) and entry.get("id") is not None:
            links[int(entry["id"])] = (int(entry["origin_id"]), int(entry["origin_slot"]))
    return links


def convert(workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """Reproduce ComfyUI's ``app.graphToPrompt()`` output for one workflow."""

    links = link_map(workflow)
    prompt: dict[str, Any] = {}
    for node in workflow.get("nodes") or []:
        class_type = node.get("type")
        node_info = object_info.get(class_type)
        if node_info is None:
            raise SystemExit(f"ComfyUI 未提供节点定义: {class_type}")
        inputs: dict[str, Any] = {}
        for entry in node.get("inputs") or []:
            link_id = entry.get("link")
            if link_id is not None and int(link_id) in links:
                origin_id, origin_slot = links[int(link_id)]
                inputs[entry["name"]] = [str(origin_id), origin_slot]
        # The saved node lists its widget inputs in UI order; that order is what
        # `widgets_values` follows (object_info is only a fallback for older files).
        widget_names = [
            entry["widget"]["name"]
            for entry in (node.get("inputs") or [])
            if isinstance(entry.get("widget"), dict)
            and isinstance(entry["widget"].get("name"), str)
        ]
        if not widget_names:
            widget_names = [name for name, _ in widget_inputs(node_info)]
        config_by_name = dict(widget_inputs(node_info))
        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            for name, value in widgets.items():
                if name not in inputs:
                    inputs[name] = value
        else:
            values = list(widgets or [])
            cursor = 0
            for name in widget_names:
                if name in inputs:
                    continue
                if cursor >= len(values):
                    break
                inputs[name] = values[cursor]
                cursor += 1
                # The UI inserts a "control_after_generate" combo right after
                # such widgets; it has no API input of its own.
                if (config_by_name.get(name) or {}).get("control_after_generate"):
                    cursor += 1
        payload: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
        mode = node.get("mode")
        if mode in (2, 4):
            payload["mode"] = mode
        prompt[str(node["id"])] = payload
    return prompt


def _node_by_class(prompt: dict[str, Any], class_type: str) -> str | None:
    for node_id, node in prompt.items():
        if node.get("class_type") == class_type:
            return node_id
    return None


def _text_source(prompt: dict[str, Any], reference: Any) -> str | None:
    """Resolve the CLIPTextEncode (or first one inside a ConditioningCombine)."""

    if not isinstance(reference, list) or len(reference) != 2:
        return None
    node = prompt.get(str(reference[0]))
    if not isinstance(node, dict):
        return None
    if node.get("class_type") == "CLIPTextEncode":
        return str(reference[0])
    for value in (node.get("inputs") or {}).values():
        found = _text_source(prompt, value)
        if found:
            return found
    return None


def infer_bindings(prompt: dict[str, Any]) -> dict[str, list[str]]:
    """Bind the parameters our UI can control onto the converted graph."""

    bindings: dict[str, list[str]] = {}
    checkpoint = _node_by_class(prompt, "CheckpointLoaderSimple")
    if checkpoint:
        bindings["checkpoint"] = [checkpoint, "inputs", "ckpt_name"]
    sampler = _node_by_class(prompt, "KSampler") or _node_by_class(prompt, "KSamplerAdvanced")
    if sampler:
        for key in ("steps", "cfg", "sampler_name", "scheduler", "seed"):
            bindings[{"sampler_name": "sampler"}.get(key, key)] = [sampler, "inputs", key]
        inputs = prompt[sampler].get("inputs") or {}
        positive = _text_source(prompt, inputs.get("positive"))
        negative = _text_source(prompt, inputs.get("negative"))
        if positive:
            bindings["positive"] = [positive, "inputs", "text"]
        if negative:
            bindings["negative"] = [negative, "inputs", "text"]
    latent = _node_by_class(prompt, "EmptyLatentImage")
    if latent:
        for key, alias in (("width", "width"), ("height", "height"), ("batch_size", "batch")):
            bindings[alias] = [latent, "inputs", key]
    save = _node_by_class(prompt, "SaveImage")
    if save:
        bindings["filename_prefix"] = [save, "inputs", "filename_prefix"]
    lora = _node_by_class(prompt, "LoraLoader")
    if lora:
        bindings["lora_name"] = [lora, "inputs", "lora_name"]
        bindings["lora_strength_model"] = [lora, "inputs", "strength_model"]
        bindings["lora_strength_clip"] = [lora, "inputs", "strength_clip"]
        bindings["lora_mode"] = [lora, "mode"]
    return bindings


def queue_for_verification(server: str, prompt: dict[str, Any]) -> tuple[bool, str]:
    """Ask ComfyUI to validate the prompt, then interrupt immediately."""

    body = json.dumps({"prompt": prompt, "client_id": "anime-pic-manage-convert"}).encode()
    request = urllib.request.Request(
        f"{server}/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        return False, f"HTTP {error.code}: {detail[:600]}"
    prompt_id = payload.get("prompt_id", "")
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{server}/interrupt", data=b"{}"), timeout=30
        ).read()
    except Exception:  # noqa: BLE001 - verification already succeeded
        pass
    return True, prompt_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-checkpoint",
        help="仅在 --verify 时替换 ckpt_name，用于旧工作流引用了已改名模型的情况",
    )
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    if "nodes" not in workflow and "links" not in workflow:
        print("输入文件已经是 API 格式，无需转换", file=sys.stderr)
        return 2

    object_info = fetch_object_info(args.server)
    prompt = convert(workflow, object_info)
    name = args.name or args.workflow.stem
    template = {
        "version": TEMPLATE_VERSION,
        "name": name,
        "description": args.description
        or f"由 {args.workflow.name} 转换而来（ComfyUI 前端等价转换）",
        "bindings": infer_bindings(prompt),
        "prompt": prompt,
    }

    if args.verify:
        verification_prompt = json.loads(json.dumps(prompt))
        if args.verify_checkpoint:
            for node in verification_prompt.values():
                if "ckpt_name" in (node.get("inputs") or {}):
                    node["inputs"]["ckpt_name"] = args.verify_checkpoint
        ok, detail = queue_for_verification(args.server, verification_prompt)
        print(("校验通过: " if ok else "校验失败: ") + detail)
        if not ok:
            return 1

    if args.output:
        target = args.output if args.output.suffix == ".json" else args.output / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写入 {target}")
    else:
        print(json.dumps(template, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
