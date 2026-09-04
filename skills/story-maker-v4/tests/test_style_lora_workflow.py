"""Structural tests for the MiniMax H3 R2V + style-LoRA workflow graph.

The renderer loads this graph via ``MINIMAX_H3_WORKFLOW``, so a broken link
table would only surface as a ComfyUI error hours into a render.
"""

from __future__ import annotations

import json
import os

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
WORKFLOW_PATH = os.path.join(
    _REPO_ROOT, "workflows", "comfyui", "minimax-h3-r2v-style-lora.json"
)


def _load() -> dict:
    with open(WORKFLOW_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _by_id(wf: dict) -> dict[int, dict]:
    return {n["id"]: n for n in wf["nodes"]}


def test_workflow_exists_and_links_are_consistent():
    wf = _load()
    link_by_id = {l[0]: l for l in wf["links"]}
    nodes = _by_id(wf)

    for node in wf["nodes"]:
        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None:
                continue
            assert link_id in link_by_id, f"node {node['id']} input {inp['name']} dangling"
            _, origin_id, _, target_id, _, _ = link_by_id[link_id]
            assert target_id == node["id"]
            assert origin_id in nodes
        for out in node.get("outputs") or []:
            for link_id in out.get("links") or []:
                assert link_id in link_by_id, f"node {node['id']} output dangling"
                assert link_by_id[link_id][1] == node["id"]


def test_lora_chain_sits_between_unet_and_attention_patch():
    wf = _load()
    nodes = _by_id(wf)
    link_by_id = {l[0]: l for l in wf["links"]}

    loras = [n for n in wf["nodes"] if n["type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 3, "three switchable style-LoRA slots are expected"

    unet = next(n for n in wf["nodes"] if n["type"] == "UNETLoader")
    sage = next(n for n in wf["nodes"] if n["type"] == "PathchSageAttentionKJ")

    # Walk back from the attention patch through the LoRA chain to the UNET.
    seen: list[int] = []
    current = link_by_id[sage["inputs"][0]["link"]][1]
    while nodes[current]["type"] == "LoraLoaderModelOnly":
        seen.append(current)
        current = link_by_id[nodes[current]["inputs"][0]["link"]][1]
    assert current == unet["id"]
    assert sorted(seen) == sorted(n["id"] for n in loras)


def test_scheduler_still_reads_the_unpatched_unet():
    # Matches Minimax_h3_4_step_lora.json: sigmas come from the raw UNET, not
    # from the LoRA-patched model.
    wf = _load()
    nodes = _by_id(wf)
    link_by_id = {l[0]: l for l in wf["links"]}
    scheduler = next(n for n in wf["nodes"] if n["type"] == "BasicScheduler")
    model_link = next(i for i in scheduler["inputs"] if i["name"] == "model")["link"]
    assert nodes[link_by_id[model_link][1]]["type"] == "UNETLoader"


def test_lora_filenames_match_the_setup_script():
    wf = _load()
    setup = os.path.join(
        _REPO_ROOT, "workflows", "setup", "minimax-h3-r2v-style-lora.sh"
    )
    with open(setup, encoding="utf-8") as fh:
        script = fh.read()
    for node in wf["nodes"]:
        if node["type"] != "LoraLoaderModelOnly":
            continue
        filename, strength = node["widgets_values"][:2]
        assert filename in script, f"{filename} is not installed by the setup script"
        assert 0.0 <= float(strength) <= 1.5


# --- end-to-end: real graph -> API prompt -> preset applied ------------------

# Comfy-core input orders for the nodes that carry widgets. Without these the
# UI->API widget mapping silently produces null lora_name values.
_OBJECT_INPUTS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["model", "lora_name", "strength_model"],
    "BasicScheduler": ["model", "scheduler", "steps", "denoise"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoadImage": ["image", "upload"],
    "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed"],
    "SaveVideo": ["video", "filename_prefix", "format", "codec"],
    "MiniMaxH3ReferenceToVideo": [
        "clip", "vae", "audio_vae", "prompt", "width", "height", "length",
        "ref_image_size",
    ],
}


def _as_api_prompt() -> dict:
    from tools.minimax_workflow import (
        DECORATIVE,
        simplify_minimax_graph,
        ui_workflow_to_api,
    )

    wf = _load()
    object_info = {}
    for node in wf["nodes"]:
        if node["type"] in DECORATIVE:
            continue
        names = _OBJECT_INPUTS.get(node["type"], [])
        object_info.setdefault(
            node["type"], {"input": {"required": {k: [] for k in names}, "optional": {}}}
        )
    api = ui_workflow_to_api(wf, object_info)
    simplify_minimax_graph(api, object_info)
    return api


def _chain(api: dict) -> list[tuple[str, float]]:
    return [
        (n["inputs"]["lora_name"], n["inputs"]["strength_model"])
        for n in api.values()
        if n["class_type"] == "LoraLoaderModelOnly"
    ]


def test_shipped_graph_converts_with_resolvable_lora_widgets():
    # A null lora_name here means the widget mapping broke, which ComfyUI would
    # only report at queue time.
    api = _as_api_prompt()
    names = [name for name, _ in _chain(api)]
    assert len(names) == 3
    assert all(isinstance(n, str) and n.endswith(".safetensors") for n in names)


def test_preset_overrides_the_graphs_baked_in_loras():
    from tools.minimax_workflow import apply_style_preset
    from tools.style_presets import resolve_lora_stack

    api = _as_api_prompt()
    assert _chain(api), "fixture should start with baked-in LoRAs"

    apply_style_preset(api, resolve_lora_stack("none"))
    assert _chain(api) == [], "preset 'none' must strip the JSON's own LoRAs"


def test_preset_applied_to_the_real_graph_keeps_it_queueable():
    import json as _json

    from tools.minimax_workflow import apply_style_preset
    from tools.style_presets import TURBO_STEPS, resolve_lora_stack

    api = _as_api_prompt()
    apply_style_preset(
        api, resolve_lora_stack("p2", turbo=True), turbo_steps=TURBO_STEPS,
    )

    assert [name for name, _ in _chain(api)] == [
        "studio1939-strong.safetensors",
        "minimax_h3_looping_sketch_anime_v1.safetensors",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    ]
    scheduler = next(n for n in api.values() if n["class_type"] == "BasicScheduler")
    assert api[scheduler["inputs"]["model"][0]]["class_type"] == "UNETLoader"
    assert scheduler["inputs"]["steps"] == TURBO_STEPS

    _json.dumps(api)  # must stay serializable for POST /prompt
    for nid, node in api.items():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in api, f"{nid} links to pruned node {value[0]}"
