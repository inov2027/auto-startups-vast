"""Unit tests for style preset resolution and LoRA-chain materialization."""

from __future__ import annotations

import pytest

from tools import style_presets as sp
from tools.minimax_workflow import apply_style_preset


# --- preset resolution -------------------------------------------------------

def test_every_preset_stays_under_the_reference_adherence_ceiling():
    for pid in sp.PRESET_IDS:
        preset = sp.get_preset(pid)
        assert preset.style_strength <= sp.MAX_STYLE_STRENGTH, pid


def test_unknown_preset_is_rejected_with_the_valid_ids():
    with pytest.raises(ValueError) as exc:
        sp.get_preset("p9")
    assert "p1" in str(exc.value)


def test_preset_lookup_is_case_and_space_tolerant():
    assert sp.get_preset("  P2 ").preset_id == "p2"


def test_none_preset_produces_an_empty_stack():
    assert sp.resolve_lora_stack("none") == []
    assert sp.resolve_lora_stack(None) == []


def test_turbo_is_appended_last_so_it_patches_the_styled_model():
    stack = sp.resolve_lora_stack("p1", turbo=True)
    assert stack[0][0] == sp.STUDIO_1939_LIGHT
    assert stack[-1] == (sp.TURBO_REF2V_4STEP, 1.0)


def test_zero_strength_entries_are_dropped():
    stack = sp.resolve_lora_stack("p1", extra="whatever.safetensors:0")
    assert all(strength > 0 for _, strength in stack)
    assert all(name != "whatever.safetensors" for name, _ in stack)


def test_duplicate_lora_in_stack_is_rejected():
    with pytest.raises(ValueError, match="twice"):
        sp.resolve_lora_stack("p1", extra=f"{sp.STUDIO_1939_LIGHT}:0.4")


def test_p4_is_flagged_as_needing_the_community_mirror_loras():
    assert sp.get_preset("p4").needs_community_loras
    assert not sp.get_preset("p4-lite").needs_community_loras
    assert not sp.get_preset("p1").needs_community_loras


def test_p4_lite_matches_p4_prompt_spine():
    # The fallback has to carry the look in the prompt instead of the weights.
    assert sp.get_preset("p4-lite").prompt_spine == sp.get_preset("p4").prompt_spine


# --- extra LoRA spec parsing -------------------------------------------------

def test_parse_extra_loras_roundtrip():
    assert sp.parse_extra_loras("a.safetensors:0.8, b.safetensors:0.5") == [
        ("a.safetensors", 0.8),
        ("b.safetensors", 0.5),
    ]
    assert sp.parse_extra_loras("") == []
    assert sp.parse_extra_loras(None) == []


def test_parse_extra_loras_rejects_malformed_specs():
    with pytest.raises(ValueError, match="expected"):
        sp.parse_extra_loras("a.safetensors")
    with pytest.raises(ValueError, match="strength"):
        sp.parse_extra_loras("a.safetensors:heavy")
    with pytest.raises(ValueError, match="empty"):
        sp.parse_extra_loras(":0.5")


# --- graph materialization ---------------------------------------------------

def _base_api() -> dict[str, dict]:
    """Minimal stand-in for the simplified H3 API graph."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
        "2": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "steps": 20}},
        "3": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["1", 0]}},
        "4": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0]}},
    }


def test_apply_style_preset_inserts_chain_on_the_guider_branch_only():
    api = _base_api()
    created = apply_style_preset(api, sp.resolve_lora_stack("p2"))

    assert len(created) == 2
    # Scheduler still reads the raw UNET.
    assert api["2"]["inputs"]["model"] == ["1", 0]
    # Attention patch now reads the end of the LoRA chain.
    assert api["3"]["inputs"]["model"] == [created[-1], 0]
    assert api[created[0]]["inputs"]["model"] == ["1", 0]
    assert api[created[1]]["inputs"]["model"] == [created[0], 0]


def test_apply_style_preset_writes_name_and_strength_in_order():
    api = _base_api()
    stack = sp.resolve_lora_stack("p2")
    created = apply_style_preset(api, stack)
    for nid, (name, strength) in zip(created, stack):
        assert api[nid]["class_type"] == "LoraLoaderModelOnly"
        assert api[nid]["inputs"]["lora_name"] == name
        assert api[nid]["inputs"]["strength_model"] == pytest.approx(strength)


def test_apply_style_preset_is_idempotent_and_replaces_a_previous_chain():
    api = _base_api()
    apply_style_preset(api, sp.resolve_lora_stack("p2"))
    apply_style_preset(api, sp.resolve_lora_stack("p1"))

    loras = [n for n in api.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert len(loras) == 1
    assert loras[0]["inputs"]["lora_name"] == sp.STUDIO_1939_LIGHT
    assert api["2"]["inputs"]["model"] == ["1", 0]


def test_empty_stack_restores_the_unpatched_graph():
    api = _base_api()
    apply_style_preset(api, sp.resolve_lora_stack("p2"))
    assert apply_style_preset(api, []) == []

    assert not [n for n in api.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert api["3"]["inputs"]["model"] == ["1", 0]
    assert api["2"]["inputs"]["model"] == ["1", 0]


def test_turbo_drops_the_scheduler_to_four_steps():
    api = _base_api()
    apply_style_preset(
        api, sp.resolve_lora_stack("p1", turbo=True), turbo_steps=sp.TURBO_STEPS,
    )
    assert api["2"]["inputs"]["steps"] == sp.TURBO_STEPS


def test_chain_works_when_the_attention_patch_was_pruned():
    # ComfyUI hosts without ComfyUI-KJNodes lose PathchSageAttentionKJ; the
    # guider then reads the UNET directly and must still get the LoRA chain.
    api = _base_api()
    api["4"]["inputs"]["model"] = ["1", 0]
    del api["3"]

    created = apply_style_preset(api, sp.resolve_lora_stack("p1"))
    assert api["4"]["inputs"]["model"] == [created[-1], 0]
    assert api["2"]["inputs"]["model"] == ["1", 0]
