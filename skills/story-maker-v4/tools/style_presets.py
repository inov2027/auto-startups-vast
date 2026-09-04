"""Illustration style presets for the MiniMax H3 LoRA stack.

The style LoRA is part of an episode's continuity contract, so it must be
selected once and applied identically to every generation. Hand-editing
``LoraLoaderModelOnly`` widgets in the workflow JSON cannot give that
guarantee — this module turns a preset name into an exact, ordered LoRA list
that :func:`tools.minimax_workflow.apply_style_preset` materializes into the
graph.

Only H3-native adapters appear here. A Flux / SDXL / Qwen-Image LoRA does not
load into the H3 DiT; see
``skills/workflow-researcher/references/minimax-h3-style-lora-discovery-2026-09-04.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Filenames as installed by workflows/setup/minimax-h3-r2v-style-lora.sh.
STUDIO_1939_LIGHT = "studio1939-light.safetensors"
STUDIO_1939_STRONG = "studio1939-strong.safetensors"
SKETCH_ANIME = "minimax_h3_looping_sketch_anime_v1.safetensors"
PAINTERLY = "h3_painterly.safetensors"
ANIME_FLAT = "h3_anime_flat_style.safetensors"
TURBO_REF2V_4STEP = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
CAMERA_MOTION = "h3_camera_motion_v1_3000_pruned.safetensors"
SPATIAL_PHYSICS = "h3_spatial_physics_clean_3000_pruned.safetensors"

# Community-mirror weights (EllaPriest45/MinimaxH3_Styles) are only installed
# when the setup script runs with STYLE_LORA_COMMUNITY=1.
COMMUNITY_MIRROR_LORAS = frozenset({PAINTERLY, ANIME_FLAT})

# Above this, H3 stops honouring the storyboard sheet's composition.
MAX_STYLE_STRENGTH = 1.2

# Turbo replaces the sampling schedule, not the style.
TURBO_STEPS = 4


@dataclass(frozen=True)
class StylePreset:
    preset_id: str
    title: str
    loras: tuple[tuple[str, float], ...]
    prompt_spine: str

    @property
    def style_strength(self) -> float:
        return round(sum(strength for _, strength in self.loras), 4)

    @property
    def needs_community_loras(self) -> bool:
        return any(name in COMMUNITY_MIRROR_LORAS for name, _ in self.loras)


_CONCEPT_ART_SPINE = (
    "semi-realistic fantasy character concept art, painterly digital rendering, "
    "visible brushwork, anime-influenced facial structure with large expressive "
    "eyes, rim light and volumetric haze, muted desaturated fantasy palette, "
    "material-accurate leather and metal"
)

_PRESETS: dict[str, StylePreset] = {
    "none": StylePreset(
        preset_id="none",
        title="No style LoRA (base H3)",
        loras=(),
        prompt_spine="",
    ),
    "p1": StylePreset(
        preset_id="p1",
        title="2D storybook illustration",
        loras=((STUDIO_1939_LIGHT, 0.85),),
        prompt_spine=(
            "hand-painted 2D storybook illustration, gouache texture on paper, "
            "soft warm palette, hand-inked contour lines, flat lighting, "
            "gentle held poses"
        ),
    ),
    "p2": StylePreset(
        preset_id="p2",
        title="Folk illustration + flat geometric shapes",
        # Slot 2 supplies the visible outline; above ~0.5 it turns the frame
        # into a sketch and destroys the flat-shape read.
        loras=((STUDIO_1939_STRONG, 0.9), (SKETCH_ANIME, 0.3)),
        prompt_spine=(
            "folk-art illustration, flat geometric shapes, bold simplified "
            "silhouettes, three-colour limited palette, decorative repeating "
            "pattern, screen-print flatness, no gradients, no rendered volume"
        ),
    ),
    "p3": StylePreset(
        preset_id="p3",
        title="Vintage editorial / storybook",
        loras=((STUDIO_1939_LIGHT, 0.7), (SKETCH_ANIME, 0.3)),
        prompt_spine=(
            "vintage editorial illustration, 1950s print, muted ochre and teal "
            "ink, visible halftone grain and misregistered plates, textured "
            "paper, graphic cropped composition"
        ),
    ),
    "p4": StylePreset(
        preset_id="p4",
        title="Semi-realistic fantasy concept art (painterly, anime-influenced)",
        loras=((PAINTERLY, 0.8), (ANIME_FLAT, 0.3)),
        prompt_spine=_CONCEPT_ART_SPINE,
    ),
    "p4-lite": StylePreset(
        preset_id="p4-lite",
        title="Semi-realistic painterly concept art (no community LoRAs)",
        loras=((STUDIO_1939_LIGHT, 0.5),),
        prompt_spine=_CONCEPT_ART_SPINE,
    ),
}

PRESET_IDS = tuple(_PRESETS)


def get_preset(preset_id: str | None) -> StylePreset:
    """Resolve a preset id (case/space tolerant). Unknown ids raise."""
    key = (preset_id or "none").strip().lower()
    if key not in _PRESETS:
        raise ValueError(
            f"unknown style preset {preset_id!r}; use one of {', '.join(PRESET_IDS)}"
        )
    return _PRESETS[key]


def parse_extra_loras(spec: str | None) -> list[tuple[str, float]]:
    """Parse ``"file.safetensors:0.7, other.safetensors:0.5"``.

    Used for the directing LoRAs (camera motion, spatial physics) which are
    per-scene decisions rather than part of the episode's style contract.
    """
    entries: list[tuple[str, float]] = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, raw_strength = chunk.rpartition(":")
        if not sep:
            raise ValueError(
                f"invalid LoRA spec {chunk!r}; expected '<filename>:<strength>'"
            )
        name = name.strip()
        if not name:
            raise ValueError(f"invalid LoRA spec {chunk!r}; filename is empty")
        try:
            strength = float(raw_strength)
        except ValueError as exc:
            raise ValueError(
                f"invalid LoRA strength in {chunk!r}: {raw_strength!r}"
            ) from exc
        entries.append((name, strength))
    return entries


def resolve_lora_stack(
    preset_id: str | None,
    *,
    turbo: bool = False,
    extra: str | None = None,
) -> list[tuple[str, float]]:
    """Build the final ordered LoRA list for a render.

    Style LoRAs come first (they define the look), then any directing LoRAs,
    then turbo last so acceleration is applied on top of the styled model.
    Zero-strength entries are dropped — an unused loader still costs a load.
    """
    preset = get_preset(preset_id)
    stack: list[tuple[str, float]] = [
        (name, strength) for name, strength in preset.loras if strength > 0
    ]

    style_total = sum(strength for _, strength in stack)
    if style_total > MAX_STYLE_STRENGTH + 1e-9:
        raise ValueError(
            f"style preset {preset.preset_id!r} totals {style_total:.2f} strength; "
            f"above {MAX_STYLE_STRENGTH} H3 stops honouring the storyboard sheet"
        )

    for name, strength in parse_extra_loras(extra):
        if strength <= 0:
            continue
        stack.append((name, strength))

    if turbo:
        stack.append((TURBO_REF2V_4STEP, 1.0))

    seen: set[str] = set()
    for name, _ in stack:
        if name in seen:
            raise ValueError(f"LoRA {name!r} is listed twice in the stack")
        seen.add(name)
    return stack
