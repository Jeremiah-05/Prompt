# -*- coding: utf-8 -*-
"""
eval/test_prompt_versions.py
Unit tests verifying prompt version dropdown choices, template loading,
alias resolution for V4, and continued operation of V5 and V6.
"""

import pytest
from prompts.builder import (
    list_versions,
    load_prompt_template,
    load_prompt_frontmatter,
    build_prompt,
    get_recommended_temp,
    PROMPT_VERSIONS,
    LEGACY_PROMPT_VERSIONS,
)


EXPECTED_VERSIONS = [
    "V1 — Naive",
    "V2 — Role + Rules",
    "V3 — Few-Shot",
    "V4 — Chain-of-Thought (CoT)",
    "V5 — ReAct",
    "V6 — RAG + ReAct",
]


def test_list_versions_exact_six_options():
    """Verify the prompt version choices contain exactly the expected 6 versions in order."""
    versions = list_versions()
    assert versions == EXPECTED_VERSIONS, (
        f"Expected versions {EXPECTED_VERSIONS}, but got {versions}"
    )
    assert len(versions) == 6, f"Expected exactly 6 versions, got {len(versions)}"
    assert "V4 — CoT" not in versions, "'V4 — CoT' should be replaced by 'V4 — Chain-of-Thought (CoT)'"
    assert "V4 — Chain-of-Thought" not in versions, "'V4 — Chain-of-Thought' should be replaced"
    assert "V4 — Chain-of-Thought (CoT)" in versions


def test_v4_template_loads_cot_content():
    """Verify selecting V4 loads the correct Chain-of-Thought prompt template."""
    v4_template = load_prompt_template("V4 — Chain-of-Thought (CoT)")
    assert v4_template, "V4 template must not be empty"
    # Verify Chain-of-Thought decomposition markers and examples
    assert "think step-by-step" in v4_template.lower()
    assert "Thought:" in v4_template
    assert "Action:" in v4_template
    assert "Action Input:" in v4_template
    assert "{user_message}" in v4_template


def test_legacy_v4_labels_point_to_same_existing_template():
    """Verify both legacy V4 labels point to the same existing V4 template without creating a new version."""
    unified_template = load_prompt_template("V4 — Chain-of-Thought (CoT)")
    cot_template = load_prompt_template("V4 — CoT")
    chain_of_thought_template = load_prompt_template("V4 — Chain-of-Thought")

    assert cot_template == unified_template, "'V4 — CoT' must load the exact same template as 'V4 — Chain-of-Thought (CoT)'"
    assert chain_of_thought_template == unified_template, "'V4 — Chain-of-Thought' must load the exact same template as 'V4 — Chain-of-Thought (CoT)'"


def test_v5_and_v6_continue_working():
    """Verify V5 (ReAct) and V6 (RAG + ReAct) prompt templates continue working correctly."""
    v5_template = load_prompt_template("V5 — ReAct")
    assert v5_template, "V5 template must load successfully"
    assert "Observation:" in v5_template
    assert "{chat_history}" in v5_template
    assert "{user_message}" in v5_template

    v6_template = load_prompt_template("V6 — RAG + ReAct")
    assert v6_template, "V6 template must load successfully"
    assert "{rag_context}" in v6_template
    assert "Observation:" in v6_template


def test_build_prompt_v4_v5_v6():
    """Verify build_prompt works correctly for V4, V5, and V6 with context injection."""
    user_msg = "Can I return order ORD1001?"

    # V4
    p4 = build_prompt(
        version="V4 — Chain-of-Thought (CoT)",
        user_message=user_msg,
    )
    assert user_msg in p4
    assert "think step-by-step" in p4.lower()

    # V5
    p5 = build_prompt(
        version="V5 — ReAct",
        user_message=user_msg,
        chat_history=[{"role": "user", "content": "Hello"}],
    )
    assert user_msg in p5
    assert "User: Hello" in p5

    # V6
    p6 = build_prompt(
        version="V6 — RAG + ReAct",
        user_message=user_msg,
        rag_context="30-day return policy for general items",
    )
    assert user_msg in p6
    assert "30-day return policy for general items" in p6


def test_v4_frontmatter_and_temperature():
    """Verify metadata and recommended temperature for V4 and its legacy aliases."""
    fm = load_prompt_frontmatter("V4 — Chain-of-Thought (CoT)")
    assert "technique" in fm
    assert "Chain-of-Thought" in fm.get("technique", "")

    # Legacy aliases also resolve frontmatter
    fm_legacy1 = load_prompt_frontmatter("V4 — CoT")
    fm_legacy2 = load_prompt_frontmatter("V4 — Chain-of-Thought")
    assert fm_legacy1 == fm
    assert fm_legacy2 == fm

    # Temperature
    assert get_recommended_temp("V4 — Chain-of-Thought (CoT)") == 0.3
    assert get_recommended_temp("V4 — CoT") == 0.3
    assert get_recommended_temp("V4 — Chain-of-Thought") == 0.3
    assert get_recommended_temp("V5 — ReAct") == 0.3
    assert get_recommended_temp("V6 — RAG + ReAct") == 0.3


def test_app_version_intel_and_card():
    """Verify app.py VERSION_INTEL and render_version_card work with the new unified option."""
    from app import VERSION_INTEL, render_version_card

    assert "V4 — Chain-of-Thought (CoT)" in VERSION_INTEL
    card_html = render_version_card("V4 — Chain-of-Thought (CoT)")
    assert "Chain-of-Thought" in card_html
    assert "Step-by-Step" in card_html

    # Legacy lookup should also not crash or fall back to default
    card_legacy = render_version_card("V4 — CoT")
    assert "Chain-of-Thought" in card_legacy
