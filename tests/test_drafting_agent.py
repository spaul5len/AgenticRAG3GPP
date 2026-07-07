from rag import drafting_agent


def gap_package():
    return {
        "topic": "AKMA privacy",
        "gap_analysis": "\n".join(
            [
                "1. Officially covered areas: official coverage [Evidence 1]",
                "2. Meeting-discussed areas: proposal [Evidence 2]",
                "3. Potential gaps: Model inference: possible gap [Evidence 1]",
                "4. Strong contribution angles: study angle [Evidence 2]",
                "5. Weak or missing evidence: missing deployment data",
                "6. Recommended next documents to inspect: related TDocs",
                "7. Evidence map: Evidence 1 official, Evidence 2 meeting",
            ]
        ),
        "evidence": "\n\n".join(
            [
                "Evidence 1\ndoc_type: official_spec\nstatus: official\ntitle: TS 33.501",
                "Evidence 2\ndoc_type: meeting_doc\nstatus: proposed\ntdoc_id: S3-230001",
            ]
        ),
        "raw_results": [{"metadata": {"doc_type": "official_spec"}}],
    }


def test_draft_sa3_contribution_calls_analyze_gap_and_returns_package(monkeypatch):
    package = gap_package()
    calls = {}

    def fake_analyze_gap(topic):
        calls["topic"] = topic
        return package

    monkeypatch.setattr(drafting_agent, "analyze_gap", fake_analyze_gap)
    monkeypatch.setattr(
        drafting_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "\n".join(
            [
                "1. Title: AKMA privacy discussion",
                "2. Abstract: SA3 is invited to discuss the topic [Evidence 1].",
                "3. Background: Official evidence exists [Evidence 1].",
                "4. Problem statement: Gap may remain [Evidence 2].",
                "5. Discussion: The following aspects may be considered.",
                "6. Proposal: It is proposed to study privacy aspects.",
                "7. Questions for SA3: Does SA3 agree to study this?",
                "8. Possible normative direction: Proposed text may use shall.",
                "9. Evidence map: Evidence 1 official; Evidence 2 meeting.",
                "10. Weak claims / missing evidence: More evidence is needed.",
            ]
        ),
    )

    output = drafting_agent.draft_sa3_contribution("AKMA privacy")

    assert calls["topic"] == "AKMA privacy"
    assert output["topic"] == "AKMA privacy"
    assert output["draft_type"] == "discussion contribution"
    assert "SA3 is invited to discuss" in output["draft"]
    assert output["gap_package"] == package


def test_draft_prompt_enforces_cautious_sa3_rules(monkeypatch):
    monkeypatch.setattr(drafting_agent, "analyze_gap", lambda topic: gap_package())
    captured = {}

    def fake_call_local_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "draft [Evidence 1]"

    monkeypatch.setattr(drafting_agent.llm, "call_local_llm", fake_call_local_llm)

    drafting_agent.draft_sa3_contribution("AKMA privacy", draft_type="CR skeleton")

    assert "1. Title" in captured["prompt"]
    assert "10. Weak claims / missing evidence" in captured["prompt"]
    assert "Do not invent clause numbers" in captured["prompt"]
    assert "Do not invent TDoc IDs" in captured["prompt"]
    assert "Do not invent companies" in captured["prompt"]
    assert "Do not treat meeting docs as approved standard text" in captured["prompt"]
    assert "SA3 is invited to discuss" in captured["prompt"]
    assert "It is proposed to study" in captured["prompt"]
    assert "The following aspects may be considered" in captured["prompt"]
    assert 'Use "shall" only' in captured["prompt"]
    assert "Do not invent clause numbers" in captured["system_prompt"]
    assert "Do not invent TDoc IDs" in captured["system_prompt"]
    assert "Do not invent company names" in captured["system_prompt"]
    assert "Use cautious SA3-style language" in captured["system_prompt"]


def test_draft_sa3_contribution_defaults_blank_draft_type(monkeypatch):
    monkeypatch.setattr(drafting_agent, "analyze_gap", lambda topic: gap_package())
    monkeypatch.setattr(
        drafting_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "draft [Evidence 1]",
    )

    output = drafting_agent.draft_sa3_contribution("  topic  ", draft_type="   ")

    assert output["topic"] == "topic"
    assert output["draft_type"] == "discussion contribution"


def test_empty_llm_response_returns_clear_draft_failure(monkeypatch):
    monkeypatch.setattr(drafting_agent, "analyze_gap", lambda topic: gap_package())
    monkeypatch.setattr(
        drafting_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: "   ",
    )

    output = drafting_agent.draft_sa3_contribution("topic")

    assert "could not be generated" in output["draft"]
