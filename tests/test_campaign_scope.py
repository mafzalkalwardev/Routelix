"""Owner limits: one Google Voice line, English and Spanish, one B2B example."""
import os

from src.campaign_scripts import STRUCTURE, TEMPLATES, objection_reply, preview_script
from src.capacity import (
    AGENT_RSS_MB,
    capacity_report,
    google_voice_concurrent_calls,
    theoretical_agents,
)
from src.voice_catalog import CATALOG, language_of_voice, voices_for


def test_one_google_voice_call_and_ram_budget():
    assert google_voice_concurrent_calls() == 1
    agents = theoretical_agents(24)
    assert agents >= 8
    assert agents < 30
    report = capacity_report(24)
    assert "1 live call" in report
    assert str(agents) in report
    assert AGENT_RSS_MB > 500


def test_voice_catalog_is_english_and_spanish_only():
    languages = {item.language for item in CATALOG}
    assert languages == {"en", "es"}
    ids = {item.voice_id for item in CATALOG}
    assert "ff_siwis" not in ids
    assert "fr_FR" not in " ".join(ids)
    assert "ef_dora" in ids and "em_alex" in ids
    assert "es_MX-ald-medium" in ids
    assert "es_ES-davefx-medium" in ids
    assert len(voices_for("en", "piper")) >= 3
    assert len(voices_for("es", "edge")) >= 3
    assert language_of_voice("es-MX-DaliaNeural") == "es"
    try:
        voices_for("fr")
    except ValueError as exc:
        assert "English and Spanish" in str(exc)
    else:
        raise AssertionError("French was accepted")


def test_only_generic_b2b_appointment_templates():
    assert set(TEMPLATES) == {"b2b-appointment", "b2b-appointment-es"}
    rendered = preview_script()
    assert list(rendered) == list(STRUCTURE)
    assert "Thursday" in rendered["close"]
    assert "Indus" not in " ".join(rendered.values())
    spanish = preview_script("b2b-appointment-es", agent="Lucía", company="Taller Norte", day="jueves")
    assert "Lucía" in spanish["opener"]
    assert objection_reply("please stop calling", "en").startswith("I will mark")
    assert objection_reply("no me interesa", "es")


def test_spanish_prompt_is_opt_in(monkeypatch):
    from src.conversation_agent import ConversationAgent

    monkeypatch.setenv("CALL_LANGUAGE", "en")
    agent = ConversationAgent.__new__(ConversationAgent)
    agent.agent_name = "Alex"
    agent.company_name = "Northwind"
    agent.company_website = ""
    agent.callback_number = "+15551212"
    agent.company_context = "A studio."
    agent.contact_name = "Sam"
    from src.dispatcher_intelligence import DispatcherConversationState
    agent.state = DispatcherConversationState()
    english = agent._build_system_prompt()
    assert "Speak only Spanish" not in english
    monkeypatch.setenv("CALL_LANGUAGE", "es")
    spanish = agent._build_system_prompt()
    assert "Speak only Spanish" in spanish
    os.environ.pop("CALL_LANGUAGE", None)
