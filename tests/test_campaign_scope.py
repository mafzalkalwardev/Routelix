"""Owner limits: one Google Voice line, English and Spanish, one B2B example."""
import os

from src.campaign_scripts import STRUCTURE, TEMPLATES, objection_reply, preview_script
from src.capacity import capacity_report, google_voice_concurrent_calls
from src.voice_catalog import CATALOG, language_of_voice, voices_for


def test_one_google_voice_call_and_cpu_is_the_limit(monkeypatch):
    assert google_voice_concurrent_calls() == 1
    monkeypatch.setenv("ROUTELIX_RAM_GB", "24")
    monkeypatch.setenv("ROUTELIX_CPU_COUNT", "4")
    monkeypatch.setenv("ROUTELIX_RESERVED_RAM_GB", "8")
    monkeypatch.setenv("ROUTELIX_RESERVED_CORES", "1")
    monkeypatch.setenv("MAX_AGENTS", "auto")
    monkeypatch.setenv("STT_LOCAL_MODEL", "tiny")
    from src.capacity import cpu_agent_limit, max_agents, ram_agent_limit

    assert ram_agent_limit() > cpu_agent_limit()
    assert max_agents() == cpu_agent_limit()
    assert max_agents() == 4
    report = capacity_report()
    assert "limiter is cpu" in report
    assert "4" in report


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


def test_fleet_admits_up_to_max_and_queues_the_rest(monkeypatch):
    monkeypatch.setenv("MAX_AGENTS", "2")
    monkeypatch.setenv("ROUTELIX_CPU_COUNT", "4")
    monkeypatch.setenv("ROUTELIX_RESERVED_CORES", "1")
    monkeypatch.setenv("ROUTELIX_RAM_GB", "24")
    monkeypatch.setenv("ROUTELIX_RESERVED_RAM_GB", "8")
    from src.agent_fleet import Fleet

    accounts = [
        {"id": str(i), "email": f"a{i}@example.com", "profile_name": f"p{i}", "enabled": True}
        for i in range(1, 5)
    ]
    plan = Fleet().plan(accounts)
    assert len(plan.slots) == 2
    assert plan.slots[0].loopback_device == "routelix_to_gv_1"
    assert plan.slots[1].capture_device == "routelix_from_gv_2.monitor"
    assert plan.slots[0].pulse_source == "routelix_to_gv_1.monitor"
    assert len(plan.queued_accounts) == 2
    assert plan.slots[0].active_calls == 0


def test_account_pacer_enforces_gap_and_hourly_cap(monkeypatch):
    monkeypatch.setenv("ACCOUNT_MIN_GAP_SECONDS", "45")
    monkeypatch.setenv("ACCOUNT_MAX_CALLS_PER_HOUR", "2")
    from src.account_pacing import AccountPacer

    pacer = AccountPacer()
    assert pacer.allow("line", now=1000)[0] is True
    pacer.mark("line", now=1000)
    ok, reason = pacer.allow("line", now=1010)
    assert ok is False
    assert "wait" in reason
    pacer.mark("line", now=1100)
    ok, reason = pacer.allow("line", now=1200)
    assert ok is False
    assert "hour" in reason


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
