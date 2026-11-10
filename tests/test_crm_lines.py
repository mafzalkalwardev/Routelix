"""Admin CRM, Ashley's seed, DNC, and one-account Chrome lines."""
import json

from src.auth import verify_password


def test_ashley_is_seeded_without_leaking_the_password(tmp_path, monkeypatch):
    users = tmp_path / "users.json"
    secret_file = tmp_path / "ashley.password"
    monkeypatch.setenv("ROUTELIX_USERS_FILE", str(users))
    monkeypatch.setenv("ROUTELIX_ASHLEY_PASSWORD_FILE", str(secret_file))
    from src.auth import ashley_password_path, ensure_ashley, list_public_users

    ensure_ashley()
    ashley = next(row for row in list_public_users() if row["username"] == "Ashley")
    assert ashley["role"] == "agent"
    assert ashley["status"] == "active"
    assert ashley["script_id"] == "truck-dispatch-sales"
    assert "password" not in ashley
    stored = json.loads(users.read_text(encoding="utf-8"))["users"]
    record = next(row for row in stored if row["username"] == "Ashley")
    secret = ashley_password_path().read_text(encoding="utf-8").strip()
    assert verify_password(secret, record["password_hash"])
    assert secret_file.stat().st_mode & 0o077 == 0
    first = secret_file.read_bytes()
    ensure_ashley()
    assert secret_file.read_bytes() == first


def test_csv_import_keeps_dnc_and_scopes_the_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("src.lead_crm.leads_path", lambda: tmp_path / "leads.json")
    from src.lead_crm import filter_callable, import_csv, is_dnc, list_leads, update_lead, user_stats

    import_csv(
        "phone,name,company\n+13175550100,Sam,Northline\n+13175550199,Pat,Other\n",
        owner="Ashley",
        list_name="truck-dispatch",
    )
    import_csv("phone,name\n+13175550999,Blocked\n", owner="other.agent", list_name="other")
    ashley = list_leads(owner="Ashley")
    assert {row["name"] for row in ashley} == {"Sam", "Pat"}
    blocked = next(row for row in list_leads(include_all=True) if row["name"] == "Blocked")
    update_lead(blocked["id"], actor="admin", is_admin=True, status="dnc", notes="asked to stop", follow_up="")
    sam = next(row for row in ashley if row["name"] == "Sam")
    update_lead(sam["id"], actor="Ashley", is_admin=False, status="booked", notes="packet sent", follow_up="2026-11-12")
    try:
        update_lead(blocked["id"], actor="Ashley", is_admin=False, notes="nope")
    except PermissionError:
        pass
    else:
        raise AssertionError("Ashley edited another user's lead")
    assert is_dnc("+1 (317) 555-0999")
    kept, skipped = filter_callable([
        {"phone": "3175550100", "name": "Sam"},
        {"phone": "3175550999", "name": "Blocked"},
    ])
    assert skipped == 1
    assert kept[0]["name"] == "Sam"
    stats = user_stats("Ashley")
    assert stats["booked"] == 1
    assert stats["calls"] == 1


def test_lines_share_one_account_and_warn_past_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTELIX_LINES", "7")
    monkeypatch.setenv("ROUTELIX_DISPLAY", ":98")
    monkeypatch.delenv("ROUTELIX_SEPARATE_DISPLAYS", raising=False)
    monkeypatch.setenv("ROUTELIX_CPU_COUNT", "4")
    monkeypatch.setenv("ROUTELIX_RESERVED_CORES", "1")
    monkeypatch.setenv("ROUTELIX_RAM_GB", "24")
    monkeypatch.setenv("ROUTELIX_RESERVED_RAM_GB", "8")
    monkeypatch.setattr("src.gv_lines._profiles_root", lambda: tmp_path)
    (tmp_path / "tony").mkdir()
    (tmp_path / "tony" / "Cookie").write_text("keep", encoding="utf-8")
    from src.gv_lines import capacity_warning, ensure_profile, line_plan

    plan = line_plan()
    assert len(plan) == 7
    assert {row["gv_account"] for row in plan} == {"Main line"}
    assert len({row["profile_name"] for row in plan}) == 7
    assert len({row["loopback_device"] for row in plan}) == 7
    assert all(row["display"] == ":98" for row in plan)
    assert all(row["vnc_port"] not in {5901, 5902, 6080, 6081} for row in plan)
    assert capacity_warning(7)
    assert ensure_profile(2) == "tony_2"
    assert (tmp_path / "tony_2" / "Cookie").read_text(encoding="utf-8") == "keep"
    monkeypatch.setenv("ROUTELIX_SEPARATE_DISPLAYS", "1")
    assert line_plan(1)[0]["display"] == ":100"
