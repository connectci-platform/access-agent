"""Tests for the READ_ONLY write-tool deny-list drift guard (issue #200)."""

from unittest.mock import patch

import pytest

from src.writeguard import looks_like_write
from src.writeguard.__main__ import main_argv
from src.writeguard.check import check_drift


class TestLooksLikeWrite:
    def test_mutating_verbs_are_flagged(self):
        for name in (
            "create_event",
            "update_announcement",
            "delete_event",
            "cancel_registration",
            "register_for_event",
            "edit_occurrence",
            "add_occurrence",
            "send_for_review",
            "restore_event",
            "report_security_incident",
        ):
            assert looks_like_write(name), name

    def test_read_verbs_are_not_flagged(self):
        for name in (
            "search_events",
            "get_chart_data",
            "list_all_software",
            "describe_realms",
            "compare_software_availability",
            "analyze_funding",
            "suggest_tags",
            "get_my_registrations",
        ):
            assert not looks_like_write(name), name


class TestCheckDrift:
    def test_clean_when_every_suspected_write_is_denied(self):
        report = check_drift(
            {"create_event", "search_events", "get_chart_data"},
            frozenset({"create_event"}),
        )
        assert report.ok
        assert report.unguarded == ()

    def test_flags_a_write_missing_from_the_deny_list(self):
        """The fail-open case this guard exists to catch: a newly shipped write
        tool that nothing strips under READ_ONLY."""
        report = check_drift(
            {"create_event", "publish_dataset", "search_events"},
            frozenset({"create_event"}),
        )
        assert not report.ok
        assert report.unguarded == ("publish_dataset",)

    def test_reports_stale_entries_without_failing(self):
        """A deny-listed tool absent from the catalog is informational — it is
        harmless to strip a tool that does not exist."""
        report = check_drift(
            {"search_events"},
            frozenset({"create_event"}),
        )
        assert report.ok
        assert report.stale == ("create_event",)

    def test_waived_exception_is_not_flagged_and_is_reported(self):
        with patch.dict(
            "src.writeguard.HEURISTIC_EXCEPTIONS",
            {"add_bookmark": "client-side only, mutates nothing server-side"},
            clear=True,
        ):
            report = check_drift({"add_bookmark", "search_events"}, frozenset())
        assert report.ok
        assert report.waived == ("add_bookmark",)
        assert report.unguarded == ()

    def test_read_only_tools_never_require_deny_listing(self):
        report = check_drift({"search_events", "get_chart_data"}, frozenset())
        assert report.ok


def _catalog_with(tool_names, servers=("announcements", "events", "jsm")):
    """A catalog carrying every write-owning server as available.

    The blindness guards require all write-owning servers to be present and
    available before the drift check runs, so CLI tests must supply them.
    """
    entries = [{"server": s, "status": "available", "tools": []} for s in servers]
    entries[0]["tools"] = [{"name": n} for n in tool_names]
    return {"servers": entries}


class TestCli:
    def test_exits_zero_when_in_sync(self, capsys):
        async def _fetch():
            return _catalog_with({"create_event", "search_events"})

        with (
            patch("src.writeguard.__main__.WRITE_MCP_TOOL_NAMES", frozenset({"create_event"})),
            pytest.raises(SystemExit) as exc,
        ):
            main_argv(_fetch=_fetch)

        assert exc.value.code == 0
        assert "OK:" in capsys.readouterr().out

    def test_exits_nonzero_and_names_the_unguarded_tool(self, capsys):
        async def _fetch():
            return _catalog_with({"create_event", "publish_dataset"})

        with (
            patch("src.writeguard.__main__.WRITE_MCP_TOOL_NAMES", frozenset({"create_event"})),
            pytest.raises(SystemExit) as exc,
        ):
            main_argv(_fetch=_fetch)

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "UNGUARDED: publish_dataset" in out
        assert "WRITE_MCP_TOOL_NAMES" in out  # tells the operator where to fix it

    def test_reports_waived_and_stale_lines(self, capsys):
        """Waivers and stale entries are surfaced, not silently swallowed: a
        waiver that stops applying, or a renamed tool, should be visible."""

        async def _fetch():
            return _catalog_with({"add_bookmark", "search_events"})

        with (
            patch.dict(
                "src.writeguard.HEURISTIC_EXCEPTIONS",
                {"add_bookmark": "client-side only"},
                clear=True,
            ),
            patch("src.writeguard.__main__.WRITE_MCP_TOOL_NAMES", frozenset({"delete_gone"})),
            pytest.raises(SystemExit) as exc,
        ):
            main_argv(_fetch=_fetch)

        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "WAIVED: add_bookmark — client-side only" in out
        assert "STALE: delete_gone" in out

    def test_main_entrypoint_delegates(self):
        """`python -m src.writeguard` path."""
        from src.writeguard.__main__ import main

        with patch("src.writeguard.__main__.main_argv") as m:
            main()
        m.assert_called_once_with()

    def test_empty_catalog_fails_rather_than_reporting_green(self, capsys):
        """A vacuous pass would be worse than a failure: it reports the
        deny-list as verified when nothing was checked."""

        async def _fetch():
            return {"servers": []}

        with pytest.raises(SystemExit) as exc:
            main_argv(_fetch=_fetch)

        assert exc.value.code == 1
        assert "cannot verify" in capsys.readouterr().out

    def test_fetch_failure_exits_nonzero_without_traceback(self, capsys):
        async def _fetch():
            raise RuntimeError("MCP unreachable")

        with pytest.raises(SystemExit) as exc:
            main_argv(_fetch=_fetch)

        assert exc.value.code == 1
        assert "failed to fetch the catalog" in capsys.readouterr().out


class TestAgainstRealDenyList:
    def test_every_known_write_tool_matches_the_heuristic(self):
        """If a real deny-list entry stops matching, the heuristic has a hole
        and would no longer suspect that tool's future siblings.

        Note this is a weak test by construction — the observed prefixes were
        derived from these same names. It guards against a rename breaking the
        match, not against the heuristic's real blind spots (see
        test_documents_known_heuristic_blind_spots)."""
        from src.agent.domains.capabilities import WRITE_MCP_TOOL_NAMES

        missed = sorted(n for n in WRITE_MCP_TOOL_NAMES if not looks_like_write(n))
        assert not missed, f"deny-listed writes the heuristic misses: {missed}"

    def test_documents_known_heuristic_blind_spots(self):
        """Pin the limitation honestly: these ARE writes and the heuristic does
        NOT catch them, so a green guard run is not proof of safety. If someone
        widens WRITE_NAME_PREFIXES to cover one, delete it from this list."""
        blind_spots = ("event_create", "eventCreate", "allocationRequest", "do_the_thing")
        for name in blind_spots:
            assert not looks_like_write(name), (
                f"{name} is now caught — remove it from this list and celebrate"
            )

    def test_servers_owning_write_tools_covers_the_capability_registry(self):
        """SERVERS_OWNING_WRITE_TOOLS is hand-maintained, so derive the real
        owners from the capability registry and require the map to cover them.

        If a write capability ships on a new server and the map is not updated,
        the guard would not notice that server going missing from the catalog —
        re-opening the blindness this map exists to close.
        """
        from src.agent.domains.capabilities import (
            WRITE_CAPABILITY_IDS,
            get_capability_registry,
        )
        from src.writeguard import SERVERS_OWNING_WRITE_TOOLS

        registry = get_capability_registry()
        owners: set[str] = set()
        for cap_id in WRITE_CAPABILITY_IDS:
            cap = registry.get_by_id(cap_id)
            backend = getattr(cap, "backend", None) if cap else None
            owners.update(getattr(backend, "servers", ()) or ())

        # Registry-derived owners are announcements + jsm. The events organizer
        # tools are deliberately NOT registry-owned (see the deny-list comment
        # in capabilities.py), so "events" is map-only and cannot be derived —
        # which is precisely why the map is hand-maintained and pinned here.
        assert owners, "expected write capabilities to declare MCP servers"
        missing = sorted(owners - SERVERS_OWNING_WRITE_TOOLS)
        assert not missing, (
            f"servers owning write capabilities but absent from "
            f"SERVERS_OWNING_WRITE_TOOLS: {missing}"
        )
        assert "events" in SERVERS_OWNING_WRITE_TOOLS, (
            "events hosts registry-less organizer write tools; it must stay mapped"
        )


class TestBlindnessGuards:
    """The guard must refuse to report green when it could not actually see
    the tools it claims to have checked. A missing tool and a deny-listed tool
    are indistinguishable to a name check."""

    def _catalog(self, servers):
        return {"servers": servers}

    def test_unreachable_server_fails_instead_of_reporting_ok(self, capsys):
        """fetch_catalog does not fail on a per-server error — it records
        status=unavailable with an empty tool list. Without this check the
        guard printed OK with most of the deny-list unverified."""

        async def _fetch():
            return self._catalog(
                [
                    {"server": "announcements", "status": "available", "tools": []},
                    {"server": "events", "status": "unavailable", "tools": []},
                    {"server": "jsm", "status": "available", "tools": [{"name": "search_x"}]},
                ]
            )

        with pytest.raises(SystemExit) as exc:
            main_argv(_fetch=_fetch)

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "unreachable" in out
        assert "events" in out
        assert "OK:" not in out

    def test_missing_write_owning_server_fails(self, capsys):
        """DISABLED_CAPABILITIES can drop a whole server (jsm and announcements
        are owned solely by write capabilities), hiding its write tools."""

        async def _fetch():
            return self._catalog(
                [
                    {"server": "announcements", "status": "available", "tools": []},
                    {"server": "events", "status": "available", "tools": [{"name": "search_x"}]},
                ]
            )

        with pytest.raises(SystemExit) as exc:
            main_argv(_fetch=_fetch)

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "no catalog entry" in out
        assert "jsm" in out
        assert "OK:" not in out

    def test_all_write_servers_present_and_available_passes(self, capsys):
        async def _fetch():
            return self._catalog(
                [
                    {"server": "announcements", "status": "available", "tools": []},
                    {"server": "events", "status": "available", "tools": [{"name": "search_x"}]},
                    {"server": "jsm", "status": "available", "tools": []},
                ]
            )

        with (
            patch("src.writeguard.__main__.WRITE_MCP_TOOL_NAMES", frozenset()),
            pytest.raises(SystemExit) as exc,
        ):
            main_argv(_fetch=_fetch)

        assert exc.value.code == 0
        assert "OK:" in capsys.readouterr().out
