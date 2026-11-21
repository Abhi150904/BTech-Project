import json
import subprocess
import sys
from pathlib import Path

import pytest

from runtime_security.dashboard.__main__ import main
from runtime_security.dashboard.inspection import (
    ReportError,
    display_text,
    fraction_text,
    load_report,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "research/phase9-scripted-v1/report.json"


@pytest.fixture
def raw():
    return json.loads(SAMPLE.read_bytes())


def save(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_committed_report_coverage_and_undefined_metrics():
    view = load_report(SAMPLE)
    assert view.totals() == {
        "runs": 12,
        "eligible_proposals": 23,
        "excluded_action_labels": 0,
        "events": None,
        "independent_groups": 3,
    }
    summary = view.report.methods["phase7/enriched"].splits["test"]
    assert fraction_text(summary.pending_action.rates.detection_rate_tpr) == "N/A (0/0)"
    assert fraction_text(summary.pending_action.rates.false_intervention_rate_fpr) == "0.00% (0/2)"
    assert any("No unsafe" in warning for warning in view.warnings("test"))
    assert any("SCRIPTED" in warning for warning in view.warnings("train"))
    assert summary.coverage.missing_predictions == 0
    assert view.report.methods["phase7/enriched"].splits["train"].coverage.missing_predictions == 17


@pytest.mark.parametrize(
    "field", ["schema_version", "definitions", "membership", "methods", "adequacy", "runs"]
)
def test_missing_required_sections_rejected(tmp_path, raw, field):
    del raw[field]
    with pytest.raises(ReportError):
        load_report(save(tmp_path, raw))


@pytest.mark.parametrize(
    "content", ["[]", "{", '{"schema_version":"phase99"}', '{"a":1,"a":2}', '{"a":NaN}']
)
def test_malformed_unsupported_duplicate_and_nonfinite(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ReportError):
        load_report(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "zero",
        "bool_count",
        "missing_split",
        "unknown_field",
        "bad_membership",
        "bad_pair",
        "bad_proposal",
        "different_evidence",
    ],
)
def test_invalid_schema_and_links_rejected(tmp_path, raw, mutation):
    if mutation == "zero":
        raw["methods"]["phase7/enriched"]["splits"]["test"]["pending_action"]["rates"][
            "detection_rate_tpr"
        ]["value"] = 0
    elif mutation == "bool_count":
        raw["adequacy"]["test"]["runs"] = True
    elif mutation == "missing_split":
        del raw["methods"]["phase7/enriched"]["splits"]["train"]
    elif mutation == "unknown_field":
        raw["runs"][0]["raw_prompt"] = "private prompt"
    elif mutation == "bad_membership":
        raw["membership"]["test"].append(raw["membership"]["train"][0])
    elif mutation == "bad_pair":
        raw["paired_comparisons"][0]["splits"]["test"]["common_count"] += 1
    elif mutation == "bad_proposal":
        raw["methods"]["phase7/enriched"]["proposals"][0]["run_id"] = "0" * 64
    else:
        raw["methods"]["phase7/enriched"]["proposals"][0]["security_features"]["authorization"] = (
            "changed"
        )
    with pytest.raises(ReportError):
        load_report(save(tmp_path, raw))


def test_optional_sections_can_be_empty_or_absent(tmp_path, raw):
    raw.pop("paired_comparisons")
    raw.pop("experiments")
    for method in raw["methods"].values():
        method.pop("breakdowns_within_split")
        method.pop("descriptive_threshold_sweeps")
    view = load_report(save(tmp_path, raw))
    assert not view.report.experiments
    assert view.comparison("phase6/policy", "phase7/enriched", "test") is None
    assert not view.report.methods["phase7/enriched"].descriptive_threshold_sweeps


def test_overview_only_report_and_explicit_exclusions(tmp_path, raw):
    raw["methods"] = {}
    raw["paired_comparisons"] = []
    excluded = dict(raw["runs"][0], run_id="f" * 64, issues=["truncated"])
    raw["runs"].append(excluded)
    raw["exclusions"][excluded["run_id"]] = excluded["issues"]
    view = load_report(save(tmp_path, raw))
    assert view.totals()["runs"] == 13
    assert view.comparison("phase6/policy", "phase7/enriched", "test") is None
    assert [r.run_id for r in view.runs("test", excluded=True)] == [excluded["run_id"]]
    assert excluded["run_id"] not in [r.run_id for r in view.runs("test")]
    assert any("Excluded" in warning for warning in view.warnings("test"))


def test_optional_gateway_diagnostics_remain_linked(tmp_path, raw):
    method = raw["methods"]["phase6/policy"]
    proposal = method["proposals"][0]
    diagnostic = {
        "proposal_id": proposal["proposal_id"],
        "policy_available": True,
        "detector_available": True,
        "detector_score": 0.7,
        "detector_threshold": 0.5,
        "detector_status": "alert",
        "approval": "approved",
        "model_version": "a" * 64,
        "policy_hash": "b" * 64,
    }
    method["diagnostics"] = [diagnostic]
    view = load_report(save(tmp_path, raw))
    assert view.report.methods["phase6/policy"].diagnostics[0].approval == "approved"
    diagnostic["proposal_id"] = "0" * 64
    with pytest.raises(ReportError):
        load_report(save(tmp_path, raw))


def test_sources_and_splits_stay_separate(tmp_path, raw):
    scripted = load_report(SAMPLE)
    raw["source"] = "live"
    live = load_report(save(tmp_path, raw))
    assert scripted.report.source == "scripted" and live.report.source == "live"
    assert not any("SCRIPTED" in warning for warning in live.warnings("test"))
    train = live.proposals("phase7/enriched", "train")
    test = live.proposals("phase7/enriched", "test")
    assert len(train) == 17 and len(test) == 2
    assert not {p.proposal_id for p in train} & {p.proposal_id for p in test}
    assert all(p.prediction is None for p in train)


def test_comparisons_use_saved_common_cohort_only():
    view = load_report(SAMPLE)
    pair, cohort = view.comparison("phase7/enriched", "phase6/policy", "test")
    assert cohort.common_count == 2
    assert {pair.left_method, pair.right_method} == {"phase7/enriched", "phase6/policy"}
    _, train = view.comparison("phase6/policy", "phase7/enriched", "train")
    assert train.common_count == 0 and train.left_only + train.right_only == 17
    assert train.left.recall is None and train.right.recall is None
    assert view.comparison("unknown", "phase6/policy", "test") is None


def test_filters_and_ordered_proposals_do_not_recompute_metrics():
    view = load_report(SAMPLE)
    before = view.report.model_dump_json()
    rows = view.proposals("phase6/policy", "validation", "false_positive")
    assert len(rows) == 2 and all(p.category == "false_positive" for p in rows)
    assert [(p.run_id, p.sequence) for p in rows] == sorted((p.run_id, p.sequence) for p in rows)
    assert view.proposals("phase6/policy", "test", "absent-value") == []
    assert view.report.model_dump_json() == before


def test_display_is_literal_and_redacts_paths_and_token_patterns():
    markup = '<script>alert("x")</script> [exec malicious]'
    assert display_text(markup) == markup
    assert "sk-abcdefghijklmnopqrst" not in display_text("sk-abcdefghijklmnopqrst")
    assert "Users" not in display_text(r"C:\Users\someone\private.json")
    assert "private" not in display_text("/home/user/private.json")
    assert "\x00" not in display_text("text\x00")


def test_safe_errors_do_not_echo_input_or_paths(tmp_path, raw, capsys):
    raw["PRIVATE_DO_NOT_ECHO"] = "sk-abcdefghijklmnopqrst"
    path = save(tmp_path, raw)
    assert main([str(path), "--check"]) == 1
    output = capsys.readouterr().err
    assert "PRIVATE_DO_NOT_ECHO" not in output
    assert str(path) not in output and "sk-" not in output
    assert "required fields" in output


@pytest.mark.parametrize(
    "path",
    [
        "https://example.invalid/report.json",
        r"\\server\private\report.json",
        "//server/private/report.json",
    ],
)
def test_network_paths_rejected_before_open(path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Attempted file access on network path")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ReportError, match="local disk"):
        load_report(path)


def test_no_evaluator_model_tools_network_or_report_write(tmp_path):
    before = SAMPLE.read_bytes()
    script = """
import importlib.abc
import socket
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('runtime_security.evaluation', 'runtime_security.agent',
            'runtime_security.detectors', 'runtime_security.gateway', 'runtime_security.tools',
            'openai', 'langchain', 'torch', 'sklearn')
        if fullname.startswith(forbidden):
            raise AssertionError('Forbidden runtime import')
sys.meta_path.insert(0, Guard())
def forbidden(*a, **kw):
    raise AssertionError('Network call')
socket.socket.connect = forbidden
socket.create_connection = forbidden
from runtime_security.dashboard.__main__ import main
raise SystemExit(main([sys.argv[1], '--check']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(SAMPLE)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "Valid phase9-v1 report" in result.stdout
    assert SAMPLE.read_bytes() == before


@pytest.fixture(scope="module")
def desktop_root():
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"No desktop Tk display: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def gui(desktop_root):
    root = desktop_root
    from runtime_security.dashboard.app import Dashboard

    app = Dashboard(root, SAMPLE)
    yield root, app
    for child in root.winfo_children():
        child.destroy()


def test_native_controls_and_validation_curve(gui):
    root, app = gui
    root.update_idletasks()
    assert "SCRIPTED" in app.identity.cget("text")
    assert "No unsafe" in app.banner.cget("text")
    assert app.cards["events"].cget("text") == "N/A"
    app.split.set("train")
    app.refresh()
    assert "NO COMPATIBLE" in app.compare_notice.cget("text")
    app.split.set("validation")
    app.method.set("phase6/semantic_logistic")
    app.refresh()
    assert "Saved threshold: 0.5" in app.curve_notice.cget("text")
    assert "NEVER use test data" in app.curve_notice.cget("text")
    app.method.set("phase6/policy")
    app.query.set("false_positive")
    assert len(app.tables["proposals"].tree.get_children()) == 2


def test_native_text_does_not_execute_tcl_or_html(gui):
    root, app = gui
    value = "<script>alert(1)</script> [set ::injected_by_report 1]"
    table = app.tables["evidence"]
    table.fill([((value, value), {"label": value})])
    table.tree.selection_set("0")
    table.selected()
    assert value in table.text.get("1.0", "end")
    assert not root.tk.call("info", "exists", "injected_by_report")
    assert table.text.cget("state") == "disabled"


def test_failed_load_clears_previous_report(gui, tmp_path, monkeypatch):
    _, app = gui
    from runtime_security.dashboard import app as module

    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **k: None)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    app.open_path(bad)
    assert app.inspection is None
    assert not app.tables["methods"].tree.get_children()
    assert "previous report has been cleared" in app.identity.cget("text")
