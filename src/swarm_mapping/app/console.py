"""The operator console: run scenarios, install rooms, browse run history.

Start it with `swarm-console` (or `streamlit run` on this file). Three pages,
chosen in the sidebar:

- **Run** — pick a scenario, swarm size, allocation variant and mode; start
  the run as a CLI subprocess and watch its log.
- **Scenarios** — every scenario with its validity; upload a new room.
- **History** — every past run: map, per-drone routes, log, metrics, and a
  side-by-side comparison of two runs.

This file is layout only. What to run, how to name it, what a row or a
comparison says, and whether a room is valid are decided in `runner`,
`history`, `records` and `validation`, which import no streamlit. Streamlit
executes this file top to bottom on every interaction, so it ends by calling
`main()` unconditionally — the `__name__ == "__main__"` guard does not work
under streamlit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from swarm_mapping.app import history, runner
from swarm_mapping.app.validation import UPLOAD_DRONES, install_scenario
from swarm_mapping.config.loader import load_config
from swarm_mapping.records import (
    LOG_FILE,
    VARIANTS,
    RunRecord,
    RunRecordError,
    list_runs,
    load_run,
    variant_label,
)

PAGES = ("Run", "Scenarios", "History")
MODES = ("Headless", "With MuJoCo view")
TAIL_LINES = 20
"""Log lines shown while a run is in progress."""

# The one run the console may have in flight: {"process", "run_dir", "request"}.
_ACTIVE = "active_run"
# The name of a room just installed, shown once after the page reruns.
_INSTALLED = "installed_scenario"


def _variant_table() -> pd.DataFrame:
    """The four allocation variants and what each passes to the CLI."""
    return pd.DataFrame(
        [
            {
                "variant": label,
                "assignment": pair[0],
                "target tolerance (cells)": pair[1],
            }
            for label, pair in VARIANTS.items()
        ]
    )


def _console_tail(run_dir: Path, lines: int = 30) -> str:
    """The end of a run's `console.txt`, for when it left no record."""
    path = run_dir / runner.CONSOLE_FILE
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _show_log_tail(run_dir: Path) -> None:
    """The live panel: tick, coverage, and the last few log lines."""
    entries = history.read_log(run_dir / LOG_FILE)
    status = history.progress(entries)
    tick, coverage = st.columns(2)
    tick.metric("Tick", "—" if status.tick is None else status.tick)
    coverage.metric(
        "Coverage", "—" if status.coverage is None else f"{status.coverage:.1%}"
    )
    st.code(
        "\n".join(str(entry) for entry in entries[-TAIL_LINES:]) or "(no log yet)",
        language=None,
    )


def _show_summary(record: RunRecord) -> None:
    """A run's headline numbers."""
    outputs = record.outputs
    cols = st.columns(4)
    cols[0].metric("Ticks", outputs.ticks)
    cols[1].metric("Coverage", f"{outputs.coverage:.1%}")
    cols[2].metric("Succeeded", "yes" if outputs.succeeded else "no")
    cols[3].metric("Failures detected", history.failures_detected(record))
    st.caption(
        f"Blocked: {outputs.blocked} · unreachable frontiers: "
        f"{outputs.unreachable_frontiers} · tick-capped: {outputs.tick_capped} · "
        f"wall time: {outputs.wall_seconds:.1f} s"
    )


def _show_finished(active: dict[str, Any]) -> None:
    """What a finished run left: exit code, summary, map."""
    process: subprocess.Popen[bytes] = active["process"]
    run_dir: Path = active["run_dir"]
    if process.returncode == 0:
        st.success(f"Run finished (exit code 0): {run_dir}")
    else:
        st.warning(
            f"Run exited with code {process.returncode}: {run_dir}. The mission "
            "did not finish on its own (tick cap), lost every drone, or "
            "crashed — see the output below."
        )
    try:
        record = load_run(run_dir)
    except RunRecordError as exc:
        st.error(str(exc))
        st.code(_console_tail(run_dir) or "(no console output)", language=None)
        return
    _show_summary(record)
    if (run_dir / "map.png").is_file():
        st.image(str(run_dir / "map.png"), caption="Map")
    st.info("Routes, the full log and comparisons are on the History page.")


def _run_page() -> None:
    """Pick a scenario and variant, start a run, follow it."""
    st.header("Run a scenario")
    choices = runner.scenario_choices(runner.scenarios_root())
    valid = [choice for choice in choices if not choice.problems]
    invalid = [choice for choice in choices if choice.problems]
    if invalid:
        with st.expander(f"{len(invalid)} scenario(s) cannot run"):
            for choice in invalid:
                st.markdown(f"**{choice.name}**")
                for problem in choice.problems:
                    st.error(problem)
    if not valid:
        st.info(f"No runnable scenarios under {runner.scenarios_root()}.")
        return

    by_name = {choice.name: choice for choice in valid}
    name = st.selectbox("Scenario", list(by_name), key="run_scenario")
    choice = by_name[name]
    spawns = choice.spawns or 1
    # `min_drones` is 1 unless the scenario has a failure schedule, in which
    # case config parsing already guarantees drone_id < spawns, so it can
    # never exceed spawns here.
    minimum = choice.min_drones
    # Keys carry the scenario name: each scenario keeps its own choices, and a
    # slider never holds a value beyond another scenario's spawn count.
    if minimum < spawns:
        drones = st.slider(
            "Drones", minimum, spawns, value=spawns, key=f"drones::{choice.name}"
        )
        if minimum > 1:
            st.caption(
                f"Minimum {minimum}: the failure schedule fails drone "
                f"{minimum - 1}, which must be present in the run."
            )
    else:
        # Only reachable when minimum == spawns: either spawns == 1 (no
        # failure schedule, nothing else to say) or minimum > 1 (a failure
        # schedule needs every declared drone) — minimum > 1 with spawns > 1
        # and minimum < spawns is the slider case above.
        drones = spawns
        if spawns == 1:
            st.caption("Drones: 1 (the scenario declares one start position)")
        else:
            st.caption(
                f"Drones: {spawns} (the failure schedule fails drone "
                f"{minimum - 1}, so every drone must be present)"
            )

    coordination = load_config(choice.config_path).coordination
    configured = variant_label(
        coordination.assignment, coordination.target_tolerance_cells
    )
    labels = list(VARIANTS)
    variant = st.radio(
        "Allocation variant",
        labels,
        index=labels.index(configured) if configured in labels else 0,
        horizontal=True,
        key=f"variant::{choice.name}",
    )
    st.caption(
        "Allocation variants change how frontiers are assigned, not the frontier "
        f"strategy. The scenario's own setting is **{configured}**."
    )
    st.dataframe(_variant_table(), hide_index=True)
    mode = st.radio("Mode", MODES, horizontal=True, key="run_mode")

    active: dict[str, Any] | None = st.session_state.get(_ACTIVE)
    running = active is not None and active["process"].poll() is None
    if st.button("Run", type="primary", disabled=running, key="run_button"):
        request = runner.RunRequest(
            config_path=choice.config_path,
            scenario=choice.name,
            drones=int(drones),
            variant=str(variant),
            view=mode == MODES[1],
            min_drones=choice.min_drones,
        )
        process, run_dir = runner.launch(request, runner.runs_root())
        st.session_state[_ACTIVE] = {
            "process": process,
            "run_dir": run_dir,
            "request": request,
        }
        # Rerun so the button is drawn again — disabled — for the new run.
        st.rerun()

    if active is None:
        return
    st.divider()
    st.subheader(f"Run: {active['run_dir'].name}")
    if running:
        # Wrapped at the call rather than decorated: without the `ui` extra (as
        # in CI) streamlit is untyped to mypy, and strict mode rejects an
        # untyped decorator. The script re-executes on every rerun, so this is
        # the same fragment either way.
        st.fragment(_live_panel, run_every=1.0)()
    else:
        _show_finished(active)


def _live_panel() -> None:
    """Refreshes about once a second while the run is in flight.

    A fragment reruns alone, so the rest of the page is not rebuilt every
    second. When the process exits it reruns the whole app, which re-enables
    the Run button and shows the results.
    """
    active: dict[str, Any] = st.session_state[_ACTIVE]
    if active["process"].poll() is not None:
        st.rerun()
    st.info("Running… the Run button is disabled until this run exits.")
    _show_log_tail(active["run_dir"])


def _scenarios_page() -> None:
    """Every scenario with its validity, and the upload form."""
    st.header("Scenarios")
    installed = st.session_state.pop(_INSTALLED, None)
    if installed:
        st.success(f"Installed '{installed}'. It is listed below and on the Run page.")
    choices = runner.scenario_choices(runner.scenarios_root())
    if choices:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "scenario": choice.name,
                        "valid": not choice.problems,
                        "start positions": choice.spawns,
                        "problems": "; ".join(choice.problems),
                    }
                    for choice in choices
                ]
            ),
            hide_index=True,
        )
    else:
        st.info(f"No scenarios under {runner.scenarios_root()}.")

    st.subheader("Add a room")
    st.caption(
        f"Upload a scenario config and the MJCF scene it uses. The room must "
        f"declare exactly {UPLOAD_DRONES} start positions, every one clear of "
        "the scene and of the others, so any drone count that includes every "
        "drone its own failure schedule names works. Nothing is saved unless "
        "it validates, and nothing is ever overwritten."
    )
    with st.form("upload", clear_on_submit=False):
        name = st.text_input("Name (lowercase letters, digits, _)", key="upload_name")
        config_file = st.file_uploader(
            "Scenario config (.yaml)", type=["yaml", "yml"], key="upload_config"
        )
        scene_file = st.file_uploader("Scene (.xml)", type=["xml"], key="upload_scene")
        submitted = st.form_submit_button("Validate and install")
    if not submitted:
        return
    if config_file is None or scene_file is None:
        st.error("Upload both the config and the scene.")
        return
    problems = install_scenario(
        name.strip(),
        config_file.getvalue().decode("utf-8", errors="replace"),
        scene_file.getvalue().decode("utf-8", errors="replace"),
        runner.scenarios_root(),
        runner.assets_dir(),
    )
    if problems:
        st.error(f"'{name}' was not installed:")
        for problem in problems:
            st.error(problem)
    else:
        # Rerun so the table above lists the new room; the message survives
        # the rerun in session state and is shown once.
        st.session_state[_INSTALLED] = name.strip()
        st.rerun()


def _history_page() -> None:
    """The run table, one run in detail, and two runs compared."""
    st.header("History")
    listing = list_runs(runner.runs_root())
    for skipped in listing.skipped:
        st.warning(f"Skipped {skipped.directory.name}: {skipped.reason}")
    if not listing.runs:
        st.info(f"No runs yet under {runner.runs_root()}.")
        return

    st.dataframe(
        pd.DataFrame([history.history_row(run) for run in listing.runs]),
        hide_index=True,
    )
    by_name = {run.directory.name: run for run in listing.runs if run.directory}
    names = list(by_name)

    st.subheader("Run details")
    selected = by_name[st.selectbox("Run", names, key="history_run")]
    _run_details(selected)

    st.subheader("Compare two runs")
    if len(names) < 2:
        st.info("Compare needs at least two runs.")
        return
    left, right = st.columns(2)
    a = by_name[left.selectbox("Run A", names, index=0, key="compare_a")]
    b = by_name[right.selectbox("Run B", names, index=1, key="compare_b")]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": delta.metric,
                    "A": str(delta.a),
                    "B": str(delta.b),
                    "B − A": "" if delta.difference is None else delta.difference,
                }
                for delta in history.compare(a, b)
            ]
        ),
        hide_index=True,
    )
    for column, record in ((left, a), (right, b)):
        assert record.directory is not None  # loaded runs always have one
        image = record.directory / "map.png"
        if image.is_file():
            column.image(str(image), caption=record.directory.name)


def _run_details(record: RunRecord) -> None:
    """One run: summary, map, per-drone routes, log, console output."""
    assert record.directory is not None  # loaded runs always have one
    directory = record.directory
    _show_summary(record)
    inputs = record.inputs
    st.caption(
        f"Scenario {history.scenario_name(record)} · variant {inputs.variant} "
        f"({inputs.assignment}, tolerance {inputs.target_tolerance_cells}) · "
        f"{inputs.drones} drone(s) · started {record.started_at}"
    )
    if (directory / "map.png").is_file():
        st.image(str(directory / "map.png"), caption="Map")

    routes = history.route_images(record)
    if routes:
        tabs = st.tabs([f"Drone {drone}" for drone, _ in routes])
        for tab, (drone, image) in zip(tabs, routes, strict=True):
            with tab:
                st.image(str(image), caption=f"Route of drone {drone}")
                stats = record.outputs.paths.get(drone, {})
                st.dataframe(pd.DataFrame([stats]), hide_index=True)

    entries = history.read_log(directory / LOG_FILE)
    st.markdown("**Log**")
    events = st.multiselect(
        "Show events", history.event_names(entries), key="log_events"
    )
    shown = [entry for entry in entries if not events or entry.get("event") in events]
    st.caption(f"{len(shown)} of {len(entries)} lines")
    if shown:
        st.dataframe(pd.DataFrame(shown), hide_index=True)
    tail = _console_tail(directory)
    if tail:
        with st.expander("Console output"):
            st.code(tail, language=None)


def main() -> None:
    """Lay out the sidebar and the chosen page."""
    st.set_page_config(page_title="Swarm mapping console", layout="wide")
    st.sidebar.title("Swarm mapping")
    page = st.sidebar.radio("Page", PAGES, key="page")
    {"Run": _run_page, "Scenarios": _scenarios_page, "History": _history_page}[
        str(page)
    ]()


main()
