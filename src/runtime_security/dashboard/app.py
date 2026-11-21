"""Native local report viewer. Tk widgets render report strings as literal text."""

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from runtime_security.dashboard.inspection import (
    ReportError,
    display_text,
    fraction_text,
    load_report,
)

BG, INK, MUTED, ACCENT = "#eef2f6", "#172c44", "#52657b", "#087f8c"


def detail_text(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return display_text(json.dumps(value, indent=2, ensure_ascii=False))


class Table(ttk.Frame):
    def __init__(self, parent, columns, *, details=True):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.payloads = {}
        pane = ttk.Panedwindow(self, orient="vertical")
        pane.pack(fill="both", expand=True)
        top = ttk.Frame(pane)
        pane.add(top, weight=3)
        self.tree = ttk.Treeview(
            top, columns=columns, show="headings", selectmode="browse", height=6
        )
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=140, minwidth=85, stretch=True)
        vertical = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(top, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.text = None
        if details:
            bottom = ttk.Frame(pane)
            pane.add(bottom, weight=2)
            ttk.Label(
                bottom, text="SELECT A ROW TO INSPECT SAVED EVIDENCE", style="Caption.TLabel"
            ).pack(anchor="w", pady=6)
            self.text = tk.Text(
                bottom,
                height=5,
                wrap="word",
                font=("Consolas", 10),
                bg="#f8fafc",
                fg=INK,
                relief="flat",
                padx=12,
                pady=8,
            )
            scroll = ttk.Scrollbar(bottom, command=self.text.yview)
            self.text.configure(yscrollcommand=scroll.set, state="disabled")
            scroll.pack(side="right", fill="y")
            self.text.pack(fill="both", expand=True)
            self.tree.bind("<<TreeviewSelect>>", self.selected)
            # Keep both panes visible when high-DPI scaling reduces the available height.
            pane.bind(
                "<Configure>",
                lambda event: pane.after_idle(
                    lambda: (
                        pane.sashpos(0, max(80, int(pane.winfo_height() * 0.58)))
                        if pane.winfo_exists() and pane.winfo_height() > 160
                        else None
                    )
                ),
            )

    def selected(self, _=None):
        selected = self.tree.selection()
        if self.text is not None:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            if selected:
                self.text.insert("1.0", detail_text(self.payloads[selected[0]]))
            self.text.configure(state="disabled")

    def fill(self, rows):
        self.tree.delete(*self.tree.get_children())
        self.payloads.clear()
        for index, (values, payload) in enumerate(rows):
            key = str(index)
            self.tree.insert("", "end", iid=key, values=[display_text(v) for v in values])
            self.payloads[key] = payload
        self.selected()


class Dashboard:
    def __init__(self, root, path=None):
        self.root, self.inspection = root, None
        root.title("Runtime Security | Evaluation Results")
        width = min(1280, root.winfo_screenwidth() - 80)
        height = min(860, root.winfo_screenheight() - 70)
        root.geometry(f"{width}x{height}")
        root.minsize(min(950, width), min(600, height))
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=BG, foreground=INK)
        style.configure("TButton", padding=(12, 8))
        style.configure("TNotebook.Tab", padding=(14, 10))
        style.configure("Treeview", background="white", fieldbackground="white", rowheight=30)
        style.configure(
            "Treeview.Heading", background="#dce5ee", padding=(8, 8), font=("Segoe UI", 10, "bold")
        )
        style.map("Treeview", background=[("selected", "#d2edf0")], foreground=[("selected", INK)])
        style.configure("Caption.TLabel", foreground=MUTED, font=("Segoe UI", 9, "bold"))
        hero = tk.Frame(root, bg=INK, padx=24, pady=10)
        hero.pack(fill="x")
        tk.Label(
            hero,
            text="RUNTIME SECURITY / RESEARCH INSPECTION",
            bg=INK,
            fg="#89d6dc",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            hero, text="Evaluation Results", bg=INK, fg="white", font=("Segoe UI", 20, "bold")
        ).pack(side="left", anchor="w")
        ttk.Button(hero, text="Open local report…", command=self.choose_file).pack(side="right")
        self.identity = tk.Label(
            root,
            text="Open a saved phase9-v1 JSON report. No network or model execution.",
            bg=BG,
            fg=MUTED,
            anchor="w",
            padx=24,
            pady=4,
        )
        self.identity.pack(fill="x")
        controls = ttk.Frame(root, padding=(24, 4))
        controls.pack(fill="x")
        self.split = tk.StringVar(value="test")
        self.method = tk.StringVar()
        ttk.Label(controls, text="Split", style="Caption.TLabel").pack(side="left", padx=(0, 8))
        self.split_box = ttk.Combobox(
            controls,
            textvariable=self.split,
            values=("train", "validation", "test"),
            state="readonly",
            width=14,
        )
        self.split_box.pack(side="left")
        ttk.Label(controls, text="Inspect method", style="Caption.TLabel").pack(
            side="left", padx=(24, 8)
        )
        self.method_box = ttk.Combobox(
            controls, textvariable=self.method, state="readonly", width=42
        )
        self.method_box.pack(side="left")
        self.split_box.bind("<<ComboboxSelected>>", self.refresh)
        self.method_box.bind("<<ComboboxSelected>>", self.refresh)
        self.banner = tk.Label(
            root,
            text="No report loaded",
            bg="#fff1cd",
            fg="#644513",
            anchor="w",
            justify="left",
            padx=14,
            pady=6,
            wraplength=1250,
            font=("Segoe UI", 10),
        )
        self.banner.pack(fill="x", padx=24, pady=10)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        self.tables = {}
        self.build_overview()
        self.tables["methods"] = Table(
            self.tab("Methods"),
            (
                "Method",
                "Predicted / missing",
                "TP",
                "FP",
                "TN",
                "FN",
                "TPR",
                "FPR",
                "Precision",
                "F1",
            ),
        )
        comparison = self.tab("Comparisons")
        self.left, self.right = tk.StringVar(), tk.StringVar()
        compare_controls = ttk.Frame(comparison)
        compare_controls.pack(fill="x", pady=6)
        self.left_box = ttk.Combobox(
            compare_controls, textvariable=self.left, state="readonly", width=38
        )
        self.right_box = ttk.Combobox(
            compare_controls, textvariable=self.right, state="readonly", width=38
        )
        self.left_box.pack(side="left", padx=(0, 12))
        self.right_box.pack(side="left")
        for box in (self.left_box, self.right_box):
            box.bind("<<ComboboxSelected>>", self.refresh_comparison)
        self.compare_notice = ttk.Label(comparison, wraplength=1150)
        self.compare_notice.pack(anchor="w", pady=10)
        self.tables["comparisons"] = Table(
            comparison, ("Saved method", "Common cohort", "TP", "FP", "TN", "FN", "TPR", "FPR")
        )
        breakdown = self.tab("Breakdowns")
        self.dimension = tk.StringVar(value="scenario")
        self.dimension_box = ttk.Combobox(
            breakdown,
            textvariable=self.dimension,
            state="readonly",
            values=("scenario", "task", "model", "version"),
            width=30,
        )
        self.dimension_box.pack(anchor="w", pady=6)
        self.dimension_box.bind("<<ComboboxSelected>>", self.refresh)
        self.tables["breakdowns"] = Table(
            breakdown, ("Category", "Predicted / missing", "TP", "FP", "TN", "FN", "TPR", "FPR")
        )
        proposals = self.tab("Proposals")
        self.query = tk.StringVar()
        ttk.Label(
            proposals,
            text="Filter scenario / task / model / outcome / tool / run ID. "
            "Filters affect rows only.",
        ).pack(anchor="w", pady=6)
        entry = ttk.Entry(proposals, textvariable=self.query)
        entry.pack(fill="x", pady=(0, 8))
        self.query.trace_add("write", lambda *_: self.refresh_records())
        self.tables["proposals"] = Table(
            proposals,
            (
                "Run ID",
                "Sequence",
                "Tool",
                "Attempt unsafe",
                "Effect completed",
                "Flag / decision",
                "Outcome",
            ),
        )
        runs = self.tab("Runs")
        self.excluded = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            runs,
            text="Show excluded runs (not assigned to a split)",
            variable=self.excluded,
            command=self.refresh_records,
        ).pack(anchor="w", pady=6)
        ttk.Label(
            runs,
            text="Uses the Proposals text filter. "
            "Select a run for ordered proposals and recorded mock effects.",
        ).pack(anchor="w", pady=6)
        self.tables["runs"] = Table(
            runs,
            (
                "Run ID",
                "Scenario",
                "Task",
                "Model / version",
                "Attack completed",
                "Benign completed",
                "Mock effects",
                "Excluded",
            ),
        )
        curves = self.tab("Thresholds")
        self.curve_notice = ttk.Label(curves, wraplength=1150)
        self.curve_notice.pack(anchor="w", pady=8)
        self.canvas = tk.Canvas(curves, height=220, bg="white", highlightthickness=0)
        self.canvas.pack(fill="x", pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _: self.draw_curve())
        self.tables["curves"] = Table(
            curves,
            ("Threshold", "TP", "FP", "TN", "FN", "TPR", "FPR", "Precision", "F1"),
            details=False,
        )
        self.tables["evidence"] = Table(
            self.tab("Evidence & definitions"), ("Section", "Saved definition / availability")
        )
        if path is not None:
            self.open_path(path)

    def tab(self, name):
        frame = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(frame, text=name)
        return frame

    def build_overview(self):
        frame = self.tab("Overview")
        cards = tk.Frame(frame, bg=BG)
        cards.pack(fill="x", pady=(0, 10))
        self.cards = {}
        for index, (key, label) in enumerate(
            (
                ("runs", "ALL RUNS"),
                ("eligible_proposals", "ELIGIBLE PROPOSALS"),
                ("events", "FULL EVENT COUNT"),
                ("independent_groups", "INDEPENDENT GROUPS"),
            )
        ):
            card = tk.Frame(cards, bg="white", padx=16, pady=8)
            card.grid(row=0, column=index, sticky="ew", padx=(0, 10))
            cards.columnconfigure(index, weight=1)
            tk.Label(card, text=label, bg="white", fg=MUTED, font=("Segoe UI", 9, "bold")).pack(
                anchor="w"
            )
            value = tk.Label(card, text="—", bg="white", fg=INK, font=("Segoe UI", 22, "bold"))
            value.pack(anchor="w")
            self.cards[key] = value
        ttk.Label(
            frame,
            text="Full event count is absent from phase9-v1. "
            "Source outcomes describe recorded runs; offline alerts do not prove prevention.",
            wraplength=1150,
        ).pack(anchor="w", pady=(0, 10))
        self.tables["overview"] = Table(
            frame, ("Measure", "Value / scope", "Meaning / denominator")
        )

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Open a local Phase 9 report", filetypes=[("JSON report", "*.json")]
        )
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            self.inspection = load_report(path)
        except ReportError as exc:
            self.inspection = None
            for table in self.tables.values():
                table.fill([])
            for card in self.cards.values():
                card.configure(text="—")
            self.canvas.delete("all")
            self.identity.configure(text="No report loaded. The previous report has been cleared.")
            self.banner.configure(text=display_text(str(exc)))
            messagebox.showerror("Report could not be loaded", str(exc), parent=self.root)
            return
        names = sorted(self.inspection.report.methods)
        for box in (self.method_box, self.left_box, self.right_box):
            box.configure(values=names)
        self.method.set(
            "phase7/enriched" if "phase7/enriched" in names else names[0] if names else ""
        )
        self.left.set("phase6/policy" if "phase6/policy" in names else names[0] if names else "")
        self.right.set(self.method.get())
        self.split.set("test")
        self.excluded.set(False)
        self.query.set("")
        self.refresh()

    def refresh(self, _=None):
        if self.inspection is None:
            return
        report, part = self.inspection.report, self.split.get()
        self.identity.configure(
            text=f"SOURCE: {report.source.upper()}   •   SPLIT: {part.upper()}   •   "
            "phase9-v1   •   "
            f"File digest: {self.inspection.fingerprint[:16]}   •   READ ONLY"
        )
        warnings = self.inspection.warnings(part)
        self.banner.configure(
            text="\n".join(warnings)
            or "Evidence adequacy is not a guarantee of generalization. "
            "Inspect saved warnings and limits."
        )
        for key, value in self.inspection.totals().items():
            if key in self.cards:
                self.cards[key].configure(text=display_text(value))
        adequacy, observed = report.adequacy[part], report.observed_source_outcomes[part]
        overview = [
            (
                ("Split runs", adequacy.runs, "Eligible runs in the selected split"),
                adequacy.model_dump(),
            ),
            (
                (
                    "Safe / unsafe / unknown actions",
                    f"{adequacy.safe_actions} / {adequacy.unsafe_actions} / "
                    f"{adequacy.unknown_actions}",
                    "Class coverage; unknowns are not negatives",
                ),
                adequacy.model_dump(),
            ),
            (
                (
                    "Observed attack success",
                    fraction_text(observed.attack_success_rate),
                    "Known injected-run outcomes",
                ),
                observed.model_dump(),
            ),
            (
                (
                    "Observed benign completion",
                    fraction_text(observed.benign_task_completion_rate),
                    "Known non-injected-run outcomes",
                ),
                observed.model_dump(),
            ),
            (
                (
                    "Unsafe attempted / completed runs",
                    f"{observed.unsafe_attempted_runs} / {observed.unsafe_completed_runs}",
                    "Proposals and completed effects remain distinct",
                ),
                observed.model_dump(),
            ),
            (
                (
                    "Excluded runs / action labels",
                    f"{len(report.exclusions)} / {report.excluded_action_labels}",
                    "Report-wide exclusions; inspect Runs",
                ),
                report.exclusions,
            ),
            (
                (
                    "Saved adequacy warnings",
                    len(adequacy.warnings),
                    "Select to inspect all report-supplied warnings",
                ),
                adequacy.model_dump(),
            ),
        ]
        overview += [
            (
                (
                    f"{scope.title()} coverage",
                    f"{a.runs} runs / {a.independent_groups} groups",
                    f"Safe {a.safe_actions}, unsafe {a.unsafe_actions}, "
                    f"unknown {a.unknown_actions}",
                ),
                a.model_dump(),
            )
            for scope, a in report.adequacy.items()
        ]
        self.tables["overview"].fill(overview)
        self.tables["methods"].fill(
            [
                (
                    (
                        name,
                        f"{m.splits[part].coverage.predicted_actions} / "
                        f"{m.splits[part].coverage.missing_predictions}",
                        *matrix_cells(m.splits[part].pending_action),
                    ),
                    {
                        "method": name,
                        "split": part,
                        "results": m.splits[part].model_dump(),
                        "threshold": m.threshold,
                        "calibration": m.calibration,
                        "representation": m.representation,
                        "latency": m.latency.model_dump(),
                    },
                )
                for name, m in sorted(report.methods.items())
            ]
        )
        method = report.methods.get(self.method.get())
        groups = (
            method.breakdowns_within_split.get(part, {}).get(self.dimension.get(), {})
            if method
            else {}
        )
        self.tables["breakdowns"].fill(
            [
                (
                    (
                        name,
                        f"{s.coverage.predicted_actions} / {s.coverage.missing_predictions}",
                        *matrix_cells(s.pending_action)[:6],
                    ),
                    s.model_dump(),
                )
                for name, s in sorted(groups.items())
            ]
        )
        evidence = [(("Metric: " + k, v), v) for k, v in report.definitions.items()]
        evidence += [
            ((k, v.status or "Availability and gaps"), v.model_dump())
            for k, v in report.experiments.items()
        ]
        evidence += [
            (("Uncertainty", report.uncertainty.reason), report.uncertainty.model_dump()),
            (("Input issues", len(report.input_issues)), report.input_issues),
            (
                ("Provenance", "Saved hashes; no external artifact lookup"),
                report.input_hashes.model_dump(),
            ),
            (
                (
                    "Unavailable detail",
                    "Raw task/injection text, matched rule IDs "
                    "and full event timeline are not saved",
                ),
                {
                    "note": "Ordered proposals, safe features, result IDs "
                    "and mock-effect counts remain inspectable."
                },
            ),
        ]
        self.tables["evidence"].fill(evidence)
        self.refresh_records()
        self.refresh_comparison()
        self.draw_curve()

    def refresh_comparison(self, _=None):
        if self.inspection is None:
            return
        found = self.inspection.comparison(self.left.get(), self.right.get(), self.split.get())
        rows = []
        if found is None:
            notice = (
                "No saved Phase 9 comparison for this pair. No ad-hoc comparison is calculated."
            )
        else:
            pair, cohort = found
            notice = (
                f"Common proposals: {cohort.common_count} | Saved left-only: {cohort.left_only} "
                f"| Saved right-only: {cohort.right_only}. "
            )
            if not cohort.common_count:
                notice += "NO COMPATIBLE SHARED PROPOSALS: comparison metrics are undefined."
            else:
                notice += "Metrics below use only the saved common cohort."
                if cohort.left_only or cohort.right_only:
                    notice += " Coverage differs: whole-method scores are not a fair comparison."
            rows = [
                ((name, cohort.common_count, *matrix_cells(matrix)[:6]), cohort.model_dump())
                for name, matrix in (
                    (pair.left_method, cohort.left),
                    (pair.right_method, cohort.right),
                )
            ]
        self.compare_notice.configure(text=notice)
        self.tables["comparisons"].fill(rows)

    def refresh_records(self):
        if self.inspection is None:
            return
        report, part, name = self.inspection.report, self.split.get(), self.method.get()
        method = report.methods.get(name)
        proposals = self.inspection.proposals(name, part, self.query.get()) if method else []
        diagnostics = {d.proposal_id: d.model_dump() for d in method.diagnostics} if method else {}
        run_map = {r.run_id: r for r in report.runs}
        self.tables["proposals"].fill(
            [
                (
                    (
                        p.run_id[:12],
                        p.sequence,
                        p.security_features.tool,
                        p.attempted_violation,
                        p.completed_violation,
                        p.prediction.disposition or str(p.prediction.flagged)
                        if p.prediction
                        else "MISSING",
                        p.category,
                    ),
                    {
                        "proposal": p.model_dump(),
                        "run": run_map[p.run_id].model_dump(),
                        "gateway": diagnostics.get(p.proposal_id),
                        "note": "A proposed call or alert does not establish execution; "
                        "inspect completed labels and recorded effects.",
                    },
                )
                for p in proposals
            ]
        )
        self.tables["runs"].fill(
            [
                (
                    (
                        r.run_id[:12],
                        r.scenario,
                        r.task,
                        f"{r.model} / {r.version}",
                        r.attack_completed,
                        r.benign_completed,
                        r.recorded_mock_effect_count,
                        bool(r.issues),
                    ),
                    {
                        "run": r.model_dump(),
                        "ordered_proposals": [
                            p.model_dump()
                            for p in sorted(method.proposals, key=lambda p: p.sequence)
                            if p.run_id == r.run_id
                        ]
                        if method
                        else [],
                        "note": "Proposal history only; the full event timeline is unavailable.",
                    },
                )
                for r in self.inspection.runs(part, self.query.get(), self.excluded.get())
            ]
        )

    def draw_curve(self):
        self.canvas.delete("all")
        if self.inspection is None:
            return
        method = self.inspection.report.methods.get(self.method.get())
        points = method.descriptive_threshold_sweeps.get(self.split.get(), []) if method else []
        saved = method.threshold if method else None
        label = saved if saved is not None else "N/A (rules or per-action signal)"
        note = f"Saved threshold: {label}. "
        note += "Descriptive validation analysis. NEVER use test data to select thresholds. "
        if self.split.get() != "validation":
            note += f"Viewing saved {self.split.get()} curves, not validation evidence. "
        if not points:
            note += "No scored curve points are available for this method/split."
        self.curve_notice.configure(text=note)
        self.tables["curves"].fill(
            [((p.threshold, *matrix_cells(p.metrics)), p.model_dump()) for p in points]
        )
        if not points:
            self.canvas.create_text(
                24,
                50,
                text="No threshold curve available",
                anchor="w",
                fill=MUTED,
                font=("Segoe UI", 12),
            )
            return
        width = max(self.canvas.winfo_width(), 500)
        x0, x1, y0, y1 = 55, width - 40, 175, 25
        self.canvas.create_line(x0, y1, x0, y0, x1, y0, fill=MUTED)
        for n in (0, 0.5, 1):
            x, y = x0 + n * (x1 - x0), y0 - n * (y0 - y1)
            self.canvas.create_text(x, y0 + 16, text=str(n), fill=MUTED)
            self.canvas.create_text(x0 - 24, y, text=f"{n:.0%}", fill=MUTED)
        self.canvas.create_text((x0 + x1) / 2, 210, text="Threshold", fill=MUTED)
        for index, (label, attr, color) in enumerate(
            (("TPR", "recall", ACCENT), ("FPR", "fpr", "#c66b25"))
        ):
            prior = None
            defined = False
            for point in sorted(points, key=lambda p: p.threshold):
                value = getattr(point.metrics, attr)
                if value is None:
                    prior = None
                    continue
                defined = True
                xy = (x0 + point.threshold * (x1 - x0), y0 - value * (y0 - y1))
                if prior is not None:
                    self.canvas.create_line(*prior, *xy, fill=color, width=2)
                self.canvas.create_oval(
                    xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3, fill=color, outline=color
                )
                prior = xy
            self.canvas.create_text(
                x0 + index * 240,
                12,
                text=label if defined else label + " N/A (undefined)",
                fill=color,
                anchor="w",
            )
        if saved is not None:
            x = x0 + saved * (x1 - x0)
            self.canvas.create_line(x, y1, x, y0, fill=INK, dash=(4, 4))
            self.canvas.create_text(x + 6, y1 + 15, text="Saved", anchor="w", fill=INK)


def matrix_cells(matrix):
    return (
        matrix.tp,
        matrix.fp,
        matrix.tn,
        matrix.fn,
        fraction_text(matrix.rates.detection_rate_tpr),
        fraction_text(matrix.rates.false_intervention_rate_fpr),
        fraction_text(matrix.rates.precision),
        fraction_text(matrix.rates.f1),
    )
