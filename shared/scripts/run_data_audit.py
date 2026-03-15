from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable when the script is called directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.audit_bids import (  # noqa: E402
    AuditConfig,
    audit_conclusion_markdown,
    build_inventory_summary,
    build_provisional_event_groups,
    collect_channel_inventory,
    collect_coordsystem_inventory,
    collect_electrode_metadata,
    collect_event_definitions,
    collect_event_occurrences,
    collect_run_inventory,
    ensure_output_dirs,
    inventory_summary_markdown,
    plot_count_bar,
    plot_electrode_radius_histogram,
    plot_event_heatmap,
    plot_event_label_counts,
    presence_check,
    save_dataframe,
    save_environment_snapshot,
    select_montage_preview_runs,
    spot_check_raw_files,
    summarise_channel_status,
    summarise_event_counts,
    write_text,
    safe_read_tsv,
)
from shared.src.montage_checks import infer_scale_plausibility, montage_sanity_markdown, plot_scale_comparison, plot_xy_layout  # noqa: E402


def run_data_audit(config: AuditConfig) -> dict[str, Path]:
    out = ensure_output_dirs(config)
    tables_dir = out["tables"]
    figures_dir = out["figures"]
    reports_dir = out["reports"]

    save_environment_snapshot(config, reports_dir)

    inventory = collect_run_inventory(config)
    save_dataframe(inventory, tables_dir / "run_inventory.csv")
    write_text(reports_dir / "bids_presence_check.txt", presence_check(config, inventory))

    runs_per_subject, runs_per_session, runs_per_task = build_inventory_summary(inventory)
    if not runs_per_subject.empty:
        plot_count_bar(runs_per_subject, "subject", "n_runs", "Runs per subject", figures_dir / "runs_per_subject.png")
    if not runs_per_task.empty:
        plot_count_bar(runs_per_task, "task", "n_runs", "Runs per task", figures_dir / "runs_per_task.png")
    write_text(
        reports_dir / "inventory_summary.md",
        inventory_summary_markdown(config, inventory, runs_per_subject, runs_per_session, runs_per_task),
    )

    event_defs = collect_event_definitions(inventory)
    save_dataframe(event_defs, tables_dir / "event_definitions_from_json.csv")

    event_occ = collect_event_occurrences(inventory, config.event_candidate_columns)
    save_dataframe(event_occ, tables_dir / "event_occurrences_long.csv")

    event_counts = summarise_event_counts(event_occ)
    save_dataframe(event_counts, tables_dir / "event_counts_by_run.csv")

    provisional = build_provisional_event_groups(event_counts)
    save_dataframe(provisional, tables_dir / "provisional_event_groups.csv")

    plot_event_label_counts(event_counts, figures_dir / "event_label_counts.png")
    plot_event_heatmap(event_counts, figures_dir / "event_counts_heatmap.png")

    channel_inventory = collect_channel_inventory(inventory)
    save_dataframe(channel_inventory, tables_dir / "channel_inventory_long.csv")
    channel_status = summarise_channel_status(channel_inventory)
    save_dataframe(channel_status, tables_dir / "channel_status_summary.csv")

    electrode_long, electrode_summary = collect_electrode_metadata(inventory)
    save_dataframe(electrode_summary, tables_dir / "electrode_coordinate_summary.csv")
    save_dataframe(collect_coordsystem_inventory(inventory), tables_dir / "coordsystem_inventory.csv")
    plot_electrode_radius_histogram(electrode_summary, figures_dir / "electrode_radius_histogram.png")

    raw_meta = spot_check_raw_files(inventory, config.raw_spotcheck_max_files, config.random_seed)
    save_dataframe(raw_meta, tables_dir / "raw_metadata_summary.csv")

    # Montage previews and scale comparisons
    preview_rows = select_montage_preview_runs(inventory, max_runs=config.montage_preview_max_runs)
    montage_report_chunks: list[str] = ["# Montage sanity report", ""]
    for preview in preview_rows.itertuples(index=False):
        elec_df = safe_read_tsv(preview.electrodes_tsv)
        if elec_df.empty:
            montage_report_chunks.append(f"## {preview.session} / {preview.task}\n\n- No electrodes.tsv found.\n")
            continue
        session_tag = str(preview.session).replace("ses-", "")
        plot_xy_layout(
            elec_df,
            figures_dir / f"montage_preview_{session_tag}_rawscale.png",
            title=f"Montage preview ({preview.session}, task={preview.task})",
        )
        plot_scale_comparison(
            elec_df,
            scales=config.coordinate_scale_checks,
            save_path=figures_dir / f"montage_scale_comparison_{session_tag}.png",
            title=f"Coordinate scale comparison ({preview.session}, task={preview.task})",
        )
        verdicts = infer_scale_plausibility(elec_df, config.coordinate_scale_checks)
        montage_report_chunks.append(montage_sanity_markdown(f"{preview.session} / task={preview.task}", verdicts))
        montage_report_chunks.append("")
    write_text(reports_dir / "montage_sanity_report.md", "\n".join(montage_report_chunks))

    report = audit_conclusion_markdown(config, inventory, provisional, electrode_summary, raw_meta)
    write_text(reports_dir / "data_audit_report.md", report)

    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ds004033 shared data audit.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "shared.yaml"),
        help="Path to the YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    config = AuditConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    out = run_full_data_audit(config)
    print(f"Audit finished. Outputs written to: {out['root']}")


if __name__ == "__main__":
    main()
