# Evaluation code

The general entry points are listed in the [release README](../README.md).
Synthetic tests and labels validate mechanics, not biological performance.
Population simulation has genotype truth only. Real evidence is unlabeled;
Candotti region categories are never per-individual triage-action labels.

Historical runners retain the original scientific parameters and safeguards. They
require separately supplied research artifacts; no panels, manifests, actual sample
lists, saved states or model assessments are distributed.

For `phase16_retry_trial`, `phase17_sensitivity` and `phase18_final_study`, set
`HAPLOTYPE_STUDY_ROOT` to an **external** study tree with the following original
relative layout. `HAPLOTYPE_CONFIG` optionally selects the YAML configuration;
default is this release's registry. These environment variables must be set before
starting Python. Use `--output` to choose a fresh output directory.

| Runner | Required artifacts beneath the external study root |
| --- | --- |
| `phase16_retry_trial` | `evaluation/phase16_panel.json` (`cases` containing `sample`, `region_key`, `stratum`); `data/reference/Egrandis_v2.fa`; target BED/category CSV under `data/ground_truth/derived/`; BAMs and `phased/ID.phased.vcf.gz`, `phased/ID.input.vcf.gz` under `data/processed/batch20/`, with indexes. `--preflight-only` avoids model/tool execution but reads these real inputs. |
| `phase17_sensitivity` | `evaluation/phase17_manifest.json` with original `pairs`, source artifact paths/hashes, case identities and counterfactual field specifications. Relative `source` paths resolve against `HAPLOTYPE_STUDY_ROOT`; absolute paths work too. `prepare` validates saved states; `run` calls the model. |
| `phase18_final_study` | `config/batch20_manifest.csv`, catalogue BED/CSV, original retry VCFs under `data/processed/batch20/phased/`, prior `outputs/batch20_real_panel_seed_20260915/panel.jsonl`, Phase 16 panel, and Phase 17 manifest. Also set `HAPLOTYPE_DEVELOPMENT_CASE_ID` to the original `sample::region_key` for the mandatory development-case exclusion; its identity must match the original frozen hash. Modes: `prepare`, `run`, `report`. |

These paths are an external input contract, not a reference to the original local
checkout. Do not point them at that checkout when testing the release. Existing
frozen hash checks remain active. External input fingerprints can record absolute
paths, so relocation is not a guarantee that old run plans can resume.

`phase18_diagnostics` accepts `--config`, required `--sources` and `--output-dir`.
The sources file is a JSON list of exactly three objects with `name` and `path`;
add `case_id` when a source is a panel JSONL. Paths resolve against the sources
file's directory. A saved audit supplies `initial_state`; panel rows supply `state`.
For example, each object has the form:

```json
{"name": "artificial-case", "path": "/path/to/external/panel.jsonl", "case_id": "ARTIFICIAL_SAMPLE::ARTIFICIAL_GENE|target|Chr01|1|10"}
```

The fixed three-call diagnostic uses a model when executed. Supply the original
three specifications in the original order; duplicate or incorrect identities are
rejected against frozen hashes before model construction. The records are not bundled.

`real_static_comparison` retains its frozen panel/model-plan hashes and cannot be
applied to a new panel through its CLI without failing those original checks.
`real_repeatability` checks the frozen configuration and panel fingerprints.
`real_retry_mechanics` requires a source audit and original retry VCF; the generic
[mechanics configuration](../config/unvalidated_real_retry_mechanics.json) is
included because it contains operational settings, not actual case records.

Phase 17/18 plans freeze the exact configured Skill filenames and raw contents.
Prepare new plans after this correction: older plans lack the required snapshots
or frozen development-exclusion hash and are deliberately rejected.
