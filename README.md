# Haplotype Mapper

Research software for constrained triage of haplotype-phasing evidence. An assessor
proposes `accept`, `retry`, or `escalate`; the deterministic controller validates
proposals, enforces acceptance safeguards and retry limits, and records decisions.
The model does not execute arbitrary commands or change genotype calls.

- `agent/`: typed state, configuration, LangGraph controller, scripted/model
  assessors, fixed-rule comparators, CLI, and bounded WhatsHap retry backend.
- `pipeline/`: input preflight, regional VCF evidence extraction, and non-executing
  GATK/bcftools/WhatsHap command plans.
- `evaluation/`: synthetic metrics, population-frequency genotype simulation,
  unlabeled real-evidence comparisons, and historical study runners.
- `scripts/`: alignment, variant calling, filtering, and batch phasing helpers.
- `tests/`: artificial unit and integration tests; fixtures are built in temporary
  directories. No biological files are needed for these tests.

This release contains software and configuration, without biological inputs, sample
lists, frozen panels, saved assessments, experimental results, model weights,
notebooks, or personal research documents.

## Install

Use Linux (or WSL), Bash, and Python 3.10, matching the working environment. Run
commands from the repository root; no editable install or custom `PYTHONPATH` is
required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r agent/requirements.txt
```

The requirements retain the source environment's exact pins, including its notebook
packages; `openpyxl==3.1.5` and `et_xmlfile==2.0.0` supply the workbook reader already
used by the source. Core imports use Pydantic, PyYAML, LangChain and LangGraph.
The requirements file is an environment snapshot, not a minimal dependency list.

For biological processing, the separate Linux Conda environment is:

```bash
conda env create -f docker/environment-bio.yml
conda activate hapmap-bio
```

It pins WhatsHap 2.8, GATK 4.6.2.0, bcftools/samtools/HTSlib 1.24, Picard 3.4.0,
SRA Toolkit 3.4.1, and classic BWA 0.7.19. The scripts call `bwa mem`; `bwa-mem2`
is also retained in the snapshot but is not its substitute. Activate this environment
for shell pipelines; the Python agent environment can remain separate. Put external
tools and Java on `PATH`. Shell helpers also require GNU utilities and `rg`
(ripgrep). These exact Linux build pins may be unavailable on other platforms;
installation from a fresh package index and Conda solve have not been verified.

## Configuration, model, and Skills

[config/parameters.yaml](config/parameters.yaml) is the canonical parameter
registry. Relative configuration paths, `paths.skills_directory`, and
`paths.output_directory` resolve against this checkout, even when a configuration
file lives elsewhere. Most CLIs accept `--config`; use absolute external input paths
to avoid ambiguity. Tool executables in the registry accept names on `PATH` or
explicit paths. Shell helpers have their own fixed settings; they do not read YAML.

The default model is Ollama `qwen2.5:7b`, temperature 0, context 8192, keep-alive
30 minutes. The configured retry strategy lowers block-cut sensitivity from 4 to 2;
the controller permits at most two retries. Scientific settings and rules are
preserved from the working source.

For model-backed runs, separately install/start [Ollama](https://docs.ollama.com/)
and provision the configured model. `langchain-ollama` is pinned in the requirements.
A remote Ollama service can be supplied through `agent.model.options.base_url` in
a private configuration copy. Other providers require their own LangChain integration
and credential environment variables named in
`agent.model.required_environment_variables`; never put credentials in YAML.
Scripted examples and tests require no model service or weights.

Both operational Skills are required and included unchanged:
[phasing_triage.md](agent/skills/phasing_triage.md) and
[retry_policy.md](agent/skills/retry_policy.md). `agent.llm.load_skills` loads
non-README Markdown in stable filename order and checks its declared status.
They are active researcher-defined policy, without biological validation.
Provenance mentions inside the preserved Skills refer to some excluded study files.

## Offline tests and artificial example

```bash
python -m unittest discover -s tests -t . -v
python -m agent.run --config config/parameters.yaml \
  --scenario retry-success --assessor scripted --output outputs/artificial-run.json
```

Other scripted scenarios are `accept`, `escalate`, and `retry-exhausted`. Use a fresh
output path for each run; replacement requires explicit `--overwrite`. Every example
record identifies its artificial origin and forbids scientific interpretation.
Tests mock model/tool calls and generate miniature BAM headers, VCFs and workbooks.

## Biological inputs and public resources

Supply these separately for real-evidence workflows:

- Coordinate-sorted BAM with an index (`.bam.bai` or `.bai`) and read groups with
  `SM` matching the selected VCF sample.
- One-sample, biallelic-SNP phased VCF. Evidence reading accepts plain or gzip
  text; plain VCF preflight does not require an index. Compressed inputs need a
  `.tbi` or `.csi` index; supply BGZF compression and an index for tool execution.
  Genotypes must have
  consistent ploidy, matching the configured value (default 4). Active retries
  additionally require the original full filtered **unphased** VCF and its index;
  do not reuse the already phased or target-intersected VCF as retry input.
- Reference FASTA, `.fai`, sequence dictionary, and aligner indexes when aligning.
  BAM, VCF, FASTA and dictionary contig names/lengths must agree.
- Zero-based, half-open target BED and a matching CSV in the same row order. CSV
  columns are `Gene`, `Target Region`, `Chromosome`, `5' Position`, `3'Position`,
  `Category`. CSV coordinates are one-based inclusive: BED start + 1 equals CSV
  start, and ends agree. Region identity includes all five identity fields.
- For population simulation: the external Candotti Appendix S1 workbook, sheet
  `Supplementary File 1`, with the five identity columns above plus `Category`,
  `Haplotype ID`, `SNP Haplotype`, `Number of SNPs`, and `E. grandis` frequencies.
  No conversion from catalogue category to per-sample triage truth is assumed.

[NCBI BioProject PRJNA873875](https://www.ncbi.nlm.nih.gov/bioproject/873875)
provides the public raw-read study. Select runs through its SRA links and use SRA
Toolkit `prefetch`/`fasterq-dump` for the chosen accessions. The project's original
reference was *E. grandis* v2.0, file `Egrandis_297_v2.0.fa.gz`, obtained through
[Phytozome's E. grandis v2.0 portal](https://phytozome-next.jgi.doe.gov/info/Egrandis_v2_0).
Use that assembly when using its catalogue coordinates; access may require a JGI
account. No alternative assembly is asserted to be interchangeable.

[Candotti et al., DOI 10.1111/tpj.16026](https://doi.org/10.1111/tpj.16026)
provides the publication and Supporting Information, including Appendix S1's target
locations and haplotypes. Obtain the workbook through the publication's supporting
files. The release does not contain a catalogue-to-BED/CSV conversion command;
prepare the exact columns and coordinate mapping above yourself. Public resources
do not supply the excluded frozen study panels or per-sample triage labels.

## Entry points

Use `python -m MODULE --help` for each runner's actual options.

| Entry point | Purpose and prerequisites |
| --- | --- |
| `agent.run` | Artificial workflow; `--assessor model` explicitly enables inference. |
| `pipeline.batch_preflight` | Validate an external sample CSV without inference. |
| `evaluation.real_evidence_census` | Summarize external manifest/BED/catalogue evidence without inference. |
| `agent.real_smoke` | Model-backed real evidence; active retries execute WhatsHap. |
| `evaluation.trials`, `evaluation.matched_comparison` | Repeated model assessment and matched artificial comparisons; these CLIs call the configured model. |
| `evaluation.population_pilot` | Requires the workbook; `--generate-only` avoids inference. |
| `evaluation.real_panel`, `evaluation.real_panel_run`, `evaluation.real_repeatability` | Prepare an external panel, run it with a model, and repeat assessments. |
| `evaluation.real_static_comparison`, `evaluation.real_retry_mechanics` | Historical comparison/mechanics code requiring saved external inputs; the latter executes tools. |

[config/batch_manifest.example.csv](config/batch_manifest.example.csv) contains
only placeholders and an artificial sample ID. Copy it to a private manifest and
replace every path. Relative manifest paths resolve against the CSV's directory.

```bash
python -m pipeline.batch_preflight --manifest /path/to/private/manifest.csv \
  --expected-ploidy 4 --output outputs/preflight.json
```

After providing real inputs and a running model, a single-sample run has this form:

```bash
python -m agent.real_smoke --config config/parameters.yaml \
  --sample YOUR_SAMPLE_ID --bam /path/to/sample.md.bam \
  --vcf /path/to/sample.phased.vcf.gz --reference /path/to/assembly.fa \
  --bed /path/to/targets.bed --region-categories /path/to/region_categories.csv \
  --retry-input-vcf /path/to/sample.filtered.unphased.vcf.gz \
  --output-dir outputs/real-run
```

The runner records `pipeline.mode=real`. This is an unscored workflow with real
external calls, distinct from the offline example above.

See [scripts/README.md](scripts/README.md) for shell input/tool overrides and
[evaluation/README.md](evaluation/README.md) for historical study prerequisites.
`pipeline.variant_calling` and `pipeline.whatshap` expose Python command-plan APIs;
they construct argument vectors without executing them.

## Limitations and reproduction

This is a software release, not an exact thesis reproduction bundle. The original
biological data, derived catalogues, manifests, frozen panels, saved model responses,
run plans and outputs are deliberately omitted. Historical hashes, exclusion rules,
seeds and model settings remain in code; they do not recreate missing artifacts.
Resuming old runs may require their exact paths and configuration fingerprints.

Real evidence has no per-sample action ground truth. Population simulation supplies
genotype truth only, and artificial test labels establish mechanics only. Regional
statistics must not be replaced with file-wide values. Retry rules are unvalidated;
short reads cannot supply absent linkage. The GATK GenotypeGVCFs ploidy provenance
question remains unresolved: its plan omits `--sample-ploidy` (null), while
HaplotypeCaller uses 4. No scientific settings were retuned for this release.

Verification uses artificial inputs and copied, isolated installed dependencies.
Real pipelines, model inference, public-data preprocessing, and fresh dependency
resolution require separate validation by the researcher.
