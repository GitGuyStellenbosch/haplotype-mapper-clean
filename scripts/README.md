# Pipeline helpers

Run with Bash from any directory. Tools use `PATH`; activate the separate
bioinformatics environment first. These scripts execute real processing and create
outputs/logs. They are not the offline example. Supply your own biological inputs,
indexes and sample lists. No list of actual study samples is bundled.

| Script | Inputs and location overrides |
| --- | --- |
| `batch_align_and_call.sh` | Required `ACCESSIONS` (one ID per line); `REF`, `BED`, `RAW`, `OUT`. FASTQs: `$RAW/ID_1.fastq`, `$RAW/ID_2.fastq`. Uses `bwa`, `samtools`, `picard`, `gatk` on `PATH`. |
| `joint_genotype.sh` | Required `ACCESSIONS`; `REF`, `INTERVALS` (Picard interval list), `OUT` (GVCFs and outputs), `TMP_DIR`. Requires `gatk`, `bcftools`, `tabix`, `rg`. Keeps 16 interval shards with all cohort samples in every shard. |
| `filter_batch20.sh` | `REF`, `OUT` (containing `joint.vcf.gz`), `TMP_DIR`; tool overrides `GATK`, `BCFTOOLS`, `TABIX`. Java uses `PATH`. Preserves its 20-sample gate. |
| `phase_batch20_pilot.sh` | Required `SAMPLE`; `PROCESSED`, `INPUT` (joint PASS SNP VCF), `BAM`, `REF`, `OUT`, `LOG_DIR`, `TMP_DIR`; tool overrides `BCFTOOLS`, `WHATSHAP`, `TABIX`. |
| `phase_batch20.sh` | Required `ACCESSIONS` with exactly 19 remaining IDs, excluding the pilot; `PROCESSED`, `INPUT`, `REF`, `OUT`, `LOG`, `TMP_DIR`; tool overrides `BCFTOOLS`, `WHATSHAP`, `TABIX`. Preserves two concurrent jobs. |

Defaults are paths inside this checkout: `data/reference/Egrandis_v2.fa`,
`data/processed/batch20`, and `data/ground_truth/derived/`. These files do not ship.
Set overrides to absolute paths. `OUT` is the phased output directory in phasing
scripts and the processed batch directory in earlier stages. Example invocation
(after replacing placeholders):

```bash
ACCESSIONS=/path/to/private/accessions.txt REF=/path/to/reference/assembly.fa \
  BED=/path/to/targets_no_mito.bed RAW=/path/to/reads OUT=/path/to/work/batch \
  bash scripts/batch_align_and_call.sh
```

The non-mitochondrial BED/interval list, FASTA/aligner indexes and GVCFs must be
prepared separately; this release provides no automatic download/preparation step.
The scripts retain fixed ploidy, filters, cohort gates, skip-existing behavior,
locks and temporary-file handling from the research implementation. Use fresh work
directories; a pre-existing output can be skipped without a provenance check.
