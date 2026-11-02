#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
REF="${REF:-$REPO/data/reference/Egrandis_v2.fa}"
BED="${BED:-$REPO/data/ground_truth/derived/target_regions_no_mito.bed}"
ACCESSIONS="${ACCESSIONS:?Set ACCESSIONS to a file containing one sample accession per line}"
RAW="${RAW:-$REPO/data/raw}"
OUT="${OUT:-$REPO/data/processed/batch20}"
for tool in bwa samtools picard gatk; do
    command -v "$tool" >/dev/null || { echo "ERROR: missing executable: $tool" >&2; exit 1; }
done
for input in "$REF" "$BED" "$ACCESSIONS"; do
    [[ -s "$input" ]] || { echo "ERROR: missing input: $input" >&2; exit 1; }
done
LOG=$OUT/logs

mkdir -p "$OUT" "$LOG"

while IFS= read -r acc || [[ -n "$acc" ]]; do
    [[ -z "$acc" ]] && continue
    echo "=== $acc ==="
    BAM="$OUT/${acc}.bam"
    MDBAM="$OUT/${acc}.md.bam"
    GVCF="$OUT/${acc}.g.vcf.gz"

    if [[ -f "$GVCF" ]]; then
        echo "Already done, skipping: $acc"
        continue
    fi

    if [[ ! -f "$MDBAM" ]]; then
        bwa mem -t 4 -R "@RG\tID:${acc}\tSM:${acc}\tPL:ILLUMINA\tLB:lib1" \
            "$REF" "$RAW/${acc}_1.fastq" "$RAW/${acc}_2.fastq" \
            2> "$LOG/${acc}.bwa.log" \
            | samtools sort -o "$BAM"
        samtools index "$BAM"

        picard MarkDuplicates \
            I="$BAM" O="$MDBAM" M="$LOG/${acc}.dup_metrics.txt" \
            2> "$LOG/${acc}.markdup.log"
        samtools index "$MDBAM"
    fi

    gatk HaplotypeCaller \
        -R "$REF" -I "$MDBAM" \
        -L "$BED" \
        -ploidy 4 -ERC GVCF \
        -O "$GVCF" \
        2> "$LOG/${acc}.haplotypecaller.log"

    echo "Done: $acc"
done < "$ACCESSIONS"

echo "All samples processed."