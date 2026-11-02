#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED="${PROCESSED:-$REPO/data/processed/batch20}"
OUT="${OUT:-$PROCESSED/phased}"; LOG_DIR="${LOG_DIR:-$PROCESSED/logs}"; TMP_DIR="${TMP_DIR:-$REPO/tmp_whatshap}"
REF="${REF:-$REPO/data/reference/Egrandis_v2.fa}"
BCFTOOLS="${BCFTOOLS:-bcftools}"; WHATSHAP="${WHATSHAP:-whatshap}"; TABIX="${TABIX:-tabix}"
SAMPLE="${SAMPLE:?Set SAMPLE to the pilot sample ID}"; INPUT="${INPUT:-$PROCESSED/joint.pass.snps.vcf.gz}"; BAM="${BAM:-$PROCESSED/$SAMPLE.md.bam}"; SAMPLE_VCF="$OUT/$SAMPLE.input.vcf.gz"; OUTPUT="$OUT/$SAMPLE.phased.vcf.gz"; LOCK_DIR="$OUT/.phase_pilot.lock"; RUN_LOG="$LOG_DIR/phase_batch20_pilot.run.log"
mkdir -p "$OUT" "$LOG_DIR" "$TMP_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then echo "ERROR: phase pilot lock exists" >&2; exit 1; fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
export TMPDIR="$TMP_DIR" TMP="$TMP_DIR" TEMP="$TMP_DIR"
for tool in "$BCFTOOLS" "$WHATSHAP" "$TABIX"; do command -v "$tool" >/dev/null || { echo "ERROR: missing $tool" >&2; exit 1; }; done
[[ -s "$INPUT" && -s "$INPUT.tbi" && -s "$BAM" && -s "$BAM.bai" ]] || { echo "ERROR: pilot input/index/BAM missing" >&2; exit 1; }
timestamp() { date --iso-8601=seconds; }
echo "[$(timestamp)] PLAN sample=$SAMPLE ploidy=4" | tee -a "$RUN_LOG"
if [[ ! -s "$SAMPLE_VCF" ]]; then
  rm -f "$OUT/.${SAMPLE}.input.tmp.vcf.gz" "$OUT/.${SAMPLE}.input.tmp.vcf.gz.tbi"; start=$(date +%s); echo "[$(timestamp)] START extraction" | tee -a "$RUN_LOG"
  "$BCFTOOLS" view -s "$SAMPLE" -Oz -o "$OUT/.${SAMPLE}.input.tmp.vcf.gz" "$INPUT" >> "$LOG_DIR/$SAMPLE.phase.log" 2>&1; "$BCFTOOLS" index -t -f "$OUT/.${SAMPLE}.input.tmp.vcf.gz"
  mv "$OUT/.${SAMPLE}.input.tmp.vcf.gz" "$SAMPLE_VCF"; mv "$OUT/.${SAMPLE}.input.tmp.vcf.gz.tbi" "$SAMPLE_VCF.tbi"; echo "[$(timestamp)] DONE extraction elapsed_seconds=$(( $(date +%s)-start ))" | tee -a "$RUN_LOG"
fi
if [[ ! -s "$OUTPUT" ]]; then
  rm -f "$OUT/.${SAMPLE}.phased.tmp.vcf.gz" "$OUT/.${SAMPLE}.phased.tmp.vcf.gz.tbi"; start=$(date +%s); echo "[$(timestamp)] START WhatsHap" | tee -a "$RUN_LOG"
  "$WHATSHAP" polyphase --reference "$REF" --ploidy 4 --output "$OUT/.${SAMPLE}.phased.tmp.vcf.gz" "$SAMPLE_VCF" "$BAM" >> "$LOG_DIR/$SAMPLE.phase.log" 2>&1; "$TABIX" -f -p vcf "$OUT/.${SAMPLE}.phased.tmp.vcf.gz"; gzip -t "$OUT/.${SAMPLE}.phased.tmp.vcf.gz"
  mv "$OUT/.${SAMPLE}.phased.tmp.vcf.gz" "$OUTPUT"; mv "$OUT/.${SAMPLE}.phased.tmp.vcf.gz.tbi" "$OUTPUT.tbi"; echo "[$(timestamp)] DONE WhatsHap elapsed_seconds=$(( $(date +%s)-start )) bytes=$(stat -c %s "$OUTPUT")" | tee -a "$RUN_LOG"
fi
samples=$("$BCFTOOLS" query -l "$OUTPUT" | wc -l); records=$("$BCFTOOLS" view -H "$OUTPUT" | wc -l); echo "[$(timestamp)] VALID sample=$SAMPLE samples=$samples records=$records" | tee -a "$RUN_LOG"
[[ "$samples" -eq 1 && "$records" -gt 0 ]] || { echo "ERROR: pilot validation failed" >&2; exit 1; }; echo "[$(timestamp)] COMPLETE phase pilot" | tee -a "$RUN_LOG"
