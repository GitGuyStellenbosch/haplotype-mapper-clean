#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED="${PROCESSED:-$REPO/data/processed/batch20}"
OUT="${OUT:-$PROCESSED/phased}"; LOG="${LOG:-$PROCESSED/logs}"; TMP="${TMP_DIR:-$REPO/tmp_whatshap}"
ACCESSIONS="${ACCESSIONS:?Set ACCESSIONS to the remaining-sample accession list (exclude the pilot)}"
[[ -s "$ACCESSIONS" ]] || { echo "ERROR: accession list missing: $ACCESSIONS" >&2; exit 1; }
mapfile -t samples < <(awk 'NF {print $0}' "$ACCESSIONS")
[[ "${#samples[@]}" -eq 19 ]] || { echo "ERROR: expected 19 remaining samples" >&2; exit 1; }
[[ "$(printf '%s\n' "${samples[@]}" | sort -u | wc -l)" -eq 19 ]] || { echo "ERROR: accession IDs must be unique" >&2; exit 1; }
BCF="${BCFTOOLS:-bcftools}"; WS="${WHATSHAP:-whatshap}"; TABIX="${TABIX:-tabix}"
INPUT="${INPUT:-$PROCESSED/joint.pass.snps.vcf.gz}"; REF="${REF:-$REPO/data/reference/Egrandis_v2.fa}"; LOCK="$OUT/.phase_batch20.lock"; RUNLOG="$LOG/phase_batch20.run.log"; MAX_JOBS=2
mkdir -p "$OUT" "$LOG" "$TMP"
if ! mkdir "$LOCK" 2>/dev/null; then echo "ERROR: phase batch lock exists" >&2; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export TMPDIR="$TMP" TMP="$TMP" TEMP="$TMP"
for x in "$BCF" "$WS" "$TABIX"; do command -v "$x" >/dev/null || { echo "ERROR: missing executable $x" >&2; exit 1; }; done
[[ -s "$INPUT" && -s "$INPUT.tbi" ]] || { echo "ERROR: filtered input missing" >&2; exit 1; }
ts(){ date --iso-8601=seconds; }
echo "[$(ts)] PLAN remaining_samples=19 max_concurrent=$MAX_JOBS ploidy=4" | tee -a "$RUNLOG"
phase_one(){
  local s="$1"; local bam="$PROCESSED/$s.md.bam"; local sv="$OUT/$s.input.vcf.gz"; local out="$OUT/$s.phased.vcf.gz"; local log="$LOG/$s.phase.log"; local start
  [[ -s "$bam" && -s "$bam.bai" ]] || { echo "[$(ts)] FAIL $s missing BAM/index" | tee -a "$RUNLOG"; return 1; }
  if [[ ! -s "$sv" ]]; then
    rm -f "$OUT/.$s.input.tmp.vcf.gz" "$OUT/.$s.input.tmp.vcf.gz.tbi"; start=$(date +%s); echo "[$(ts)] START $s extraction" | tee -a "$RUNLOG"
    "$BCF" view -s "$s" -Oz -o "$OUT/.$s.input.tmp.vcf.gz" "$INPUT" >> "$log" 2>&1; "$BCF" index -t -f "$OUT/.$s.input.tmp.vcf.gz"; mv "$OUT/.$s.input.tmp.vcf.gz" "$sv"; mv "$OUT/.$s.input.tmp.vcf.gz.tbi" "$sv.tbi"
    echo "[$(ts)] DONE $s extraction elapsed_seconds=$(( $(date +%s)-start ))" | tee -a "$RUNLOG"
  fi
  if [[ ! -s "$out" ]]; then
    rm -f "$OUT/.$s.phased.tmp.vcf.gz" "$OUT/.$s.phased.tmp.vcf.gz.tbi"; start=$(date +%s); echo "[$(ts)] START $s WhatsHap" | tee -a "$RUNLOG"
    "$WS" polyphase --reference "$REF" --ploidy 4 --output "$OUT/.$s.phased.tmp.vcf.gz" "$sv" "$bam" >> "$log" 2>&1; "$TABIX" -f -p vcf "$OUT/.$s.phased.tmp.vcf.gz"; gzip -t "$OUT/.$s.phased.tmp.vcf.gz"; mv "$OUT/.$s.phased.tmp.vcf.gz" "$out"; mv "$OUT/.$s.phased.tmp.vcf.gz.tbi" "$out.tbi"
    echo "[$(ts)] DONE $s WhatsHap elapsed_seconds=$(( $(date +%s)-start )) bytes=$(stat -c %s "$out")" | tee -a "$RUNLOG"
  fi
  gzip -t "$out"; [[ -s "$out.tbi" ]]; local n; n=$("$BCF" query -l "$out" | wc -l); [[ "$n" -eq 1 ]] || { echo "[$(ts)] FAIL $s validation samples=$n" | tee -a "$RUNLOG"; return 1; }; echo "[$(ts)] VALID $s records=$("$BCF" view -H "$out" | wc -l)" | tee -a "$RUNLOG"
}
active=0; failed=0
for s in "${samples[@]}"; do phase_one "$s" & active=$((active+1)); if (( active >= MAX_JOBS )); then if ! wait -n; then failed=$((failed+1)); fi; active=$((active-1)); fi; done
while (( active > 0 )); do if ! wait -n; then failed=$((failed+1)); fi; active=$((active-1)); done
echo "[$(ts)] COMPLETE remaining_phasing failed=$failed" | tee -a "$RUNLOG"; [[ "$failed" -eq 0 ]]
