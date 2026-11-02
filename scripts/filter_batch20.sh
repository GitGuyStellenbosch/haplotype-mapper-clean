#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT="${OUT:-$REPO/data/processed/batch20}"
REF="${REF:-$REPO/data/reference/Egrandis_v2.fa}"
LOG_DIR="$OUT/logs"
TMP_DIR="${TMP_DIR:-$REPO/tmp_gatk}"
VCF="$OUT/joint.vcf.gz"
FILTERED="$OUT/joint.filtered.vcf.gz"
PASS_SNPS="$OUT/joint.pass.snps.vcf.gz"
LOCK_DIR="$OUT/.filter_batch20.lock"
GATK="${GATK:-gatk}"
BCFTOOLS="${BCFTOOLS:-bcftools}"
TABIX="${TABIX:-tabix}"
JAVA_OPTIONS="-Djava.io.tmpdir=$TMP_DIR -Xmx4g"
RUN_LOG="$LOG_DIR/joint_filter.run.log"

mkdir -p "$LOG_DIR" "$TMP_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "ERROR: another batch20 filtering run appears to be active: $LOCK_DIR" >&2
    exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

available_mb() { df -Pm "$1" | awk 'NR == 2 {print $4}'; }
if (( $(available_mb /tmp) < 512 )); then
    echo "ERROR: /tmp has less than 512 MB available." >&2
    exit 1
fi
if (( $(available_mb "$REPO") < 10240 )); then
    echo "ERROR: project disk has less than 10 GB available." >&2
    exit 1
fi
export TMPDIR="$TMP_DIR" TMP="$TMP_DIR" TEMP="$TMP_DIR"

for tool in "$GATK" "$BCFTOOLS" "$TABIX" java; do
    command -v "$tool" >/dev/null || { echo "ERROR: missing executable: $tool" >&2; exit 1; }
done
[[ -s "$VCF" && -s "$VCF.tbi" ]] || {
    echo "ERROR: joint VCF or index is missing: $VCF" >&2
    exit 1
}

timestamp() { date --iso-8601=seconds; }
echo "[$(timestamp)] PLAN input=$VCF filtered=$FILTERED pass_snps=$PASS_SNPS java_heap=4g" | tee -a "$RUN_LOG"

if [[ ! -s "$FILTERED" ]]; then
    filtered_tmp="$OUT/.joint.filtered.tmp.vcf.gz"
    rm -f "$filtered_tmp" "$filtered_tmp.tbi"
    start=$(date +%s)
    echo "[$(timestamp)] START VariantFiltration" | tee -a "$RUN_LOG"
    "$GATK" --java-options "$JAVA_OPTIONS" VariantFiltration \
        -R "$REF" \
        -V "$VCF" \
        --tmp-dir "$TMP_DIR" \
        --filter-expression "QD < 2.0" --filter-name "QD2" \
        --filter-expression "QUAL < 30.0" --filter-name "QUAL30" \
        --filter-expression "SOR > 3.0" --filter-name "SOR3" \
        --filter-expression "FS > 60.0" --filter-name "FS60" \
        --filter-expression "MQ < 40.0" --filter-name "MQ40" \
        --filter-expression "MQRankSum < -12.5" --filter-name "MQRankSum-12.5" \
        --filter-expression "ReadPosRankSum < -8.0" --filter-name "ReadPosRankSum-8" \
        -O "$filtered_tmp" \
        >> "$LOG_DIR/joint.variantfiltration.log" 2>&1
    "$TABIX" -f -p vcf "$filtered_tmp"
    gzip -t "$filtered_tmp"
    mv "$filtered_tmp" "$FILTERED"
    mv "$filtered_tmp.tbi" "$FILTERED.tbi"
    end=$(date +%s)
    echo "[$(timestamp)] DONE VariantFiltration elapsed_seconds=$((end - start)) bytes=$(stat -c %s "$FILTERED")" | tee -a "$RUN_LOG"
else
    echo "[$(timestamp)] SKIP VariantFiltration output_exists=1" | tee -a "$RUN_LOG"
fi

if [[ ! -s "$PASS_SNPS" ]]; then
    pass_tmp="$OUT/.joint.pass.snps.tmp.vcf.gz"
    rm -f "$pass_tmp" "$pass_tmp.tbi"
    start=$(date +%s)
    echo "[$(timestamp)] START bcftools PASS_biallelic_SNP" | tee -a "$RUN_LOG"
    "$BCFTOOLS" view -f PASS -m2 -M2 -v snps \
        -Oz -o "$pass_tmp" "$FILTERED" \
        >> "$LOG_DIR/joint.pass_snps.log" 2>&1
    "$BCFTOOLS" index -t -f "$pass_tmp"
    gzip -t "$pass_tmp"
    mv "$pass_tmp" "$PASS_SNPS"
    mv "$pass_tmp.tbi" "$PASS_SNPS.tbi"
    end=$(date +%s)
    echo "[$(timestamp)] DONE bcftools PASS_biallelic_SNP elapsed_seconds=$((end - start)) bytes=$(stat -c %s "$PASS_SNPS")" | tee -a "$RUN_LOG"
else
    echo "[$(timestamp)] SKIP bcftools PASS_biallelic_SNP output_exists=1" | tee -a "$RUN_LOG"
fi

sample_count=$("$BCFTOOLS" query -l "$PASS_SNPS" | wc -l)
record_count=$("$BCFTOOLS" view -H "$PASS_SNPS" | wc -l)
echo "[$(timestamp)] VALID pass_snps samples=$sample_count records=$record_count" | tee -a "$RUN_LOG"
if [[ "$sample_count" -ne 20 || "$record_count" -le 0 ]]; then
    echo "ERROR: filtered VCF failed the 20-sample/non-zero-record gate." | tee -a "$RUN_LOG" >&2
    exit 1
fi
echo "[$(timestamp)] COMPLETE batch20 filtering" | tee -a "$RUN_LOG"
