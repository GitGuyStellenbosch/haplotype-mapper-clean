#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ACCESSIONS="${ACCESSIONS:?Set ACCESSIONS to the cohort accession list}"
OUT="${OUT:-$REPO/data/processed/batch20}"
INTERVALS="${INTERVALS:-$REPO/data/ground_truth/derived/target_regions_no_mito.interval_list}"
REF="${REF:-$REPO/data/reference/Egrandis_v2.fa}"
TMP_DIR="${TMP_DIR:-$REPO/tmp_gatk}"
JAVA_OPTIONS="-Djava.io.tmpdir=$TMP_DIR -Xmx4g"
SHARD_DIR="$OUT/joint_shards"
SHARD_COUNT=16

for tool in gatk bcftools tabix rg; do
    command -v "$tool" >/dev/null || { echo "ERROR: missing executable: $tool" >&2; exit 1; }
done
for input in "$ACCESSIONS" "$REF" "$INTERVALS"; do
    [[ -s "$input" ]] || { echo "ERROR: missing input: $input" >&2; exit 1; }
done
mkdir -p "$TMP_DIR"

available_mb() {
    df -Pm "$1" | awk 'NR == 2 {print $4}'
}

if (( $(available_mb /tmp) < 512 )); then
    echo "ERROR: /tmp has less than 512 MB available; refusing to start." >&2
    df -h /tmp >&2
    exit 1
fi

if (( $(available_mb "$REPO") < 10240 )); then
    echo "ERROR: project disk has less than 10 GB available; refusing to start." >&2
    df -h "$REPO" >&2
    exit 1
fi

# Keep shell, native-library, and Java temporary files off WSL's small /tmp tmpfs.
export TMPDIR="$TMP_DIR"
export TMP="$TMP_DIR"
export TEMP="$TMP_DIR"

expected=$(awk 'NF {n++} END {print n + 0}' "$ACCESSIONS")
actual=$(find "$OUT" -maxdepth 1 -name '*.g.vcf.gz' | wc -l)
if [[ "$actual" -ne "$expected" ]]; then
    echo "Only $actual of $expected GVCFs present. Not proceeding." >&2
    exit 1
fi

if [[ ! -s "$INTERVALS" ]]; then
    echo "ERROR: interval list is missing or empty: $INTERVALS" >&2
    exit 1
fi

GVCF_ARGS=()
while IFS= read -r acc || [[ -n "$acc" ]]; do
    [[ -z "$acc" ]] && continue
    GVCF_ARGS+=( -V "$OUT/${acc}.g.vcf.gz" )
done < "$ACCESSIONS"

run_timed() {
    local label="$1"
    shift
    local start end
    start=$(date +%s)
    echo "[$(date --iso-8601=seconds)] START $label"
    "$@"
    end=$(date +%s)
    echo "[$(date --iso-8601=seconds)] DONE $label elapsed_seconds=$((end - start))"
}

mkdir -p "$SHARD_DIR"

interval_count=$(rg -v '^@' "$INTERVALS" | wc -l)
echo "[$(date --iso-8601=seconds)] PLAN samples=$expected intervals=$interval_count shards=$SHARD_COUNT java_heap=4g"

# Split the interval list while retaining its Picard header. Each shard still
# contains all samples, so joint genotyping remains cohort-wide per interval.
for ((shard = 1; shard <= SHARD_COUNT; shard++)); do
    shard_file=$(printf '%s/intervals_%02d.interval_list' "$SHARD_DIR" "$shard")
    if [[ ! -s "$shard_file" ]]; then
        {
            rg '^@' "$INTERVALS"
            rg -v '^@' "$INTERVALS" | awk -v shard="$shard" -v count="$SHARD_COUNT" \
                '((NR - 1) % count) + 1 == shard'
        } > "$shard_file"
    fi
done

completed=0
for ((shard = 1; shard <= SHARD_COUNT; shard++)); do
    shard_id=$(printf '%02d' "$shard")
    shard_file="$SHARD_DIR/intervals_${shard_id}.interval_list"
    shard_vcf="$SHARD_DIR/joint_${shard_id}.vcf.gz"
    shard_db="$SHARD_DIR/genomicsdb_${shard_id}"
    shard_intervals=$(rg -v '^@' "$shard_file" | wc -l)

    if [[ -s "$shard_vcf" ]]; then
        completed=$((completed + 1))
        echo "[$(date --iso-8601=seconds)] SKIP shard=$shard/$SHARD_COUNT intervals=$shard_intervals output_exists=1 completed=$completed/$SHARD_COUNT"
        continue
    fi

    rm -rf "$shard_db"
    echo "[$(date --iso-8601=seconds)] PROGRESS shard=$shard/$SHARD_COUNT intervals=$shard_intervals completed=$completed/$SHARD_COUNT"
    run_timed "GenomicsDBImport shard=$shard_id" \
        gatk --java-options "$JAVA_OPTIONS" GenomicsDBImport \
        "${GVCF_ARGS[@]}" \
        --genomicsdb-workspace-path "$shard_db" \
        --tmp-dir "$TMP_DIR" \
        -L "$shard_file"

    run_timed "GenotypeGVCFs shard=$shard_id" \
        gatk --java-options "$JAVA_OPTIONS" GenotypeGVCFs \
        -R "$REF" \
        -V "gendb://$shard_db" \
        -O "$shard_vcf" \
        -L "$shard_file"

    if [[ ! -s "$shard_vcf" ]]; then
        echo "ERROR: shard VCF is missing or empty: $shard_vcf" >&2
        exit 1
    fi
    tabix -f -p vcf "$shard_vcf"
    rm -rf "$shard_db"
    completed=$((completed + 1))
    echo "[$(date --iso-8601=seconds)] COMPLETE shard=$shard/$SHARD_COUNT completed=$completed/$SHARD_COUNT vcf_bytes=$(stat -c %s "$shard_vcf")"
done

shard_vcfs=()
for ((shard = 1; shard <= SHARD_COUNT; shard++)); do
    shard_vcfs+=("$SHARD_DIR/joint_$(printf '%02d' "$shard").vcf.gz")
done

run_timed "bcftools concat all shards" \
    bcftools concat -a -Oz -o "$OUT/joint.vcf.gz" "${shard_vcfs[@]}"
tabix -f -p vcf "$OUT/joint.vcf.gz"

if [[ ! -s "$OUT/joint.vcf.gz" ]]; then
    echo "ERROR: joint VCF was not created or is empty: $OUT/joint.vcf.gz" >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] VALID joint VCF size=$(stat -c %s "$OUT/joint.vcf.gz") bytes"
sample_count=$(bcftools query -l "$OUT/joint.vcf.gz" | wc -l)
echo "[$(date --iso-8601=seconds)] VALID joint VCF samples=$sample_count expected=$expected"
if [[ "$sample_count" -ne "$expected" ]]; then
    echo "ERROR: joint VCF sample count does not match expected cohort." >&2
    exit 1
fi
