"""Deterministic compatibility checks for BAM, VCF, and reference inputs."""

from __future__ import annotations

import gzip
import re
import struct
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.state import InputFiles


class InputValidationReport(BaseModel):
    """Non-throwing, auditable result of real-input inspection."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    errors: list[str] = Field(default_factory=list)
    checks_completed: list[str] = Field(default_factory=list)
    discovered_indexes: dict[str, Path] = Field(default_factory=dict)
    vcf_samples: list[str] = Field(default_factory=list)
    bam_read_group_samples: list[str] = Field(default_factory=list)
    observed_genotype_ploidies: list[int] = Field(default_factory=list)
    contig_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _reference_dictionary_path(reference: Path) -> Path:
    return reference.with_suffix(".dict")


def _read_exact(handle, size: int, description: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"truncated BAM while reading {description}")
    return value


def _parse_header_tags(line: str) -> dict[str, str]:
    return {
        field[:2]: field[3:]
        for field in line.rstrip("\n").split("\t")[1:]
        if len(field) >= 4 and field[2] == ":"
    }


def _read_bam_header(path: Path) -> tuple[dict[str, int], list[str], int]:
    """Read BAM header metadata directly through its BGZF-compatible gzip stream."""

    with gzip.open(path, "rb") as handle:
        if _read_exact(handle, 4, "magic") != b"BAM\x01":
            raise ValueError("file does not begin with BAM magic")
        text_length = struct.unpack("<i", _read_exact(handle, 4, "header length"))[0]
        if text_length < 0 or text_length > 100_000_000:
            raise ValueError(f"invalid BAM header length: {text_length}")
        header_text = _read_exact(handle, text_length, "header text").decode("utf-8")

    contigs: dict[str, int] = {}
    read_group_samples: list[str] = []
    read_group_count = 0
    for line in header_text.splitlines():
        if line.startswith("@SQ\t"):
            tags = _parse_header_tags(line)
            if "SN" not in tags or "LN" not in tags:
                raise ValueError("BAM @SQ entry lacks SN or LN")
            contigs[tags["SN"]] = int(tags["LN"])
        elif line.startswith("@RG\t"):
            read_group_count += 1
            tags = _parse_header_tags(line)
            if tags.get("SM"):
                read_group_samples.append(tags["SM"])
    return contigs, read_group_samples, read_group_count


def _read_fai(path: Path) -> dict[str, int]:
    contigs: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"FAI row {line_number} has fewer than two fields")
            contigs[fields[0]] = int(fields[1])
    return contigs


def _read_sequence_dictionary(path: Path) -> dict[str, int]:
    contigs: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("@SQ\t"):
                continue
            tags = _parse_header_tags(line)
            if "SN" not in tags or "LN" not in tags:
                raise ValueError("reference dictionary @SQ entry lacks SN or LN")
            contigs[tags["SN"]] = int(tags["LN"])
    if not contigs:
        raise ValueError("reference dictionary has no @SQ entries")
    return contigs


def _vcf_contig(line: str) -> tuple[str, int] | None:
    identifier = re.search(r"(?:<|,)ID=([^,>]+)", line)
    length = re.search(r"(?:^|,)length=([0-9]+)", line, flags=re.IGNORECASE)
    if identifier is None or length is None:
        return None
    return identifier.group(1), int(length.group(1))


def _read_vcf_metadata(
    path: Path,
    expected_sample: str | None,
) -> tuple[list[str], dict[str, int], set[int]]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    samples: list[str] = []
    contigs: dict[str, int] = {}
    ploidies: set[int] = set()
    sample_column: int | None = None

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##contig=<"):
                parsed = _vcf_contig(line)
                if parsed is not None:
                    contigs[parsed[0]] = parsed[1]
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                if expected_sample is not None and expected_sample in samples:
                    sample_column = 9 + samples.index(expected_sample)
                elif expected_sample is None and len(samples) == 1:
                    sample_column = 9
                continue
            if line.startswith("#"):
                continue
            if not samples:
                raise ValueError("VCF data appeared before the #CHROM header")
            if sample_column is None:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= sample_column:
                raise ValueError(f"VCF row {line_number} lacks the selected sample")
            format_names = fields[8].split(":")
            if "GT" not in format_names:
                continue
            sample_values = fields[sample_column].split(":")
            genotype_index = format_names.index("GT")
            if genotype_index >= len(sample_values):
                continue
            genotype = sample_values[genotype_index]
            if genotype in {"", "."}:
                continue
            ploidies.add(len(re.split(r"[/|]", genotype)))

    if not samples:
        raise ValueError("VCF has no #CHROM header")
    return samples, contigs, ploidies


def _summarize(values: list[str], limit: int = 5) -> str:
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def _check_contigs(
    source_name: str,
    source: dict[str, int],
    reference: dict[str, int],
    errors: list[str],
    *,
    exact: bool = False,
) -> None:
    if not source:
        errors.append(f"{source_name} declares no contigs")
        return
    missing = sorted(set(source) - set(reference))
    if missing:
        errors.append(
            f"{source_name} contigs absent from reference: {_summarize(missing)}"
        )
    length_mismatches = sorted(
        name
        for name in set(source) & set(reference)
        if source[name] != reference[name]
    )
    if length_mismatches:
        errors.append(
            f"{source_name}/reference length mismatches: "
            f"{_summarize(length_mismatches)}"
        )
    if exact:
        absent = sorted(set(reference) - set(source))
        if absent:
            errors.append(
                f"{source_name} is missing reference contigs: {_summarize(absent)}"
            )


def inspect_real_inputs(
    inputs: InputFiles,
    *,
    expected_sample: str | None = None,
    expected_ploidy: int | None = None,
) -> InputValidationReport:
    """Inspect required files, indexes, sample metadata, ploidy, and contigs."""

    if expected_ploidy is not None and expected_ploidy < 1:
        raise ValueError("expected_ploidy must be at least 1")

    errors: list[str] = []
    checks: list[str] = []
    indexes: dict[str, Path] = {}
    required = {
        "bam": inputs.bam,
        "vcf": inputs.vcf,
        "reference_fasta": inputs.reference_fasta,
    }
    for label, path in required.items():
        if path is None:
            errors.append(f"{label} path is required")
        elif not path.is_file():
            errors.append(f"{label} file does not exist: {path}")

    bam_contigs: dict[str, int] = {}
    read_group_samples: list[str] = []
    read_group_count = 0
    if inputs.bam is not None and inputs.bam.is_file():
        if inputs.bam.suffix.lower() != ".bam":
            errors.append(f"bam must use the .bam extension: {inputs.bam}")
        bam_index = _first_existing(
            [Path(f"{inputs.bam}.bai"), inputs.bam.with_suffix(".bai")]
        )
        if bam_index is None:
            errors.append(f"BAM index not found for: {inputs.bam}")
        else:
            indexes["bam"] = bam_index
        try:
            bam_contigs, read_group_samples, read_group_count = _read_bam_header(
                inputs.bam
            )
            checks.append("bam_header")
        except (OSError, EOFError, UnicodeDecodeError, ValueError, struct.error) as exc:
            errors.append(f"could not inspect BAM header: {exc}")

    vcf_samples: list[str] = []
    vcf_contigs: dict[str, int] = {}
    ploidies: set[int] = set()
    if inputs.vcf is not None and inputs.vcf.is_file():
        vcf_name = inputs.vcf.name.lower()
        if not (vcf_name.endswith(".vcf") or vcf_name.endswith(".vcf.gz")):
            errors.append(f"vcf must use .vcf or .vcf.gz: {inputs.vcf}")
        if vcf_name.endswith(".vcf.gz"):
            vcf_index = _first_existing(
                [Path(f"{inputs.vcf}.tbi"), Path(f"{inputs.vcf}.csi")]
            )
            if vcf_index is None:
                errors.append(f"compressed VCF index not found for: {inputs.vcf}")
            else:
                indexes["vcf"] = vcf_index
        try:
            vcf_samples, vcf_contigs, ploidies = _read_vcf_metadata(
                inputs.vcf, expected_sample
            )
            checks.append("vcf_header_and_genotypes")
        except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
            errors.append(f"could not inspect VCF: {exc}")

    reference_contigs: dict[str, int] = {}
    dictionary_contigs: dict[str, int] = {}
    if inputs.reference_fasta is not None and inputs.reference_fasta.is_file():
        reference = inputs.reference_fasta
        if not reference.name.lower().endswith((".fa", ".fasta", ".fna")):
            errors.append(
                "reference_fasta must use .fa, .fasta, or .fna: " f"{reference}"
            )
        fasta_index = Path(f"{reference}.fai")
        if not fasta_index.is_file():
            errors.append(f"FASTA index not found: {fasta_index}")
        else:
            indexes["reference_fasta"] = fasta_index
            try:
                reference_contigs = _read_fai(fasta_index)
                checks.append("reference_fai")
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"could not inspect FASTA index: {exc}")

        dictionary = _reference_dictionary_path(reference)
        if not dictionary.is_file():
            errors.append(f"reference sequence dictionary not found: {dictionary}")
        else:
            indexes["reference_dictionary"] = dictionary
            try:
                dictionary_contigs = _read_sequence_dictionary(dictionary)
                checks.append("reference_dictionary")
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"could not inspect reference dictionary: {exc}")

    if read_group_count == 0 and "bam_header" in checks:
        errors.append("BAM header has no read groups")
    elif read_group_count > len(read_group_samples):
        errors.append("one or more BAM read groups lack an SM sample value")

    if expected_sample is not None:
        if vcf_samples and expected_sample not in vcf_samples:
            errors.append(
                f"VCF samples do not contain expected sample '{expected_sample}'"
            )
        if read_group_count and expected_sample not in read_group_samples:
            errors.append(
                f"BAM read-group samples do not contain expected sample "
                f"'{expected_sample}'"
            )
    elif vcf_samples and read_group_samples and not (
        set(vcf_samples) & set(read_group_samples)
    ):
        errors.append("VCF samples and BAM read-group samples do not overlap")
    elif expected_sample is None and len(vcf_samples) > 1:
        errors.append("expected_sample is required for a multi-sample VCF")
    if vcf_samples or read_group_count:
        checks.append("sample_and_read_groups")

    if vcf_samples:
        if not ploidies:
            errors.append("could not determine genotype ploidy from the selected sample")
        elif len(ploidies) > 1:
            errors.append(
                "selected VCF sample contains inconsistent genotype ploidies: "
                + ", ".join(str(value) for value in sorted(ploidies))
            )
        elif expected_ploidy is not None and expected_ploidy not in ploidies:
            observed = next(iter(ploidies))
            errors.append(
                f"VCF genotype ploidy {observed} does not match configured ploidy "
                f"{expected_ploidy}"
            )
        checks.append("genotype_ploidy")

    if reference_contigs:
        _check_contigs("BAM", bam_contigs, reference_contigs, errors)
        _check_contigs("VCF", vcf_contigs, reference_contigs, errors)
        if dictionary_contigs:
            _check_contigs(
                "reference dictionary",
                dictionary_contigs,
                reference_contigs,
                errors,
                exact=True,
            )
        checks.append("contig_compatibility")

    return InputValidationReport(
        errors=errors,
        checks_completed=checks,
        discovered_indexes=indexes,
        vcf_samples=vcf_samples,
        bam_read_group_samples=sorted(set(read_group_samples)),
        observed_genotype_ploidies=sorted(ploidies),
        contig_counts={
            "bam": len(bam_contigs),
            "vcf": len(vcf_contigs),
            "reference_fai": len(reference_contigs),
            "reference_dictionary": len(dictionary_contigs),
        },
    )


__all__ = [
    "InputValidationReport",
    "inspect_real_inputs",
]
