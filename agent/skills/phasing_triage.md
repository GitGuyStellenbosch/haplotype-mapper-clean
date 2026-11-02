# Phasing triage

**Status:** Active research policy, version research-skills-20261006-v3; not scientifically validated.

Assess one sample and target region. Accept retains the observed result, retry tests one allowed intervention, and escalate stops automation for review. These are operational decisions, not certified biological findings. [POLICY: researcher-defined]

## Read the evidence

A heterozygous SNP is a single-base variant with different called alleles. Phasing assigns alleles to chromosome copies. `outcome=phased` means every observed heterozygous genotype uses phased separators, including single-site cases; it does not establish correct linkage. `call_rate` is fully called / retained regional SNP records, not phased fraction or a family quality measure. A phase set groups phased variants; `additional_metrics.phase_set_count` counts distinct PS identifiers. [SOURCE: pipeline/real_evidence.py]

Treat null fields and absent metrics as unknown. In particular, unknown `block_count`, `block_n50`, `haplotype_count`, and `mendelian_errors` are not zero. With `family_data_available=false`, claim no family validation. Use actual values in the rationale, not invented depth, linkage, pedigree, or error measurements. Confidence expresses policy judgment, not a calibrated biological probability. [POLICY; schema: agent/state.py]

## Choose an action in this order

1. **Escalate** `not_run`, `failed`, `unexpected_haplotype_count`, inconsistent evidence, or unresolved sample/ploidy/PS format problems in `tool_messages` or `warnings`. Counts must satisfy phased + unphased = heterozygous when known. `phased` requires all observed heterozygous SNPs phased; `partially_phased` requires both phased and unphased SNPs. Distinguish input problems from biological uncertainty. Multiple valid PS identifiers are fragmentation, not a PS format error. Routine no-truth/no-pedigree notices alone do not prohibit assessment. [POLICY]
2. **Escalate after genotype changes.** If `additional_metrics.retry_genotype_change_count>0`, stop for review. Active phasing-only retries now reject such outputs before regional reassessment; this check also protects older records. [POLICY; metric: agent/real_retry.py]
3. **No informative variants:** with `heterozygous_snp_count=0` and `outcome=no_phase_informative_variants`, escalate as “phasing not applicable.” This is neither phasing failure nor success. `call_rate=0` means no fully called SNP genotypes; an intermediate value means incomplete calls; `call_rate=1` means called genotypes without observed heterozygosity. Do not retry this condition alone. The biological handling of not-applicable cases remains [OPEN]. [POLICY]
4. **Accept complete observed linkage:** require `outcome=phased`, at least two heterozygous SNPs, phased count equal to heterozygous count, unphased count 0, `call_rate=1`, and `additional_metrics.phase_set_count=1`, with no problem above. Any unknown required value prevents acceptance. These exact conditions reuse the recorded Phase 14 operational rule; they are research choices, not universal biological thresholds. [POLICY; docs/2026-10-06_phase14_static_comparator.md]
5. **Retry or escalate:** for partial/unphased evidence or multiple phase sets, apply `retry_policy.md`. Select an eligible unused allowed strategy; otherwise escalate with the missing evidence or stopping reason. Unknown block statistics, haplotype count, and unavailable Mendelian evidence do not veto a diagnostic block-cut retry when its stated eligibility is met. A single heterozygous site has no within-region relationship to phase, so do not retry merely to create one. [POLICY]

## Scope and interpretation

The catalogue has 14,071 targets, median recorded target size 124 bases, and 542 single-position SNP spans. Nearby/overlapping targets sometimes share phase sets. Short reads cannot connect every distant variant; parameters cannot manufacture absent spanning reads. Fragmentation alone does not identify its cause. [SOURCE: data/ground_truth/derived/region_categories.csv; docs/2026-09-24_retry_execution_scope_and_output_isolation.md; W28]

The ploidy-four pipeline is a calling/phasing representation, not proof of four distinct biological haplotypes. Candotti used subsequent genotype adjustment and family checks that this reconstruction does not reproduce. Do not diagnose duplication from haplotype count, change ploidy to force a result, or map Category 1/2/3 labels or population frequencies to actions. [SOURCE: Candotti et al. 2023, methods; POLICY]

Sources: [WhatsHap 2.8 guide (W28)](https://raw.githubusercontent.com/whatshap/whatshap/v2.8/doc/guide.rst); [Candotti et al. 2023](https://doi.org/10.1111/tpj.16026). Published facts support tool interpretation; the action mappings are active researcher policy. Real evidence has no action truth, population-frequency simulation has genotype truth only, and synthetic fixtures test mechanics. Real action truth is unavailable. [POLICY]
