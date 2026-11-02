# Retry policy

**Status:** Active research policy, version research-skills-20261006-v3; not scientifically validated.

Use the supplied Allowed retry strategies, `evidence`, `tool_messages`, `warnings`, `attempt_history`, `strategies_tried`, and `retries_used`. Parameters are exact overrides on the recorded baseline. Try each ID once; the configured maximum of two retries remains an upper guard, but only one strategy is active. This allowance is a project compute choice. [SOURCE: agent/llm.py; agent/nodes.py; config/parameters.yaml]

## Select the intervention

- **`lower-block-cut-sensitivity-v1`**, exactly `{"block_cut_sensitivity": 2}`: when at least two heterozygous SNPs and more than one known regional phase set show fragmentation, with no unresolved input problem or documented absent read connections, select this strategy if it is available and unused. Unknown block statistics, haplotype count, Mendelian evidence, or read connectivity do not by themselves disqualify this diagnostic trial. The trial is not a promise of recovery. Lower sensitivity allows longer blocks but increases switch-error risk, meaning incorrect chromosome assignment. The value 2 is a fixed project proposal, not an optimum established for Eucalyptus. [POLICY; W28; S20; N21]
`--distrust-genotypes` is excluded because WhatsHap `polyphase` may overwrite input genotype calls; this project assesses phasing after genotyping, not genotype revision. Do not invent values, change ploidy, toggle reference usage, or request additional flags. [POLICY; W28]

## Stop and reassess

Do not retry zero informative SNPs alone, a single-site linkage problem, missing input, sample/ploidy mismatch, documented absent spanning reads, an already tried ID, or an exhausted allowance. Missing connectivity is unknown, not confirmed absence. If no listed strategy applies, escalate. [POLICY]

After a retry, apply `phasing_triage.md` to its evidence. Compare known SNP counts, call rate, and phase-set count with the preceding attempt. If those observations are unchanged, stop and escalate. The backend rejects outputs with changed genotype calls. Completion and longer blocks are not biological improvement. A run that satisfies the acceptance rule is accepted operationally, without a correctness claim. [POLICY; agent/nodes.py; agent/real_retry.py]

The backend runs the original full filtered sample VCF once per exact strategy, caches regional evidence, and preserves a separate immutable artifact. Only a requesting region adopts its alternative evidence; other case records are not updated. Exact-region slicing is not used because it changed boundary-case results. The backend enforces a 120-second tool limit; failure stops that attempt and is audited. [SOURCE: agent/real_retry.py; config/parameters.yaml; 2026-09-24 scope receipt]

For `retry`, return the exact listed strategy object in `retry_strategy`. For `accept` or `escalate`, return `retry_strategy: null`. Return one allowed action, confidence from 0 to 1, and a short rationale naming the evidence and the rule used. Never output commands or claims about unsupplied measurements. [SOURCE: agent/llm.py; POLICY]

Sources: [WhatsHap 2.8 guide (W28)](https://raw.githubusercontent.com/whatshap/whatshap/v2.8/doc/guide.rst), polyphase; [Schrinner et al. 2020 (S20)](https://doi.org/10.1186/s13059-020-02158-1), block cuts; [Abou Saada et al. 2021 (N21)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8082856/), Fig. 5. Their long-read findings do not validate this short-read action policy. [SOURCE; POLICY]
