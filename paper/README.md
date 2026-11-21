# Behavioral Runtime Security Monitoring for Prompt-Injected LLM Agents

## Introduction

LLM agents consume external data and propose tool actions. Indirect prompt
injection uses that data channel to redirect behavior beyond the legitimate
task. This project investigates a host-controlled action boundary with trusted
metadata, behavioral features, learned scores and inspectable containment.

## Related work

Greshake et al. demonstrate indirect prompt injection through retrieved data in
LLM-integrated applications ([paper](https://arxiv.org/abs/2302.12173)).
AgentDojo provides an extensible environment for evaluating agent tasks, attacks
and defenses over untrusted tool data ([Debenedetti et al.](https://arxiv.org/abs/2406.13352)).
These establish relevant problem and evaluation settings; our tiny synthetic
fixture is not a reproduction or competitive evaluation of either study.

## Threat model and method

The attacker controls synthetic document content, while tool registration,
policy metadata, detector checkpoint and host approval remain trusted. See
[methodology](../docs/methodology.md) for the architecture and trust-boundary
diagram. The system observes causal action prefixes, extracts semantic context,
and routes proposals through a policy/detector gateway before dispatch.

## Experiments

Frozen grouped data feeds rules, conventional ML and a BiLSTM with attention.
Phase 9 reports preserve source/split coverage, explicit metric denominators,
unknown labels, and counterexamples. The final offline demonstration compares
existing benign and injected scripted runs without and with the gateway.
Outcome assertions use recorded mock side effects and refusal telemetry.

## Results

See [recorded results and validation](../docs/results.md). Software tests and
packaging checks establish exercised workflow behavior. They do not establish
research superiority. The held-out data contain no unsafe proposals, so held-out
detection rate is undefined. Authorized sharing can trigger an incorrect review
refusal; this utility loss must remain visible.

## Limitations and next experiments

The corpus contains 12 scripted runs and three independent groups. There are no
matched executed four-mode trials, coherent-malicious/reliability comparison,
representative unseen-case labels, or independent repeated-trial confidence
intervals. Tools and effects are synthetic; live model behavior can vary. Future
work must expand independently grouped positive and negative examples, freeze
test sets, measure benign utility/latency, and report negative results. No claim
of production security or detector generalization is made.
