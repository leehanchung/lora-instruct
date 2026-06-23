"""Prompt templates for the web domain (kept together, versioned with the code)."""

EXPLORE = """\
You are generating a fact-seeking research question grounded in real sources.
Given the seed topic below, produce:
  - a specific question with a single verifiable answer,
  - the answer (truth),
  - 1-3 supporting items: {{source, quote}} that entail the answer.
Seed: {seed}
"""

VERIFY = """\
Decide whether the supporting quotes entail the answer. Answer yes/no with a reason.
Question: {question}
Answer: {truth}
Quotes: {quotes}
"""
