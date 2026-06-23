"""Shared sampling logic across benchmarks.

Factored out so every benchmark reuses one sampler instead of copy-pasting
(the DR-Tulu evaluation/samplers/ pattern). Add provider/model samplers here.
"""
