"""datagen — synthetic deep-research task generation.

core/ holds domain-agnostic stage base classes; domains/<name>/ subclass them
with an identical, stage-named file layout (explore -> verify -> distract ->
extend -> index). Generated tasks carry ground truth, so the output doubles as
both training and eval data. (Pattern from chroma context-1-data-gen.)
"""
