"""search_server — the tool/search HTTP service behind a stable contract.

One contract (/search, /visit, /retrieve), swappable backends (BM25, dense,
live web). The deep-research agent (libs/dr_agent) is a thin client; training,
eval, and serving all hit this same service so tool behaviour never drifts.
"""
