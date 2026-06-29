"""Bootstrap layer: source registry, downloader, and orchestrator.

The bootstrap layer is the only place the offline MVP reaches out to
the network. Once sources are downloaded and validated, everything
else operates on local files.
"""
