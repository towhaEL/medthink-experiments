"""
Zero-dependency HTTP Web Server and REST API package for Kaggle Hub.
"""

from .http_server import KaggleHubServer, run_server

__all__ = ["KaggleHubServer", "run_server"]
