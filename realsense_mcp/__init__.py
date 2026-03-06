"""RealSense MCP Server - Intel RealSense D405/D435 depth camera integration."""

__all__ = ["server"]


def __getattr__(name):
    if name == "server":
        from .server import server
        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
