"""Network helper: prefer the reachable addresses of a host with several A records.

overpass-api.de resolves to two IPv4 addresses, and from the author's network one of
them (65.109.112.52) is unreachable while the other answers in ~0.2 s. Clients that hit
the bad one first wait out a 180 s connect timeout and fail, which made Overpass look
"intermittently blocked". `prefer_reachable` probes each address and restricts name
resolution for that host, in this process only, to the ones that accept a connection.
"""

from __future__ import annotations

import socket

_patched: dict[str, list] = {}
_original_getaddrinfo = socket.getaddrinfo


def _getaddrinfo(host, *args, **kwargs):
    if host in _patched:
        return _patched[host]
    return _original_getaddrinfo(host, *args, **kwargs)


def prefer_reachable(host: str, port: int = 443, timeout: float = 3.0) -> list[str]:
    """Probe every address of `host`; resolve it to the reachable ones only.

    Returns the reachable addresses. Raises if none accepts a TCP connection.
    """
    infos = _original_getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    good = []
    for info in infos:
        try:
            with socket.create_connection(info[4], timeout=timeout):
                good.append(info)
        except OSError:
            continue
    if not good:
        tried = [i[4][0] for i in infos]
        raise ConnectionError(f"No reachable address for {host}:{port} (tried {tried})")
    _patched[host] = good
    socket.getaddrinfo = _getaddrinfo
    return [i[4][0] for i in good]
