"""Deny IP networking for the credential-free Kaiwu conformance process tree."""

import errno
import socket
from typing import NoReturn

_MESSAGE = "network access is disabled by the Kaiwu local-conformance guard"


def _deny_network(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise OSError(errno.ENETUNREACH, _MESSAGE)


class _OfflineSocket(socket.socket):
    def connect(self, address: object) -> NoReturn:
        del address
        _deny_network()

    def connect_ex(self, address: object) -> NoReturn:
        del address
        _deny_network()


socket.socket = _OfflineSocket
socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.gethostbyname = _deny_network
socket.gethostbyname_ex = _deny_network
socket.gethostbyaddr = _deny_network
