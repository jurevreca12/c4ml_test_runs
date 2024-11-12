import chisel4ml
from chisel4ml.chisel4ml_server import Chisel4mlServer
import subprocess
import tempfile
import socket
import atexit
from pathlib import Path

_server_list = []
_server_num = 0

def close_servers():
    global _server_list
    for subp, server in _server_list:
        server.stop()
        subp.terminate()

atexit.register(close_servers)


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def create_server(c4ml_jar, num_servers=1):
    global _server_list
    for num in range(num_servers):
        tmp_dir = Path(tempfile.TemporaryDirectory(prefix="chisel4ml").name)
        c4ml_jar = Path(c4ml_jar).resolve()
        free_port = get_free_port()
        assert c4ml_jar.exists()
        command = ["java", "-jar", f"{c4ml_jar}", "-p", f"{free_port}", "-d", f"{tmp_dir}"]
        c4ml_subproc = subprocess.Popen(command)
        c4ml_server = Chisel4mlServer(tmp_dir, free_port)
        _server_list.append((c4ml_subproc, c4ml_server))

def get_server():
    global _server_list, _server_num
    assert len(_server_list) >= _server_num
    _, c4ml_server = _server_list[_server_num]
    _server_num = (_server_num + 1) % len(_server_list)
    return c4ml_server
