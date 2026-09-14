from __future__ import annotations

import json
import socket
import threading
import unittest

from firered_rl.backends.mgba_bridge import MGBABridgeBackend


class TestMGBABridgeControlProtocol(unittest.TestCase):
    def test_status_and_close_are_newline_delimited(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        requests: list[dict[str, object]] = []

        def serve() -> None:
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as reader:
                for response in (
                    {
                        "ok": True,
                        "adapter_version": "0.1.1",
                        "profile": "firered_us_v1_0",
                        "game_code": "BPRE",
                        "raw_game_code": "AGB-BPRE",
                        "rom_crc32": "dd88761c",
                        "supported": True,
                    },
                    {
                        "ok": True,
                        "state_ready": False,
                        "frame": 123,
                        "ram_fnv1a": "12345678",
                    },
                    {
                        "ok": True,
                        "state_ready": False,
                        "action": 1,
                        "frames_requested": 1,
                        "frame_start": 100,
                        "frame_end": 101,
                    },
                    {"ok": True},
                ):
                    requests.append(json.loads(reader.readline()))
                    connection.sendall(
                        (json.dumps(response, separators=(",", ":")) + "\n").encode()
                    )
            listener.close()

        thread = threading.Thread(target=serve)
        thread.start()

        bridge = MGBABridgeBackend(host, port)
        status = bridge.status()
        reset = bridge.reset_response(42)
        step = bridge.step_response(1, 1)
        bridge.close()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(status["supported"])
        self.assertEqual(reset["frame"], 123)
        self.assertEqual(status["rom_crc32"], "dd88761c")
        self.assertEqual(step["frame_end"] - step["frame_start"], 1)
        self.assertEqual(
            requests,
            [
                {"command": "status"},
                {"command": "reset", "seed": 42},
                {"command": "step", "action": 1, "frames": 1},
                {"command": "close"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
