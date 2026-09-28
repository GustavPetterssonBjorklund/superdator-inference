from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import websocket

log = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray], None]
ErrorCallback = Callable[[Exception], None]


class InferenceClient:
    """Streams frames to a superdator-inference server over its `/stream` websocket
    and receives annotated frames back.

    Frames are sent and received as BGR numpy arrays (as returned/expected by
    OpenCV); JPEG encoding/decoding for the wire format is handled internally.

    Usage:
        with InferenceClient("ws://localhost:8000/stream") as client:
            client.send(frame)
            annotated = client.latest_frame
    """

    def __init__(
        self,
        url: str = "ws://localhost:8000/stream",
        *,
        jpeg_quality: int = 80,
        on_frame: Optional[FrameCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ):
        self.url = url
        self.jpeg_quality = jpeg_quality
        self._on_frame = on_frame
        self._on_error = on_error

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._connected = threading.Event()

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._handle_open,
            on_close=self._handle_close,
            on_message=self._handle_message,
            on_error=self._handle_error,
        )
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> "InferenceClient":
        """Start the background connection. Idempotent."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        return self

    def close(self, timeout: float = 5.0) -> None:
        self._ws.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._connected.clear()

    def __enter__(self) -> "InferenceClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- state -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """The most recently received annotated frame, or None if none has arrived yet."""
        with self._frame_lock:
            return self._latest_frame

    # -- sending -----------------------------------------------------------

    def send(self, frame: np.ndarray) -> bool:
        """Encode a BGR frame as JPEG and send it. Returns False if not connected."""
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        return self.send_jpeg(buffer.tobytes())

    def send_jpeg(self, data: bytes) -> bool:
        """Send raw JPEG bytes. Returns False if not connected."""
        if not self.connected:
            return False
        try:
            self._ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
            return True
        except Exception as e:
            log.error("Send failed: %s", e)
            return False

    # -- websocket callbacks -----------------------------------------------

    def _handle_open(self, ws):
        self._connected.set()
        log.info("Connected to %s", self.url)

    def _handle_close(self, ws, close_status_code, close_msg):
        self._connected.clear()
        log.warning("Connection closed (code=%s, reason=%s)", close_status_code, close_msg)

    def _handle_message(self, ws, message):
        if isinstance(message, str):
            log.error("Server sent text message instead of a frame: %s", message)
            return

        frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("Received %d bytes but failed to decode as an image", len(message))
            return

        with self._frame_lock:
            self._latest_frame = frame

        if self._on_frame is not None:
            self._on_frame(frame)

    def _handle_error(self, ws, error):
        log.error("Websocket error: %s", error)
        if self._on_error is not None:
            self._on_error(error)
