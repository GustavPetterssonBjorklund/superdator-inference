import argparse
import logging
import threading
import time

import cv2
import numpy as np
import websocket

JPEG_QUALITY = 80
STATS_INTERVAL_SEC = 5.0

log = logging.getLogger("client")


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam client for superdator-inference")
    parser.add_argument("--url", default="ws://localhost:8000/stream", help="Server websocket URL")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--debug", action="store_true", help="Enable verbose/trace logging")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    websocket.enableTrace(args.debug)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}")

    latest_frame = None
    frame_lock = threading.Lock()
    sent_count = 0
    received_count = 0

    def on_open(ws):
        log.info("Connected to %s", args.url)

    def on_close(ws, close_status_code, close_msg):
        log.warning("Connection closed (code=%s, reason=%s)", close_status_code, close_msg)

    def on_message(ws, message):
        nonlocal latest_frame, received_count
        if isinstance(message, str):
            log.error("Server sent text message instead of a frame: %s", message)
            return

        frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("Received %d bytes but failed to decode as an image", len(message))
            return

        received_count += 1
        log.debug("Decoded annotated frame #%d (%d bytes)", received_count, len(message))
        with frame_lock:
            latest_frame = frame

    def on_error(ws, error):
        log.error("Websocket error: %s", error)

    ws = websocket.WebSocketApp(
        args.url,
        on_open=on_open,
        on_close=on_close,
        on_message=on_message,
        on_error=on_error,
    )
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    last_stats_time = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Failed to read frame from webcam")
                break

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                if ws.sock and ws.sock.connected:
                    try:
                        ws.send(buffer.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                        sent_count += 1
                    except Exception as e:
                        log.error("Send failed: %s", e)
                else:
                    log.debug("Skipping send: websocket not connected")

            now = time.monotonic()
            if now - last_stats_time >= STATS_INTERVAL_SEC:
                log.info("Frames sent: %d, frames received: %d", sent_count, received_count)
                last_stats_time = now

            with frame_lock:
                display_frame = latest_frame

            if display_frame is not None:
                cv2.imshow("Inference result", display_frame)
            else:
                cv2.imshow("Inference result", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        log.info("Shutting down (frames sent: %d, frames received: %d)", sent_count, received_count)
        cap.release()
        cv2.destroyAllWindows()
        ws.close()


if __name__ == "__main__":
    main()
