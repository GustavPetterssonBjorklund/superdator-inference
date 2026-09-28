import argparse
import logging
import time

import cv2
import websocket

from superdator_client import InferenceClient

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

    sent_count = 0
    received_count = 0

    def on_frame(frame):
        nonlocal received_count
        received_count += 1
        log.debug("Decoded annotated frame #%d", received_count)

    def on_error(error):
        log.error("Websocket error: %s", error)

    client = InferenceClient(args.url, on_frame=on_frame, on_error=on_error)
    client.connect()

    last_stats_time = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Failed to read frame from webcam")
                break

            if client.send(frame):
                sent_count += 1
            else:
                log.debug("Skipping send: not connected")

            now = time.monotonic()
            if now - last_stats_time >= STATS_INTERVAL_SEC:
                log.info("Frames sent: %d, frames received: %d", sent_count, received_count)
                last_stats_time = now

            display_frame = client.latest_frame
            cv2.imshow("Inference result", display_frame if display_frame is not None else frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        log.info("Shutting down (frames sent: %d, frames received: %d)", sent_count, received_count)
        cap.release()
        cv2.destroyAllWindows()
        client.close()


if __name__ == "__main__":
    main()
