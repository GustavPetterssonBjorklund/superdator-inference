import asyncio
import os
import queue
import threading

# Put server on GPU1, this will leave GPU0 for training and other processes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# TensorRT engine file
MODEL_PATH = "best.engine"

JPEG_QUALITY = 80

app = FastAPI()

infer_queue: "queue.Queue[tuple]" = queue.Queue()
main_loop: asyncio.AbstractEventLoop | None = None

def inference_worker():
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    
    # Warm up with dummy input
    model.predict(np.zeros(
        (640, 640, 3),
        dtype=np.uint8),
        verbose=False
    )
    
    while True:
        frame, future = infer_queue.get()
        
        try:
            result = model.predict(frame, verbose=False)[0]
            annotated = result.plot()
        
        except Exception as e:
            main_loop.call_soon_threadsafe(future.set_exception, e)
            continue
        main_loop.call_soon_threadsafe(future.set_result, annotated)
        
class FrameMailbox:
    def __init__(self):
        self.frame = None
        self._event = asyncio.Event()
        
    def put(self, frame):
        self.frame = frame
        self._event.set()
        
    async def get(self):
        await self._event.wait()
        frame = self.frame
        self._event.clear()
        return frame

@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_event_loop()
    threading.Thread(target=inference_worker, daemon=True).start()

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    mailbox = FrameMailbox()
    
    async def receive_frames():
        try:
            while True:
                data = await ws.receive_bytes()
                mailbox.put(data)
        except WebSocketDisconnect:
            pass
        
    async def processor():
        while True:
            data = await mailbox.get()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            
            future = main_loop.create_future()
            infer_queue.put((frame, future))
            
            try: 
                annotated = await future
            except Exception as e:
                await ws.send_text(f"Error during inference: {str(e)}")
                continue
            
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            
            if not ok:
                await ws.send_text("Error encoding image")
                continue
                
            try:
                await ws.send_bytes(buffer.tobytes())
            except Exception as e:
                break
        
    recv_task = asyncio.create_task(receive_frames())
    proc_task = asyncio.create_task(processor())
    done, pending = await asyncio.wait(
        {recv_task, proc_task},
        return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    for task in done:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            raise exc

if __name__ == "__main__":
    # Single worker only: the inference thread, queue, and TensorRT engine
    # are process-global, so extra workers would each load their own copy
    # of the model and fight over the GPU.
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)