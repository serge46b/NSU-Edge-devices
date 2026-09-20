import asyncio
import socket
import threading
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
JPEG_QUALITY = 80

_lock = threading.Lock()
_latest_bgr = None
_frame_ver = 0
_start_lock = threading.Lock()
_started = False
_loop = None
_thread = None
_runner = None


def _jpeg_bytes(frame):
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


def set_frame(frame):
    global _latest_bgr, _frame_ver
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("imshow expects a BGR HxWx3 array")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    with _lock:
        _latest_bgr = np.ascontiguousarray(arr.copy())
        _frame_ver += 1


def _listen_urls(host, port):
    urls = [f"http://127.0.0.1:{port}/"]
    if host in ("127.0.0.1", "localhost"):
        return [f"http://{host}:{port}/"]
    if host not in ("0.0.0.0", "::", ""):
        urls.append(f"http://{host}:{port}/")
        return urls
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            urls.append(f"http://{ip}:{port}/")
    except OSError:
        pass
    return urls


def _latest():
    with _lock:
        ver = _frame_ver
        frame = _latest_bgr
        if not isinstance(frame, np.ndarray):
            return ver, None
        return ver, frame.copy()


async def _index(_request):
    return web.FileResponse(
        INDEX_HTML,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


async def _jpeg(_request):
    _ver, frame = _latest()
    if frame is None:
        return web.Response(status=503, text="no frame")
    data = await asyncio.get_running_loop().run_in_executor(None, _jpeg_bytes, frame)
    if not data:
        return web.Response(status=500, text="jpeg encode failed")
    return web.Response(
        body=data,
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


async def _drain(request):
    transport = request.transport
    if transport is None:
        return
    while not transport.is_closing() and transport.get_write_buffer_size() > 0:
        await asyncio.sleep(0.005)


async def _stream(request):
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)
    sock = request.transport.get_extra_info("socket") if request.transport else None
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    loop = asyncio.get_running_loop()
    last_ver = -1
    try:
        while True:
            await _drain(request)
            ver, frame = _latest()
            if frame is None or ver == last_ver:
                await asyncio.sleep(0.01)
                continue
            last_ver = ver
            data = await loop.run_in_executor(None, _jpeg_bytes, frame)
            if not data:
                continue
            payload = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(data)).encode()
                + b"\r\n\r\n"
                + data
                + b"\r\n"
            )
            await resp.write(payload)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    return resp


async def _start_app(host, port):
    global _runner
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/stream", _stream)
    app.router.add_get("/frame.jpg", _jpeg)
    _runner = web.AppRunner(app)
    await _runner.setup()
    site = web.TCPSite(_runner, host, port)
    await site.start()


def _run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _shutdown():
    global _runner
    if _runner is not None:
        await _runner.cleanup()
        _runner = None


def start(host="0.0.0.0", port=8080):
    global _loop, _thread, _started
    with _start_lock:
        if _started:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
        thread.start()
        asyncio.run_coroutine_threadsafe(_start_app(host, port), loop).result()
        _loop = loop
        _thread = thread
        _started = True
        urls = " ".join(_listen_urls(host, port))
        print(f"tmpwebrtc: {urls}")


def stop():
    global _started, _loop, _thread
    with _start_lock:
        if not _started:
            return
        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), _loop).result(timeout=10)
        except Exception:
            pass
        _loop.call_soon_threadsafe(_loop.stop)
        _thread.join(timeout=5)
        _started = False
        _loop = None
        _thread = None


def imshow(arg1, frame=None, *, host="0.0.0.0", port=8080):
    if frame is None:
        frame = arg1
    start(host=host, port=port)
    set_frame(frame)
