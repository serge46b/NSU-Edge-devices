import serial
from typing import Literal
from queue import Queue, Full, Empty
from threading import Event, Thread

PORT = "/dev/ttyS7"
BAUD_RATE = 115200

AllowedCommands = Literal["OK", "PENDING", "NOT_RECOGNIZED"]


class Opener:
    __device: serial.Serial
    __in_process_event: Event
    __queue: Queue[AllowedCommands]
    __thread: Thread
    __stop_event: Event

    def __init__(self, in_process_event: Event, stop_event: Event) -> None:
        self.__device = serial.Serial(PORT, BAUD_RATE, timeout=None)
        self.__in_process_event = in_process_event
        self.__queue = Queue(maxsize=3)
        self.__thread = Thread(target=self._run, daemon=True)
        self.__stop_event = stop_event

        print("Opener: Waiting for device...")
        self._reset()
        print("Opener: Device ready")
        self.__thread.start()

    def __del__(self) -> None:
        self.__device.cancel_read()
        self.__device.close()
        self.__in_process_event.clear()

    def _run(self) -> None:
        print("Opener: Thread started")
        while not self.__stop_event.is_set():
            try:
                command = self.__queue.get(timeout=1)
            except Empty:
                continue
            if not self.__in_process_event.is_set():
                self.__in_process_event.set()
            print(f"Opener: Processing command: {command}")
            match command:
                case "OK":
                    self._open()
                case "PENDING":
                    self._pending_notify()
                case "NOT_RECOGNIZED":
                    self._nr_notify()
            print(f"Opener: Command processed")
            self.__in_process_event.clear()
        print("Opener: Thread stopped")

    def _open(self) -> None:
        self._send_command("FACE_OK")
        if not self._wait_for_line("BUSY") or not self._wait_for_line("DONE"):
            print("Opener: OPEN command failed")
            self._reset()

    def _pending_notify(self) -> None:
        self._send_command("PROCESSING")
        if not self._wait_for_line("DETECTION"):
            print("Opener: PROCESSING command failed")
            self._reset()

    def _nr_notify(self) -> None:
        self._send_command("FACE_BAD")
        if not self._wait_for_line("DENIED"):
            print("Opener: CLOSE command failed")
            self._reset()

    def _reset(self) -> None:
        self._send_command("RESET")
        if not self._wait_for_line("RESET_OK"):
            raise ValueError("Opener: RESET command failed")

    def _wait_for_line(self, expected: str) -> bool:
        line = self.__device.readline().decode("utf-8").strip()
        if line != expected:
            return False
        return True

    def _send_command(self, command: str) -> None:
        self.__device.write(f"{command}\n".encode("utf-8"))

    def run_command(self, command: AllowedCommands) -> None:
        try:
            self.__queue.put_nowait(command)
        except Full:
            self.__queue.get_nowait()
            self.__queue.put_nowait(command)
