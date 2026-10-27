"""Bounded PDF text extraction in a killable child process. No OCR or JS execution."""
from io import BytesIO
import logging
import multiprocessing
from time import monotonic

import psutil
from pypdf import PdfReader

from app.domain.documents import InvalidPDF, PDFLimitExceeded, PageText


def _parse_pdf(data: bytes, max_pages: int, max_chars: int) -> list[PageText]:
    reader = PdfReader(BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise InvalidPDF("Encrypted PDFs are not supported.")
    if len(reader.pages) > max_pages:
        raise PDFLimitExceeded("PDF exceeds the configured page limit.")
    pages = []
    count = 0
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").replace("\x00", "")
        count += len(text)
        if count > max_chars:
            raise PDFLimitExceeded("PDF exceeds the configured extracted-text limit.")
        pages.append(PageText(number, text))
    return pages


def _worker(connection, data: bytes, max_pages: int, max_chars: int):
    # Parser messages may contain PDF content; never forward them to server logs.
    logging.disable(logging.CRITICAL)
    try:
        connection.send(("ok", _parse_pdf(data, max_pages, max_chars)))
    except (InvalidPDF, PDFLimitExceeded) as exc:
        connection.send((type(exc).__name__, str(exc)))
    except Exception:
        connection.send(("InvalidPDF", "PDF is malformed or could not be parsed."))
    finally:
        connection.close()


class IsolatedPDFExtractor:
    def __init__(self, max_pages: int, max_chars: int, timeout: float, memory_mb: int):
        self.max_pages = max_pages
        self.max_chars = max_chars
        self.timeout = timeout
        self.memory_bytes = memory_mb * 1024 * 1024

    def extract(self, data: bytes) -> list[PageText]:
        if not data.startswith(b"%PDF-"):
            raise InvalidPDF("File does not have a valid PDF header.")
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(sender, data, self.max_pages, self.max_chars))
        started = monotonic()
        try:
            process.start()
            sender.close()
            monitor = psutil.Process(process.pid)
            while True:
                if monotonic() - started > self.timeout:
                    raise PDFLimitExceeded("PDF parsing exceeded the configured time limit.")
                try:
                    if monitor.memory_info().rss > self.memory_bytes:
                        raise PDFLimitExceeded("PDF parsing exceeded the configured memory limit.")
                except psutil.NoSuchProcess:
                    pass
                if receiver.poll(0.02):
                    try:
                        status, value = receiver.recv()
                    except EOFError as exc:
                        raise InvalidPDF("PDF parser exited before extraction completed.") from exc
                    if status == "ok":
                        return value
                    if status == "PDFLimitExceeded":
                        raise PDFLimitExceeded(value)
                    raise InvalidPDF(value)
                if not process.is_alive():
                    raise InvalidPDF("PDF parser exited before extraction completed.")
        finally:
            receiver.close()
            sender.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
