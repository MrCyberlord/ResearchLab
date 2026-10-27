"""Bound the multipart body before Starlette spools it to disk."""
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class UploadBodyLimit:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") != "/api/documents" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        total = 0
        headers = dict(scope.get("headers", []))
        detail = {"code": "file_too_large", "message": "Upload exceeds the configured request-size limit."}
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    # Starlette closes multipart temporary files for HTTPException.
                    raise HTTPException(status_code=413, detail=detail)
            return message
        await self.app(scope, limited_receive, send)
