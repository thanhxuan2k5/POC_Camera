import logging
import asyncio
from typing import Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        # Store connections per room (e.g., camera_id)
        self.room_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_name: str):
        await websocket.accept()
        if room_name not in self.room_connections:
            self.room_connections[room_name] = set()
        self.room_connections[room_name].add(websocket)
        logger.info(f"WebSocket connected to room '{room_name}'. Total clients: {len(self.room_connections[room_name])}")

    def disconnect(self, websocket: WebSocket, room_name: str):
        if room_name in self.room_connections:
            self.room_connections[room_name].discard(websocket)
            if not self.room_connections[room_name]:
                del self.room_connections[room_name]
        logger.info(f"WebSocket disconnected from room '{room_name}'.")

    def call_soon_threadsafe(self, loop, callback):
        """Helper to schedule a coroutine from a different thread."""
        if loop:
            loop.call_soon_threadsafe(asyncio.create_task, callback)

    async def _broadcast_to_room_async(self, message: dict, room_name: str):
        if room_name in self.room_connections:
            for connection in list(self.room_connections[room_name]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(connection, room_name)

    def broadcast_to_room_from_thread(self, loop, message: dict, room_name: str):
        """Thread-safe method to broadcast to a room."""
        self.call_soon_threadsafe(loop, self._broadcast_to_room_async(message, room_name))

manager = ConnectionManager()

@router.websocket("/live/{camera_id}")
async def live_stream(websocket: WebSocket, camera_id: int):
    room = str(camera_id)
    await manager.connect(websocket, room)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)

# The generic /events endpoint might not be needed if events are also per-camera
@router.websocket("/events")
async def events_stream(websocket: WebSocket):
    room = "events"
    await manager.connect(websocket, room)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
