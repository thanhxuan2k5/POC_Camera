import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from web.database import get_db, Event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

class EventResponse(BaseModel):
    id: int
    camera_id: int
    token_track_id: str
    result: str
    confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    image_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    occurred_at: datetime
    class Config:
        from_attributes = True

@router.get("/", response_model=List[EventResponse])
def list_events(
    skip: int = 0, 
    limit: int = 100, 
    result: Optional[str] = None,
    camera_id: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Event)
    if result:
        query = query.filter(Event.result == result)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if date_from:
        query = query.filter(Event.occurred_at >= date_from)
    if date_to:
        query = query.filter(Event.occurred_at <= date_to)
        
    events = query.order_by(Event.occurred_at.desc()).offset(skip).limit(limit).all()
    
    import os
    for event in events:
        if event.image_path:
            filename = os.path.basename(event.image_path.replace('\\', '/'))
            event.image_path = f"/storage/events/{filename}"
        if event.thumbnail_path:
            filename = os.path.basename(event.thumbnail_path.replace('\\', '/'))
            event.thumbnail_path = f"/storage/events/{filename}"
            
    return events

@router.get("/stats")
def get_event_stats(db: Session = Depends(get_db)):
    total = db.query(Event).count()
    ok_count = db.query(Event).filter(Event.result == 'OK').count()
    ng_count = db.query(Event).filter(Event.result == 'NG').count()
    return {
        "total": total,
        "ok_count": ok_count,
        "ng_count": ng_count
    }

@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
        
    import os
    if event.image_path:
        filename = os.path.basename(event.image_path.replace('\\', '/'))
        event.image_path = f"/storage/events/{filename}"
    if event.thumbnail_path:
        filename = os.path.basename(event.thumbnail_path.replace('\\', '/'))
        event.thumbnail_path = f"/storage/events/{filename}"
        
    return event

@router.delete("/")
def delete_events_by_date(date_from: datetime, date_to: datetime, db: Session = Depends(get_db)):
    deleted = db.query(Event).filter(Event.occurred_at >= date_from, Event.occurred_at <= date_to).delete()
    db.commit()
    return {"deleted": deleted}

class BatchDeleteRequest(BaseModel):
    ids: List[int]

@router.delete("/batch")
def delete_batch_events(data: BatchDeleteRequest, db: Session = Depends(get_db)):
    deleted = db.query(Event).filter(Event.id.in_(data.ids)).delete(synchronize_session=False)
    db.commit()
    return {"deleted": deleted}

@router.delete("/all")
def delete_all_events(db: Session = Depends(get_db)):
    deleted = db.query(Event).delete()
    db.commit()
    return {"deleted": deleted}

@router.delete("/{event_id}")
def delete_single_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(event)
    db.commit()
    return {"status": "deleted"}

class EventUpdate(BaseModel):
    result: str

@router.put("/{event_id}", response_model=EventResponse)
def update_event(event_id: int, data: EventUpdate, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event.result = data.result
    db.commit()
    db.refresh(event)
    
    import os
    if event.image_path:
        filename = os.path.basename(event.image_path.replace('\\', '/'))
        event.image_path = f"/storage/events/{filename}"
    if event.thumbnail_path:
        filename = os.path.basename(event.thumbnail_path.replace('\\', '/'))
        event.thumbnail_path = f"/storage/events/{filename}"
        
    return event

@router.get("/export")
def export_events(db: Session = Depends(get_db)):
    import csv
    import io
    from fastapi.responses import StreamingResponse
    
    events = db.query(Event).order_by(Event.occurred_at.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Camera ID", "Track ID", "Result", "Confidence", "Similarity", "Occurred At", "Image Path"])
    
    for e in events:
        writer.writerow([
            e.id, e.camera_id, e.token_track_id, e.result,
            round(e.confidence, 4) if e.confidence else 0,
            round(e.similarity_score, 4) if e.similarity_score else 0,
            e.occurred_at.isoformat() if e.occurred_at else "",
            e.image_path or ""
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=events_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )
