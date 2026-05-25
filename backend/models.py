from pydantic import BaseModel
from typing import List, Optional


class ResearchRequest(BaseModel):
    query: str


class Source(BaseModel):
    url: str
    title: str
    content: str


class ResearchFinding(BaseModel):
    subquestion: str
    sources: List[Source]


class ReportResponse(BaseModel):
    markdown_file: str
    pdf_file: str
    markdown_content: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
