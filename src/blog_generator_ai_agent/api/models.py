from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any


class TopicRequest(BaseModel):
    pillar: str = Field(..., description="Content pillar for topic generation")


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""
    topic: str = Field(..., min_length=3, max_length=200, description="Topic to research")
    mode: str = Field(..., pattern="^(SERP|RAG|reference)$", description="Research mode: SERP, RAG, or reference")
    urls: List[HttpUrl] = Field(default_factory=list, description="List of URLs to include in research")
    uploads: List[str] = Field(default_factory=list, description="List of uploaded file references")
    research_method: str = Field("SERP", pattern="^(SERP|RAG|reference)$", description="Research method to use (SERP, RAG, reference)")
    pillar: Optional[str] = Field(None, min_length=3, max_length=100, description="Content pillar for the research")


class CompetitorAnalysisRequest(BaseModel):
    urls: List[str] = Field(..., description="Competitor URLs to analyze")
    topic: str = Field(..., description="Topic for competitor analysis")
    research_method: str = Field("SERP", description="Research method to use (SERP, RAG, reference)")


class KeywordRequest(BaseModel):
    topic: str = Field(..., description="Topic for keyword generation")
    findings: Optional[str] = Field(None, description="Research findings")
    primary_keyword: Optional[str] = Field(None, description="Optional primary keyword")
    research_method: str = Field(default="SERP", description="Research method to use")
    pillar: Optional[str] = Field(None, description="Content pillar")


class TitleRequest(BaseModel):
    topic: str = Field(..., description="Topic for title generation")
    primary: str = Field(..., description="Primary keyword")
    secondary: List[str] = Field(..., description="Secondary keywords")
    research_method: str = Field("SERP", description="Research method to use")


class StructureRequest(BaseModel):
    topic: str = Field(..., description="Topic for structure")
    structure_type: str = Field(..., description="Structure type")
    keywords: Dict[str, Any] = Field(..., description="Keyword strategy")
    primary_keyword: str = Field(..., description="Primary keyword for the content")
    research_method: str = Field("SERP", description="Research method to use")


class OutlineRequest(BaseModel):
    topic: str = Field(..., description="Topic for outline")
    structure: str = Field(..., description="Content structure type")
    keywords: Dict[str, Any] = Field(..., description="Keyword strategy")
    research_method: str = Field("SERP", description="Research method to use")


class BlogGenerationRequest(BaseModel):
    topic: str = Field(..., description="Blog topic")
    structure_type: str = Field(..., description="Structure type")
    primary_keyword: str = Field(..., description="Primary keyword")
    keywords: Dict[str, Any] = Field(..., description="Keywords")
    brand_voice: str = Field(default="professional, helpful, concise", description="Brand voice guidelines")
    research_method: str = Field("SERP", description="Research method to use")


class WorkflowRequest(BaseModel):
    topic: str = Field(..., description="Blog topic")
    pillar: str = Field(..., description="Content pillar")
    mode: str = Field(default="SERP", description="Research mode")
    urls: Optional[List[str]] = None
    uploads: Optional[List[str]] = None
    structure: str = Field(default="How-to Guide", description="Blog structure type")
    brand_voice: str = Field(default="professional, helpful, concise", description="Brand voice")


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


