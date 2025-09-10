from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Dict, Optional, Any
from datetime import datetime

# Topic Generation Models
class TopicSuggestion(BaseModel):
    id: Optional[str] = Field(default=None,description="Unique identifier for the topic")
    title: str = Field(description="Topic title")
    description: str = Field(description="Brief description of the topic")
    target_audience: str = Field(description="Target audience for this topic")
    content_pillar_alignment: str = Field(description="How well this aligns with the content pillar")
    estimated_engagement_potential: str = Field(description="Estimated engagement potential (High/Medium only)")
    relevance_score: float = Field(ge=0.0, le=10.0, description="Relevance score out of 10")

class TopicSuggestionsResponse(BaseModel):
    topics: List[TopicSuggestion] = Field(
        description="List of generated topic suggestions (exactly 6 items)"
    )

class TopicGenerationOutput(BaseModel):
    topics: List[TopicSuggestion] = Field(description="List of topic suggestions")
    pillar: str = Field(description="Content pillar used for generation")
    generation_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_suggestions: int = Field(description="Total number of suggestions generated")

# Research Models
class ResearchFinding(BaseModel):
    insight: str = Field(description="Key insight or finding")
    source: str = Field(description="Source of the information")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score for this finding")
    category: Optional[str] = Field(default="", description="Category of the finding (statistics, trend, opportunity, etc.)")
    citation: Optional[str] = Field(default="", description="Proper citation format")

class ResearchOutput(BaseModel):
    topic: str = Field(description="Researched topic")
    research_method: str = Field(description="Method used for research (SERP/RAG/reference)")
    key_insights: List[str] = Field(description="Key insights extracted")
    statistics: List[str] = Field(description="Relevant statistics found")
    findings: List[ResearchFinding] = Field(description="Detailed research findings")
    content_gaps: List[str] = Field(description="Identified content gaps")
    talking_points: List[str] = Field(description="Recommended talking points")
    sources: List[Dict[str, str]] = Field(description="Source citations with URLs")
    research_date: str = Field(default_factory=lambda: datetime.now().isoformat())

# Competitor Analysis Models
class CompetitorContent(BaseModel):
    url: str = Field(description="Competitor URL")
    title: str = Field(description="Content title")
    summary: str = Field(description="Content summary")
    tone_style: str = Field(description="Tone and style analysis")
    structure_pattern: str = Field(description="Content structure pattern")
    keyword_strategy: Optional[List[str]] = Field(description="Keywords identified in content")
    strengths: List[str] = Field(description="Content strengths")
    weaknesses: List[str] = Field(description="Content weaknesses")

class CompetitorAnalysisOutput(BaseModel):
    topic: str = Field(description="Analyzed topic")
    competitor_summaries: List[CompetitorContent] = Field(description="Individual competitor analysis")
    common_patterns: List[str] = Field(description="Common patterns across competitors")
    tone_analysis: Dict[str, str] = Field(description="Overall tone and style analysis")
    structure_patterns: List[str] = Field(description="Common structure patterns")
    keyword_strategies: List[str] = Field(description="Competitor keyword strategies")
    differentiation_opportunities: List[str] = Field(default= "",description="Opportunities for differentiation")
    what_to_replicate: List[str] = Field(description="Best practices to replicate")
    what_to_avoid: List[str] = Field(description="Practices to avoid")
    analysis_date: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator("tone_analysis", mode="before")
    @classmethod
    def normalize_tone_analysis(cls, value: Any) -> Dict[str, str]:
        # Accept string or list and convert to a dict; pass through dict
        if value is None:
            return {}
        if isinstance(value, str):
            return {"summary": value}
        if isinstance(value, list):
            return {"summary": ", ".join(str(v) for v in value)}
        return value

# Keyword Strategy Models
class KeywordStrategy(BaseModel):
    primary_keyword: str = Field(description="Primary target keyword")
    primary_rationale: str = Field(description="Rationale for primary keyword selection")
    secondary_keywords: List[str] = Field(description="Secondary keywords (2-3)")
    long_tail_opportunities: List[str] = Field(description="Long-tail keyword opportunities")
    search_intent: str = Field(description="Search intent analysis")
    competition_level: str = Field(description="Competition level assessment")
    recommended_density: str = Field(description="Recommended keyword density")
    semantic_keywords: List[str] = Field(description="Semantically related keywords")

class KeywordStrategyOutput(BaseModel):
    topic: str = Field(description="Topic for keyword strategy")
    strategy: KeywordStrategy = Field(description="Comprehensive keyword strategy")
    implementation_notes: List[str] = Field(description="Implementation guidance")
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    serp_results: Optional[Dict[str, Any]] = Field(default=None, description="SERP data used to inform the strategy")

# Title Generation Models
class TitleOption(BaseModel):
    title: str = Field(description="Title option")
    type: str = Field(description="Title type (SEO, Listicle, How-to, etc.)")
    seo_score: float = Field(ge=0.0, le=10.0, description="SEO optimization score")
    engagement_score: float = Field(ge=0.0, le=10.0, description="Estimated engagement score")
    rationale: Optional[str] = Field(default="", description="Rationale for this title option")

    @field_validator("rationale", mode="before")
    @classmethod
    def default_rationale(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

class TitleGenerationOutput(BaseModel):
    topic: str = Field(description="Topic for title generation")
    title_options: List[TitleOption] = Field(description="Generated title options")
    recommended_title: str = Field(description="Best recommended title")
    recommendation_rationale: str = Field(description="Why this title was recommended")
    seo_considerations: List[str] = Field(description="SEO considerations for titles")
    generated_date: str = Field(default_factory=lambda: datetime.now().isoformat())

# Content Structure Models
class ContentSection(BaseModel):
    heading_level: str = Field(description="Heading level (H1, H2, H3)")
    heading_text: str = Field(description="Heading text")
    key_points: List[str] = Field(description="Key points for this section")
    content_intent: str = Field(description="Intent and purpose of this section")
    estimated_word_count: int = Field(description="Estimated word count for section")
    keywords_to_include: List[str] = Field(description="Keywords to naturally include")

class ContentStructureOutput(BaseModel):
    topic: str = Field(description="Topic for content structure")
    structure_type: str = Field(description="Type of content structure")
    heading_structure: List[str] = Field(description="H1, H2, H3 heading structure")
    sections: List[ContentSection] = Field(description="Detailed section breakdown")
    cta_placements: List[str] = Field(description="Recommended CTA placement locations")
    faq_section: Optional[List[Dict[str, str]]] = Field(description="FAQ section if applicable")
    total_estimated_words: int = Field(description="Total estimated word count")
    tone_notes: str = Field(description="Tone and style guidance")
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat())

# Blog Generation Models
class BlogMetadata(BaseModel):
    title: str = Field(description="Blog post title")
    meta_description: str = Field(description="Meta description for SEO")
    word_count: int = Field(description="Total word count")
    reading_time: str = Field(description="Estimated reading time")
    primary_keyword: str = Field(description="Primary keyword used")
    secondary_keywords: List[str] = Field(description="Secondary keywords used")
    internal_links: List[str] = Field(description="Suggested internal linking opportunities")
    seo_score: float = Field(ge=0.0, le=10.0, description="SEO optimization score")
    readability_score: str = Field(description="Readability assessment")

class BlogGenerationOutput(BaseModel):
    topic: str = Field(description="Blog topic")
    blog_content: str = Field(description="Complete blog post content in markdown")
    metadata: BlogMetadata = Field(description="Blog metadata and SEO information")
    content_quality_assessment: Dict[str, str] = Field(description="Quality assessment metrics")
    brand_voice_alignment: str = Field(description="Brand voice consistency assessment")
    sources_used: List[Dict[str, str]] = Field(description="Sources and citations used")
    optimization_suggestions: List[str] = Field(description="Additional optimization suggestions")
    generated_date: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator("content_quality_assessment", mode="before")
    @classmethod
    def normalize_quality_assessment(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, str):
            return {"summary": value}
        if isinstance(value, list):
            return {"summary": ", ".join(str(v) for v in value)}
        return value

# Workflow Models
class WorkflowStep(BaseModel):
    step_name: str = Field(description="Name of the workflow step")
    status: str = Field(description="Status of the step (completed, failed, skipped)")
    output_summary: str = Field(description="Summary of step output")
    execution_time: str = Field(description="Time taken for execution")
    error_message: Optional[str] = Field(description="Error message if step failed")

class CompleteWorkflowOutput(BaseModel):
    session_id: str = Field(description="Session identifier")
    topic: str = Field(description="Blog topic")
    pillar: str = Field(description="Content pillar")
    research_method: str = Field(description="Research method used")
    
    # Individual step outputs
    topic_suggestions: Optional[TopicGenerationOutput] = None
    research_findings: Optional[ResearchOutput] = None
    competitor_analysis: Optional[CompetitorAnalysisOutput] = None
    keyword_strategy: Optional[KeywordStrategyOutput] = None
    title_options: Optional[TitleGenerationOutput] = None
    content_structure: Optional[ContentStructureOutput] = None
    blog_post: Optional[BlogGenerationOutput] = None
    
    # Workflow metadata
    workflow_steps: List[WorkflowStep] = Field(description="Execution status of each step")
    total_execution_time: str = Field(description="Total workflow execution time")
    final_status: str = Field(description="Final workflow status")
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Quick access to final deliverables
    final_blog_content: Optional[str] = Field(description="Final blog content")
    final_metadata: Optional[BlogMetadata] = Field(description="Final blog metadata")
    success_metrics: Dict[str, Any] = Field(description="Success metrics and KPIs")