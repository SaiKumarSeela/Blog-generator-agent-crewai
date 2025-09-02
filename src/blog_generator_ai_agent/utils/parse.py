import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class TopicSuggestion(BaseModel):
    """Model for topic suggestions with validation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    target_audience: str = Field(min_length=1, max_length=200)
    content_pillar_alignment: str = Field(min_length=1, max_length=100)
    estimated_engagement_potential: str = Field(pattern="^(Low|Medium|High)$")


class ResearchFinding(BaseModel):
    """Model for research findings with validation"""
    source: str = Field(min_length=1, max_length=200)
    snippet: str = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def extract_crew_output(result: Any) -> str:
    """
    Extract the actual output from CrewAI result with comprehensive fallback handling.
    
    Args:
        result: CrewAI result object or string
        
    Returns:
        str: Extracted content or string representation of result
    """
    if result is None:
        return ""
    
    # Direct string return
    if isinstance(result, str):
        return result.strip()
    
    # Try common CrewAI result attributes
    for attr in ['raw', 'output', 'content']:
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, str) and value.strip():
                return value.strip()
    
    # Handle tasks_output for multi-task results
    if hasattr(result, 'tasks_output') and result.tasks_output:
        last_task = result.tasks_output[-1]
        for attr in ['raw', 'output', 'content']:
            if hasattr(last_task, attr):
                value = getattr(last_task, attr)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(last_task)
    
    # Final fallback
    return str(result)


def _clean_text(text: str) -> str:
    """Clean text by removing ANSI codes and normalizing whitespace"""
    # Remove ANSI color codes
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean up newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text.strip()


def _parse_table_format(raw_output: str) -> List[TopicSuggestion]:
    """Parse topic suggestions from table format"""
    topics = []
    lines = raw_output.split('\n')
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # Detect table header
        if "topic" in line.lower() and "title" in line.lower() and "|" in line:
            in_table = True
            continue
        
        # Skip separator lines
        if in_table and re.match(r'^[\|\-\s]+$', line):
            continue
        
        # Parse table rows
        if in_table and "|" in line and line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 3:
                try:
                    topic = TopicSuggestion(
                        title=parts[0][:200] if parts[0] else f"Topic {len(topics) + 1}",
                        description=parts[1][:1000] if len(parts) > 1 else "AI-generated topic suggestion",
                        target_audience=parts[2][:200] if len(parts) > 2 else "General audience",
                        content_pillar_alignment=parts[3][:100] if len(parts) > 3 else "Technology",
                        estimated_engagement_potential=_normalize_engagement_level(
                            parts[4] if len(parts) > 4 else "Medium"
                        )
                    )
                    topics.append(topic)
                except Exception:
                    continue  # Skip invalid rows
    
    return topics


def _parse_list_format(raw_output: str) -> List[TopicSuggestion]:
    """Parse topic suggestions from numbered or bulleted lists"""
    topics = []
    lines = raw_output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Match numbered lists (1., 2., etc.) or bullet points
        list_patterns = [
            r'^\d+\.\s*(.+)',
            r'^[•\-\*]\s*(.+)',
            r'^[a-zA-Z]\.\s*(.+)'
        ]
        
        for pattern in list_patterns:
            match = re.match(pattern, line)
            if match and len(match.group(1)) > 10:  # Minimum meaningful length
                content = match.group(1)
                
                # Extract title (first sentence or up to colon)
                title = content.split('.')[0].strip()
                if ':' in title:
                    title = title.split(':')[0].strip()
                
                title = title[:200]  # Truncate if too long
                
                try:
                    topic = TopicSuggestion(
                        title=title or f"AI Topic Suggestion {len(topics) + 1}",
                        description=content[:1000],
                        target_audience="Tech enthusiasts, professionals",
                        content_pillar_alignment="Technology",
                        estimated_engagement_potential="High" if len(topics) < 3 else "Medium"
                    )
                    topics.append(topic)
                    
                    if len(topics) >= 7:  # Limit to 7 topics
                        return topics
                        
                except Exception:
                    continue  # Skip invalid topics
                
                break  # Found a match, move to next line
    
    return topics


def _generate_default_topics() -> List[TopicSuggestion]:
    """Generate default topic suggestions as fallback"""
    default_topics = [
        ("AI and Machine Learning Trends", "Explore the latest developments in artificial intelligence and machine learning technologies"),
        ("Future of Technology in Business", "How emerging technologies are reshaping business operations and strategies"),
        ("Digital Transformation Strategies", "Comprehensive guide to implementing digital transformation in organizations"),
        ("Emerging Tech Solutions", "Overview of cutting-edge technological solutions across various industries"),
        ("Innovation in Software Development", "Latest trends and practices in modern software development methodologies")
    ]
    
    topics = []
    for i, (title, description) in enumerate(default_topics):
        try:
            topic = TopicSuggestion(
                title=title,
                description=description,
                target_audience="Tech professionals, decision makers",
                content_pillar_alignment="Technology",
                estimated_engagement_potential="High" if i < 2 else "Medium"
            )
            topics.append(topic)
        except Exception:
            continue  # Skip if validation fails
    
    return topics


def _normalize_engagement_level(level: str) -> str:
    """Normalize engagement level to valid values"""
    level = level.lower().strip()
    if any(word in level for word in ['high', 'excellent', 'great']):
        return "High"
    elif any(word in level for word in ['low', 'poor', 'minimal']):
        return "Low"
    else:
        return "Medium"


def parse_topic_suggestions(raw_output: str) -> List[TopicSuggestion]:
    """
    Parse topic suggestions from crew output with multiple parsing strategies.
    
    Args:
        raw_output: Raw output string from CrewAI
        
    Returns:
        List[TopicSuggestion]: List of parsed topic suggestions
    """
    if not raw_output or not isinstance(raw_output, str):
        return _generate_default_topics()
    
    cleaned_output = _clean_text(raw_output)
    topics = []
    
    try:
        # Strategy 1: Parse table format
        if "|" in cleaned_output and "topic" in cleaned_output.lower():
            topics = _parse_table_format(cleaned_output)
        
        # Strategy 2: Parse list format if table parsing failed
        if not topics:
            topics = _parse_list_format(cleaned_output)
        
        # Strategy 3: Fallback to default topics
        if not topics:
            topics = _generate_default_topics()
    
    except Exception:
        # Final fallback
        topics = _generate_default_topics()
    
    # Ensure we have at least one topic
    if not topics:
        try:
            topics = [TopicSuggestion(
                title="Generated Technology Topic",
                description="AI-generated topic based on your content pillar",
                target_audience="General audience",
                content_pillar_alignment="Technology",
                estimated_engagement_potential="Medium"
            )]
        except Exception:
            pass  # If even this fails, return empty list
    
    return topics


def parse_research_findings(raw_output: str) -> List[ResearchFinding]:
    """
    Parse research findings from crew output.
    
    Args:
        raw_output: Raw output string from CrewAI
        
    Returns:
        List[ResearchFinding]: List of research findings
    """
    findings = []
    
    try:
        if not raw_output or not isinstance(raw_output, str):
            raise ValueError("Invalid raw output")
        
        cleaned_output = _clean_text(raw_output)
        
        if not cleaned_output:
            raise ValueError("Empty cleaned output")
        
        # Create finding with cleaned output
        finding = ResearchFinding(
            source="Research Analysis",
            snippet=cleaned_output,
            tags=["research", "analysis"],
            confidence=0.8
        )
        findings.append(finding)
        
    except Exception as e:
        # Create error finding with limited raw output
        error_snippet = f"Error processing research findings: {str(e)}"
        if raw_output:
            error_snippet += f"\n\nRaw output preview:\n{str(raw_output)[:1000]}..."
        
        try:
            finding = ResearchFinding(
                source="Processing Error",
                snippet=error_snippet,
                tags=["error", "research"],
                confidence=0.0
            )
            findings.append(finding)
        except Exception:
            # If even error finding fails, return empty list
            pass
    
    return findings


def format_crew_result(result: Any, step: str) -> Dict[str, Any]:
    """
    Format crew AI result for API response with comprehensive error handling.
    
    Args:
        result: CrewAI result object
        step: Processing step name
        
    Returns:
        Dict[str, Any]: Formatted result dictionary
    """
    timestamp = datetime.now().isoformat()
    
    try:
        content = extract_crew_output(result)
        
        if not content:
            return {
                "step": step,
                "content": "",
                "error": "No content extracted from result",
                "timestamp": timestamp,
                "status": "warning"
            }
        
        return {
            "step": step,
            "content": content,
            "timestamp": timestamp,
            "status": "completed"
        }
        
    except Exception as e:
        return {
            "step": step,
            "content": str(result)[:1000] if result else "",
            "error": f"Error processing result: {str(e)}",
            "timestamp": timestamp,
            "status": "error"
        }