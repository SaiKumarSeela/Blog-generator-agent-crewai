import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64

# Page configuration
st.set_page_config(
    page_title="Blog Generator AI Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        border-left: 4px solid #2e8b57;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #1f77b4, #2e8b57);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    
    .workflow-step {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    
    .error-details {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    .suggestion-item {
        background-color: #e8f5e8;
        border: 1px solid #c3e6cb;
        border-radius: 3px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
class Config:
    API_BASE_URL = "http://localhost:8085"
    
    # API endpoints
    ENDPOINTS = {
        "topic_generate": "/topic/generate",
        "research_run": "/research/run",
        "competitors_analyse": "/competitors/analyse",
        "seo_keywords": "/seo/keywords",
        "titles_generate": "/titles/generate",
        "structure_create": "/structure/create",
        "blog_generate": "/blog/generate",
        "workflow_run": "/workflow/run",
        "upload": "/upload",
        "health": "/health",
        "session_status": "/sessions/{session_id}",
        "step_output": "/sessions/{session_id}/step/{step_name}"
    }

config = Config()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    default_values = {
        'current_step': 'overview',
        'session_id': None,
        'workflow_data': {},
        'api_responses': {},
        'workflow_complete': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced Error Display Components
def display_error_message(error_data: Dict[str, Any], title: str = "⚠️ Error Occurred"):
    """Display user-friendly error message with suggestions"""
    if not error_data:
        return
    
    st.markdown(f"### {title}")
    
    # Main error message
    if "user_message" in error_data:
        st.error(error_data["user_message"])
    
    # Error code and technical details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if "error_code" in error_data:
            st.info(f"**Error Code:** {error_data['error_code']}")
    
    with col2:
        if "technical_details" in error_data:
            with st.expander("🔧 Technical Details (for support)", expanded=False):
                st.code(error_data["technical_details"], language="text")
    
    # Suggestions
    if "suggestions" in error_data and error_data["suggestions"]:
        st.markdown("**💡 Suggestions to resolve:**")
        for suggestion in error_data["suggestions"]:
            st.markdown(f'<div class="suggestion-item">• {suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown("---")

def display_api_error_response(response: Dict[str, Any], operation: str = "API operation"):
    """Display error information from API response"""
    if "error" not in response:
        return
    
    error_data = response["error"]
    
    # Check if it's a structured error response
    if isinstance(error_data, dict) and "user_message" in error_data:
        display_error_message(error_data, f"⚠️ {operation} Failed")
    else:
        # Fallback for simple error strings
        st.error(f"**{operation} failed:** {error_data}")
    
    # Show session ID if available
    if "session_id" in response:
        st.info(f"**Session ID:** {response['session_id']}")

def check_response_for_errors(response: Dict[str, Any], operation: str = "API operation") -> bool:
    """Check if response contains errors and display them"""
    if "error" in response:
        display_api_error_response(response, operation)
        return True
    
    if "error_info" in response:
        display_error_message(response["error_info"], f"⚠️ {operation} Completed with Errors")
        return True
    
    if response.get("status") == "error":
        st.error(f"**{operation} failed:** {response.get('message', 'Unknown error')}")
        return True
    
    return False

# API utility functions
def make_api_request(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> dict:
    """Make API request with enhanced error handling"""
    url = f"{config.API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=300)
            else:
                response = requests.post(url, json=data, timeout=300)
        else:
            response = requests.get(url, timeout=60)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        error_response = {
            "error": {
                "user_message": "The request took too long to complete. Please try again with a simpler request.",
                "technical_details": "Request timeout exceeded",
                "error_code": "REQUEST_TIMEOUT",
                "suggestions": [
                    "Try breaking down your request into smaller parts",
                    "Check your internet connection",
                    "Wait a few minutes and try again"
                ]
            },
            "status": "error"
        }
        return error_response
    
    except requests.exceptions.ConnectionError:
        error_response = {
            "error": {
                "user_message": "Cannot connect to the server. Please check if the service is running.",
                "technical_details": "Connection refused or server unavailable",
                "error_code": "CONNECTION_ERROR",
                "suggestions": [
                    "Check if the FastAPI server is running",
                    "Verify the API URL is correct",
                    "Check your internet connection"
                ]
            },
            "status": "error"
        }
        return error_response
    
    except requests.exceptions.RequestException as e:
        error_response = {
            "error": {
                "user_message": "Network request failed. Please check your connection and try again.",
                "technical_details": f"Request error: {str(e)}",
                "error_code": "REQUEST_ERROR",
                "suggestions": [
                    "Check your internet connection",
                    "Try refreshing the page",
                    "Contact support if the issue persists"
                ]
            },
            "status": "error"
        }
        return error_response
    
    except json.JSONDecodeError as e:
        error_response = {
            "error": {
                "user_message": "Received invalid response from server. Please try again.",
                "technical_details": f"JSON decode error: {str(e)}",
                "error_code": "INVALID_RESPONSE",
                "suggestions": [
                    "Try refreshing the page",
                    "Check if the server is responding correctly",
                    "Contact support if the issue persists"
                ]
            },
            "status": "error"
        }
        return error_response
    
    except Exception as e:
        error_response = {
            "error": {
                "user_message": "An unexpected error occurred. Please try again or contact support.",
                "technical_details": f"Unexpected error: {str(e)}",
                "error_code": "UNKNOWN_ERROR",
                "suggestions": [
                    "Try refreshing the page",
                    "Check your input data",
                    "Contact support with error details"
                ]
            },
            "status": "error"
        }
        return error_response

def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = make_api_request(config.ENDPOINTS["health"])
        return response.get("status") == "healthy"
    except:
        return False

# UI Components
def display_api_status():
    """Display API connection status"""
    with st.sidebar:
        st.subheader("🔗 API Connection")
        
        if check_api_health():
            st.success("✅ API Connected")
            st.write(f"**Base URL:** {config.API_BASE_URL}")
        else:
            st.error("❌ API Disconnected")
            st.write(f"**Base URL:** {config.API_BASE_URL}")
            st.warning("Please ensure the FastAPI server is running on the specified URL.")

def display_workflow_progress():
    """Display workflow progress"""
    steps = [
        ("📋 Overview", "overview"),
        ("💡 Topic Generation", "topic"),
        ("🔍 Research", "research"),
        ("🏆 Competitor Analysis", "competitors"),
        ("🔑 Keywords", "keywords"),
        ("📰 Titles", "titles"),
        ("🏗️ Structure", "structure"),
        ("📝 Blog Generation", "blog"),
        ("⚡ Complete Workflow", "workflow")
    ]
    
    with st.sidebar:
        st.subheader("📊 Workflow Progress")
        
        for step_name, step_key in steps:
            if st.session_state.current_step == step_key:
                st.markdown(f"**🔵 {step_name}**")
            elif step_key in st.session_state.workflow_data:
                st.markdown(f"✅ {step_name}")
            else:
                st.markdown(f"⚪ {step_name}")

def display_session_info():
    """Display current session information"""
    with st.sidebar:
        st.subheader("📱 Session Info")
        
        if st.session_state.session_id:
            st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
            st.write(f"**Started:** {datetime.now().strftime('%H:%M:%S')}")
            
            # Session data summary
            completed_steps = len([k for k in st.session_state.workflow_data.keys() if k != 'session_id'])
            st.write(f"**Completed Steps:** {completed_steps}/8")
        else:
            st.write("No active session")

def render_json_viewer(data: dict, title: str = "Data"):
    """Render JSON data in an expandable viewer"""
    with st.expander(f"🔍 View {title} (JSON)", expanded=False):
        st.json(data)

def render_metrics_dashboard(data: dict):
    """Render metrics dashboard for workflow data"""
    if not data:
        return
    
    st.subheader("📊 Workflow Metrics")
    
    # Count completed steps
    completed_steps = len([k for k in data.keys() if k not in ['session_id']])
    
    # Create columns based on available data
    available_metrics = []
    
    # Topic metrics
    if 'topic_generation' in data:
        topic_data = data['topic_generation']
        available_metrics.append(("Topics Generated", len(topic_data.get('topics', []))))
    
    # Research metrics
    if 'research' in data:
        research_data = data['research']
        available_metrics.append(("Research Findings", len(research_data.get('findings', []))))
    
    # Competitor metrics
    if 'competitor_analysis' in data:
        comp_data = data['competitor_analysis']
        available_metrics.append(("Competitors Analyzed", len(comp_data.get('competitor_summaries', []))))
    
    # Keywords metrics
    if 'keywords_strategy' in data:
        keywords_data = data['keywords_strategy']
        strategy = keywords_data.get('strategy', {})
        available_metrics.append(("Keywords", len(strategy.get('secondary_keywords', [])) + 1))
    
    # Title metrics
    if 'title_generation' in data:
        title_data = data['title_generation']
        available_metrics.append(("Title Options", len(title_data.get('title_options', []))))
    
    # Structure metrics
    if 'content_structure' in data:
        structure_data = data['content_structure']
        available_metrics.append(("Content Sections", len(structure_data.get('sections', []))))
    
    # Blog metrics
    if 'blog_generation' in data:
        blog_data = data['blog_generation']
        metadata = blog_data.get('metadata', {})
        available_metrics.append(("Word Count", metadata.get('word_count', 0)))
    
    # Complete workflow metrics
    if 'complete_workflow' in data:
        workflow_data = data['complete_workflow']
        available_metrics.append(("Workflow Status", workflow_data.get('final_status', 'completed').title()))
    
    # Create columns dynamically
    if available_metrics:
        cols = st.columns(min(len(available_metrics), 4))
        
        for i, (label, value) in enumerate(available_metrics):
            with cols[i % len(cols)]:
                if isinstance(value, str):
                    st.metric(label, value)
                else:
                    st.metric(label, value, help=f"{label} from completed workflow")
    
    # Show completion status
    st.info(f"✅ **{completed_steps} workflow steps completed**")

# Page Functions
def render_overview_page():
    """Render overview/home page"""
    st.markdown('<div class="main-header">📝 Blog Generator AI Agent</div>', unsafe_allow_html=True)
    st.markdown("""
    ## Welcome to the AI-Powered Blog Generation System
    This comprehensive platform empowers you to create SEO-optimized, brand-consistent blog posts in minutes through an intelligent workflow  delivering automatically researched and keyword-optimized content, professional quality aligned with your brand voice, significant time savings, and a cost-effective alternative to traditional content creation
    #### 🚀 **How to Use**
    **Step 1:** Enter your content topic or pillar  
    **Step 2:** Choose your preferred workflow (step-by-step or automated)  
    **Step 3:** Get your complete, ready-to-publish blog post
    #### 🔧 **How It Works**
    Our AI agents automatically handle research, competitor analysis, keyword strategy, and content creation. You get a complete blog post that's ready to publish or customize further.
    """)
    st.markdown("---")
    # Quick start options
    st.subheader("🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Start Step-by-Step workflow", use_container_width=True):
            st.session_state.current_step = 'topic'
            st.rerun()
    
    with col2:
        if st.button("⚡ Run Complete Workflow", use_container_width=True):
            st.session_state.current_step = 'workflow'
            st.rerun()
    
    with col3:
        if st.button("📊 View Session data", use_container_width=True):
            if st.session_state.workflow_data:
                render_metrics_dashboard(st.session_state.workflow_data)
            else:
                st.info("No workflow data available yet. Start by generating topics!")

def render_topic_generation_page():
    """Render topic generation page"""
    st.markdown('<div class="step-header">💡 Step 1: Topic Generation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate topic suggestions based on your content pillar. The AI will analyze your pillar and suggest relevant, 
    SEO-friendly topics that align with your content strategy.
    """)
    
    with st.form("topic_generation_form"):
        pillar = st.text_input(
            "Content Pillar",
            placeholder="e.g., Customer Loyalty Programs, Digital Marketing, E-commerce",
            help="Enter the main theme or category for your content"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("Generate Topics", use_container_width=True)
    
    if submitted and pillar:
        with st.spinner("🔄 Generating topic suggestions..."):
            request_data = {"pillar": pillar}
            response = make_api_request(
                config.ENDPOINTS["topic_generate"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Topic Generation"):
                # If there are errors, still show the response data if available
                if "topics" in response and response["topics"]:
                    st.warning("⚠️ Topic generation completed with some errors, but topics were generated.")
                    st.session_state.session_id = response.get("session_id")
                    st.session_state.workflow_data['topic_generation'] = response
                else:
                    st.error("❌ Topic generation failed. Please try again.")
                    return
            else:
                st.session_state.session_id = response.get("session_id")
                st.session_state.workflow_data['topic_generation'] = response
                st.success("✅ Topic generation completed!")
            
            # Display generated topics
            topics = response.get("topics", [])
            if topics:
                st.subheader("💡 Generated Topics")
                
                selected_topic = st.selectbox(
                    "Select a topic to proceed:",
                    options=range(len(topics)),
                    format_func=lambda x: f"{topics[x].get('title', 'Untitled')}"
                )
                
                if selected_topic is not None:
                    topic_details = topics[selected_topic]
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Title:** {topic_details.get('title', 'N/A')}")
                        st.write(f"**Description:** {topic_details.get('description', 'N/A')}")
                        st.write(f"**Pillar:** {topic_details.get('pillar', pillar)}")
                    
                    with col2:
                        if st.button("Continue to Research →", use_container_width=True):
                            st.session_state.current_step = 'research'
                            st.rerun()
            
            render_json_viewer(response, "Topic Generation Response")
    
    elif submitted:
        st.error("Please enter a content pillar to generate topics.")

def render_research_page():
    """Render research page"""
    st.markdown('<div class="step-header">🔍 Step 2: Research</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Choose your research method to gather comprehensive information about your topic.
    """)
    
    # Show previous topic if available, otherwise ask for input
    previous_topic = ""
    previous_pillar = ""
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
            previous_pillar = st.session_state.workflow_data['topic_generation'].get('pillar', '')
    
    with st.form("research_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Topic",
                value=previous_topic,
                placeholder="Enter topic to research",
                help="Topic to research"
            )
            
            research_mode = st.selectbox(
                "Research Mode",
                options=["SERP", "RAG", "reference"],
                help="Choose how to conduct research"
            )
        
        with col2:
            pillar = st.text_input(
                "Content Pillar",
                value=previous_pillar,
                placeholder="Enter content pillar",
                help="Content pillar for context"
            )
        
        # Additional options based on mode
        urls = []
        uploads = []
        
        if research_mode in ["SERP", "reference"]:
            urls_input = st.text_area(
                "Additional URLs (optional)",
                placeholder="https://example.com\nhttps://another-example.com",
                help="One URL per line"
            )
            if urls_input:
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Reference Files (optional)",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'docx'],
            help="Upload documents to include in research"
        )
        
        submitted = st.form_submit_button("Start Research", use_container_width=True)
    
    if submitted and topic:
        with st.spinner("🔄 Conducting research..."):
            # Handle file uploads first
            file_references = []
            if uploaded_files:
                st.info(f"📁 Uploading {len(uploaded_files)} files...")
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    upload_response = make_api_request(
                        config.ENDPOINTS["upload"],
                        method="POST",
                        files=files
                    )
                    if "error" not in upload_response:
                        file_references.append(upload_response.get("file_path", ""))
            
            request_data = {
                "topic": topic,
                "mode": research_mode,
                "research_method": research_mode,
                "urls": urls,
                "uploads": file_references,
                "pillar": pillar
            }
            
            response = make_api_request(
                config.ENDPOINTS["research_run"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Research"):
                # If there are errors, still show the response data if available
                if "findings" in response and response["findings"]:
                    st.warning("⚠️ Research completed with some errors, but findings were generated.")
                    st.session_state.workflow_data['research'] = response
                else:
                    st.error("❌ Research failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['research'] = response
                st.success("✅ Research completed successfully!")
            
            # Display research results
            findings = response.get("findings", [])
            if findings:
                st.subheader(f"📊 Research Results ({len(findings)} findings)")
                
                # Display findings in expandable sections
                for i, finding in enumerate(findings[:5]):  # Show first 5
                    with st.expander(f"Finding {i+1}: "):
                        st.write(f"**Source:** {finding.get('source', 'N/A')}")
                        st.write(f"**Content:** {finding.get('insight', 'N/A')[:200]}...")
                        st.write(f"**Citation:** {finding.get('citation', 'N/A')}")
                        
            # Key insights
            key_insights = response.get("key_insights", [])
            if key_insights:
                st.subheader("💡 Key Insights")
                for insight in key_insights[:3]:
                    st.info(insight)
            
            if st.button("Continue to Competitor Analysis →", use_container_width=True):
                st.session_state.current_step = 'competitors'
                st.rerun()
            
            render_json_viewer(response, "Research Response")
            
    elif submitted:
        st.error("Please enter a topic for research.")

def render_competitor_analysis_page():
    """Render competitor analysis page"""
    st.markdown('<div class="step-header">🏆 Step 3: Competitor Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze competitor content to understand the competitive landscape and identify opportunities 
    for your blog post.
    """)
    
    # Show previous topic if available
    previous_topic = ""
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
    
    with st.form("competitor_analysis_form"):
        topic = st.text_input(
            "Topic",
            value=previous_topic,
            placeholder="Enter topic for competitor analysis",
            help="Topic for competitor analysis"
        )
        
        urls_input = st.text_area(
            "Competitor URLs",
            placeholder="https://competitor1.com/blog-post\nhttps://competitor2.com/article\nhttps://competitor3.com/guide",
            help="Enter competitor URLs (one per line). Leave empty for automatic discovery."
        )
        
        submitted = st.form_submit_button("Analyze Competitors", use_container_width=True)
    
    if submitted and topic:
        # Parse URLs
        urls = []
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        else:
            # Use default competitor URLs for demo
            urls = [
                "https://blog.hubspot.com",
                "https://neilpatel.com/blog",
                "https://contentmarketinginstitute.com"
            ]
        
        with st.spinner("🔄 Analyzing competitors..."):
            request_data = {
                "topic": topic,
                "urls": urls,
                "research_method": "SERP"
            }
            
            response = make_api_request(
                config.ENDPOINTS["competitors_analyse"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Competitor Analysis"):
                # If there are errors, still show the response data if available
                if "competitor_summaries" in response and response["competitor_summaries"]:
                    st.warning("⚠️ Competitor analysis completed with some errors, but results were generated.")
                    st.session_state.workflow_data['competitor_analysis'] = response
                else:
                    st.error("❌ Competitor analysis failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['competitor_analysis'] = response
                st.success(f"✅ Analyzed {len(urls)} competitors!")
            
            # Display competitor analysis results
            summaries = response.get("competitor_summaries", [])
            if summaries:
                st.subheader("📊 Competitor Analysis Results")
                
                for summary in summaries:
                    with st.expander(f"🔍 {summary.get('url', 'Unknown URL')}"):
                        st.write(f"**Title:** {summary.get('title', 'N/A')}")
                        st.write(f"**Summary:** {summary.get('summary', 'N/A')}")
                        st.write(f"**Tone:** {summary.get('tone_style', 'N/A')}")
                        st.write(f"**Keywords:** {summary.get('keyword_strategy', 'N/A')}")
                        st.write(f"**Strengths:** {summary.get('strengths', 'N/A')}")
                        st.write(f"**Weaknesses:** {summary.get('weaknesses', 'N/A')}")
                        st.write(f"**Structure:** {summary.get('structure_pattern', 'N/A')}")
                        
            
            if st.button("Continue to Keywords →", use_container_width=True):
                st.session_state.current_step = 'keywords'
                st.rerun()
            
            render_json_viewer(response, "Competitor Analysis Response")

def render_keywords_page():
    """Render keywords generation page"""
    st.markdown('<div class="step-header">🔑 Step 4: SEO Keywords</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate a strategic keyword plan including primary and secondary keywords optimized for SEO.
    """)
    
    # Show previous data if available
    previous_topic = ""
    previous_pillar = ""
    previous_findings = ""
    
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
            previous_pillar = st.session_state.workflow_data['topic_generation'].get('pillar', '')
    
    if st.session_state.workflow_data.get('research'):
        previous_findings = json.dumps(st.session_state.workflow_data['research'], indent=2)
    
    with st.form("keywords_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Topic",
                value=previous_topic,
                placeholder="Enter topic for keyword generation",
                help="Topic for keyword generation"
            )
            
            primary_keyword = st.text_input(
                "Primary Keyword (optional)",
                placeholder="e.g., customer loyalty program",
                help="Leave empty for AI suggestion"
            )
        
        with col2:
            pillar = st.text_input(
                "Content Pillar",
                value=previous_pillar,
                placeholder="Enter content pillar",
                help="Content pillar for context"
            )
        
        findings = st.text_area(
            "Research Findings (optional)",
            value=previous_findings,
            height=100,
            placeholder="Enter research findings or leave empty",
            help="Research findings to inform keyword strategy"
        )
        
        submitted = st.form_submit_button("Generate Keywords", use_container_width=True)
    
    if submitted and topic:
        with st.spinner("🔄 Generating keyword strategy..."):
            request_data = {
                "topic": topic,
                "findings": findings,
                "primary_keyword": primary_keyword,
                "pillar": pillar,
                "research_method": "SERP"
            }
            
            response = make_api_request(
                config.ENDPOINTS["seo_keywords"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Keyword Strategy"):
                # If there are errors, still show the response data if available
                if "strategy" in response and response["strategy"]:
                    st.warning("⚠️ Keyword strategy completed with some errors, but strategy was generated.")
                    st.session_state.workflow_data['keywords_strategy'] = response
                else:
                    st.error("❌ Keyword strategy failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['keywords_strategy'] = response
                st.success("✅ Keyword strategy generated!")
            
            # Display keyword strategy
            strategy = response.get("strategy", {})
            if strategy:
                st.subheader("🎯 Keyword Strategy")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Primary Keyword**")
                    primary = strategy.get("primary_keyword", "N/A")
                    st.code(primary, language=None)
                
                with col2:
                    st.markdown("**Secondary Keywords**")
                    secondary = strategy.get("secondary_keywords", [])
                    for kw in secondary:
                        st.write(f"• {kw}")
                
                with col3:
                    st.markdown("**Long-tail Keywords**")
                    long_tail = strategy.get("long_tail_keywords", [])
                    for kw in long_tail[:3]:  # Show first 3
                        st.write(f"• {kw}")
            
            if st.button("Continue to Titles →", use_container_width=True):
                st.session_state.current_step = 'titles'
                st.rerun()
            
            render_json_viewer(response, "Keywords Response")

def render_titles_page():
    """Render title generation page"""
    st.markdown('<div class="step-header">📰 Step 5: Title Generation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate multiple title variations optimized for SEO and user engagement.
    """)
    
    # Show previous data if available
    previous_topic = ""
    previous_primary = ""
    previous_secondary = []
    
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
    
    if st.session_state.workflow_data.get('keywords_strategy'):
        strategy = st.session_state.workflow_data['keywords_strategy'].get("strategy", {})
        previous_primary = strategy.get("primary_keyword", "")
        previous_secondary = strategy.get("secondary_keywords", [])
    
    with st.form("titles_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Topic",
                value=previous_topic,
                placeholder="Enter topic for title generation",
                help="Topic for title generation"
            )
            
            primary = st.text_input(
                "Primary Keyword",
                value=previous_primary,
                placeholder="Enter primary keyword",
                help="Primary keyword to include in titles"
            )
        
        with col2:
            secondary_input = st.text_area(
                "Secondary Keywords",
                value="\n".join(previous_secondary),
                placeholder="Enter secondary keywords (one per line)",
                help="One keyword per line"
            )
        
        submitted = st.form_submit_button("Generate Titles", use_container_width=True)
    
    if submitted and topic and primary:
        secondary_list = [kw.strip() for kw in secondary_input.split('\n') if kw.strip()]
        
        with st.spinner("🔄 Generating title options..."):
            request_data = {
                "topic": topic,
                "primary": primary,
                "secondary": secondary_list,
                "research_method": "SERP"
            }
            
            response = make_api_request(
                config.ENDPOINTS["titles_generate"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Title Generation"):
                # If there are errors, still show the response data if available
                if "title_options" in response and response["title_options"]:
                    st.warning("⚠️ Title generation completed with some errors, but titles were generated.")
                    st.session_state.workflow_data['title_generation'] = response
                else:
                    st.error("❌ Title generation failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['title_generation'] = response
                st.success("✅ Title options generated!")
            
            # Display title options
            title_options = response.get("title_options", [])
            if title_options:
                st.subheader("📝 Title Options")
                
                selected_title_idx = st.radio(
                    "Select a title:",
                    options=range(len(title_options)),
                    format_func=lambda x: f"**{title_options[x].get('type', 'Standard')}:** {title_options[x].get('title', 'N/A')}"
                )
                
                if selected_title_idx is not None:
                    selected_title = title_options[selected_title_idx]
                    
                    # Show title details
                    st.info(f"**Selected:** {selected_title.get('title', 'N/A')}")
                    st.write(f"**Type:** {selected_title.get('type', 'N/A')}")
                    st.write(f"**SEO Score:** {selected_title.get('seo_score', 'N/A')}")
                    
                    if st.button("Continue to Structure →", use_container_width=True):
                        st.session_state.current_step = 'structure'
                        st.rerun()
            
            render_json_viewer(response, "Title Generation Response")

def render_structure_page():
    """Render content structure page"""
    st.markdown('<div class="step-header">🏗️ Step 6: Content Structure</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Choose and customize your blog post structure.
    """)
    
    # Show previous data if available
    previous_topic = ""
    previous_primary = ""
    previous_keywords = {}
    
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
    
    if st.session_state.workflow_data.get('keywords_strategy'):
        strategy = st.session_state.workflow_data['keywords_strategy'].get("strategy", {})
        previous_primary = strategy.get("primary_keyword", "")
        previous_keywords = strategy
    
    with st.form("structure_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Topic",
                value=previous_topic,
                placeholder="Enter topic for structure creation",
                help="Topic for structure creation"
            )
            
            structure_type = st.selectbox(
                "Structure Type",
                options=["How-to Guide", "Listicle", "Thought Leadership", "Deep-dive Explainer"],
                help="Choose the blog post structure type"
            )
        
        with col2:
            primary_keyword = st.text_input(
                "Primary Keyword",
                value=previous_primary,
                placeholder="Enter primary keyword",
                help="Primary keyword for the structure"
            )
        
        submitted = st.form_submit_button("Create Structure", use_container_width=True)
    
    if submitted and topic and structure_type:
        with st.spinner("🔄 Creating content structure..."):
            request_data = {
                "topic": topic,
                "structure_type": structure_type,
                "keywords": previous_keywords,
                "primary_keyword": primary_keyword,
                "research_method": "SERP"
            }
            
            response = make_api_request(
                config.ENDPOINTS["structure_create"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Content Structure"):
                # If there are errors, still show the response data if available
                if "sections" in response and response["sections"]:
                    st.warning("⚠️ Content structure completed with some errors, but structure was generated.")
                    st.session_state.workflow_data['content_structure'] = response
                else:
                    st.error("❌ Content structure failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['content_structure'] = response
                st.success("✅ Content structure created!")
            
            # Display structure
            sections = response.get("sections", [])
            if sections:
                st.subheader("📋 Content Structure")
                
                for i, section in enumerate(sections):
                    with st.expander(f"Section {i+1}: {section.get('heading', 'N/A')}"):
                        st.write(f"**Type:** {section.get('type', 'N/A')}")
                        st.write(f"**Content:** {section.get('description', 'N/A')}")
                        if section.get('word_count'):
                            st.write(f"**Target Words:** {section.get('word_count', 'N/A')}")
            
            if st.button("Continue to Blog Generation →", use_container_width=True):
                st.session_state.current_step = 'blog'
                st.rerun()
            
            render_json_viewer(response, "Content Structure Response")

def render_blog_generation_page():
    """Render blog generation page"""
    st.markdown('<div class="step-header">📝 Step 7: Blog Generation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate the complete blog post based on your structure and keywords.
    """)
    
    # Show previous data if available
    previous_topic = ""
    previous_structure = "How-to Guide"
    previous_primary = ""
    previous_keywords = {}
    
    if st.session_state.workflow_data.get('topic_generation'):
        topics = st.session_state.workflow_data['topic_generation'].get('topics', [])
        if topics:
            previous_topic = topics[0].get('title', '')
    
    if st.session_state.workflow_data.get('content_structure'):
        previous_structure = st.session_state.workflow_data['content_structure'].get('structure_type', 'How-to Guide')
    
    if st.session_state.workflow_data.get('keywords_strategy'):
        strategy = st.session_state.workflow_data['keywords_strategy'].get("strategy", {})
        previous_primary = strategy.get("primary_keyword", "")
        previous_keywords = strategy
    
    with st.form("blog_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Topic",
                value=previous_topic,
                placeholder="Enter blog topic",
                help="Blog topic"
            )
            
            structure_type = st.selectbox(
                "Structure Type",
                options=["How-to Guide", "Listicle", "Thought Leadership", "Deep-dive Explainer"],
                index=0,
                help="Blog structure type"
            )
        
        with col2:
            primary_keyword = st.text_input(
                "Primary Keyword",
                value=previous_primary,
                placeholder="Enter primary keyword",
                help="Primary keyword for SEO"
            )
            
            brand_voice = st.text_input(
                "Brand Voice",
                value="professional, helpful, concise",
                help="Brand voice guidelines"
            )
        
        submitted = st.form_submit_button("Generate Blog Post", use_container_width=True)
    
    if submitted and topic:
        with st.spinner("🔄 Generating blog post... This may take a few minutes."):
            request_data = {
                "topic": topic,
                "structure_type": structure_type,
                "primary_keyword": primary_keyword,
                "keywords": previous_keywords,
                "brand_voice": brand_voice,
                "research_method": "SERP"
            }
            
            response = make_api_request(
                config.ENDPOINTS["blog_generate"],
                method="POST",
                data=request_data
            )
            
            # Check for errors first
            if check_response_for_errors(response, "Blog Generation"):
                # If there are errors, still show the response data if available
                if "blog_content" in response and response["blog_content"]:
                    st.warning("⚠️ Blog generation completed with some errors, but content was generated.")
                    st.session_state.workflow_data['blog_generation'] = response
                else:
                    st.error("❌ Blog generation failed. Please try again.")
                    return
            else:
                st.session_state.workflow_data['blog_generation'] = response
                st.success("✅ Blog post generated successfully!")
            
            # Display blog content
            blog_content = response.get("blog_content", "")
            metadata = response.get("metadata", {})
            
            if blog_content:
                st.subheader("📄 Generated Blog Post")
                
                # Blog metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", metadata.get("word_count", 0))
                with col2:
                    st.metric("Reading Time", f"{metadata.get('reading_time', 0)} min")
                with col3:
                    st.metric("SEO Score", metadata.get("seo_score", "N/A"))
                with col4:
                    st.metric("Keywords Used", metadata.get("keywords_count", 0))
                
                # Blog content display
                with st.expander("📖 View Full Blog Post", expanded=True):
                    st.markdown(blog_content, unsafe_allow_html=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download as Markdown",
                        blog_content,
                        file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Convert to HTML for download
                    html_content = f"""
                    <html>
                    <head>
                        <title>{topic}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                            h1, h2, h3 {{ color: #333; }}
                            p {{ line-height: 1.6; }}
                        </style>
                    </head>
                    <body>
                        {blog_content}
                    </body>
                    </html>
                    """
                    st.download_button(
                        "📥 Download as HTML",
                        html_content,
                        file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            
            # Citations and sources
            sources = response.get("sources", [])
            if sources:
                st.subheader("📚 Sources & Citations")
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source.get('title', 'Source')}]({source.get('url', '#')})")
            
            render_json_viewer(response, "Blog Generation Response")
            
            # Mark workflow as complete
            st.session_state.workflow_complete = True

def render_complete_workflow_page():
    """Render complete workflow automation page"""
    st.markdown('<div class="step-header">⚡ Complete Workflow Automation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Run the entire blog generation workflow automatically. This will execute all steps 
    from topic generation to final blog post in one streamlined process.
    """)
    
    with st.form("complete_workflow_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Blog Topic",
                placeholder="e.g., How to Build Customer Loyalty Programs",
                help="Main topic for your blog post"
            )
            
            pillar = st.text_input(
                "Content Pillar",
                placeholder="e.g., Customer Engagement",
                help="Content pillar/category"
            )
            
            mode = st.selectbox(
                "Research Mode",
                options=["SERP", "RAG", "reference"],
                help="How to conduct research"
            )
        
        with col2:
            structure = st.selectbox(
                "Blog Structure",
                options=["How-to Guide", "Listicle", "Thought Leadership", "Deep-dive Explainer"],
                help="Structure type for the blog"
            )
            
            brand_voice = st.text_input(
                "Brand Voice",
                value="professional, helpful, concise",
                help="Brand voice guidelines"
            )
        
        # Additional options
        st.markdown("**Additional Options (Optional)**")
        
        urls_input = st.text_area(
            "Reference URLs",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Additional URLs for research (one per line)"
        )
        
        uploaded_files = st.file_uploader(
            "Upload Reference Files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'docx'],
            help="Upload documents for reference"
        )
        
        submitted = st.form_submit_button("🚀 Run Complete Workflow", use_container_width=True)
    
    if submitted and topic and pillar:
        # Prepare URLs
        urls = []
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        # Handle file uploads
        uploads = []
        if uploaded_files:
            with st.spinner("📁 Uploading files..."):
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    upload_response = make_api_request(
                        config.ENDPOINTS["upload"],
                        method="POST",
                        files=files
                    )
                    if "error" not in upload_response:
                        uploads.append(upload_response.get("file_path", ""))
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔄 Running complete workflow..."):
            status_text.text("Starting workflow...")
            progress_bar.progress(0.1)
            
            request_data = {
                "topic": topic,
                "pillar": pillar,
                "mode": mode,
                "structure": structure,
                "brand_voice": brand_voice
            }
            
            if urls:
                request_data["urls"] = urls
            if uploads:
                request_data["uploads"] = uploads
            
            status_text.text("Executing workflow steps...")
            progress_bar.progress(0.5)
            
            response = make_api_request(
                config.ENDPOINTS["workflow_run"],
                method="POST",
                data=request_data
            )
            
            progress_bar.progress(1.0)
            
            # Check for errors first
            if check_response_for_errors(response, "Complete Workflow"):
                # If there are errors, still show the response data if available
                if "final_blog_content" in response and response["final_blog_content"]:
                    st.warning("⚠️ Complete workflow completed with some errors, but content was generated.")
                    st.session_state.session_id = response.get("session_id")
                    st.session_state.workflow_data['complete_workflow'] = response
                    st.session_state.workflow_complete = True
                else:
                    st.error("❌ Complete workflow failed. Please try again.")
                    progress_bar.empty()
                    status_text.empty()
                    return
            else:
                st.session_state.session_id = response.get("session_id")
                st.session_state.workflow_data['complete_workflow'] = response
                st.session_state.workflow_complete = True
                st.success("✅ Complete workflow executed successfully!")
            
            status_text.empty()
            progress_bar.empty()
            
            # Display workflow results
            st.subheader("🎉 Workflow Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            success_metrics = response.get("success_metrics", {})
            
            with col1:
                st.metric("Status", response.get("final_status", "completed").title())
            with col2:
                st.metric("Execution Time", response.get("execution_time", "N/A"))
            with col3:
                st.metric("Steps Completed", success_metrics.get("steps_completed", 0))
            with col4:
                st.metric("Word Count", success_metrics.get("final_word_count", 0))
            
            # Generated content preview
            if response.get("final_blog_content"):
                st.subheader("📄 Generated Blog Post")
                
                with st.expander("📖 View Generated Blog", expanded=False):
                    st.markdown(response.get("final_blog_content", ""), unsafe_allow_html=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Blog (Markdown)",
                        response.get("final_blog_content", ""),
                        file_name=f"complete_workflow_blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Create workflow summary
                    workflow_summary = {
                        "topic": topic,
                        "pillar": pillar,
                        "execution_time": response.get("execution_time"),
                        "final_status": response.get("final_status"),
                        "success_metrics": success_metrics,
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        "📥 Download Summary (JSON)",
                        json.dumps(workflow_summary, indent=2),
                        file_name=f"workflow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            render_json_viewer(response, "Complete Workflow Response")

def render_workflow_navigation():
    """Render navigation between workflow steps"""
    steps = [
        ("📋 Overview", "overview"),
        ("💡 Topics", "topic"),
        ("🔍 Research", "research"),
        ("🏆 Competitors", "competitors"),
        ("🔑 Keywords", "keywords"),
        ("📰 Titles", "titles"),
        ("🏗️ Structure", "structure"),
        ("📝 Blog", "blog"),
        ("⚡ Workflow", "workflow")
    ]
    
    with st.sidebar:
        st.subheader("🧭 Navigation")
        
        for step_name, step_key in steps:
            if st.button(f"{step_name}", key=f"nav_{step_key}", use_container_width=True):
                st.session_state.current_step = step_key
                st.rerun()

def main():
    """Main application function"""
    init_session_state()
    
    # Sidebar
    display_api_status()
    display_workflow_progress()
    display_session_info()
    render_workflow_navigation()
    
    # Main content area
    if st.session_state.current_step == 'overview':
        render_overview_page()
    elif st.session_state.current_step == 'topic':
        render_topic_generation_page()
    elif st.session_state.current_step == 'research':
        render_research_page()
    elif st.session_state.current_step == 'competitors':
        render_competitor_analysis_page()
    elif st.session_state.current_step == 'keywords':
        render_keywords_page()
    elif st.session_state.current_step == 'titles':
        render_titles_page()
    elif st.session_state.current_step == 'structure':
        render_structure_page()
    elif st.session_state.current_step == 'blog':
        render_blog_generation_page()
    elif st.session_state.current_step == 'workflow':
        render_complete_workflow_page()
    
    # Footer with workflow summary
    if st.session_state.workflow_data:
        st.markdown("---")
        render_metrics_dashboard(st.session_state.workflow_data)
        
        if st.session_state.workflow_complete:
            st.success("🎉 Workflow completed successfully! You can now download your generated content.")

if __name__ == "__main__":
    main()