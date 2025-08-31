from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools.custom_tool import WebSearchTool, EnhancedRAGTool, ResearchModeTool, CompetitorAnalysisTool
from dotenv import load_dotenv
import os
from pathlib import Path
from .utils import load_yaml_config

load_dotenv()



@CrewBase
class BlogGeneratorCrew():
    """Blog Generator Crew for creating SEO-optimized content"""
    
    def __init__(self) -> None:
        # Get the project root directory (blog_generator_ai_agent folder)
        project_root = Path(__file__).parent.parent.parent
        
        # Create memory directory if it doesn't exist
        memory_dir = project_root / 'memory'
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Set memory path
        self.memory_path = str(memory_dir / 'crew_memory.json')
        
        # Load configurations with proper paths
        agents_config_path = project_root / 'src' / 'blog_generator_ai_agent' / 'config' / 'agents.yaml'
        tasks_config_path = project_root / 'src' / 'blog_generator_ai_agent' / 'config' / 'tasks.yaml'
        
        try:
            self.agents_config = load_yaml_config(str(agents_config_path))
            self.tasks_config = load_yaml_config(str(tasks_config_path))
            
            # Debug: Check if configs loaded properly
            if not isinstance(self.agents_config, dict):
                raise ValueError(f"agents_config is not a dict: {type(self.agents_config)}")
            if not isinstance(self.tasks_config, dict):
                raise ValueError(f"tasks_config is not a dict: {type(self.tasks_config)}")
                
        except Exception as e:
            print(f"Error loading configurations: {e}")
            print(f"Agents config path: {agents_config_path}")
            print(f"Tasks config path: {tasks_config_path}")
            raise
        
        # Initialize Gemini LLM with rate limiting
        self.llm = LLM(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini/gemini-1.5-flash",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            max_retries=2,
            timeout=30
        )
        
        # Initialize tools based on custom_tool.py
        self.web_search_tool = WebSearchTool()
        self.enhanced_rag_tool = EnhancedRAGTool()
        self.research_mode_tool = ResearchModeTool()
        self.competitor_analysis_tool = CompetitorAnalysisTool()
    
    @agent
    def topic_researcher(self) -> Agent:
        """Agent specialized in topic research and suggestion generation"""
        return Agent(
            config=self.agents_config['topic_researcher'],
            llm=self.llm,
            tools=[self.web_search_tool, self.research_mode_tool],
            verbose=True
        )
    
    @agent
    def knowledge_retriever(self) -> Agent:
        """Agent for retrieving and processing knowledge from various sources"""
        return Agent(
            config=self.agents_config['knowledge_retriever'],
            llm=self.llm,
            tools=[self.enhanced_rag_tool, self.research_mode_tool, self.web_search_tool],
            verbose=True
        )
    
    @agent
    def competitor_analyst(self) -> Agent:
        """Agent specialized in competitor content analysis"""
        return Agent(
            config=self.agents_config['competitor_analyst'],
            llm=self.llm,
            tools=[self.competitor_analysis_tool, self.web_search_tool],
            verbose=True
        )
    
    @agent
    def seo_strategist(self) -> Agent:
        """Agent focused on SEO strategy and keyword research"""
        return Agent(
            config=self.agents_config['seo_strategist'],
            llm=self.llm,
            tools=[self.web_search_tool, self.research_mode_tool],
            verbose=True
        )
    
    @agent
    def content_structurer(self) -> Agent:
        """Agent responsible for structuring and organizing content"""
        return Agent(
            config=self.agents_config['content_structurer'],
            llm=self.llm,
            tools=[],
            verbose=True
        )
    
    @agent
    def blog_writer(self) -> Agent:
        """Agent specialized in writing comprehensive blog content"""
        return Agent(
            config=self.agents_config['blog_writer'],
            llm=self.llm,
            tools=[],
            verbose=True
        )
    
    @task
    def generate_topic_suggestions(self) -> Task:
        """Generate topic suggestions using SERP analysis"""
        return Task(
            config=self.tasks_config['generate_topic_suggestions'],
            agent=self.topic_researcher(),
            verbose=True
        )
    
    @task
    def research_topic(self) -> Task:
        """Research the topic using multiple research modes"""
        return Task(
            config=self.tasks_config['research_topic'],
            agent=self.knowledge_retriever(),
            verbose=True
        )
    
    @task
    def analyze_competitors(self) -> Task:
        """Analyze competitor content for insights"""
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst(),
            verbose=True
        )
    
    @task
    def develop_keyword_strategy(self) -> Task:
        """Develop comprehensive keyword strategy"""
        return Task(
            config=self.tasks_config['develop_keyword_strategy'],
            agent=self.seo_strategist(),
            context=[self.research_topic(), self.analyze_competitors()],
            verbose=True
        )
    
    @task
    def generate_title_options(self) -> Task:
        """Generate optimized title options"""
        return Task(
            config=self.tasks_config['generate_title_options'],
            agent=self.content_structurer(),
            context=[self.develop_keyword_strategy()],
            verbose=True
        )
    
    @task
    def create_content_structure(self) -> Task:
        """Create detailed content structure and outline"""
        return Task(
            config=self.tasks_config['create_content_structure'],
            agent=self.content_structurer(),
            context=[self.research_topic(), self.develop_keyword_strategy()],
            verbose=True
        )
    
    @task
    def generate_full_blog(self) -> Task:
        """Generate the complete blog post"""
        return Task(
            config=self.tasks_config['generate_full_blog'],
            agent=self.blog_writer(),
            context=[
                self.research_topic(), 
                self.develop_keyword_strategy(), 
                self.create_content_structure()
            ],
            output_file='generated_blog.md',
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """Create the Blog Generator crew with enhanced tools"""
        # Create memory directory if it doesn't exist
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        return Crew(
            agents=[
                self.topic_researcher(),
                self.knowledge_retriever(),
                self.competitor_analyst(),
                self.seo_strategist(),
                self.content_structurer(),
                self.blog_writer()
            ],
            tasks=[
                self.generate_topic_suggestions(),
                self.research_topic(),
                self.analyze_competitors(),
                self.develop_keyword_strategy(),
                self.generate_title_options(),
                self.create_content_structure(),
                self.generate_full_blog()
            ],
            process=Process.sequential,
            verbose=True,
            memory=True,
            full_output=True,
            manager_llm=self.llm,
            embedder={
                "provider": "google",
                "config": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "text-embedding-004"  # or "text-embedding-preview-0409"
                }
            },
            memory_config={
                'type': 'file',
                'path': str(memory_dir / 'crew_memory.json'),
                'embedder': {
                    'provider': 'google',
                    'config': {
                        'api_key': os.getenv("GEMINI_API_KEY"),
                        'model': 'text-embedding-004'
                    }
                }
            }
        )
    
    def add_knowledge_documents(self, pdf_paths=None, markdown_paths=None, urls=None):
        """Add documents to the knowledge base before running workflow"""
        results = {}
        
        if pdf_paths:
            results['pdf_results'] = self.enhanced_rag_tool.add_pdf_documents(pdf_paths)
        
        if markdown_paths:
            results['markdown_results'] = self.enhanced_rag_tool.add_markdown_documents(markdown_paths)
        
        if urls:
            results['url_results'] = self.enhanced_rag_tool.add_url_documents(urls)
        
        return results
    
    def run_workflow(self, inputs: dict) -> dict:
        """Execute the complete blog generation workflow with enhanced research capabilities"""
        
        # Validate required inputs
        required_keys = ['topic', 'pillar']
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Set default values for optional inputs
        inputs.setdefault('research_method', 'SERP')  # Options: SERP, RAG, reference
        inputs.setdefault('structure_type', 'How-to Guide')
        inputs.setdefault('primary_keyword', inputs['topic'].lower())
        inputs.setdefault('keywords', {
            'primary': inputs['topic'].lower(),
            'secondary': [f"how to {inputs['topic'].lower()}", f"{inputs['topic'].lower()} guide"]
        })
        
        # Add variables that tasks expect
        inputs.setdefault('research_findings', '')
        inputs.setdefault('content_structure', '')
        
        print(f"Starting workflow with research method: {inputs['research_method']}")
        
        try:
            # Create the content structure first
            content_structure = self.crew().kickoff(inputs=inputs)
            return content_structure
            
        except Exception as e:
            print(f"Error during workflow execution: {e}")
            raise
    
    def run_research_mode(self, mode: str, topic: str, **kwargs):
        """
        Run specific research modes independently
        
        Args:
            mode: 'serp', 'rag', or 'reference' 
            topic: Research topic
            **kwargs: Additional parameters for research mode
        """
        return self.research_mode_tool._run(mode, topic, **kwargs)
    
    def analyze_competitor_content(self, topic: str, competitor_urls=None, num_competitors=5):
        """
        Run competitor analysis independently
        
        Args:
            topic: Topic to analyze
            competitor_urls: Optional list of competitor URLs
            num_competitors: Number of competitors to analyze if URLs not provided
        """
        return self.competitor_analysis_tool._run(topic, competitor_urls, num_competitors)