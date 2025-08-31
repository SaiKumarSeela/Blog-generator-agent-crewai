from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools.custom_tool import WebSearchTool, RAGTool, CompetitorAnalysisTool, SEOTool
from dotenv import load_dotenv
import os
from pathlib import Path
from .utils import load_yaml_config

load_dotenv()

# Set dummy OpenAI key if not present to avoid CrewAI validation error
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "12345466"

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
        self.llm_fast = LLM(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini/gemini-1.5-flash",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            max_retries=3,
            timeout=30
        )
        
        # Use flash model for both fast and smart to avoid pro model quota
        self.llm_smart = LLM(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini/gemini-1.5-flash",
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            max_retries=3,
            timeout=30
        )
        
        # Initialize tools
        self.web_search_tool = WebSearchTool()
        self.rag_tool = RAGTool()
        self.competitor_tool = CompetitorAnalysisTool()
        self.seo_tool = SEOTool()
    
    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['topic_researcher'],
            llm=self.llm_fast,
            tools=[self.web_search_tool]
        )
    
    @agent
    def knowledge_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['knowledge_retriever'],
            llm=self.llm_smart,
            tools=[self.rag_tool, self.web_search_tool]
        )
    
    @agent
    def competitor_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['competitor_analyst'],
            llm=self.llm_smart,
            tools=[self.competitor_tool, self.web_search_tool]
        )
    
    @agent
    def seo_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_strategist'],
            llm=self.llm_fast,
            tools=[self.seo_tool, self.web_search_tool]
        )
    
    @agent
    def content_structurer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_structurer'],
            llm=self.llm_smart,
            tools=[]
        )
    
    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['blog_writer'],
            llm=self.llm_smart,
            tools=[]
        )
    
    @task
    def generate_topic_suggestions(self) -> Task:
        return Task(
            config=self.tasks_config['generate_topic_suggestions'],
            agent=self.topic_researcher()
        )
    
    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config['research_topic'],
            agent=self.knowledge_retriever()
        )
    
    @task
    def analyze_competitors(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst()
        )
    
    @task
    def develop_keyword_strategy(self) -> Task:
        return Task(
            config=self.tasks_config['develop_keyword_strategy'],
            agent=self.seo_strategist(),
            context=[self.research_topic(), self.analyze_competitors()]
        )
    
    @task
    def generate_title_options(self) -> Task:
        return Task(
            config=self.tasks_config['generate_title_options'],
            agent=self.content_structurer(),
            context=[self.develop_keyword_strategy()]
        )
    
    @task
    def create_content_structure(self) -> Task:
        return Task(
            config=self.tasks_config['create_content_structure'],
            agent=self.content_structurer(),
            context=[self.research_topic(), self.develop_keyword_strategy()]
        )
    
    @task
    def generate_full_blog(self) -> Task:
        return Task(
            config=self.tasks_config['generate_full_blog'],
            agent=self.blog_writer(),
            context=[self.research_topic(), self.develop_keyword_strategy(), self.create_content_structure()],
            output_file='generated_blog.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create the Blog Generator crew"""
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
            # Explicitly set the LLM for the crew
            manager_llm=self.llm_smart,
            memory_config={
                'type': 'file',
                'path': self.memory_path,
                'embedder': {
                    'provider': 'huggingface',
                    'config': {
                        'model': 'sentence-transformers/all-mpnet-base-v2',
                        'device': 'cpu'
                    }
                }
            }
        )
    
    def run_workflow(self, inputs: dict) -> dict:
        """Execute the complete blog generation workflow"""
        result = self.crew().kickoff(inputs=inputs)
        return result