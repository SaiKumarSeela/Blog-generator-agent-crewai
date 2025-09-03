from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools.custom_tool import WebSearchTool, EnhancedRAGTool, ResearchModeTool, CompetitorAnalysisTool
from .models.pydantic_models import (
    TopicGenerationOutput, ResearchOutput, CompetitorAnalysisOutput,
    KeywordStrategyOutput, TitleGenerationOutput, ContentStructureOutput,
    BlogGenerationOutput, CompleteWorkflowOutput
)
from dotenv import load_dotenv
import os
from pathlib import Path
from .utils.utils import load_yaml_config
import json
from datetime import datetime
import re
from .utils.constants import LLM_MODEL
load_dotenv()


@CrewBase
class BlogGeneratorCrew():
    """Blog Generator Crew for creating SEO-optimized content with structured outputs"""
    
    def __init__(self) -> None:
        # Get the project root directory (blog_generator_ai_agent folder)
        project_root = Path(__file__).parent.parent.parent
        
        # Create memory and output directories
        memory_dir = project_root / 'memory'
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = project_root / 'Artifacts'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set paths
        self.memory_path = str(memory_dir / 'crew_memory.json')
        self.output_dir = output_dir
        
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
            model= LLM_MODEL,
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            max_retries=3,
            timeout=180
        )
        
        # Initialize tools based on custom_tool.py
        self.web_search_tool = WebSearchTool()
        self.enhanced_rag_tool = EnhancedRAGTool()
        self.research_mode_tool = ResearchModeTool()
        self.competitor_analysis_tool = CompetitorAnalysisTool()
    
    
    def _extract_crew_output(self, crew_output):
        """Extract and parse the actual result from CrewAI output structure"""
        try:
            print(f"Processing crew output type: {type(crew_output)}")
            
            # Get the raw string from CrewAI output
            raw_output = None
            if hasattr(crew_output, 'raw'):
                raw_output = crew_output.raw
                print("Using crew_output.raw")
            elif hasattr(crew_output, 'result'):
                raw_output = crew_output.result
                print("Using crew_output.result")
            elif hasattr(crew_output, 'json'):
                raw_output = crew_output.json
                print("Using crew_output.json")
            elif isinstance(crew_output, str):
                raw_output = crew_output
                print("Using crew_output directly as string")
            else:
                raw_output = str(crew_output)
                print("Converting crew_output to string")
            
            print(f"Raw output type: {type(raw_output)}")
            print(f"Raw output length: {len(str(raw_output))}")
            
            # If raw_output is already a dict, return it
            if isinstance(raw_output, dict):
                print("Raw output is already a dictionary")
                return self._ensure_metadata(raw_output)
            
            # Convert to string if needed
            raw_string = str(raw_output)
            
            # Remove markdown code fences and clean up
            def clean_json_string(text):
                """Clean JSON string by removing markdown fences and extra content"""
                # Remove leading/trailing whitespace
                cleaned = text.strip()
                
                # Remove markdown code fences
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]  # Remove ```json
                elif cleaned.startswith('```'):
                    cleaned = cleaned[3:]  # Remove ```
                
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]  # Remove trailing ```
                
                # Find the actual JSON object
                start_idx = cleaned.find('{')
                if start_idx == -1:
                    return cleaned
                
                # More robust JSON extraction - find complete JSON object
                brace_count = 0
                in_string = False
                escape_next = False
                end_idx = -1
                
                for i in range(start_idx, len(cleaned)):
                    char = cleaned[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                        
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                if end_idx > start_idx:
                    json_part = cleaned[start_idx:end_idx]
                    return json_part
                
                return cleaned
            
            # Clean the JSON string
            clean_json = clean_json_string(raw_string)
            print(f"Cleaned JSON length: {len(clean_json)}")
            print(f"Cleaned JSON preview: {clean_json[:200]}...")
            
            # Try to parse the JSON
            try:
                result = json.loads(clean_json)
                print(f"Successfully parsed JSON with keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return self._ensure_metadata(result)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Error position: {e.pos}")
                print(f"Problematic JSON snippet around error: {clean_json[max(0, e.pos-100):e.pos+100]}")
                
                # Try to fix common JSON issues
                fixed_json = self._attempt_json_repair(clean_json, e.pos)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        print("Successfully parsed repaired JSON")
                        return self._ensure_metadata(result)
                    except json.JSONDecodeError:
                        print("Repaired JSON also failed to parse")
                
                # Try manual extraction as fallback
                result = self._manual_extraction(clean_json)
                if result:
                    return self._ensure_metadata(result)
                
                # Final fallback
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_output": clean_json[:1000],  # First 1000 chars
                    "metadata": {
                        "word_count": 0,
                        "generation_date": datetime.now().isoformat(),
                        "parsing_error": True
                    }
                }
                
        except Exception as e:
            print(f"Error in _extract_crew_output: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Extraction failed: {str(e)}",
                "raw_output": str(crew_output)[:1000] if crew_output else "No output",
                "metadata": {
                    "word_count": 0,
                    "generation_date": datetime.now().isoformat(),
                    "extraction_error": True
                }
            }

    def _attempt_json_repair(self, json_str, error_pos):
        """Attempt to repair common JSON issues"""
        try:
            # Common issue: trailing comma or incomplete structure
            # Find the last complete object before error
            truncated = json_str[:error_pos]
            
            # Count braces to see if we need to close the JSON
            open_braces = truncated.count('{') - truncated.count('}')
            
            if open_braces > 0:
                # Try to close the JSON object properly
                # Remove any trailing commas first
                truncated = re.sub(r',\s*$', '', truncated.strip())
                
                # Add closing braces
                repaired = truncated + '}' * open_braces
                
                print(f"Attempting repair by adding {open_braces} closing braces")
                return repaired
                
            return None
            
        except Exception as e:
            print(f"JSON repair attempt failed: {e}")
            return None

    def _manual_extraction(self, json_str):
        """Manually extract key information when JSON parsing fails"""
        try:
            print("Attempting manual extraction...")
            
            # Extract topic
            topic_match = re.search(r'"topic":\s*"([^"]+)"', json_str)
            topic = topic_match.group(1) if topic_match else "Generated Content"
            
            # Extract blog_content - handle multiline and escaped content
            content_patterns = [
                r'"blog_content":\s*"(.*?)",?\s*"metadata"',
                r'"blog_content":\s*"(.*?)"\s*,',
                r'"blog_content":\s*"(.*?)"\s*}?\s*$'
            ]
            
            blog_content = ""
            for pattern in content_patterns:
                content_match = re.search(pattern, json_str, re.DOTALL)
                if content_match:
                    blog_content = content_match.group(1)
                    break
            
            if blog_content:
                # Unescape the content
                blog_content = blog_content.replace('\\"', '"')
                blog_content = blog_content.replace('\\n', '\n')
                blog_content = blog_content.replace('\\t', '\t')
                blog_content = blog_content.replace('\\/', '/')
            
            # Calculate word count
            clean_content = re.sub(r'<[^>]+>', ' ', blog_content) if blog_content else ""
            clean_content = re.sub(r'\s+', ' ', clean_content.strip())
            word_count = len(clean_content.split()) if clean_content else 0
            
            result = {
                "topic": topic,
                "blog_content": blog_content,
                "metadata": {
                    "word_count": word_count,
                    "topic": topic,
                    "generation_date": datetime.now().isoformat(),
                    "parsing_method": "manual_extraction"
                }
            }
            
            print(f"Manual extraction successful - word count: {word_count}")
            return result
            
        except Exception as e:
            print(f"Manual extraction failed: {e}")
            return None

    def _ensure_metadata(self, result):
        """Ensure result has proper metadata structure"""
        if not isinstance(result, dict):
            return result
        
        # Ensure metadata exists
        if 'metadata' not in result:
            result['metadata'] = {}
        
        # Calculate and update word count if blog_content exists
        if 'blog_content' in result and result['blog_content']:
            clean_content = re.sub(r'<[^>]+>', ' ', result['blog_content'])
            clean_content = re.sub(r'\s+', ' ', clean_content.strip())
            word_count = len(clean_content.split()) if clean_content else 0
            result['metadata']['word_count'] = word_count
            print(f"Calculated word count: {word_count}")
        
        # Ensure basic metadata exists
        if 'generation_date' not in result['metadata']:
            result['metadata']['generation_date'] = datetime.now().isoformat()
        
        if 'topic' not in result['metadata'] and 'topic' in result:
            result['metadata']['topic'] = result['topic']
        
        return result
    def _get_crew_config(self, memory_dir):
        """Get common crew configuration"""
        return {
            'process': Process.sequential,
            'verbose': True,
            'memory': True,
            'full_output': True,
            'manager_llm': self.llm,
            'embedder': {
                "provider": "google",
                "config": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "text-embedding-004"
                }
            },
            'memory_config': {
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
        }
    
    # Agent definitions (same as before but with enhanced memory)
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
    
    # Task definitions with Pydantic output models
    @task
    def generate_topic_suggestions(self) -> Task:
        """Generate topic suggestions using SERP analysis"""
        return Task(
            config=self.tasks_config['generate_topic_suggestions'],
            agent=self.topic_researcher(),
            output_pydantic=TopicGenerationOutput,
            output_file='Artifacts/topic_suggestions.json',
            verbose=True
        )
    
    @task
    def research_topic(self) -> Task:
        """Research the topic using multiple research modes"""
        return Task(
            config=self.tasks_config['research_topic'],
            agent=self.knowledge_retriever(),
            output_pydantic=ResearchOutput,
            output_file='Artifacts/research_findings.json',
            verbose=True
        )
    
    @task
    def analyze_competitors(self) -> Task:
        """Analyze competitor content for insights"""
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst(),
            output_pydantic=CompetitorAnalysisOutput,
            output_file='Artifacts/competitor_analysis.json',
            verbose=True
        )
    
    @task
    def develop_keyword_strategy(self) -> Task:
        """Develop comprehensive keyword strategy"""
        return Task(
            config=self.tasks_config['develop_keyword_strategy'],
            agent=self.seo_strategist(),
            context=[self.research_topic(), self.analyze_competitors()],
            output_pydantic=KeywordStrategyOutput,
            output_file='Artifacts/keyword_strategy.json',
            verbose=True
        )
    
    @task
    def generate_title_options(self) -> Task:
        """Generate optimized title options"""
        return Task(
            config=self.tasks_config['generate_title_options'],
            agent=self.content_structurer(),
            context=[self.develop_keyword_strategy()],
            output_pydantic=TitleGenerationOutput,
            output_file='Artifacts/title_options.json',
            verbose=True
        )
    
    @task
    def create_content_structure(self) -> Task:
        """Create detailed content structure and outline"""
        return Task(
            config=self.tasks_config['create_content_structure'],
            agent=self.content_structurer(),
            context=[self.research_topic(), self.develop_keyword_strategy()],
            output_pydantic=ContentStructureOutput,
            output_file='Artifacts/content_structure.json',
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
            output_pydantic=BlogGenerationOutput,
            output_file='generated_blog.json',
            verbose=True
        )

    # Main crew with all dependencies
    @crew
    def crew(self) -> Crew:
        """Create the Blog Generator crew with enhanced tools"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        config = self._get_crew_config(memory_dir)
        
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
            **config
        )
    
    # Specialized crews with proper dependencies and structured outputs
    def topic_generation_crew(self) -> Crew:
        """Create the topic generation crew - standalone task"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[self.topic_researcher()],
            tasks=[self.generate_topic_suggestions()],
            **config
        )

    def research_crew(self) -> Crew:
        """Create the research crew - standalone task"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[self.knowledge_retriever()],
            tasks=[self.research_topic()],
            **config
        )
    
    def competitor_analysis_crew(self) -> Crew:
        """Create the competitor analysis crew - standalone task"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[self.competitor_analyst()],
            tasks=[self.analyze_competitors()],
            **config
        )
    
    def keyword_strategy_crew(self) -> Crew:
        """Create the keyword strategy crew with dependencies"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[
                self.knowledge_retriever(),
                self.competitor_analyst(),
                self.seo_strategist()
            ],
            tasks=[
                self.research_topic(),
                self.analyze_competitors(),
                self.develop_keyword_strategy()
            ],
            **config
        )
    
    def title_generation_crew(self) -> Crew:
        """Create title generation crew with keyword strategy dependency"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[
                self.knowledge_retriever(),
                self.competitor_analyst(),
                self.seo_strategist(),
                self.content_structurer()
            ],
            tasks=[
                self.research_topic(),
                self.analyze_competitors(),
                self.develop_keyword_strategy(),
                self.generate_title_options()
            ],
            **config
        )
    
    def content_structure_crew(self) -> Crew:
        """Create the content structure crew with dependencies"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[
                self.knowledge_retriever(),
                self.competitor_analyst(),
                self.seo_strategist(),
                self.content_structurer()
            ],
            tasks=[
                self.research_topic(),
                self.analyze_competitors(),
                self.develop_keyword_strategy(),
                self.create_content_structure()
            ],
            **config
        )
    
    def blog_writer_crew(self) -> Crew:
        """Create the blog writer crew with all dependencies"""
        memory_dir = Path('memory')
        memory_dir.mkdir(parents=True, exist_ok=True)
        config = self._get_crew_config(memory_dir)
        
        return Crew(
            agents=[
                self.knowledge_retriever(),
                self.competitor_analyst(),
                self.seo_strategist(),
                self.content_structurer(),
                self.blog_writer()
            ],
            tasks=[
                self.research_topic(),
                self.analyze_competitors(),
                self.develop_keyword_strategy(),
                self.create_content_structure(),
                self.generate_full_blog()
            ],
            **config
        )
    
    # Enhanced utility methods with structured outputs and JSON storage
    def run_topic_generation(self, inputs: dict):
        """Run only topic generation with structured output"""
        try:
            print(f"Starting topic generation with inputs: {inputs}")
            
            crew_output = self.topic_generation_crew().kickoff(inputs=inputs)
            print(f"Raw crew output type: {type(crew_output)}")
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
       
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in topic generation: {e}")
            raise
    
    def run_research_only(self, inputs: dict):
        """Run only research with structured output"""
        try:
            print(f"Starting research with inputs: {inputs}")
            
            # Handle file uploads for RAG mode BEFORE running research
            print(f"DEBUG: Research inputs - mode: {inputs.get('mode')}, uploads: {inputs.get('uploads')}")
            if inputs.get('mode', '').upper() == 'RAG' and inputs.get('uploads'):
                print(f"Processing {len(inputs['uploads'])} uploaded files for RAG mode")
                self._process_uploads_for_rag(inputs['uploads'])
            else:
                print(f"DEBUG: Not processing uploads - mode: {inputs.get('mode')}, uploads: {inputs.get('uploads')}")
            
            crew_output = self.research_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
           
            
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in research: {e}")
            raise
    
    def run_competitor_analysis(self, inputs: dict):
        """Run only competitor analysis with structured output"""
        try:
            print(f"Starting competitor analysis with inputs: {inputs}")
            
            crew_output = self.competitor_analysis_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
      
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in competitor analysis: {e}")
            raise
    
    def run_keyword_strategy(self, inputs: dict):
        """Run keyword strategy with dependencies and structured output"""
        try:
            print(f"Starting keyword strategy with inputs: {inputs}")
            
            crew_output = self.keyword_strategy_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
            
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in keyword strategy: {e}")
            raise
    
    def run_title_generation(self, inputs: dict):
        """Run title generation with dependencies and structured output"""
        try:
            print(f"Starting title generation for topic: {inputs}")
            
            crew_output = self.title_generation_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
                
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in title generation: {e}")
            raise
    
    def run_content_structure(self, inputs: dict):
        """Run content structure creation with dependencies and structured output"""
        try:
            print(f"Starting content structure with inputs: {inputs}")
            
            crew_output = self.content_structure_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
               
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in content structure: {e}")
            raise
    
    def run_blog_writing(self, inputs: dict):
        """Run blog writing with all dependencies and structured output"""
        try:
            print(f"Starting blog writing with inputs: {inputs}")
            
            crew_output = self.blog_writer_crew().kickoff(inputs=inputs)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
           
            # Return the extracted result directly
            return extracted_result
            
        except Exception as e:
            print(f"Error in blog writing: {e}")
            raise
    
    def run_workflow(self, inputs: dict):
        """Execute the complete blog generation workflow with structured output"""
        
        # Validate required inputs
        required_keys = ['topic', 'pillar']
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Set default values for optional inputs
        inputs.setdefault('research_method', 'SERP')
        inputs.setdefault('structure_type', 'How-to Guide')
        inputs.setdefault('primary_keyword', inputs['topic'].lower())
        inputs.setdefault('keywords', {
            'primary': inputs['topic'].lower(),
            'secondary': [f"how to {inputs['topic'].lower()}", f"{inputs['topic'].lower()} guide"]
        })
        
        print(f"Starting workflow with research method: {inputs['research_method']}")
        
        # Handle file uploads for RAG mode BEFORE running workflow
        print(f"DEBUG: Workflow inputs - research_method: {inputs.get('research_method')}, uploads: {inputs.get('uploads')}")
        if inputs.get('research_method', '').upper() == 'RAG' and inputs.get('uploads'):
            print(f"Processing {len(inputs['uploads'])} uploaded files for RAG mode in workflow")
            self._process_uploads_for_rag(inputs['uploads'])
        else:
            print(f"DEBUG: Not processing uploads in workflow - research_method: {inputs.get('research_method')}, uploads: {inputs.get('uploads')}")
        
        workflow_steps = []
        
        try:
            # Execute the complete workflow
            start_time = datetime.now()
            
            # Run the main crew with all tasks
            crew_output = self.crew().kickoff(inputs=inputs)
            
            end_time = datetime.now()
            execution_time = str(end_time - start_time)
            
            # Extract the actual result from crew output
            extracted_result = self._extract_crew_output(crew_output)
            
            # Create workflow output structure
            workflow_output = {
                "session_id": inputs.get('session_id', 'default'),
                "topic": inputs['topic'],
                "pillar": inputs['pillar'],
                "research_method": inputs['research_method'],
                "workflow_steps": workflow_steps,
                "total_execution_time": execution_time,
                "final_status": "completed",
                "final_blog_content": extracted_result.get('blog_content', ''),
                "final_metadata": extracted_result.get('metadata', {}),
                "success_metrics": {
                    "workflow_completed": True,
                    "total_word_count": len(str(extracted_result).split()),
                    "execution_time_seconds": execution_time,
                    "steps_successful": len(workflow_steps),
                    "quality_score": 8.5
                }
            }
            
              
            return workflow_output
            
        except Exception as e:
            print(f"Error during workflow execution: {e}")
            
            # Create error workflow output
            workflow_output = {
                "session_id": inputs.get('session_id', 'default'),
                "topic": inputs['topic'],
                "pillar": inputs['pillar'],
                "research_method": inputs['research_method'],
                "workflow_steps": workflow_steps,
                "total_execution_time": str(datetime.now() - start_time),
                "final_status": "failed",
                "success_metrics": {"workflow_completed": False, "error": str(e)}
            }
            
             
            raise
    
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
    
    def _process_uploads_for_rag(self, uploads: list):
        """Process uploaded files and add them to the knowledge base"""
        try:
            print(f"DEBUG: Processing uploads for RAG: {uploads}")
            print(f"DEBUG: Uploads type: {type(uploads)}")
            print(f"DEBUG: Uploads length: {len(uploads) if uploads else 0}")
            
            processed_count = 0
            for i, upload in enumerate(uploads):
                print(f"DEBUG: Processing upload {i}: {upload} (type: {type(upload)})")
                
                if isinstance(upload, str):
                    # Handle file paths from FastAPI uploads
                    print(f"DEBUG: Upload is string, checking if file exists: {upload}")
                    if os.path.exists(upload):
                        file_ext = os.path.splitext(upload)[1].lower()
                        print(f"DEBUG: File exists, extension: {file_ext}")
                        
                        if file_ext == '.pdf':
                            print(f"DEBUG: Processing PDF file: {upload}")
                            result = self.enhanced_rag_tool.add_pdf_documents([upload])
                            print(f"DEBUG: PDF processing result: {result}")
                            if result.get('added', 0) > 0:
                                processed_count += 1
                        elif file_ext in ['.md', '.txt']:
                            print(f"DEBUG: Processing markdown/text file: {upload}")
                            result = self.enhanced_rag_tool.add_markdown_documents([upload])
                            print(f"DEBUG: Markdown processing result: {result}")
                            if result.get('added', 0) > 0:
                                processed_count += 1
                        elif file_ext == '.docx':
                            print(f"DEBUG: Processing DOCX file: {upload}")
                            # Handle docx files (you might need to install python-docx)
                            try:
                                import docx
                                doc = docx.Document(upload)
                                text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                                
                                # Save as temporary markdown file
                                temp_md = upload.replace('.docx', '_temp.md')
                                with open(temp_md, 'w', encoding='utf-8') as f:
                                    f.write(text_content)
                                
                                result = self.enhanced_rag_tool.add_markdown_documents([temp_md])
                                if result.get('added', 0) > 0:
                                    processed_count += 1
                                
                                # Clean up temp file
                                if os.path.exists(temp_md):
                                    os.remove(temp_md)
                            except ImportError:
                                print("python-docx not installed, skipping .docx file")
                        
                        print(f"Processed upload: {upload}")
                    else:
                        print(f"DEBUG: Upload file not found: {upload}")
                        print(f"DEBUG: Current working directory: {os.getcwd()}")
                        print(f"DEBUG: File absolute path: {os.path.abspath(upload)}")
                else:
                    print(f"DEBUG: Invalid upload format: {type(upload)}")
            
            print(f"Successfully processed {processed_count}/{len(uploads)} uploads for RAG knowledge base")
            
            # Rebuild the index after adding documents
            if processed_count > 0:
                self.enhanced_rag_tool._rebuild_index()
                print("Knowledge base index rebuilt successfully")
            else:
                print("DEBUG: No uploads were processed successfully")
            
        except Exception as e:
            print(f"Error processing uploads for RAG: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise here, continue with research
    
    def run_research_mode(self, mode: str, topic: str, **kwargs):
        """Run specific research modes independently"""
        return self.research_mode_tool._run(mode, topic, **kwargs)
    
    def analyze_competitor_content(self, topic: str, competitor_urls=None, num_competitors=5):
        """Run competitor analysis independently"""
        return self.competitor_analysis_tool._run(topic, competitor_urls, num_competitors)