# Blog Generator AI Agent

Agentic RAG-enabled system for research, SEO planning, content structuring, and blog generation using crewAI, FastAPI, and Google Gemini.

### Features
- **API-first** FastAPI service with structured outputs for every step (topics, research, competitors, keywords, structure, blog, workflow)
- **CLI** runners for demos and interactive generation
- **Observability** via Azure Application Insights (OpenTelemetry)

### Requirements
- Python >=3.10,<3.14
- Google Gemini API access
- SERP API key (for web search) optional but recommended

### Environment variables
Create a `.env` file in the project root with:

```bash
GEMINI_API_KEY=your_gemini_key
SERP_API_KEY=your_serp_key
FIRECRAWL_API_KEY=optional_firecrawl_key
APPLICATIONINSIGHTS_CONNECTION_STRING=your_azure_appinsights_connection_string
```

Note: The FastAPI app requires `APPLICATIONINSIGHTS_CONNECTION_STRING`. If absent, it will raise an error at startup. If you don't have that key, you can feel free to comment the telemetry code in `fastapi_main.py` file.

### Install and run locally
You can use either `uv` or plain `pip`. You can check this [documentation](https://docs.crewai.com/en/installation) as well.

- Using uv (recommended):
```bash
pip install uv
uv pip install -e . # or
pip install -r requirements.txt

```

- Using pip:
```bash
pip install -e .
```
- install Crewai
```bash
uv tool install crewai
```

### CLI usage (optional)
Entry points are defined in `pyproject.toml` and `src/blog_generator_ai_agent/main.py`:

```bash
# Run default demo inputs
python -m src.blog_generator_ai_agent.main

# Interactive mode
python -m src.blog_generator_ai_agent.main --interactive

# Research demo
python -m src.blog_generator_ai_agent.main --research-demo

# Custom topic
python -m src.blog_generator_ai_agent.main "Your Topic Here"
```
or you can also directly run crew by following this below commands in the root directory

```bash
crewai install 
uv add <package-name> # to install any additional packages
crewai run 
```

### Run the FastAPI service
Default host/port are defined in `src/blog_generator_ai_agent/utils/constants.py` (`0.0.0.0:8085`).

```bash
uvicorn fastapi_main:app --host 0.0.0.0 --port 8085
```

Health check: `GET /health` → {"status":"healthy"}

Key endpoints (JSON request bodies are defined in `src/blog_generator_ai_agent/api/models.py`):
- `POST /topic/generate`
- `POST /research/run`
- `POST /competitors/analyse`
- `POST /seo/keywords`
- `POST /titles/generate`
- `POST /structure/select`
- `POST /outline/create`
- `POST /blog/generate`
- `POST /workflow/run`


### Streamlit app (frontend)
There is a simple UI in `streamlit_app.py` if you want to experiment:

```bash
streamlit run streamlit_app.py
```

### Docker
The repository includes a production-ready `Dockerfile` that builds a slim Python 3.12 image and serves the FastAPI app with Uvicorn.

Build:
```bash
docker build --no-cache -t blog-generator-ai-agent:latest .
```

Run:
```bash
docker run --rm -p 8085:8085 \
  -e GEMINI_API_KEY=your_gemini_key \
  -e SERP_API_KEY=your_serp_key \
  -e FIRECRAWL_API_KEY=your_firecrawl_key \
  -e APPLICATIONINSIGHTS_CONNECTION_STRING=your_azure_appinsights_connection_string \
  blog-generator-ai-agent:latest
```

Health check will hit `http://localhost:8085/health` automatically.

### Architecture
![architecture](https://github.com/user-attachments/assets/85a90711-8080-4169-b4d5-1f5d8677ffb6)

## Test Cases
To run the test cases, use the following command
```bash
pytest -q
```
### Project structure
- `fastapi_main.py`: FastAPI service entrypoint
- `src/blog_generator_ai_agent/crew.py`: crewAI orchestration
- `src/blog_generator_ai_agent/main.py`: CLI flows and demos
- `src/blog_generator_ai_agent/tools/`: Custom tools
- `src/blog_generator_ai_agent/api/`: Pydantic models and exception handling
- `src/blog_generator_ai_agent/utils/`: constants and telemetry setup
- `Artifacts/`: generated JSON/MD/HTML outputs

