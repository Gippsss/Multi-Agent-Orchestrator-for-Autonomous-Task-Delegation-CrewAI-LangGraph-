# Multi-Agent Orchestrator for Autonomous Task Delegation

## Overview
This is an AI workflow automation system where multiple specialized agents collaborate dynamically using CrewAI and LangGraph. The system features context retention via Chroma vector memory, improving reasoning and decision efficiency across agent interactions.

## Last Updated
November 6, 2025

## Project Architecture

### Core Components

1. **Vector Memory System** (`vector_memory.py`)
   - ChromaDB-based persistent storage
   - Semantic search capabilities
   - Agent-specific memory tracking
   - Context retention across sessions

2. **Multi-Agent System** (`agents.py`)
   - CrewAI-based orchestrator
   - 4 specialized agents:
     - **Researcher**: Information gathering and analysis
     - **Writer**: Content creation and documentation
     - **Analyst**: Data analysis and strategic recommendations
     - **Reviewer**: Quality assurance and validation
   - Sequential and hierarchical workflow processes
   - Memory-enhanced task execution

3. **LangGraph Workflow** (`workflow_graph.py`)
   - State-based graph orchestration
   - Dynamic agent routing
   - Coordinator node for workflow management
   - Automatic state transitions

4. **Streamlit Application** (`app.py`)
   - Interactive web dashboard
   - Multiple workflow execution interfaces
   - Real-time visualization
   - Memory management UI
   - Analytics and execution history

### Key Features

- **Dual Orchestration Systems**: Both CrewAI and LangGraph implementations
- **Vector Memory**: Persistent context retention using ChromaDB
- **Specialized Agents**: 4 distinct agents with unique capabilities
- **Workflow Visualization**: Interactive graph displays
- **Pre-built Examples**: Market research, content creation, strategic planning
- **Custom Workflow Builder**: User-defined task configurations
- **Memory Search**: Semantic search across stored contexts
- **Analytics Dashboard**: Execution history and agent usage statistics

### Technology Stack

- **Frontend**: Streamlit
- **Agent Framework**: CrewAI
- **Workflow Orchestration**: LangGraph
- **LLM Integration**: LangChain + OpenAI (gpt-5)
- **Vector Database**: ChromaDB
- **Visualization**: Plotly
- **Data Processing**: Pandas

### Dependencies

All required packages are installed via uv:
- crewai, crewai-tools
- langgraph, langchain, langchain-openai, langchain-community
- chromadb
- streamlit
- plotly, pandas, networkx

### Environment Configuration

**Required Secrets:**
- `OPENAI_API_KEY`: Powers all AI agents with gpt-5 model

**Server Configuration:**
- Port: 5000 (configured in `.streamlit/config.toml`)
- Address: 0.0.0.0
- Headless mode enabled

### Workflow

The application runs on port 5000 with the command:
```bash
streamlit run app.py --server.port 5000
```

### Usage

1. **Home Tab**: Overview and quick start guide
2. **CrewAI Workflow**: Execute pre-built or custom collaborative tasks
3. **LangGraph Workflow**: Dynamic state-based agent orchestration
4. **Agents Tab**: View agent capabilities and roles
5. **Memory Tab**: Search and manage vector memory
6. **Analytics Tab**: Review execution history and statistics

### Recent Changes

**November 6, 2025 - Phase 2 Enhancements**
- Added custom agent creation interface with role, goal, and backstory customization
- Implemented workflow template system with 5 reusable patterns (market research, content pipeline, strategic planning, data analysis, quick research)
- Created enhanced performance analytics with execution time tracking and agent-specific metrics
- Added export functionality for execution history (JSON/CSV), analytics reports, and memory data
- Implemented advanced memory retrieval with similarity scoring, agent filtering, and adjustable result counts
- Added comprehensive performance dashboard with visualizations
- Integrated download capabilities for all major data exports

**November 6, 2025 - Initial Build**
- Initial implementation of multi-agent orchestrator
- Created vector memory system with ChromaDB
- Implemented 4 specialized agents (researcher, writer, analyst, reviewer)
- Built LangGraph workflow with state management
- Developed comprehensive Streamlit dashboard
- Added pre-built workflow examples
- Integrated memory-enhanced context retention
- Created workflow visualization with Plotly
