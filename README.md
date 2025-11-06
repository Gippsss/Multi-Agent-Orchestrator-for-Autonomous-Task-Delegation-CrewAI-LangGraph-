# 🧠 Multi-Agent Orchestrator for Autonomous Task Delegation
### (CrewAI + LangGraph + Chroma Vector Memory)

## 🚀 Overview
This project implements an **AI-driven multi-agent orchestration system** where multiple intelligent agents collaborate dynamically to complete complex workflows autonomously.  
Built using **CrewAI** for agent coordination, **LangGraph** for workflow management, and **Chroma** for semantic memory, this system demonstrates how AI agents can plan, reason, and delegate tasks effectively — with persistent context retention.

---

## 🧩 Key Features
- **🧠 Multi-Agent Collaboration** — Agents (Researcher, Writer, Analyst, Reviewer, etc.) work together using CrewAI for structured task execution.
- **⚙️ Dynamic Workflow Management** — LangGraph handles the orchestration logic, agent routing, and dependency tracking.
- **💾 Context Retention via Chroma** — All agent interactions are vectorized and stored, enabling contextual reasoning and memory recall across sessions.
- **📊 Interactive Dashboard (Streamlit)** — Visualizes task flow, agent reasoning, and delegation chains in real time.
- **🪄 Extensible Architecture** — New agents or skills can be easily added to scale the workflow.
- **🔐 LLM-Agnostic Design** — Supports OpenAI, Groq, and OpenRouter APIs for flexible LLM integration.

---

## 🧠 Example Workflow
1. **Researcher Agent** gathers contextual data from a query.  
2. **Writer Agent** structures findings into coherent output.  
3. **Analyst Agent** evaluates consistency, adds analytics, and generates summaries.  
4. **Reviewer Agent** provides feedback and refinement.  
5. The system uses **LangGraph** to control task sequencing and **Chroma** to maintain vector-based semantic memory.

---

## 🏗️ Tech Stack
| Component | Technology |
|------------|-------------|
| Agent Framework | [CrewAI](https://github.com/joaomdmoura/crewAI) |
| Workflow Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Vector Memory | [ChromaDB](https://www.trychroma.com/) |
| Frontend / Dashboard | [Streamlit](https://streamlit.io/) |
| LLM Integration | OpenAI / Groq / OpenRouter |
| Language | Python 3.10+ |

---

## ⚙️ Installation
```bash
# Clone this repository
git clone https://github.com/<your-username>/AgentOrchestrator.git
cd AgentOrchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
