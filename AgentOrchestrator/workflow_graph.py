import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class AgentState(TypedDict):
    """State object for the multi-agent workflow graph."""
    messages: Annotated[List[Any], operator.add]
    current_agent: str
    task_description: str
    research_output: str
    analysis_output: str
    writing_output: str
    review_output: str
    final_output: str
    next_step: str
    iterations: int
    max_iterations: int

class LangGraphWorkflow:
    """
    LangGraph-based workflow orchestration with state management.
    Provides dynamic routing and state persistence for multi-agent systems.
    """
    
    def __init__(self, max_iterations: int = 5):
        """Initialize the LangGraph workflow."""
        self.max_iterations = max_iterations
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-5",
            api_key=OPENAI_API_KEY
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
        self.execution_log = []
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("coordinator", self._coordinator_node)
        
        # Define the workflow edges
        workflow.set_entry_point("coordinator")
        
        # Coordinator decides the next step
        workflow.add_conditional_edges(
            "coordinator",
            self._route_next_step,
            {
                "research": "researcher",
                "analyze": "analyst",
                "write": "writer",
                "review": "reviewer",
                "end": END
            }
        )
        
        # After each agent, go back to coordinator
        workflow.add_edge("researcher", "coordinator")
        workflow.add_edge("analyst", "coordinator")
        workflow.add_edge("writer", "coordinator")
        workflow.add_edge("reviewer", "coordinator")
        
        return workflow.compile()
    
    def _researcher_node(self, state: AgentState) -> AgentState:
        """Research agent node - gathers information and conducts research."""
        messages = [
            SystemMessage(content="""You are a Senior Research Analyst. Your role is to:
            1. Conduct thorough research on the given topic
            2. Gather comprehensive information
            3. Identify key facts, trends, and insights
            4. Present your findings in a structured format"""),
            HumanMessage(content=f"Research Task: {state['task_description']}")
        ]
        
        response = self.llm.invoke(messages)
        
        state['research_output'] = response.content
        state['current_agent'] = 'researcher'
        state['messages'].append(AIMessage(content=f"[RESEARCHER]: {response.content[:200]}..."))
        state['iterations'] += 1
        
        self._log_execution(state, 'researcher', response.content)
        
        return state
    
    def _analyst_node(self, state: AgentState) -> AgentState:
        """Analyst agent node - analyzes data and provides insights."""
        context = state.get('research_output', 'No research available')
        
        messages = [
            SystemMessage(content="""You are a Strategic Business Analyst. Your role is to:
            1. Analyze the research findings
            2. Identify patterns and trends
            3. Evaluate options and alternatives
            4. Provide strategic recommendations"""),
            HumanMessage(content=f"Task: {state['task_description']}\n\nResearch Context:\n{context}")
        ]
        
        response = self.llm.invoke(messages)
        
        state['analysis_output'] = response.content
        state['current_agent'] = 'analyst'
        state['messages'].append(AIMessage(content=f"[ANALYST]: {response.content[:200]}..."))
        state['iterations'] += 1
        
        self._log_execution(state, 'analyst', response.content)
        
        return state
    
    def _writer_node(self, state: AgentState) -> AgentState:
        """Writer agent node - creates compelling content."""
        research = state.get('research_output', '')
        analysis = state.get('analysis_output', '')
        
        context = f"Research:\n{research}\n\nAnalysis:\n{analysis}"
        
        messages = [
            SystemMessage(content="""You are a Professional Content Writer. Your role is to:
            1. Create well-structured, engaging content
            2. Present complex information clearly
            3. Tailor content to the intended audience
            4. Ensure grammatical correctness and flow"""),
            HumanMessage(content=f"Writing Task: {state['task_description']}\n\nContext:\n{context}")
        ]
        
        response = self.llm.invoke(messages)
        
        state['writing_output'] = response.content
        state['current_agent'] = 'writer'
        state['messages'].append(AIMessage(content=f"[WRITER]: {response.content[:200]}..."))
        state['iterations'] += 1
        
        self._log_execution(state, 'writer', response.content)
        
        return state
    
    def _reviewer_node(self, state: AgentState) -> AgentState:
        """Reviewer agent node - reviews and validates output."""
        writing = state.get('writing_output', 'No content to review')
        
        messages = [
            SystemMessage(content="""You are a Quality Assurance Reviewer. Your role is to:
            1. Review content for quality and accuracy
            2. Identify gaps or areas for improvement
            3. Ensure completeness and coherence
            4. Provide constructive feedback or approve"""),
            HumanMessage(content=f"Review Task: {state['task_description']}\n\nContent to Review:\n{writing}")
        ]
        
        response = self.llm.invoke(messages)
        
        state['review_output'] = response.content
        state['final_output'] = state['writing_output']
        state['current_agent'] = 'reviewer'
        state['messages'].append(AIMessage(content=f"[REVIEWER]: {response.content[:200]}..."))
        state['iterations'] += 1
        
        self._log_execution(state, 'reviewer', response.content)
        
        return state
    
    def _coordinator_node(self, state: AgentState) -> AgentState:
        """Coordinator node - determines next step in workflow."""
        current_agent = state.get('current_agent', 'none')
        iterations = state.get('iterations', 0)
        
        # Determine next step based on current state
        if iterations >= state['max_iterations']:
            state['next_step'] = 'end'
        elif not state.get('research_output'):
            state['next_step'] = 'research'
        elif not state.get('analysis_output'):
            state['next_step'] = 'analyze'
        elif not state.get('writing_output'):
            state['next_step'] = 'write'
        elif not state.get('review_output'):
            state['next_step'] = 'review'
        else:
            state['next_step'] = 'end'
        
        return state
    
    def _route_next_step(self, state: AgentState) -> str:
        """Route to the next step based on coordinator decision."""
        return state.get('next_step', 'end')
    
    def _log_execution(self, state: AgentState, agent: str, output: str):
        """Log execution details for tracking."""
        self.execution_log.append({
            'agent': agent,
            'iteration': state['iterations'],
            'output_preview': output[:150] + '...' if len(output) > 150 else output
        })
    
    def execute(self, task_description: str) -> Dict[str, Any]:
        """
        Execute the workflow for a given task.
        
        Args:
            task_description: Description of the task to execute
            
        Returns:
            Execution results with outputs from all agents
        """
        # Initialize state
        initial_state = AgentState(
            messages=[],
            current_agent="",
            task_description=task_description,
            research_output="",
            analysis_output="",
            writing_output="",
            review_output="",
            final_output="",
            next_step="research",
            iterations=0,
            max_iterations=self.max_iterations
        )
        
        # Clear execution log
        self.execution_log = []
        
        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)
            
            return {
                'success': True,
                'task': task_description,
                'research': final_state.get('research_output', ''),
                'analysis': final_state.get('analysis_output', ''),
                'writing': final_state.get('writing_output', ''),
                'review': final_state.get('review_output', ''),
                'final_output': final_state.get('final_output', ''),
                'iterations': final_state.get('iterations', 0),
                'execution_log': self.execution_log
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'task': task_description,
                'execution_log': self.execution_log
            }
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get workflow structure for visualization."""
        return {
            'nodes': [
                {'id': 'coordinator', 'label': 'Coordinator', 'type': 'decision'},
                {'id': 'researcher', 'label': 'Researcher', 'type': 'agent'},
                {'id': 'analyst', 'label': 'Analyst', 'type': 'agent'},
                {'id': 'writer', 'label': 'Writer', 'type': 'agent'},
                {'id': 'reviewer', 'label': 'Reviewer', 'type': 'agent'}
            ],
            'edges': [
                {'from': 'coordinator', 'to': 'researcher', 'label': 'research'},
                {'from': 'coordinator', 'to': 'analyst', 'label': 'analyze'},
                {'from': 'coordinator', 'to': 'writer', 'label': 'write'},
                {'from': 'coordinator', 'to': 'reviewer', 'label': 'review'},
                {'from': 'researcher', 'to': 'coordinator', 'label': 'complete'},
                {'from': 'analyst', 'to': 'coordinator', 'label': 'complete'},
                {'from': 'writer', 'to': 'coordinator', 'label': 'complete'},
                {'from': 'reviewer', 'to': 'coordinator', 'label': 'complete'}
            ]
        }
