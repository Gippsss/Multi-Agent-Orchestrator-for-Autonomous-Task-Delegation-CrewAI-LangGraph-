import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import json

from agents import MultiAgentOrchestrator
from workflow_graph import LangGraphWorkflow
from vector_memory import VectorMemory

st.set_page_config(
    page_title="Multi-Agent Orchestrator",
    page_icon="🤖",
    layout="wide"
)

def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please configure OPENAI_API_KEY in your environment secrets.")
        st.info("This system requires an OpenAI API key to function. The key will be used to power the AI agents.")
        st.stop()
    return True

def create_workflow_graph_viz(workflow_data):
    """Create a network graph visualization of the workflow."""
    nodes = workflow_data['nodes']
    edges = workflow_data['edges']
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    positions = {
        'coordinator': (0.5, 0.9),
        'researcher': (0.2, 0.5),
        'analyst': (0.4, 0.5),
        'writer': (0.6, 0.5),
        'reviewer': (0.8, 0.5)
    }
    
    color_map = {
        'decision': '#FF6B6B',
        'agent': '#4ECDC4'
    }
    
    for node in nodes:
        x, y = positions.get(node['id'], (0.5, 0.5))
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['label'])
        node_color.append(color_map.get(node['type'], '#95E1D3'))
    
    edge_x = []
    edge_y = []
    
    for edge in edges:
        from_node = next(n for n in nodes if n['id'] == edge['from'])
        to_node = next(n for n in nodes if n['id'] == edge['to'])
        
        from_pos = positions.get(edge['from'], (0.5, 0.5))
        to_pos = positions.get(edge['to'], (0.5, 0.5))
        
        edge_x.extend([from_pos[0], to_pos[0], None])
        edge_y.extend([from_pos[1], to_pos[1], None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            size=40,
            color=node_color,
            line_width=2,
            line_color='white'
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)',
                       height=400
                   ))
    
    return fig

def main():
    st.title("🤖 Multi-Agent Orchestrator")
    st.markdown("### Autonomous Task Delegation with CrewAI + LangGraph")
    
    check_api_key()
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = MultiAgentOrchestrator(memory_enabled=True)
    
    if 'langgraph_workflow' not in st.session_state:
        st.session_state.langgraph_workflow = LangGraphWorkflow()
    
    tabs = st.tabs([
        "🏠 Home",
        "🎯 CrewAI Workflow",
        "🔄 LangGraph Workflow",
        "👥 Agents",
        "💾 Memory",
        "📊 Analytics"
    ])
    
    with tabs[0]:
        st.header("Welcome to the Multi-Agent Orchestrator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            
            **Multi-Agent Collaboration**
            - 4 specialized agents (Researcher, Writer, Analyst, Reviewer)
            - Autonomous task delegation
            - Dynamic workflow orchestration
            
            **Vector Memory System**
            - Persistent context retention with ChromaDB
            - Semantic search for relevant information
            - Agent memory tracking
            
            **Dual Orchestration**
            - **CrewAI**: Sequential & hierarchical workflows
            - **LangGraph**: State-based graph workflows
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Quick Start
            
            1. **Explore Agents**: View agent capabilities and roles
            2. **Run CrewAI Workflow**: Execute multi-step collaborative tasks
            3. **Run LangGraph Workflow**: Dynamic state-based orchestration
            4. **Check Memory**: View stored context and insights
            5. **Analyze Performance**: Review execution history
            
            ### 📋 Sample Use Cases
            - Market research and analysis
            - Content creation pipelines
            - Strategic planning
            - Quality assurance workflows
            """)
        
        st.info("💡 **Tip**: Start with the 'CrewAI Workflow' or 'LangGraph Workflow' tabs to see the agents in action!")
    
    with tabs[1]:
        st.header("🎯 CrewAI Workflow Execution")
        st.markdown("Execute collaborative tasks using CrewAI's sequential or hierarchical processes.")
        
        workflow_type = st.radio(
            "Select Workflow Type",
            ["Workflow Templates", "Pre-built Example", "Custom Workflow"],
            horizontal=True
        )
        
        if workflow_type == "Workflow Templates":
            st.markdown("### 📋 Workflow Templates")
            st.info("Quick-start templates for common multi-agent patterns. Just enter your topic!")
            
            templates = st.session_state.orchestrator.get_workflow_templates()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                template_options = {v['name']: k for k, v in templates.items()}
                selected_template_name = st.selectbox(
                    "Choose Template",
                    list(template_options.keys())
                )
                template_id = template_options[selected_template_name]
                template = templates[template_id]
                
                st.markdown(f"**Description:** {template['description']}")
                st.markdown(f"**Process Type:** {template['process_type'].title()}")
                st.markdown(f"**Number of Tasks:** {len(template['tasks'])}")
                
                with st.expander("📝 View Template Tasks"):
                    for i, task in enumerate(template['tasks'], 1):
                        st.markdown(f"**{i}. {task['agent'].title()}**")
                        st.text(task['description'])
                        st.caption(f"Expected: {task['expected_output']}")
                        st.markdown("---")
            
            with col2:
                topic = st.text_input(
                    "Topic",
                    placeholder="e.g., Electric vehicles",
                    help="Enter the topic to apply this template to"
                )
                
                process_override = st.selectbox(
                    "Process Type",
                    ["Default", "sequential", "hierarchical"],
                    help="Override default process type"
                )
            
            if st.button("🚀 Execute Template", type="primary", use_container_width=True):
                if not topic:
                    st.warning("Please enter a topic")
                else:
                    with st.spinner(f"Executing {selected_template_name} workflow..."):
                        process_type = None if process_override == "Default" else process_override
                        result = st.session_state.orchestrator.execute_template(
                            template_id=template_id,
                            topic=topic,
                            process_type=process_type
                        )
                        
                        if result.get('success') == False:
                            st.error(f"Error: {result.get('error')}")
                        else:
                            st.success("✅ Workflow completed!")
                            
                            st.markdown("### Execution Results")
                            
                            for task_detail in result.get('tasks', []):
                                status_icon = "✅" if task_detail['status'] == 'completed' else "⏳"
                                st.markdown(f"{status_icon} **Task {task_detail['task_id']}** ({task_detail['agent']}): {task_detail['description']}")
                            
                            if 'result' in result:
                                with st.expander("📄 Final Output", expanded=True):
                                    st.write(result['result'])
        
        elif workflow_type == "Pre-built Example":
            example = st.selectbox(
                "Choose Example",
                [
                    "Market Research & Analysis",
                    "Content Creation Pipeline",
                    "Strategic Planning"
                ]
            )
            
            example_configs = {
                "Market Research & Analysis": [
                    {
                        'description': 'Research current trends in artificial intelligence and machine learning, focusing on practical applications in business',
                        'agent': 'researcher',
                        'expected_output': 'Comprehensive research report on AI/ML trends'
                    },
                    {
                        'description': 'Analyze the research findings and identify top 3 opportunities for businesses',
                        'agent': 'analyst',
                        'expected_output': 'Strategic analysis with business opportunities'
                    },
                    {
                        'description': 'Write an executive summary based on the research and analysis',
                        'agent': 'writer',
                        'expected_output': 'Clear, professional executive summary'
                    },
                    {
                        'description': 'Review the executive summary for quality and completeness',
                        'agent': 'reviewer',
                        'expected_output': 'Quality review with approval or feedback'
                    }
                ],
                "Content Creation Pipeline": [
                    {
                        'description': 'Research best practices for remote work productivity and team collaboration',
                        'agent': 'researcher',
                        'expected_output': 'Research findings on remote work best practices'
                    },
                    {
                        'description': 'Create an engaging blog post about remote work productivity tips',
                        'agent': 'writer',
                        'expected_output': 'Well-written blog post (800-1000 words)'
                    },
                    {
                        'description': 'Review the blog post for readability, accuracy, and engagement',
                        'agent': 'reviewer',
                        'expected_output': 'Review with improvements or approval'
                    }
                ],
                "Strategic Planning": [
                    {
                        'description': 'Research emerging technologies that could disrupt the retail industry',
                        'agent': 'researcher',
                        'expected_output': 'Technology trend research for retail'
                    },
                    {
                        'description': 'Analyze the impact and feasibility of these technologies for retail businesses',
                        'agent': 'analyst',
                        'expected_output': 'Impact analysis with recommendations'
                    },
                    {
                        'description': 'Write a strategic plan outline for digital transformation in retail',
                        'agent': 'writer',
                        'expected_output': 'Strategic plan document'
                    }
                ]
            }
            
            tasks_config = example_configs[example]
            
            st.markdown("**Workflow Steps:**")
            for i, task in enumerate(tasks_config, 1):
                st.markdown(f"{i}. **{task['agent'].title()}**: {task['description'][:80]}...")
        
        else:
            st.markdown("**Build Your Custom Workflow**")
            num_tasks = st.number_input("Number of tasks", min_value=1, max_value=5, value=2)
            
            tasks_config = []
            for i in range(num_tasks):
                with st.expander(f"Task {i+1}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        description = st.text_area(
                            f"Task Description {i+1}",
                            key=f"task_desc_{i}",
                            placeholder="Describe what this agent should do..."
                        )
                    with col2:
                        agent = st.selectbox(
                            f"Assign Agent {i+1}",
                            ['researcher', 'writer', 'analyst', 'reviewer'],
                            key=f"agent_{i}"
                        )
                    
                    if description:
                        tasks_config.append({
                            'description': description,
                            'agent': agent,
                            'expected_output': f'Output for task {i+1}'
                        })
        
        col1, col2 = st.columns([1, 4])
        with col1:
            process_type = st.selectbox("Process", ["sequential", "hierarchical"])
        
        if st.button("🚀 Execute Workflow", type="primary", use_container_width=True):
            if tasks_config:
                with st.spinner("Executing workflow... This may take a minute."):
                    result = st.session_state.orchestrator.execute_workflow(
                        tasks_config=tasks_config,
                        process_type=process_type
                    )
                    
                    st.success("✅ Workflow completed!")
                    
                    st.markdown("### Execution Results")
                    
                    for task_detail in result.get('tasks', []):
                        status_icon = "✅" if task_detail['status'] == 'completed' else "⏳"
                        st.markdown(f"{status_icon} **Task {task_detail['task_id']}** ({task_detail['agent']}): {task_detail['description']}")
                    
                    if 'result' in result:
                        with st.expander("📄 Final Output", expanded=True):
                            st.write(result['result'])
                    
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
            else:
                st.warning("Please configure at least one task.")
    
    with tabs[2]:
        st.header("🔄 LangGraph Workflow Execution")
        st.markdown("Dynamic state-based workflow with automatic agent routing.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            viz_data = st.session_state.langgraph_workflow.get_workflow_visualization()
            fig = create_workflow_graph_viz(viz_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            **Workflow Flow:**
            
            1. 🎯 Coordinator evaluates state
            2. 📊 Routes to appropriate agent
            3. 🔄 Agent processes task
            4. ↩️ Returns to coordinator
            5. 🔁 Repeats until complete
            """)
        
        st.markdown("---")
        
        task_description = st.text_area(
            "Enter Task Description",
            placeholder="Example: Analyze the impact of AI on healthcare and create a summary report...",
            height=100
        )
        
        if st.button("🚀 Execute LangGraph Workflow", type="primary", use_container_width=True):
            if task_description:
                with st.spinner("Executing LangGraph workflow..."):
                    result = st.session_state.langgraph_workflow.execute(task_description)
                    
                    if result.get('success'):
                        st.success(f"✅ Workflow completed in {result['iterations']} iterations")
                        
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📊 Research",
                            "📈 Analysis",
                            "✍️ Writing",
                            "✅ Review",
                            "📋 Execution Log"
                        ])
                        
                        with tab1:
                            st.markdown("### Research Output")
                            st.write(result.get('research', 'No research output'))
                        
                        with tab2:
                            st.markdown("### Analysis Output")
                            st.write(result.get('analysis', 'No analysis output'))
                        
                        with tab3:
                            st.markdown("### Writing Output")
                            st.write(result.get('writing', 'No writing output'))
                        
                        with tab4:
                            st.markdown("### Review Output")
                            st.write(result.get('review', 'No review output'))
                        
                        with tab5:
                            st.markdown("### Execution Log")
                            for log_entry in result.get('execution_log', []):
                                st.markdown(f"**Iteration {log_entry['iteration']}** - {log_entry['agent'].title()}")
                                st.text(log_entry['output_preview'])
                                st.markdown("---")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
            else:
                st.warning("Please enter a task description.")
    
    with tabs[3]:
        st.header("👥 Agent Directory")
        
        agent_tab1, agent_tab2 = st.tabs(["📋 All Agents", "➕ Create Custom Agent"])
        
        with agent_tab1:
            agents = st.session_state.orchestrator.get_agent_info()
            custom_agents = st.session_state.orchestrator.list_custom_agents()
            
            st.markdown(f"**Total Agents:** {len(agents)} (Default: 4, Custom: {len(custom_agents)})")
            st.markdown("---")
            
            cols = st.columns(2)
            for idx, agent in enumerate(agents):
                with cols[idx % 2]:
                    with st.container():
                        is_custom = agent['name'] in custom_agents
                        badge = "🔧 Custom" if is_custom else "⭐ Default"
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {agent['name'].title()} 🤖")
                        with col2:
                            st.caption(badge)
                        
                        st.markdown(f"**Role:** {agent['role']}")
                        st.markdown(f"**Goal:** {agent['goal']}")
                        with st.expander("View Backstory"):
                            st.write(agent['backstory'])
                        
                        if is_custom:
                            if st.button(f"🗑️ Delete", key=f"delete_{agent['name']}"):
                                result = st.session_state.orchestrator.delete_custom_agent(agent['name'])
                                if result['success']:
                                    st.success(result['message'])
                                    st.rerun()
                                else:
                                    st.error(result['message'])
                        
                        st.markdown("---")
        
        with agent_tab2:
            st.markdown("### Create a Custom Agent")
            st.info("Define a new agent with custom role, goal, and capabilities for your workflows.")
            
            with st.form("create_agent_form"):
                agent_name = st.text_input(
                    "Agent Name (ID)*",
                    placeholder="e.g., data_scientist",
                    help="Unique identifier (lowercase, no spaces)"
                )
                
                role = st.text_input(
                    "Role/Title*",
                    placeholder="e.g., Senior Data Scientist",
                    help="The agent's professional role"
                )
                
                goal = st.text_area(
                    "Primary Goal*",
                    placeholder="e.g., Analyze datasets and build predictive models...",
                    help="What the agent is trying to achieve",
                    height=80
                )
                
                backstory = st.text_area(
                    "Backstory*",
                    placeholder="e.g., You are an experienced data scientist with expertise in...",
                    help="The agent's background, expertise, and working style",
                    height=120
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    allow_delegation = st.checkbox("Allow Delegation", value=True, help="Can this agent delegate tasks to others?")
                with col2:
                    max_iter = st.number_input("Max Iterations", min_value=1, max_value=10, value=3, help="Maximum task execution iterations")
                
                submitted = st.form_submit_button("✨ Create Agent", type="primary", use_container_width=True)
                
                if submitted:
                    if not agent_name or not role or not goal or not backstory:
                        st.error("Please fill in all required fields marked with *")
                    else:
                        result = st.session_state.orchestrator.create_custom_agent(
                            agent_name=agent_name.lower().replace(" ", "_"),
                            role=role,
                            goal=goal,
                            backstory=backstory,
                            allow_delegation=allow_delegation,
                            max_iter=max_iter
                        )
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
    
    with tabs[4]:
        st.header("💾 Vector Memory System")
        
        memory_stats = st.session_state.orchestrator.get_memory_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Memories", memory_stats.get('total_memories', 0))
        with col2:
            st.metric("Unique Agents", memory_stats.get('unique_agents', 0))
        with col3:
            st.metric("Active Collection", "agent_memory")
        
        st.markdown("---")
        
        st.markdown("### 🔍 Advanced Memory Search")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_query = st.text_input("Search Query", placeholder="Enter semantic search query...")
        
        with col2:
            n_results = st.number_input("Results", min_value=1, max_value=20, value=5)
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔍 Search", use_container_width=True)
        
        agent_filter = st.multiselect(
            "Filter by Agent",
            options=list(memory_stats.get('agents', [])),
            help="Filter results by specific agents"
        )
        
        if search_clicked and search_query:
            vector_memory = st.session_state.orchestrator.vector_memory
            
            filter_metadata = {"agent_name": agent_filter[0]} if agent_filter and len(agent_filter) == 1 else None
            
            results = vector_memory.search_memory(
                search_query, 
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
            st.markdown(f"### Search Results ({len(results)} found)")
            
            if results:
                for i, result in enumerate(results, 1):
                    similarity = 1 - result.get('distance', 0) if result.get('distance') else 0
                    
                    with st.expander(f"Result {i} - {result['metadata'].get('agent_name', 'Unknown')} | Similarity: {similarity:.2%}"):
                        st.write(result['content'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"🕒 {result['metadata'].get('timestamp', 'Unknown')[:19]}")
                        with col2:
                            st.caption(f"📌 Task: {result['metadata'].get('task_name', 'N/A')}")
            else:
                st.info("No results found. Try a different query.")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Memory", type="secondary", use_container_width=True):
                st.session_state.orchestrator.clear_memory()
                st.success("Memory cleared successfully!")
                st.rerun()
        
        with col2:
            if st.button("📥 Export Memory Data", type="secondary", use_container_width=True):
                if memory_stats.get('total_memories', 0) > 0:
                    vector_memory = st.session_state.orchestrator.vector_memory
                    recent_memories = vector_memory.get_recent_memories(n_results=100)
                    
                    memory_export = json.dumps(recent_memories, indent=2)
                    st.download_button(
                        "⬇️ Download JSON",
                        memory_export,
                        "memory_export.json",
                        "application/json"
                    )
                else:
                    st.warning("No memory data to export")
    
    with tabs[5]:
        st.header("📊 Analytics & Performance Dashboard")
        
        analytics = st.session_state.orchestrator.get_performance_analytics()
        
        if 'message' in analytics:
            st.info(analytics['message'])
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Workflows", analytics['total_workflows'])
            with col2:
                st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
            with col3:
                st.metric("Avg Time", f"{analytics['avg_execution_time']:.1f}s")
            with col4:
                st.metric("Most Used Agent", analytics['most_used_agent'].title() if analytics['most_used_agent'] else "N/A")
            
            st.markdown("---")
            
            perf_tab1, perf_tab2, perf_tab3 = st.tabs(["📈 Agent Performance", "⏱️ Execution History", "📋 Process Types"])
            
            with perf_tab1:
                st.markdown("### Agent Performance Metrics")
                
                agent_stats = analytics.get('agent_stats', {})
                if agent_stats:
                    agent_data = []
                    for agent_name, stats in agent_stats.items():
                        agent_data.append({
                            'Agent': agent_name.title(),
                            'Usage Count': stats['usage_count'],
                            'Avg Task Time (s)': round(stats['avg_task_time'], 2)
                        })
                    
                    df_agents = pd.DataFrame(agent_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.bar(df_agents, x='Agent', y='Usage Count', 
                                     title='Agent Usage Distribution',
                                     color='Agent')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.bar(df_agents, x='Agent', y='Avg Task Time (s)',
                                     title='Average Task Execution Time',
                                     color='Agent')
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.dataframe(df_agents, use_container_width=True, hide_index=True)
                else:
                    st.info("No agent statistics available yet.")
            
            with perf_tab2:
                st.markdown("### Recent Execution History")
                
                history = st.session_state.orchestrator.execution_history
                
                for execution in reversed(history[-10:]):
                    status_badge = "✅ Success" if execution.get('success') else "❌ Failed"
                    exec_time = execution.get('execution_time', 0)
                    
                    with st.expander(f"Workflow #{execution['workflow_id']} - {execution['process_type'].title()} | {status_badge} | {exec_time:.1f}s"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Timestamp:** {execution.get('timestamp', 'N/A')[:19]}")
                            st.markdown(f"**Agents Used:** {', '.join(execution['agents_used'])}")
                            st.markdown(f"**Number of Tasks:** {execution.get('num_tasks', 0)}")
                        
                        with col2:
                            st.metric("Execution Time", f"{exec_time:.2f}s")
                        
                        st.markdown("**Tasks:**")
                        for task in execution.get('tasks', []):
                            status_icon = "✅" if task['status'] == 'completed' else "⏳"
                            st.markdown(f"{status_icon} {task['agent'].title()}: {task['description']}")
                        
                        if 'error' in execution:
                            st.error(f"Error: {execution['error']}")
                        elif 'result' in execution and execution['result']:
                            with st.expander("View Result"):
                                st.text(execution['result'][:800] + "..." if len(execution['result']) > 800 else execution['result'])
            
            with perf_tab3:
                st.markdown("### Process Type Statistics")
                
                process_stats = analytics.get('process_type_stats', {})
                if process_stats:
                    process_data = []
                    for process_type, stats in process_stats.items():
                        success_rate = (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0
                        process_data.append({
                            'Process Type': process_type.title(),
                            'Total Workflows': stats['count'],
                            'Successful': stats['success'],
                            'Success Rate (%)': round(success_rate, 1)
                        })
                    
                    df_process = pd.DataFrame(process_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig3 = px.pie(df_process, values='Total Workflows', names='Process Type',
                                     title='Workflow Distribution by Process Type')
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        st.dataframe(df_process, use_container_width=True, hide_index=True)
                else:
                    st.info("No process type statistics available yet.")
            
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_format = st.selectbox("Export Format", ["JSON", "CSV"])
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📤 Export History", type="primary", use_container_width=True):
                    format_type = export_format.lower()
                    export_data = st.session_state.orchestrator.export_execution_history(format=format_type)
                    
                    if export_data:
                        file_ext = "json" if format_type == "json" else "csv"
                        mime_type = "application/json" if format_type == "json" else "text/csv"
                        
                        st.download_button(
                            f"⬇️ Download {export_format}",
                            export_data,
                            f"execution_history.{file_ext}",
                            mime_type
                        )
                    else:
                        st.warning("No execution history to export")
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📊 Export Analytics", type="secondary", use_container_width=True):
                    analytics_export = json.dumps(analytics, indent=2)
                    st.download_button(
                        "⬇️ Download Analytics JSON",
                        analytics_export,
                        "analytics_report.json",
                        "application/json"
                    )

if __name__ == "__main__":
    main()
