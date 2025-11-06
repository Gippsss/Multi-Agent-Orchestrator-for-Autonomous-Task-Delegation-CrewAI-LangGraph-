import os
import time
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from vector_memory import VectorMemory
from typing import List, Dict, Any

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class MultiAgentOrchestrator:
    """
    Multi-agent orchestrator using CrewAI for autonomous task delegation.
    Manages specialized agents with vector memory for context retention.
    """
    
    def __init__(self, memory_enabled: bool = True):
        """Initialize the orchestrator with optional memory."""
        self.memory_enabled = memory_enabled
        self.vector_memory = VectorMemory() if memory_enabled else None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-5",
            api_key=OPENAI_API_KEY
        )
        
        # Create specialized agents
        self.agents = self._create_agents()
        self.execution_history = []
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents with distinct roles and capabilities."""
        
        researcher = Agent(
            role='Senior Research Analyst',
            goal='Conduct thorough research and gather comprehensive information on given topics',
            backstory="""You are an experienced research analyst with expertise in 
            information gathering, data analysis, and synthesis. You excel at finding 
            relevant information, identifying patterns, and presenting comprehensive 
            research findings. Your research is always well-sourced and objective.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=3
        )
        
        writer = Agent(
            role='Professional Content Writer',
            goal='Create compelling, well-structured content based on research and analysis',
            backstory="""You are a talented content writer with years of experience 
            in creating engaging, informative content. You excel at taking complex 
            information and presenting it in a clear, accessible manner. Your writing 
            is always well-organized, grammatically correct, and tailored to the 
            intended audience.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=3
        )
        
        analyst = Agent(
            role='Strategic Business Analyst',
            goal='Analyze data, identify insights, and provide strategic recommendations',
            backstory="""You are a strategic analyst with deep expertise in business 
            analysis, data interpretation, and strategic planning. You excel at 
            identifying trends, evaluating options, and providing actionable 
            recommendations. Your analysis is always thorough and data-driven.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=3
        )
        
        reviewer = Agent(
            role='Quality Assurance Reviewer',
            goal='Review work for quality, accuracy, and completeness',
            backstory="""You are a meticulous quality assurance expert with an eye 
            for detail. You excel at reviewing content, identifying gaps, ensuring 
            accuracy, and suggesting improvements. Your reviews are thorough, 
            constructive, and focused on delivering the highest quality output.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        return {
            'researcher': researcher,
            'writer': writer,
            'analyst': analyst,
            'reviewer': reviewer
        }
    
    def create_task(
        self, 
        description: str, 
        agent_name: str,
        expected_output: str = None,
        context_tasks: List[Task] = None
    ) -> Task:
        """
        Create a task for a specific agent with optional context.
        
        Args:
            description: Task description
            agent_name: Name of the agent to assign the task
            expected_output: Description of expected output
            context_tasks: List of tasks to use as context
            
        Returns:
            Created Task object
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # Enhance description with memory context if enabled
        enhanced_description = description
        if self.memory_enabled and self.vector_memory:
            memory_context = self.vector_memory.get_agent_context(
                agent_name=agent_name,
                task_context=description,
                n_results=3
            )
            if memory_context and memory_context != "No previous context available.":
                enhanced_description = f"{description}\n\nRelevant Context:\n{memory_context}"
        
        task = Task(
            description=enhanced_description,
            agent=agent,
            expected_output=expected_output or f"Completed output for: {description[:50]}...",
            context=context_tasks
        )
        
        return task
    
    def execute_workflow(
        self, 
        tasks_config: List[Dict[str, Any]],
        process_type: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Execute a workflow with multiple tasks and agents.
        
        Args:
            tasks_config: List of task configurations with 'description', 'agent', 'expected_output'
            process_type: 'sequential' or 'hierarchical'
            
        Returns:
            Workflow execution results
        """
        tasks = []
        task_details = []
        
        # Create tasks
        for i, task_config in enumerate(tasks_config):
            context_tasks = tasks if task_config.get('use_context', True) and i > 0 else None
            
            task = self.create_task(
                description=task_config['description'],
                agent_name=task_config['agent'],
                expected_output=task_config.get('expected_output'),
                context_tasks=context_tasks
            )
            tasks.append(task)
            
            task_details.append({
                'task_id': i + 1,
                'agent': task_config['agent'],
                'description': task_config['description'][:100] + '...' if len(task_config['description']) > 100 else task_config['description'],
                'status': 'pending'
            })
        
        # Create and execute crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential if process_type == "sequential" else Process.hierarchical,
            verbose=True
        )
        
        start_time = time.time()
        
        try:
            result = crew.kickoff()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store results in memory if enabled
            if self.memory_enabled and self.vector_memory:
                for i, task_config in enumerate(tasks_config):
                    self.vector_memory.add_memory(
                        content=f"Task: {task_config['description']}\nResult: Completed successfully",
                        agent_name=task_config['agent'],
                        task_name=f"workflow_task_{i+1}",
                        metadata={'workflow_id': len(self.execution_history) + 1}
                    )
            
            # Update task statuses
            for task_detail in task_details:
                task_detail['status'] = 'completed'
            
            execution_record = {
                'workflow_id': len(self.execution_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'tasks': task_details,
                'result': str(result),
                'process_type': process_type,
                'agents_used': [t['agent'] for t in tasks_config],
                'num_tasks': len(tasks_config),
                'success': True
            }
            
            self.execution_history.append(execution_record)
            
            return execution_record
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_record = {
                'workflow_id': len(self.execution_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'tasks': task_details,
                'error': str(e),
                'process_type': process_type,
                'agents_used': [t['agent'] for t in tasks_config],
                'num_tasks': len(tasks_config),
                'success': False
            }
            self.execution_history.append(error_record)
            return error_record
    
    def get_agent_info(self) -> List[Dict[str, str]]:
        """Get information about all agents."""
        agent_info = []
        for name, agent in self.agents.items():
            agent_info.append({
                'name': name,
                'role': agent.role,
                'goal': agent.goal,
                'backstory': agent.backstory[:150] + '...' if len(agent.backstory) > 150 else agent.backstory
            })
        return agent_info
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.memory_enabled and self.vector_memory:
            return self.vector_memory.get_memory_stats()
        return {'message': 'Memory not enabled'}
    
    def clear_memory(self):
        """Clear all agent memories."""
        if self.memory_enabled and self.vector_memory:
            self.vector_memory.clear_memory()
            return {'status': 'Memory cleared'}
        return {'status': 'Memory not enabled'}
    
    def create_custom_agent(
        self,
        agent_name: str,
        role: str,
        goal: str,
        backstory: str,
        allow_delegation: bool = True,
        max_iter: int = 3
    ) -> Dict[str, Any]:
        """
        Create a custom agent with user-defined characteristics.
        
        Args:
            agent_name: Unique identifier for the agent
            role: Agent's role/title
            goal: Agent's primary goal
            backstory: Agent's background and expertise
            allow_delegation: Whether agent can delegate tasks
            max_iter: Maximum iterations for task execution
            
        Returns:
            Status dictionary with agent information
        """
        if agent_name in self.agents:
            return {
                'success': False,
                'message': f"Agent '{agent_name}' already exists"
            }
        
        try:
            custom_agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                llm=self.llm,
                verbose=True,
                allow_delegation=allow_delegation,
                max_iter=max_iter
            )
            
            self.agents[agent_name] = custom_agent
            
            return {
                'success': True,
                'message': f"Agent '{agent_name}' created successfully",
                'agent_name': agent_name,
                'role': role
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error creating agent: {str(e)}"
            }
    
    def delete_custom_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Delete a custom agent (cannot delete default agents).
        
        Args:
            agent_name: Name of the agent to delete
            
        Returns:
            Status dictionary
        """
        default_agents = ['researcher', 'writer', 'analyst', 'reviewer']
        
        if agent_name in default_agents:
            return {
                'success': False,
                'message': f"Cannot delete default agent '{agent_name}'"
            }
        
        if agent_name not in self.agents:
            return {
                'success': False,
                'message': f"Agent '{agent_name}' not found"
            }
        
        del self.agents[agent_name]
        
        return {
            'success': True,
            'message': f"Agent '{agent_name}' deleted successfully"
        }
    
    def list_custom_agents(self) -> List[str]:
        """Get list of custom (non-default) agents."""
        default_agents = ['researcher', 'writer', 'analyst', 'reviewer']
        return [name for name in self.agents.keys() if name not in default_agents]
    
    def get_workflow_templates(self) -> Dict[str, Any]:
        """Get predefined workflow templates."""
        return {
            'market_research': {
                'name': 'Market Research & Analysis',
                'description': 'Comprehensive market research with analysis and reporting',
                'process_type': 'sequential',
                'tasks': [
                    {
                        'description': 'Research current market trends, competitor analysis, and customer insights for {topic}',
                        'agent': 'researcher',
                        'expected_output': 'Detailed market research report'
                    },
                    {
                        'description': 'Analyze the research findings and identify key opportunities, threats, and strategic recommendations',
                        'agent': 'analyst',
                        'expected_output': 'Strategic analysis with actionable insights'
                    },
                    {
                        'description': 'Create an executive summary report based on research and analysis',
                        'agent': 'writer',
                        'expected_output': 'Professional executive summary'
                    },
                    {
                        'description': 'Review the report for accuracy, completeness, and clarity',
                        'agent': 'reviewer',
                        'expected_output': 'Quality review and approval'
                    }
                ]
            },
            'content_pipeline': {
                'name': 'Content Creation Pipeline',
                'description': 'End-to-end content creation from research to publication',
                'process_type': 'sequential',
                'tasks': [
                    {
                        'description': 'Research best practices, examples, and key information about {topic}',
                        'agent': 'researcher',
                        'expected_output': 'Research findings and sources'
                    },
                    {
                        'description': 'Write engaging, well-structured content about {topic} based on the research',
                        'agent': 'writer',
                        'expected_output': 'Complete draft content (800-1200 words)'
                    },
                    {
                        'description': 'Review content for quality, accuracy, readability, and SEO optimization',
                        'agent': 'reviewer',
                        'expected_output': 'Reviewed content with suggestions'
                    }
                ]
            },
            'strategic_planning': {
                'name': 'Strategic Planning',
                'description': 'Strategic analysis and planning framework',
                'process_type': 'sequential',
                'tasks': [
                    {
                        'description': 'Research industry trends, emerging technologies, and competitive landscape for {topic}',
                        'agent': 'researcher',
                        'expected_output': 'Comprehensive industry research'
                    },
                    {
                        'description': 'Analyze opportunities, risks, and develop strategic recommendations for {topic}',
                        'agent': 'analyst',
                        'expected_output': 'Strategic analysis and recommendations'
                    },
                    {
                        'description': 'Create a strategic plan document with clear goals, initiatives, and timelines',
                        'agent': 'writer',
                        'expected_output': 'Strategic plan document'
                    }
                ]
            },
            'data_analysis': {
                'name': 'Data Analysis & Reporting',
                'description': 'Analyze data and create comprehensive reports',
                'process_type': 'sequential',
                'tasks': [
                    {
                        'description': 'Analyze the provided data about {topic} and identify key patterns, trends, and insights',
                        'agent': 'analyst',
                        'expected_output': 'Data analysis with key findings'
                    },
                    {
                        'description': 'Create a clear, visual report summarizing the analysis and insights',
                        'agent': 'writer',
                        'expected_output': 'Professional data report'
                    },
                    {
                        'description': 'Review the report for accuracy and clarity of insights',
                        'agent': 'reviewer',
                        'expected_output': 'Validated report'
                    }
                ]
            },
            'quick_research': {
                'name': 'Quick Research Task',
                'description': 'Fast research and summary for time-sensitive topics',
                'process_type': 'sequential',
                'tasks': [
                    {
                        'description': 'Quickly research and gather key information about {topic}',
                        'agent': 'researcher',
                        'expected_output': 'Concise research summary'
                    },
                    {
                        'description': 'Create a brief summary of the research findings',
                        'agent': 'writer',
                        'expected_output': 'Short summary (300-500 words)'
                    }
                ]
            }
        }
    
    def execute_template(self, template_id: str, topic: str, process_type: str = None) -> Dict[str, Any]:
        """
        Execute a workflow template with a given topic.
        
        Args:
            template_id: ID of the template to execute
            topic: Topic to apply to the template
            process_type: Override template's default process type
            
        Returns:
            Execution results
        """
        templates = self.get_workflow_templates()
        
        if template_id not in templates:
            return {
                'success': False,
                'error': f"Template '{template_id}' not found"
            }
        
        template = templates[template_id]
        
        tasks_config = []
        for task in template['tasks']:
            tasks_config.append({
                'description': task['description'].replace('{topic}', topic),
                'agent': task['agent'],
                'expected_output': task['expected_output']
            })
        
        process = process_type or template['process_type']
        
        return self.execute_workflow(tasks_config=tasks_config, process_type=process)
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get detailed performance analytics from execution history."""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_workflows = len(self.execution_history)
        successful_workflows = sum(1 for w in self.execution_history if w.get('success', False))
        failed_workflows = total_workflows - successful_workflows
        
        total_execution_time = sum(w.get('execution_time', 0) for w in self.execution_history)
        avg_execution_time = total_execution_time / total_workflows if total_workflows > 0 else 0
        
        agent_usage = {}
        agent_task_times = {}
        
        for workflow in self.execution_history:
            exec_time = workflow.get('execution_time', 0)
            num_tasks = workflow.get('num_tasks', 1)
            avg_task_time = exec_time / num_tasks if num_tasks > 0 else 0
            
            for agent_name in workflow.get('agents_used', []):
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
                if agent_name not in agent_task_times:
                    agent_task_times[agent_name] = []
                agent_task_times[agent_name].append(avg_task_time)
        
        agent_stats = {}
        for agent_name, count in agent_usage.items():
            times = agent_task_times.get(agent_name, [0])
            agent_stats[agent_name] = {
                'usage_count': count,
                'avg_task_time': sum(times) / len(times) if times else 0
            }
        
        process_type_stats = {}
        for workflow in self.execution_history:
            process_type = workflow.get('process_type', 'unknown')
            if process_type not in process_type_stats:
                process_type_stats[process_type] = {'count': 0, 'success': 0}
            process_type_stats[process_type]['count'] += 1
            if workflow.get('success', False):
                process_type_stats[process_type]['success'] += 1
        
        return {
            'total_workflows': total_workflows,
            'successful_workflows': successful_workflows,
            'failed_workflows': failed_workflows,
            'success_rate': (successful_workflows / total_workflows * 100) if total_workflows > 0 else 0,
            'total_execution_time': total_execution_time,
            'avg_execution_time': avg_execution_time,
            'agent_stats': agent_stats,
            'process_type_stats': process_type_stats,
            'most_used_agent': max(agent_usage.items(), key=lambda x: x[1])[0] if agent_usage else None
        }
    
    def export_execution_history(self, format: str = 'json') -> str:
        """
        Export execution history in JSON or CSV format.
        
        Args:
            format: 'json' or 'csv'
            
        Returns:
            Formatted string data
        """
        if not self.execution_history:
            return ""
        
        if format == 'json':
            return json.dumps(self.execution_history, indent=2)
        elif format == 'csv':
            csv_lines = []
            csv_lines.append("Workflow ID,Timestamp,Process Type,Agents Used,Num Tasks,Execution Time,Success,Result Preview")
            
            for workflow in self.execution_history:
                workflow_id = workflow.get('workflow_id', '')
                timestamp = workflow.get('timestamp', '')
                process_type = workflow.get('process_type', '')
                agents_used = ';'.join(workflow.get('agents_used', []))
                num_tasks = workflow.get('num_tasks', 0)
                exec_time = workflow.get('execution_time', 0)
                success = workflow.get('success', False)
                result_preview = workflow.get('result', workflow.get('error', ''))[:100].replace(',', ';').replace('\n', ' ')
                
                csv_lines.append(f"{workflow_id},{timestamp},{process_type},{agents_used},{num_tasks},{exec_time:.2f},{success},{result_preview}")
            
            return '\n'.join(csv_lines)
        
        return ""
