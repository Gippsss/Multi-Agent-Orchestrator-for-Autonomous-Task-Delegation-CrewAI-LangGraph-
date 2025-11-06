import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from datetime import datetime
import hashlib

class VectorMemory:
    """
    Vector memory system using ChromaDB for persistent context retention.
    Enables semantic search and long-term memory for multi-agent systems.
    """
    
    def __init__(self, collection_name: str = "agent_memory", persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB with persistent storage."""
        self.persist_directory = persist_directory
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Multi-agent context and memory storage"}
            )
    
    def add_memory(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        agent_name: str = None,
        task_name: str = None
    ) -> str:
        """
        Add a memory entry to the vector database.
        
        Args:
            content: The text content to store
            metadata: Additional metadata about the memory
            agent_name: Name of the agent creating this memory
            task_name: Name of the task associated with this memory
            
        Returns:
            The ID of the stored memory
        """
        timestamp = datetime.now().isoformat()
        
        memory_id = hashlib.md5(
            f"{content}{timestamp}".encode()
        ).hexdigest()
        
        memory_metadata = {
            "timestamp": timestamp,
            "agent_name": agent_name or "unknown",
            "task_name": task_name or "general",
            **(metadata or {})
        }
        
        self.collection.add(
            documents=[content],
            metadatas=[memory_metadata],
            ids=[memory_id]
        )
        
        return memory_id
    
    def search_memory(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using semantic similarity.
        
        Args:
            query: The search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant memories with metadata
        """
        where_filter = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        memories = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                memories.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return memories
    
    def get_recent_memories(
        self, 
        n_results: int = 10,
        agent_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent memories, optionally filtered by agent.
        
        Args:
            n_results: Number of results to return
            agent_name: Optional filter by agent name
            
        Returns:
            List of recent memories
        """
        where_filter = {"agent_name": agent_name} if agent_name else None
        
        try:
            all_items = self.collection.get(
                where=where_filter,
                limit=n_results
            )
            
            memories = []
            if all_items['documents']:
                for i in range(len(all_items['documents'])):
                    memories.append({
                        'content': all_items['documents'][i],
                        'metadata': all_items['metadatas'][i] if all_items['metadatas'] else {},
                        'id': all_items['ids'][i] if all_items['ids'] else None
                    })
            
            memories.sort(
                key=lambda x: x['metadata'].get('timestamp', ''),
                reverse=True
            )
            
            return memories[:n_results]
        except Exception as e:
            return []
    
    def get_agent_context(
        self, 
        agent_name: str,
        task_context: str = None,
        n_results: int = 5
    ) -> str:
        """
        Get relevant context for an agent based on task and history.
        
        Args:
            agent_name: Name of the agent
            task_context: Optional task description for context search
            n_results: Number of memories to retrieve
            
        Returns:
            Formatted context string
        """
        if task_context:
            memories = self.search_memory(
                query=task_context,
                n_results=n_results,
                filter_metadata={"agent_name": agent_name}
            )
        else:
            memories = self.get_recent_memories(
                n_results=n_results,
                agent_name=agent_name
            )
        
        if not memories:
            return "No previous context available."
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            timestamp = memory['metadata'].get('timestamp', 'Unknown time')
            content = memory['content']
            context_parts.append(f"{i}. [{timestamp}] {content}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self, agent_name: str = None):
        """
        Clear memories, optionally filtered by agent.
        
        Args:
            agent_name: Optional agent name to clear only that agent's memories
        """
        if agent_name:
            items = self.collection.get(where={"agent_name": agent_name})
            if items['ids']:
                self.collection.delete(ids=items['ids'])
        else:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "Multi-agent context and memory storage"}
            )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        count = self.collection.count()
        
        all_items = self.collection.get()
        agents = set()
        if all_items['metadatas']:
            agents = {m.get('agent_name', 'unknown') for m in all_items['metadatas']}
        
        return {
            'total_memories': count,
            'unique_agents': len(agents),
            'agents': list(agents)
        }
