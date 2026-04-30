"""
Conversation Memory Module
Manages multi-turn conversations with context tracking
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from openai import OpenAI


@dataclass
class Message:
    """Single message in conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None


@dataclass
class Conversation:
    """Conversation session with memory"""
    conversation_id: str
    messages: List[Message]
    created_at: str
    last_updated: str
    metadata: Dict[str, Any] = None


class ConversationMemory:
    """Manages conversation history and context"""

    def __init__(
        self,
        max_history: int = 5,
        storage_dir: str = "data/conversations"
    ):
        self.max_history = max_history
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI for query rewriting
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)

    def create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Create new conversation"""
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now().isoformat()
        return Conversation(
            conversation_id=conversation_id,
            messages=[],
            created_at=now,
            last_updated=now,
            metadata={}
        )

    def add_message(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Conversation:
        """Add message to conversation"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            sources=sources
        )

        conversation.messages.append(message)
        conversation.last_updated = datetime.now().isoformat()

        # Trim history if needed
        if len(conversation.messages) > self.max_history * 2:  # *2 for user+assistant pairs
            conversation.messages = conversation.messages[-(self.max_history * 2):]

        return conversation

    def get_conversation_context(
        self,
        conversation: Conversation,
        include_sources: bool = False
    ) -> str:
        """Format conversation history as context string"""
        context_parts = []

        for msg in conversation.messages[-self.max_history * 2:]:  # Last N exchanges
            context_parts.append(f"{msg.role.upper()}: {msg.content}")

        return "\n\n".join(context_parts)

    def rewrite_query_with_context(
        self,
        current_query: str,
        conversation: Conversation
    ) -> str:
        """
        Rewrite user query to be standalone using conversation context

        Example:
        History: "What are treatments for prostate cancer?" -> "ADT, chemo..."
        Current: "What about side effects?"
        Rewritten: "What are the side effects of ADT and chemotherapy for prostate cancer?"
        """
        # If no history, return as-is
        if not conversation.messages:
            return current_query

        # Get recent context (last 3 exchanges)
        recent_context = self.get_conversation_context(
            conversation
        )

        # Rewrite query
        prompt = f"""Given this conversation history, rewrite the user's latest question to be self-contained and clear.

Conversation History:
{recent_context}

Latest User Question: {current_query}

Rewrite the question to include necessary context from the history. Make it specific and searchable.
Output ONLY the rewritten question, nothing else."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )

            rewritten = response.choices[0].message.content.strip()

            # Remove quotes if present
            rewritten = rewritten.strip('"').strip("'")

            return rewritten

        except Exception as e:
            print(f"⚠️  Query rewrite failed: {e}")
            return current_query  # Fallback to original

    def save_conversation(self, conversation: Conversation):
        """Save conversation to disk"""
        filepath = self.storage_dir / f"{conversation.conversation_id}.json"

        # Convert to dict
        conv_dict = asdict(conversation)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conv_dict, f, indent=2, ensure_ascii=False)

    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from disk"""
        filepath = self.storage_dir / f"{conversation_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conv_dict = json.load(f)

            # Reconstruct conversation
            messages = [
                Message(**msg) for msg in conv_dict['messages']
            ]

            return Conversation(
                conversation_id=conv_dict['conversation_id'],
                messages=messages,
                created_at=conv_dict['created_at'],
                last_updated=conv_dict['last_updated'],
                metadata=conv_dict.get('metadata', {})
            )

        except Exception as e:
            print(f"⚠️  Failed to load conversation: {e}")
            return None

    def list_conversations(self) -> List[str]:
        """List all conversation IDs"""
        return [
            f.stem for f in self.storage_dir.glob("*.json")
        ]

    def get_conversation_summary(self, conversation: Conversation) -> str:
        """Generate short summary of conversation"""
        if not conversation.messages:
            return "Empty conversation"

        first_user_msg = next(
            (msg.content for msg in conversation.messages if msg.role == 'user'),
            "No messages"
        )

        return f"{first_user_msg[:50]}..." if len(first_user_msg) > 50 else first_user_msg


class ConversationalRAG:
    """RAG system with conversation memory"""

    def __init__(self, rag_retriever, memory: ConversationMemory = None):
        self.retriever = rag_retriever
        self.memory = memory or ConversationMemory()

    def query(
        self,
        question: str,
        conversation: Conversation,
        model: str = "gpt-4o-mini",
        save_conversation: bool = True
    ) -> Dict[str, Any]:
        """
        Query with conversation context

        Args:
            question: User's question
            conversation: Current conversation
            model: LLM model to use
            save_conversation: Whether to save after query

        Returns:
            Response dict with answer and metadata
        """
        # Rewrite query with context if needed
        standalone_query = self.memory.rewrite_query_with_context(
            question, conversation
        )

        print(f"🔄 Query rewrite:")
        print(f"   Original: {question}")
        print(f"   Rewritten: {standalone_query}")

        # Query RAG system with rewritten query
        response = self.retriever.query(
            question=standalone_query,
            model=model,
            return_sources=True
        )

        # Add user message to history
        self.memory.add_message(
            conversation,
            role='user',
            content=question  # Store original, not rewritten
        )

        # Add assistant message to history
        self.memory.add_message(
            conversation,
            role='assistant',
            content=response['answer'],
            sources=response.get('sources', [])
        )

        # Save conversation
        if save_conversation:
            self.memory.save_conversation(conversation)

        # Add conversation metadata to response
        response['conversation_id'] = conversation.conversation_id
        response['rewritten_query'] = standalone_query
        response['conversation_turns'] = len(conversation.messages) // 2

        return response

    def get_conversation_history(
        self,
        conversation: Conversation
    ) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []

        for msg in conversation.messages:
            history.append({
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            })

        return history


def demo_conversation():
    """Demo conversational RAG"""
    from src.retrieval_optimized import OptimizedRAGRetriever, OptimizedRetrievalConfig

    print("="*70)
    print(" "*15 + "🧠 CONVERSATIONAL RAG DEMO")
    print("="*70)
    print()

    # Initialize
    retriever = OptimizedRAGRetriever(
        config=OptimizedRetrievalConfig(top_k=5)
    )

    memory = ConversationMemory(max_history=5)
    conv_rag = ConversationalRAG(retriever, memory)

    # Create conversation
    conversation = memory.create_conversation()
    print(f"📝 Conversation ID: {conversation.conversation_id}")
    print()

    # Simulate multi-turn conversation
    queries = [
        "What are the current treatment options for prostate cancer?",
        "What about side effects?",  # Context: side effects of those treatments
        "Compare them to radiotherapy",  # Context: ADT/chemo vs radiotherapy
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Turn {i}: {query}")
        print(f"{'='*70}")

        response = conv_rag.query(query, conversation)

        print(f"\n📄 Answer:")
        print(response['answer'][:300] + "...")

        print(f"\n📊 Metadata:")
        print(f"   Conversation turns: {response['conversation_turns']}")
        print(f"   Sources used: {response['num_sources']}")

        input("\nPress Enter for next turn...")

    print("\n" + "="*70)
    print("✅ Demo complete!")
    print(f"💾 Conversation saved: data/conversations/{conversation.conversation_id}.json")
    print("="*70)


if __name__ == "__main__":
    demo_conversation()
