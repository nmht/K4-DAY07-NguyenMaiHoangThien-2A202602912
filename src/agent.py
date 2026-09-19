from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self._store.search(query=question, top_k=top_k)

        context_parts = []
        for rank, hit in enumerate(retrieved, 1):
            context_parts.append(
                f"\n--- Reference Document (Rank {rank}, Score: {hit['score']:.3f}) ---\n"
                f"{hit['content']}"
            )
        context_block = "".join(context_parts)

        prompt = f"""
You are an AI assistant answering a question about our documents.
Use the following retrieved documents to answer the question.
If the answer is not in the documents, say so.

Question: {question}

Retrieved Documents:{context_block}

Answer:"""

        return self._llm_fn(prompt)
