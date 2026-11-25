# backend/evaluate_ragas.py  -- run from the backend folder: uv run evaluate_ragas.py
import json, time, uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference, ResponseRelevancy,
)

from config import GROQ_API_KEY, GROQ_MODEL
from agent import rag_agent                 # your LangGraph agent
from vectorstore import get_retriever, embeddings

retriever = get_retriever()

def run_agent(question: str) -> str:
    """Call the real agent. New thread_id per question so chat memory never leaks between tests.
    Web search is OFF so answers come only from your Pinecone knowledge base."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4()), "web_search_enabled": False}}
    state = rag_agent.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage))

golden = json.load(open("golden_set.json", encoding="utf-8"))
rows = []
for i, item in enumerate(golden, 1):
    q = item["question"]
    print(f"[{i}/{len(golden)}] {q}")
    contexts = [d.page_content for d in retriever.invoke(q)]   # chunks Pinecone returns for this question
    answer = run_agent(q)
    rows.append({
        "user_input": q,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": item["ground_truth"],
    })
    time.sleep(2)   # be gentle with Groq rate limits

# Judge = Groq (same provider you already use). RAGAS needs its own LLM + embeddings to score.
judge = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY))
judge_emb = LangchainEmbeddingsWrapper(embeddings)

result = evaluate(
    EvaluationDataset.from_list(rows),
    metrics=[
        Faithfulness(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        ResponseRelevancy(strictness=1),   # Groq rejects n>1 completions, so keep this at 1
    ],
    llm=judge,
    embeddings=judge_emb,
    run_config=RunConfig(max_workers=2, timeout=180),   # low parallelism avoids Groq 429 errors
)
print(result)
df = result.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("Saved ragas_results.csv")
