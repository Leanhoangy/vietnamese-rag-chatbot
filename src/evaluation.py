import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retrieval import qa_chain
from llm_chain import model
from embedder import embeddings
from ragas.run_config import RunConfig
# Bước 1: Config RAGAS dùng Groq
ragas_llm = LangchainLLMWrapper(model)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Bước 2: Load test set
with open("test_set.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)
test_cases = test_cases[:5] 
# Bước 3: Chạy pipeline lấy câu trả lời
questions, answers, contexts, ground_truths = [], [], [], []
for i, case in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] {case['question']}")
    result = qa_chain.invoke({"query": case["question"]})
    source_docs = result.get("source_documents", [])
    context = [doc.page_content for doc in source_docs]
    questions.append(case["question"])
    answers.append(result["result"])
    contexts.append(context if context else [result["result"]])
    ground_truths.append(case["ground_truth"])
    time.sleep(1)

# Bước 4: Tạo dataset
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

# Bước 5: Chạy RAGAS
print("\nĐang chạy RAGAS evaluation...")
result = evaluate(
    dataset,
    metrics=[faithfulness,answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(max_workers=2, timeout=300)
)

# Bước 6: In và lưu kết quả
print("\n=== KẾT QUẢ RAGAS ===")
df = result.to_pandas()
print(f"Faithfulness:     {df['faithfulness'].mean():.3f}")
print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
print(f"Context Precision: {df['context_precision'].mean():.3f}")
print(f"Context Recall:    {df['context_recall'].mean():.3f}")
df.to_csv("ragas_results.csv", index=False)
print("✓ Đã lưu vào ragas_results.csv")