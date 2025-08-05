# Task 3: Comparing vLLM and ChatGPT (GPT-4o-mini)

## Objective
This experiment evaluates the performance of a locally hosted **LLaMA 3.2 1B model via vLLM** against **ChatGPT (gpt-4o-mini)** across 10 factual and arithmetic questions. The test measures **accuracy**, **inference time**, and **output quality**, offering insights into the strengths and limitations of each model.

---

## Test Overview

| Test # | Prompt                                       | Expected Answer | vLLM Correct | ChatGPT Correct |
|--------|----------------------------------------------|------------------|--------------|------------------|
| 1      | What is the capital of France?               | Paris            | ✅            | ✅                |
| 2      | 2 + 2 =                                      | 4                | ✅            | ✅                |
| 3      | What is the capital of Japan?                | Tokyo            | ✅            | ✅                |
| 4      | Who wrote *Pride and Prejudice*?             | Jane Austen      | ❌            | ✅                |
| 5      | Boiling point of water (Celsius)?            | 100              | ✅            | ✅                |
| 6      | What is 9 * 8?                               | 72               | ❌            | ✅                |
| 7      | Which planet is known as the Red Planet?     | Mars             | ❌            | ✅                |
| 8      | What is the chemical symbol for gold?        | Au               | ❌            | ✅                |
| 9      | Who painted the Mona Lisa?                   | Leonardo da Vinci| ✅            | ✅                |
| 10     | What is the square root of 144?              | 12               | ✅            | ✅                |

---

## Performance Summary

| Metric           | vLLM (LLaMA 3.2 1B) | ChatGPT (GPT-4o-mini) |
|------------------|---------------------|------------------------|
| Total Correct     | 6 / 10              | 10 / 10                |
| Average Time      | ~0.32s              | ~1.08s                 |
| Cost              | Free (local)        | API-based ($)          |

---

## ⚖️ Analysis

- **Accuracy:**  
  ChatGPT achieved a perfect 10/10. vLLM returned correct answers 60% of the time. Incorrect vLLM outputs tended to include excessive repetition or unclear responses.

- **Speed:**  
  vLLM was consistently faster (~0.3s), making it ideal for real-time or resource-constrained environments.

- **Cost Consideration:**  
  vLLM runs locally at no incremental cost, while ChatGPT incurs token-based charges through OpenAI’s API.

- **Output Quality:**  
  ChatGPT produced fluent, coherent, and accurate responses. vLLM often echoed prompts or failed to answer definitively—likely due to limited model size and decoding configuration.

---

## Output File
The full output log was saved as:  
**`comparison_results.txt`**  
It contains detailed response times and correctness flags for each test case.

---

## Final Summary

| Model   | Accuracy | Speed  | Cost  | Summary |
|---------|----------|--------|-------|---------|
| **vLLM** | 6/10     | Fast   | Free  | Great for local/fast use, but inconsistent accuracy |
| **ChatGPT** | 10/10    | Moderate | Paid  | Ideal for accurate, polished results |

**Conclusion:**  
While vLLM shows promise in performance and cost-efficiency, it falls short in consistency and correctness. ChatGPT remains the more reliable model for general-purpose use—but at a higher cost and slower response time. The choice depends on use case: **ChatGPT for quality**, **vLLM for affordability and speed**.
