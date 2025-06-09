## 📅 6-Week Curriculum to Learn Large Language Models (LLMs)

This curriculum is designed for learners with a deep learning and transformer background (e.g., computer vision) who are new to language models. It covers theory, hands-on tools, codebases, and essential papers from training to deployment.

---

## ✅ WEEK 1: Transformer Decoder, Tokenization, and Causal LM

### 🎯 Goal:

Understand how GPT-style models process text via self-attention and how tokenization works.

### 📘 Topics:

* Transformer decoder stack: self-attention, FFN, residuals, LayerNorm
* Causal attention masking
* Positional encodings: sinusoidal vs learned
* Tokenization: BPE, WordPiece, SentencePiece
* Special tokens (PAD, BOS, EOS, CLS)
* Autoregressive language modeling objective
* Perplexity and cross-entropy loss

### 🔗 Key Papers:

* Attention is All You Need – Vaswani et al. (2017)
* Language Models are Few-Shot Learners (GPT-2)

### 💻 GitHub Repos:

* nanoGPT: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
* picoGPT: [https://github.com/jaymody/picoGPT](https://github.com/jaymody/picoGPT)

---

## ✅ WEEK 2: Pretraining and Sampling Strategies

### 🎯 Goal:

Understand LLM training pipelines and generation strategies.

### 📘 Topics:

* Pretraining datasets: C4, The Pile, BookCorpus, Common Crawl
* Optimizer: AdamW, learning rate scheduling, weight decay
* Context length, batch size, token masking
* Gradient checkpointing, FSDP, ZeRO, mixed precision
* Model sharding, tensor/parameter parallelism
* Text generation: greedy, beam search, top-k, top-p, temperature, contrastive decoding

### 🔗 Key Papers:

* Scaling Laws for Neural Language Models (Kaplan et al.)
* Better Language Models and Their Implications (GPT-2)

### 💻 GitHub:

* DeepSpeed Chat Examples: [https://github.com/microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)
* Megatron-LM: [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* HuggingFace Trainer: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
* HuggingFace Sampling Blog: [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)

---

## ✅ WEEK 3: Evaluation Metrics and Benchmarks

### 🎯 Goal:

Learn how to evaluate LLMs using standardized metrics and benchmarks.

### 📘 Topics:

* Metrics: perplexity, cross-entropy, BLEU, ROUGE

### 🔗 Key Papers:

* BLEU (Papineni et al. 2002)
* ROUGE (Lin 2004)

### 💻 GitHub:

* BERTScore: [https://github.com/Tiiiger/bert\_score](https://github.com/Tiiiger/bert_score)
* nlg-eval: [https://github.com/Maluuba/nlg-eval](https://github.com/Maluuba/nlg-eval)
* lm-eval-harness: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* HuggingFace Open LLM Leaderboard: [https://huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

## ✅ WEEK 4: Finetuning, Instruction Tuning, LoRA, QLoRA

### 🎯 Goal:

Learn how to adapt pretrained LLMs efficiently.

### 📘 Topics:

* Supervised instruction tuning: FLAN, Alpaca, Dolly
* Prompt-response dataset formatting
* LoRA (Low-Rank Adaptation)
* Prefix tuning
* PEFT trade-offs: compute vs accuracy

### 🔗 Key Papers:
* LoRA (Hu et al., 2021)

### 💻 GitHub:

* HuggingFace PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
---

## ✅ WEEK 5: Inference Optimization + Prompting

### 🎯 Goal:

Serve LLMs efficiently and maximize performance via prompting.

### 📘 Topics:

* Prompting strategies: zero-shot, few-shot, chain-of-thought, self-refinement
* FlashAttention, KV caching, Rotary Positional Embedding (RoPE)
* Quantization: GPTQ, AWQ
* Accelerated inference: vLLM, sglang, TensorRT-LLM

### 🔗 Key Papers:

* Chain-of-Thought Prompting (Wei et al., 2022)
* FlashAttention (Dao et al., 2022)

### 💻 GitHub:

* vLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
* GPTQ: [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
* AWQ: [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
---

## ✅ WEEK 6: RLHF + RAG + Applications + Agents

### 🎯 Goal:

Learn alignment, retrieval-based generation, and advanced applications.

### 📘 Topics:

* RLHF: preference modeling, reward modeling, PPO, DPO
* RAG: query embedding, document chunking, FAISS, Chroma, context injection
* Tools & agents: ReAct, Toolformer, AutoGPT
* Use cases: QA, summarization, code generation, dialog systems
* Safety & guardrails: hallucination detection, content filtering, Constitutional AI

### 🔗 Key Papers:

* InstructGPT (Ouyang et al., 2022)
* DPO (Rafailov et al., 2023)
* RAG (Lewis et al., 2020)
* Atlas (Izacard et al., 2022)
* Constitutional AI (Anthropic, 2022)

---
