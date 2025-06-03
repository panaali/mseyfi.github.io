Here’s your **complete 6-week study plan** for learning LLMs from the ground up — **topics**, **papers**, **GitHub repos**, and **goals** are all included. Each week focuses on mastering a coherent module of LLM development, tuning, evaluation, and deployment, optimized for your deep learning background.

---

# 📅 Full 6-Week Calendar to Learn LLMs

---

## ✅ WEEK 1: Transformer Decoder, Tokenization, and Causal LM

### 🎯 Goal: Understand how GPT-style models represent and process text.

### 📘 Topics:

* Transformer decoder stack: self-attention, feed-forward, residual + LayerNorm
* Causal attention mask
* Position encodings: sinusoidal vs learned
* Byte-Pair Encoding (BPE), WordPiece, SentencePiece
* Tokenizer input/output formats
* Special tokens (PAD, BOS, EOS, CLS)
* Autoregressive language modeling objective
* Perplexity definition and cross-entropy loss

### 🔗 Key Papers:

* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### 💻 GitHub Repos:

* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [minGPT](https://github.com/karpathy/minGPT)
* [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
* [tiktoken](https://github.com/openai/tiktoken)

---

## ✅ WEEK 2: LLM Pretraining and Sampling Strategies

### 🎯 Goal: Learn how LLMs are pretrained on large corpora and how text is generated.

### 📘 Topics:

* Pretraining datasets: C4, The Pile, BookCorpus, Common Crawl
* Optimizer: AdamW, LR schedules, weight decay
* Gradient checkpointing, FSDP, ZeRO
* Sequence truncation and chunking
* Context window, batch size, masking mechanics
* Sampling techniques:

  * Greedy
  * Beam search
  * Top-k, top-p (nucleus)
  * Temperature
  * Contrastive decoding

### 🔗 Key Papers:

* [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
* [Better Language Models (GPT-2)](https://openai.com/research/better-language-models)

### 💻 GitHub Repos:

* [DeepSpeed Examples – ChatGPT training](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* [HuggingFace Transformers Trainer](https://github.com/huggingface/transformers)
* [Text generation sampling demo](https://huggingface.co/blog/how-to-generate)

---

## ✅ WEEK 3: Evaluation Metrics and Benchmarks

### 🎯 Goal: Evaluate LLM quality across generation, classification, and reasoning tasks.

### 📘 Topics:

* Perplexity, cross-entropy
* BLEU (translation), ROUGE (summarization), METEOR
* BERTScore (semantic similarity)
* Exact Match, F1 (for QA)
* Pass\@k (code generation)
* HumanEval, MT-Bench
* Benchmarks:

  * MMLU
  * TruthfulQA
  * HellaSwag
  * ARC
  * GSM8K

### 🔗 Key Papers:

* [BLEU](https://aclanthology.org/P02-1040/)
* [ROUGE](https://aclanthology.org/W04-1013/)
* [BERTScore](https://arxiv.org/abs/1904.09675)
* [MMLU Benchmark](https://arxiv.org/abs/2009.03300)

### 💻 GitHub Repos:

* [nlp-metrics (nlg-eval)](https://github.com/Maluuba/nlg-eval)
* [BERTScore](https://github.com/Tiiiger/bert_score)
* [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

---

## ✅ WEEK 4: Fine-Tuning, Instruction Tuning, LoRA, QLoRA

### 🎯 Goal: Adapt pretrained LLMs to downstream tasks with limited compute.

### 📘 Topics:

* Supervised instruction tuning (e.g. FLAN, Alpaca)
* Prompt–response dataset structure
* Hyperparameter tuning for finetuning
* Parameter-efficient tuning:

  * LoRA (Low-Rank Adaptation)
  * QLoRA (4-bit training)
  * Prefix tuning
  * BitFit
* Evaluation of finetuned models
* Data formatting: prompt engineering for finetuning

### 🔗 Key Papers:

* [Self-Instruct](https://arxiv.org/abs/2212.10560)
* [LoRA](https://arxiv.org/abs/2106.09685)
* [QLoRA](https://arxiv.org/abs/2305.14314)

### 💻 GitHub Repos:

* [HuggingFace PEFT](https://github.com/huggingface/peft)
* [alpaca-lora](https://github.com/tloen/alpaca-lora)
* [QLoRA (artidoro)](https://github.com/artidoro/qlora)
* [FastChat](https://github.com/lm-sys/FastChat)

---

## ✅ WEEK 5: Inference Optimization + Prompt Engineering

### 🎯 Goal: Serve LLMs efficiently and improve zero-shot/few-shot capabilities.

### 📘 Topics:

* Prompting:

  * Zero-shot, few-shot, chain-of-thought (CoT)
  * Tool-use, self-refinement, scratchpad prompting
* Quantization:

  * GPTQ (INT4)
  * AWQ (asymmetric quantization)
  * LLM.int8()
* KV caching
* FlashAttention, Rotary Positional Embedding (RoPE)
* Efficient inference libraries (vLLM, TensorRT-LLM)

### 🔗 Key Papers:

* [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
* [LLM.int8()](https://arxiv.org/abs/2208.07339)
* [FlashAttention](https://arxiv.org/abs/2205.14135)

### 💻 GitHub Repos:

* [vLLM](https://github.com/vllm-project/vllm)
* [GPTQ](https://github.com/IST-DASLab/gptq)
* [AWQ](https://github.com/mit-han-lab/llm-awq)
* [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
* [LangChain](https://github.com/langchain-ai/langchain)

---

## ✅ WEEK 6: RLHF + Applications (QA, Summarization, Coding, Agents)

### 🎯 Goal: Understand how models are aligned with human preferences and deployed in real-world tasks.

### 📘 Topics:

* RLHF:

  * Reward model training
  * Preference pair generation
  * PPO algorithm
  * DPO (Direct Preference Optimization)
* Constitutional AI (Anthropic)
* Use cases:

  * Summarization
  * Question Answering
  * Retrieval Augmented Generation (RAG)
  * Code generation (e.g., HumanEval, CodeT)
  * Multi-modal LLMs (BLIP, Flamingo, LLaVA – optional)

### 🔗 Key Papers:

* [InstructGPT](https://arxiv.org/abs/2203.02155)
* [DPO](https://arxiv.org/abs/2305.18290)
* [Constitutional AI](https://arxiv.org/abs/2212.08073)

### 💻 GitHub Repos:

* [trlx](https://github.com/CarperAI/trlx)
* [HuggingFace TRL](https://github.com/huggingface/trl)
* [Axolotl (LoRA + RLHF trainer)](https://github.com/OpenAccess-AI-Collective/axolotl)
* [OpenChatKit](https://github.com/togethercomputer/OpenChatKit)

---
