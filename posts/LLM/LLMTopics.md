1- what is Tokenizer, what are different tokenizers?

2- Embedding 

3- detokenizatin

4-Transformer architechtures:
  - Encoder only models (BERT) sequence-to-sequence, (input output same size) or use case: sentiment, classification
  - Encoder-decoder models sequence-to-sequence with different length (BArt, T5)
  - Decoder only architechtures (Jurassic, Lama, GPT, BLUE)


- text you enter to the model is prompt
- output of the model is completion

- Prompt ---> inference ----> completion
- context window: full amount of text available to memory

5- In context Learning and zero/one/few shot inference

6- greedy(select the word according to the softmax highest probability) vs random sampling text generation (select a text randomly based on the softmax probability- if a word has 2% chance it can still be selected however in the greedy case there is no way a word with less maximum softmax probability be chosen)

7- top-k and top-p sampling and temperature

8- Encoder only models (BERT, ROBERTA) Autoencoding models ,  
  - masked languege modeling (MLM)
  - mask a word and predict a word (self supervisedly)
  - use bidirectional context, meaning both the past and the future words can take part to predicting the missing word.  - 
  - used for Named Entity Recognition)
  - Sentiment analysis
  - token/sentence classification
    -BERT and ROBERTA

9- Decoder only (GPT, Lamma) or autoregressive models (Causal Language modeling ) --> Text prediction from previsous tokens. 
The model has no knowledge of the end of the sentence.Unidirectional context, very good in zero shot inference

10- Encoder-Decoder models(T5, Bart) summarization, translation, question answering   
  - pretrain using span corruption
  - Sentinel tokens
# computational Challenges of LLMs
BFLOAT16 and FP16
Model Sharding FSDP
Chinchilla Rule 
## Finetuning

ICL (in-contect learning zeor/one/few shot learning doesnt work for smaller models and may take a lot of space in the context window)

1- Instruction fine-tuning 
  - works on pairs of prompt-completion.
  - For a single task fine-tuning we need around 1K examples.
  - it's a full fine-tuning mechanism that updates all the model weights
  - Finetuning on a single task may cause catastrophic forgetting
  - If model performance drop on other tasks are important, you need to finetune over multiple tasks
    - Often 50K to 100K examples across multiple taks are needed.
  - **PEFT** or parameter efficient fine-tuning only adapt task-specific weights.
  - One important technique to prevent catasrophic forgetting is regularization on gradients norms. Using this technique, we prevent big updates on the backpropagation.
  - ## FLAN Finetuned LAnguage NET.
2- Metrics for evaluating LLMS
  - ## Definitions:
    -  **Unigram** is a single word  
    -  **Bigram** is a two words
    -  **n-gram** is a group of n *consecutive* words
  
  - Rouge: Used for text summarization
    - compares a summary to one or multiple other reference summaries. 
    
    EX:

    Reference: It is cold outside
    
    Generated output: it is very cols outside 
    
    $$
    Rouge-1_{\text{recall}} = \frac{\text{unigram matches}}{\text{unnigrams in reference}}
    $$

    
    $$
    Rouge-1_{\text{precision}} = \frac{\text{unigram matches}}{\text{unnigrams in output}}
    $$

    $$
    Rouge-1_{\text{F1}} = 2.\frac{\text{precision}\times \text{recall}}{\text{precision} + \text{recall}}
    $$
    
  
  - BLEU: Used for translation task
    - Compares translation to human-generated translation   
        
  
