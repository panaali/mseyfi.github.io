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
    
