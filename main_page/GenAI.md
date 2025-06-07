[![Home](https://img.shields.io/badge/Home-Click%20Here-blue?style=flat&logo=homeadvisor&logoColor=white)](../)

<br>


## [![ControlNet](https://img.shields.io/badge/ControlNet-Adding_Conditional_Control_to_Text_to_Image_Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/ControlNet)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
ControlNet is an advanced extension of diffusion models that introduces additional control mechanisms, allowing for precise guidance over the generation process. By integrating control signals (e.g., edge maps, segmentation masks, poses), ControlNet enables the generation of images that adhere to specific structural or semantic constraints provided by the user.
<p></p>
</div>


## [![CFG](https://img.shields.io/badge/CFD-Classifier_Free_Diffusion-blue?style=for-the-badge&logo=github)](../posts/CFG)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Classifier-Free Diffusion is a powerful technique in generative modeling, particularly within diffusion models, that enhances the quality and controllability of generated outputs without relying on an external classifier. This comprehensive guide will delve into the intricacies of classifier-free diffusion, covering its principles, training and inference processes, intuitive explanations, and practical implementations in tasks like image inpainting, super-resolution, and text-to-image generation.
<p></p>
</div>



## [![Classifier-Guided Diffusion](https://img.shields.io/badge/CGD-Classifier_Guided_Diffusion-blue?style=for-the-badge&logo=github)](../posts/Classifier_Guided_Diffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Classifier-guided diffusion is a powerful technique in generative modeling that leverages an external classifier to steer the generation process toward desired attributes or classes. This method enhances the quality and controllability of generated data, such as images, by integrating class-specific information during the diffusion process.
<p></p>
</div>


## [![Latent Diffusion](https://img.shields.io/badge/LDM-Latent_Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/StableDiffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Stable Diffusion is a powerful generative model that synthesizes high-quality images guided by textual/another modality descriptions. It leverages the strengths of Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs) to produce images efficiently and effectively.
 <p></p>
</div>

## [![NCSN](https://img.shields.io/badge/NCSN-Noise_Conditional_Score_Networks/Score_Based_Generative_Models-blue?style=for-the-badge&logo=github)](../posts/NCSN)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
A **Score-based Generative Model** learns the gradient of the log-probability (the “score”) for a family of **noisy** versions of data. Instead of directly learning a generative model $ p(x) $, we train a network $ s_\theta(x, \sigma) $ that approximates:

$$
\nabla_x \log p_\sigma(x) \quad \text{where} \quad p_\sigma(x)
$$

is the distribution of data *corrupted* by noise of scale $\sigma$. Once we learn a good approximation of the score $\nabla_x \log p_\sigma(x)$, we can sample from the (clean) distribution by **progressively denoising** data using **Langevin dynamics** (or an equivalent Stochastic Differential Equation).
<p></p>
</div>


## [![Diffusion Models DDPM](https://img.shields.io/badge/DDPM-Diffusion_Models-blue?style=for-the-badge&logo=github)](../posts/Diffusion)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
In this guide, we'll provide sample code for training and inference of a diffusion model, specifically focusing on a Denoising Diffusion Probabilistic Model (DDPM). We'll define the structure for the encoder and decoder using a simplified UNet architecture. Each line of code includes inline comments explaining its purpose, along with the tensor shapes.
 <p></p>
</div>


## [![GLIP](https://img.shields.io/badge/GLIP-Grounded_Language_Image_Pre_training-blue?style=for-the-badge&logo=github)](../posts/GLIP)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
GLIP (Grounded Language-Image Pre-training) is a unified model architecture that bridges the gap between vision and language by integrating object detection and phrase grounding tasks. It leverages both visual and textual data to perform object detection conditioned on textual descriptions, enabling the model to recognize objects based on their semantic meanings.
<p></p>
_Last updated: {{ site.time | date: "%B %d, %Y" }}_
</div>


## [![CLIP](https://img.shields.io/badge/CLIP-Learning_Transferable_Visual_Models_From_Natural_Language_Supervision-blue?style=for-the-badge&logo=github)](../posts/CLIP)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Learning Transferable Visual Models From Natural Language Supervision" is a groundbreaking paper by OpenAI that introduces CLIP (Contrastive Language-Image Pre-training). CLIP learns visual concepts from natural language supervision by jointly training an image encoder and a text encoder to predict the correct pairings of images and texts.
<p></p>
</div>


## [![texttoimage](https://img.shields.io/badge/Text_to_Image-grey?style=for-the-badge&logo=github)](../posts/TextToImage)

<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
notable text-to-image generation models along with their corresponding research papers, sorted by the year they were published:
 <p></p>
</div>

## [![LORA](https://img.shields.io/badge/LORA-Low_Rank_Adaptation-blue?style=for-the-badge&logo=github)](../posts/LORA)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large-scale pre-trained models. It allows us to adapt a model by introducing low-rank trainable matrices into certain parts of the network while keeping the original pre-trained weights frozen. <p></p>
</div>

## [![EVALUATION](https://img.shields.io/badge/Evaluation-Evaluation_Metrics_for_LLMs-blue?style=for-the-badge&logo=github)](../posts/EvaluationMetricsLLM)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
Language models are evaluated across diverse tasks such as next-token prediction, text classification, summarization, translation, code generation, and question answering. Each task requires a suitable metric that reflects model performance both quantitatively and qualitatively.

</div>
## [![BERT](https://img.shields.io/badge/BERT-Bidirectional_Encoder_Representations_from_Transformers-blue?style=for-the-badge&logo=github)](../posts/LLM/BERT)
<div style="background-color: #f0f8ff; color: #555;font-weight: 485; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #ccc;">
BERT (Bidirectional Encoder Representations from Transformers), developed by Google AI in 2018, marked a paradigm shift in natural language processing by introducing deep bidirectional context modeling through the Transformer architecture. Unlike traditional left-to-right or right-to-left language models, BERT reads entire sequences bidirectionally, enabling better understanding of linguistic context.
</div>

