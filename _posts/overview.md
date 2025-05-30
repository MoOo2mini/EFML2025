---
layout: distill
title: The Effect of Batch Size in LoRA Training.
description: TODO.



date: 2025-05-29
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Sangyoon Lee
    url: https://sangyoon-lee99.github.io/
    affiliations:
      name: Postech GSAI
 
# must be the exact same name as your blogpost
bibliography: overview.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Pre-training of Foundation Adapters
  - name: Experiments
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## Introduction

Low-Rank Adaptation (LoRA) has emerged as a widely adopted technique for efficiently fine-tuning large language models (LLMs) by injecting lightweight trainable low-rank matrices into the model's weights. Despite its growing popularity, the optimal training configurations for LoRA, particularly the role of batch size, remain underexplored. This presents a challenge in real-world scenarios, where LoRA is often used in resource-constrained environments that demand quick, reliable hyperparameter choices without exhaustive tuning.

Compounding this issue, recent LoRA variants such as PiSSA and MiLoRA propose seemingly contradictory initialization strategies (principal vs. minor singular components), making it difficult to discern best practices since each work uses different experimental setups. However, these findings are based on divergent experimental setups, making it difficult to draw consistent conclusions or establish best practices. This lack of standardization contributes to performance discrepancies across studies, obscuring the true impact of design decisions.

In this post, we explore how batch size influences the training of LoRA-based methods, and outline two promising research directions to advance our understanding and methodology:

1. Main Finding
2. Main Message

## Motivation

Figrue 1. 기본 PiSSA config.에서의 PiSSA, MiLoRA, LoRA results

We fine-tuned <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Llama-2-7B</a> using LoRA and its recent variants, PiSSA and MiLoRA, adopting the hyperparameter configuration from ? et al. (XXXX). Figure 1 demonstrates the test accuracy on HumanEval benchmark after training on the Code-Feedback dataset for a single epoch. While ? et al. (XXXX) achieves strong performance under their original settings, we observe that MiLoRA outperforms other methods when simply reducing the batch size. This result highlights a key insight: training effectiveness is highly sensitive to configuration choices, especially batch size.

To demystify the impact of batch size on fine-tuning with low-rank adaptation, we focus on the following two questions:

1. How does batch size affect the training dynamics of LoRA-based methods when paired with an optimally tuned learning rate?
2. Given a fixed data budget, how can we select the batch size that yields the best performance?
   
Through this lens, we aim to clarify the relationship between batch size and fine-tuning efficacy in LoRA-style adaptation methods.

## Background

### General Effect of Batch Size

In traditional mini-batch stochastic gradient descent (SGD), batch size plays a critical role in the trade-off between training speed, model generalization, and computational efficiency.

Smaller batches tend to provide noisier but more frequent gradient updates. This noise introduced by smaller batches acts as a form of regularization, allowing it to explore the loss landscape more robustly and avoid over-fitting to the training data. Despite each steps being based on fewer examples, frequent updates indicate that the model's parameters are adjusted more often, potentially leading to faster convergence in terms of epochs. However, increasing in update steps can result in longer total training in terms of wall-clock time.

Larger batches provide more stable and accurate gradients by aggregating more information per update, which typically leads to fewer steps for convergence. Each gradient update becomes closer to the true direction of descent for the data distribution, allowing the model to make more optimal progress with each step. However, large batch training comes with its own challenges. Larger batches often require proportionally higher learning rates to maintain step size. Without increasing the learning rate, large batches can result in smaller steps and slow down convergence. Moreover, very large batches, approaching full-batch gradient descent, are known to risk converging to sharper minima in the loss landscape, which generalize poorly to unseen data. This phenomenon has been observed empirically by Keskar et al. (2017), with showing significant performance drops up to a 5% lower test accuracy when using excessively large batches.

Therefore, finding the optimal batch size is key to balancing the trade-off between computational efficiency and generalization. This has prompted extensive research in the deep learning community, particularly regarding the scaling of batch sizes and their effects on training dynamics. A critical element of this research is the concept of Critical Batch Size (CBS), which refers to the batch size beyond which increasing the size no longer significantly reduces the number of training steps. This area of study is critical because choosing the right batch size can significantly impact both the speed of training and the final performance of the model. The ability to train large models efficiently while maintaining or improving generalization is a central challenge, and ongoing research continues to refine strategies for batch size selection and optimization.

### Interplay with LoRA

LoRA variants 발전 + PiSSA & MiLoRA 간략 설명

이건 GPT에서 추출 정리

LoRA기 때문에 다른 점 (고려사항) ex) LoRA is often used in resource-constrained environments that demand quick, reliable hyperparameter choices without exhaustive tuning.기 때문에 관련 연구가 있다거나
Fine tuning 상황

그리고 아직 관련 연구가 부족하다.

## The Role of Batch Size in LoRA Training

### Experimental setup 

In all our experiments, we adopt <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Llama-2-7B</a> as the backbone for fine-tuning across a range of natural language understanding (NLU) downstream tasks.

We explore batch sizes in the range of $${2^2, 2^3, 2^4, 2^6, 2^7}$$. Since Figure 1 only modifies the batch size of training, the result is somewhat misleading while the proper learning rate is a critical factor for varying batch size training. To througly analyze the sole impact of the batch size, we search for the optimal learning rate for each batch size in $${1e-3, 3e-4, 5e-5, 2e-5, 5e-6, 1e-6}$$, grounded in commonly used configurations across recent PEFT literatures. 

We conduct all experiments using the Hugging Face transformers and peft libraries, and monitor training dynamics such as convergence speed and final performance under each configuration

### 뭐라하지1

Figure X. test accuracy for same epoch

In Figure X, we compare the test performance of LoRA and its recent variants, PiSSA and MiLoRA, across a range of batch sizes, each paired with an optimally tuned learning rate. We observe that batch size has a substantial impact on LoRA based methods, with accuracy fluctuations of up to X% on the ? task. Notably, when the batch size is properly configured, vanilla LoRA matches or even outperforms other complex methods.

This suggests that the performance gap previously attributed to architectural improvements may, in part, stem from suboptimal training configurations. Our findings highlight that standard LoRA, without any structural changes, remains a strong baseline, so long as traditional hyperparameters like batch size and learning rate are carefully tuned. Appendix 언급 + Table..?

### 뭐라하지2

Figure W. LoRA-4to512_various_tasks

## Conclusion

## Apendix..?

동일 step 비교

learning rate 별 plot
