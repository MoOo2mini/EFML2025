---
layout: distill
title: Understanding the Role of Batch Size in Low-Rank Adaptation
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

**Low-Rank Adaptation (LoRA)** has emerged as a widely adopted technique for efficiently fine-tuning large language models (LLMs). By injecting lightweight, trainable low-rank matrices into pretrained weights, LoRA offers a practical solution for adapting massive models without the full cost of end-to-end training. Despite its growing popularity, its sensitivity to key training hyperparameters, especially **batch size**, remains underexplored. This presents a challenge in real-world scenarios, where LoRA is often used in resource-constrained environments that demand quick, reliable hyperparameter choices without exhaustive tuning.

Complicating matters further, recent LoRA variants such as PiSSA and MiLoRA propose seemingly contradictory initialization strategies (principal vs. minor singular components), yet each work reports gain based on different experimental setups. This lack of consistency makes it difficult to discern whether observed improvements stem from algorithmic advances or simply from favorable training configurations. As a result, best practices remain unclear, and the actual influence of design choices like initialization and batch size is frequently masked.

In this post, we explore **how batch size influences the training of LoRA-based methods**.

> **Our main contributions are as follows:**
> 
> **1. We show that batch size plays a critical role in LoRA fine-tuning, with up to X% variation in test accuracy depending on its setting.**
> 
> **2. We demonstrate that vanilla LoRA can match or even outperform recent variants like PiSSA and MiLoRA, simply by tuning the batch size appropriately.**
> 
> **3. We uncover non-monotonic trends in LoRA’s performance as batch size increases, underscoring the need for a deeper understanding of its optimization behavior.**


## Motivation

Figrue 1. 기본 PiSSA config.에서의 PiSSA, MiLoRA, LoRA results

We fine-tune <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Llama-2-7B</a> using LoRA and its recent variants, PiSSA and MiLoRA, following the hyperparameter configuration proposed by ? et al. (XXXX). As shown in Figure 1, we evaluate their test accuracy on the HumanEval benchmark after a single epoch of training on the CodeFeedback dataset. While the original configuration favors PiSSA, we find that MiLoRA outperforms other methods simply by reducing the batch size—without any structural modification.

This result highlights a key insight: training effectiveness in LoRA-based methods is highly sensitive to configuration choices, particularly batch size. Yet, these factors are often overlooked in comparative studies.

To demystify the impact of batch size, we focus on following two key questions:

1. How does batch size affect the LoRA training dynamics when paired with an optimally tuned learning rate?
2. Under a fixed data budget, how can we choose a batch size that baalnces perfomrance and efficiency?
   
Through this lens, we aim to reveal the underappreciated role of batch size in LoRA-based adaptation and provide practical guidance for future studies and real-world deployments.

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

For all experiments, we adopt <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Llama-2-7B</a> as the backbone for fine-tuning across a diverse set of natural language understanding (NLU) downstream tasks:

- GSM8K: A benchmark for grade-school level mathematical reasoning and arithmetic problems.

- MATH: A challenging dataset focused on high school and competition-level math problems.

- HumanEval: A code generation benchmark where the model is asked to complete Python functions.

- MBPP: A dataset for Python function generation based on simple problem descriptions.

- MT-Bench: A multi-turn instruction-following evaluation, automatically scored using Gemini?.

To examine the impact of batch size, we evaluate LoRA and its variants across a range of values $$\{4, 8, 16, 64, 128, 512\}$$. While Figure 1 varies only the batch size during training, such results can be misleading, learning rate is a critical co-factor, especially when changing the batch size. To isolate the effect of batch size itself, we conduct a small grid search for the optimal learning rate for each batch size, selecting from $$\{1e-3, 3e-4, 5e-5, 2e-5, 5e-6, 1e-6\}$$, based on commonly used ranges in recent PEFT literature.
Since Figure 1 only modifies the batch size of training, the result is somewhat misleading while the proper learning rate is a critical factor for varying batch size training. To througly analyze the sole impact of the batch size, we search for the optimal learning rate for each batch size in $${1e-3, 3e-4, 5e-5, 2e-5, 5e-6, 1e-6}$$, grounded in commonly used configurations across recent PEFT literatures. 

### 뭐라하지1

Figure X. test accuracy for same epoch

In Figure X, we compare the test performance of LoRA and its recent variants, PiSSA and MiLoRA, across a range of batch sizes, each paired with an optimally tuned learning rate. We observe that batch size has a substantial impact on LoRA based methods, with accuracy fluctuations of up to X% on the ? task. Notably, when the batch size is properly configured, vanilla LoRA matches or even outperforms other complex methods.

This suggests that the performance gap previously attributed to architectural improvements may, in part, stem from suboptimal training configurations. Our findings highlight that standard LoRA, without any structural changes, remains a strong baseline, so long as traditional hyperparameters like batch size and learning rate are carefully tuned. Appendix 언급 + Table.. (추가할까 아니면 애시당초 table로 깔까)?

### 뭐라하지2

Figure W. LoRA-4to512_various_tasks

To better understand the overall impact of batch size, we extend our analysis by evaluating LoRA across a broader range, from asmall to exteremely large batch sizes (up to 512). Surprisingly, we observe a non-monotonic trend in test accuracy: performance initially drops, then rises unexectedly at intermediate batch sizes. before degrading again at the largest setting.

This unpredictable behavior makes it challenging to select a batch size that balances training efficiency and generalization performance. Our findings suggest that larger batch sizes do not always lead to worse performance, while they show poor performance in the very large batch. This underscores the need for a deeper understanding of how batch size interacts with optimization dynamics, and highlights the importance of careful configuration tuning in LoRA-based training pipelines. 여기도 Appendix 연결 무언가

## Conclusion

In this study, we revisit the performance of LoRA and its recent variants using the LLaMA-2-7B model, with a particular focus on the often-overlooked role of batch size. Our findings reveal two key insights:

**First**, when the batch size is properly tuned, vanilla LoRA can achieve performance on par with, or even better than, its variants, without requiring any structural modifications. **Second**, we observe that the relationship between batch size and test performance is non-monotonic: performance increases at intermediate batch sizes but drops at larger ones, forming a nonlinear pattern that defies standard assumptions. These observations highlight the importance of carefully controlling for hyperparameter configurations when evaluating new parameter-efficient fine-tuning (PEFT) methods. Without such control, reported improvements may reflect differences in tuning rather than actual algorithmic gains.

Finally, the unexpected fluctuations in performance across batch sizes suggest that subtle dynamics in optimization, which requires further theoretical investigation and large-scale empirical validation. We hope this work encourages the community to revisit current evaluation practices and adopt more rigorous, configuration-aware comparisons when assessing PEFT techniques.

## Apendix..?

동일 step 비교

learning rate 별 plot

위에 안 넣은 task에 대한 plot
