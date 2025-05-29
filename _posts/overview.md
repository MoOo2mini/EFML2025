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

기본 PiSSA config.에서의 PiSSA, MiLoRA, LoRA results

## Background

### General Effect of Batch Size

In traditional SGD-based training, batch size involves a well-known trade-off:

Smaller batches provide noisier but more frequent gradient updates. This noise can act as a form of regularization, potentially helping the model generalize by avoiding over-fitting to sharp minima. Frequent updates mean the model’s parameters are adjusted more times per epoch, which can lead to faster convergence in terms of number of epochs (though each epoch sees fewer examples per update).

Larger batches yield more stable, accurate gradients (approaching full-batch gradient descent). This can converge in fewer steps since each step is a nearly optimal descent direction for the current data distribution. However, large batches often require a proportionally higher learning rate to maintain step size – otherwise each update, though precise, may move parameters only slightly. If the learning rate is not tuned, large batches can converge slowly in terms of wall-clock time or get stuck in suboptimal minima. Additionally, very large batches are known to risk converging to sharper minima that generalize poorly. Empirically, training with extremely large batches has shown test accuracy drops (a generalization gap), compared to using smaller batches. The underlying reason is that small-batch methods tend to find “flatter” minima (with many small eigenvalues of the Hessian), whereas large-batch methods can overfit to a narrow basin of the loss surface. These sharp minima fit the training data well but don’t transfer as robustly to test data. This phenomenon was documented by Keskar et al. (2017), who observed up to 5% lower test accuracy when using large batches, and linked it to the sharpness of the solutions.

Large batch training 관련
Accurate mini batch, Data Parallelism, Critical Batch Size

### Interplay with LoRA

이건 GPT에서 추출 정리

## Experiments

### Experimental setup 

In all experiments, we use [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) as the baseline model. We compare 3 different methods (LoRA, PiSSA, MiLoRA) on various downstream tasks.
itemize로 설명..?

We conduct our experiments on 

## Conclusion

## Apendix..?
