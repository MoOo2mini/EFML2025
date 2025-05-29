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

Low-Rank Adaptation (LoRA) has become a popular method for fine-tuning large language models (LLMs) efficiently by injecting small trainable low-rank matrices into the model’s weights. While LoRA and its variants show impressive performance, the optimal training settings for LoRA still remain under-explored, particularly the impact of batch size. This is particularly problematic in practical settings, as LoRA is frequently employed in resource-constrained environments where practitioners must make fast and effective hyperparameter decisions without exhaustive tuning. Moreover, recent LoRA variants like PiSSA and MiLoRA propose seemingly contradictory initialization strategies (principal vs. minor singular components), making it difficult to discern best practices since each work uses different experimental setups. This lack of standardization contributes to performance discrepancies across studies, obscuring the true impact of design decisions. From this view, we aim to demystify the following question: 
“How batch size affects the training dynamics of LoRA-based methods”

In this post, we provides an insights on batch size in LoRA fine-tuning, and outlines two promising research directions to advance our understanding and methodology:

1. Main Finding
2. Main Message


## Pre-training of Foundation Adapters

Given a "frozen" pre-trained LLM $$S$$ with trainable adapters $$A$$ attached to each of its block layers, our research question is: *How can we pre-train the adapters $$A$$*? We may follow a previous approach <d-cite key="gunter2024appleintelligencefoundationlanguage,cui2024efficienteffectivetextencoding"></d-cite> to perform continual pre-training with $$S$$ to learn $$A$$. However, we further extend the previous approach with classical knowledge distillation <d-cite key="wu2024rethinkingkullbackleiblerdivergenceknowledge,muralidharan2024compactlanguagemodelspruning"></d-cite>. In particular, we introduce a joint training method to pre-train the adapters. The method comprises two key modules: knowledge distillation and continual pre-training, as illustrated in Figure 1.

- Knowledge distillation (KD): A larger, frozen pre-trained LLM is employed as the teacher model to facilitate knowledge distillation, transferring its knowledge to a smaller student model that consists of the frozen $$S$$ augmented with trainable adapters $$A$$. Specifically, the Kullback-Leibler divergence is used to measure the difference between the LM head logits of the teacher and student models, resulting in a corresponding knowledge distillation loss $$\mathcal{L}_{\text{KD}}$$ during training, which guides the student to mimic the teacher’s output distributions.


- Continual pre-training (CPT): We continually pre-train the student model (here, $$S$$ with adapters $$A$$) using a causal language modeling objective. By keeping all the original weights of $$S$$ frozen and updating only the adapters' weights, we efficiently adapt the model to new data without overwriting its existing knowledge. A corresponding cross-entropy loss $$\mathcal{L}_{\text{LM}}$$ is computed based on the LM head logit outputs of the student model during training.


- Joint training:  The knowledge distillation and continual pre-training tasks are jointly optimized on a text corpus, with the final objective loss computed as a linear combination of the knowledge distillation loss and the cross-entropy loss: $$\mathcal{L} = \alpha\mathcal{L}_{\text{KD}} + (1 - \alpha)\mathcal{L}_{\text{LM}}$$.



{% include figure.html path="assets/img/2025-04-28-foundation-adapter/Model.png" class="img-fluid" %}
<div class="caption">
    Figure 1. Illustration of our joint approach for pre-training foundation adapters.
</div>



## Experiments

### General setup 

In all experiments, we use Low-Rank Adaptation (LoRA) <d-cite key="hu2022lora"></d-cite> exclusively as our adapters, applying them across all linear layers in the model architecture. In addition, we set the mixture weight $$\alpha$$ in the final loss equation to 0.5, i.e. $$\mathcal{L} = 0.5\mathcal{L}_{\text{KD}} + 0.5\mathcal{L}_{\text{LM}}$$.


In  a preliminary experiment, we perform continual pre-training using the "frozen" [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) as the student model with LoRA rank 8 on the [QuRatedPajama-260B](https://huggingface.co/datasets/princeton-nlp/QuRatedPajama-260B) dataset <d-cite key="pmlr-v235-wettig24a"></d-cite>. We find that the loss and evaluation scores converge at 5B training tokens. Therefore, we use a subset of 5B tokens from QuRatedPajama-260B as our pre-training data for all pre-training settings. 

### Knowledge distillation helps improve performance


{% include figure.html path="assets/img/2025-04-28-foundation-adapter/Ablation.png" class="img-fluid" %}
<div class="caption">
    Table 1. Ablation results with the teacher model Llama-3.1-8B-Instruct. CPT and KD denote continual pre-training and knowledge distillation, respectively. With a LoRA rank of 8, the total number of adapter parameters is 5.64 million.
</div>

To investigate the effectiveness of knowledge distillation, our initial experiments utilize the student model [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) with trainable LoRA of rank 8 and employ [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) <d-cite key="dubey2024llama3herdmodels"></d-cite> as the teacher model. We explore two scenarios: first, performing only CPT (i.e., the loss function is $$\mathcal{L} = \mathcal{L}_{\text{LM}}$$), and second, combining CPT with KD.


Table 1 presents results obtained on 6 key Open LLM benchmarks using the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), including AI2 Reasoning Challenge (ARC; 25-shot) <d-cite key="allenaiarc"></d-cite>, HellaSwag (10-shot) <d-cite key="zellers-etal-2019-hellaswag"></d-cite>, MMLU (5-shot) <d-cite key="hendrycks2021measuring"></d-cite>, TruthfulQA (0-shot) <d-cite key="lin-etal-2022-truthfulqa"></d-cite>, Winogrande (5-shot) <d-cite key="WinoGrande"></d-cite> and GSM8k (5-shot) <d-cite key="cobbe2021trainingverifierssolvemath"></d-cite>. 

Compared to the baseline model Llama-3.2-1B, applying CPT alone reveals small performance changes in most benchmarks, with an average increase from 40.55 to 40.97. This improvement is primarily due to increases in TruthfulQA (from 37.58 to 39.35) and Winogrande (from 62.43 to 63.69), while the remaining benchmarks exhibit negligible or negative changes. In contrast, the model with both CPT and KD demonstrates a more substantial improvement over applying CPT alone. This is evident in the increased average score of 41.62, driven by improvements in MMLU, TruthfulQA, and particularly GSM8K, which increases from 6.90 to 8.87. These results suggest that combining CPT with KD yields more comprehensive performance improvements across multiple tasks.




### Effects of different LoRA ranks


{% include figure.html path="assets/img/2025-04-28-foundation-adapter/Ranks2.png" class="img-fluid" %}
<div class="caption">
    Table 2. Experimental results with the teacher model Llama-3.1-8B-Instruct regarding different LoRA ranks. The total number of adapter parameters varies across ranks: 22.54 million parameters for a LoRA rank of 32; 45.09 million parameters for a LoRA rank of 64; and 90.17 million parameters for a LoRA rank of 128.
</div>


We study the effects of different LoRA ranks—8, 32, 64, and 128—using the student model [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and the teacher model Llama-3.1-8B-Instruct, applying both CPT and KD. Table 2 shows the obtained results on the 6 key Open LLM benchmarks, revealing improvements of over 1.0 points across all ranks compared to the baseline Llama-3.2-1B-Instruct. The model with a LoRA rank of 64 stands out as the best performer, achieving an average score of 48.95 and excelling in the GSM8K benchmark with a score of 37.91. Notably, the overall average scores—48.50, 48.56, 48.95, and 48.62 for ranks 8, 32, 64, and 128, respectively—show minimal variation, indicating that changes in LoRA rank have little impact on performance differences. Benchmarks such as ARC and Winogrande also exhibit only slight fluctuations, while small improvements in GSM8K from rank 8 to 64 are not significant enough to indicate a clear advantage for any particular rank.


### Use-case: Summarization

{% include figure.html path="assets/img/2025-04-28-foundation-adapter/RougeL.png" class="img-fluid" %}
<div class="caption">
    Figure 2. ROUGE-L scores on the QMSum test set with different numbers of training examples. "Random" and "Foundation" indicate that when fine-tuning, the LoRA weights are either randomly initialized or initialized using the pre-trained foundation LoRA rank 64.
</div>

We also examine the effectiveness of pre-trained foundation LoRA adapters for downstream task fine-tuning, specifically supervised fine-tuning. We utilize the base LLM [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and employ the [QMSum](https://github.com/Yale-LILY/QMSum) dataset <d-cite key="zhong-etal-2021-qmsum"></d-cite>, focusing on query-specific summarization with 1,095 training examples. We compare two strategies for LoRA-based LLM fine-tuning: (1) Random initialization, where the LoRA weights are randomly initialized, and (2) Foundation initialization, where the LoRA weights are initialized using the pre-trained foundation LoRA rank 64 from previous experiments exploring the "effects of different LoRA ranks."

Figure 2 presents the ROUGE-L results on the QMSum  test set, when fine-tuning with varying numbers of training examples at 1095 (i.e. using the full QMSum training set), 548 (i.e. one half of the training set), 274 (i.e. one-fourth of the training set) and 137 (i.e. one-eighth of the training set). When the number of training examples is 0, the ROUGE-L score of 15.73​ reflects the baseline model's performance with zero-shot prompting. Clearly, fine-tuning significantly improves the summarization score, even with just over 100 training examples. Notably, using pre-trained LoRA weights for initialization consistently outperforms the random initialization across all training sizes, clearly demonstrating the effectiveness of foundation initialization from pre-trained LoRA for this specific use case.

## Conclusion

In this blog post, we present a joint training approach that combines continual pre-training and knowledge distillation to pre-train foundation adapters for adapter weight initialization in LLM fine-tuning. Our experiments demonstrate that this approach achieves performance improvements across multiple tasks. Especially, we show that for a specific use case in summarization, using weight initialization from a pre-trained foundation LoRA enhances performance compared to random initialization. In future work, we plan to pre-train LoRA adapters for various models and evaluate these pre-trained adapters on additional downstream tasks.

