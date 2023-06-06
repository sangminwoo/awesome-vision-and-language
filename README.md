# Awesome Vision-and-Language: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome vision and language resources, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

**Table Of Contents**
* [Survey](#survey)
* [Dataset](#dataset)
* [Image Captioning](#image-captioning)
* [Image Retrieval](#image-retrieval)
* [Scene Text Recognition (OCR)](#scene-text-recognition)
* [Scene Graph](#scene-graph)
* [text2image](#text2image)
* [Video Captioning](#video-captioning)
* [Video Question Answering](#video-question-answering)
* [Video Understanding](#video-understanding)
* [Vision and Language Navigation](#vision-and-language-navigation)
* [Vision and Language Pretraining](#vision-and-language-pretraining)
* [Visual Dialog](#visual-dialog)
* [Visual Grounding](#visual-grounding)
* [Visual Question Answering (VQA)](#visual-question-answering)
* [Visual Reasoning](#visual-reasoning)
* [Visual Relationship Detection](#visual-relationship-detection)
* [Visual Storytelling](#visual-storytelling)

## Survey
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| A Survey of Current Datasets for Vision and Language Research | 2015 EMNLP | [1506.06833](https://arxiv.org/abs/1506.06833) | []() | []() |
| Multimodal Machine Learning: A Survey and Taxonomy |  | [1705.09406](https://arxiv.org/abs/1705.09406) | []() | []() |
| A Comprehensive Survey of Deep Learning for Image Captioning |  | [1810.04020](https://arxiv.org/abs/1810.04020) | []() | []() |
| Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods |  | [1907.09358](https://arxiv.org/abs/1907.09358) | []() | []() |
| A Survey of Scene Graph Generation and Application |  | [Scene-Graph-Survey](https://www.xiaojun.ai/papers/Scene-Graph-Survey.pdf) | []() | []() |
| Challenges and Prospects in Vision and Language Research |  | [1904.09317](https://arxiv.org/abs/1904.09317) | []() | []() |
| Deep Multimodal Representation Learning: A Survey | 2019 ACCESS | [ACCESS 2019](https://ieeexplore.ieee.org/document/8715409) | []() | []() |
| Multimodal Intelligence: Representation Learning, Information Fusion, and Applications |  | [1911.03977](https://arxiv.org/abs/1911.03977) | []() | []() |
| Vision and Language: from Visual Perception to Content Creation | 2020 APSIPA | [1912.11872](https://arxiv.org/abs/1912.11872) | []() | []() |
| Multimodal Research in Vision and Language: A Review of Current and Emerging Trends |  | [2010.09522](https://arxiv.org/abs/2010.09522) | []() | []() |
|  |  | []() | []() | []() |

## Dataset

| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| VQA: Visual Question Answering              | 2015 ICCV  | [1505.00468](https://arxiv.org/abs/1505.00468) |  | [visualqa](https://visualqa.org/) |
| Visual Storytelling | 2016 NAACL  | [1604.03968](https://arxiv.org/abs/1604.03968) | [ai-visual-storytelling-seq2seq](https://github.com/ai-visual-storytelling-seq2seq) | [VIST](http://visionandlanguage.net/VIST/) |
| Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations | 2017 IJCV  | [1602.07332](https://arxiv.org/abs/1602.07332) | [visual_genome_python_driver](https://github.com/ranjaykrishna/visual_genome_python_driver) | [visualgenome](https://visualgenome.org/) |
| CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning | 2017 CVPR | [1612.06890](https://arxiv.org/abs/1612.06890) | []() | []() |
| AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions | 2018 CVPR | [1705.08421](https://arxiv.org/abs/1705.08421) |  | [AVA](http://thoth.inrialpes.fr/ava/) |
| Embodied Question Answering | 2018 CVPR | [1711.11543](https://arxiv.org/abs/1711.11543) | []() | [embodiedqa](https://embodiedqa.org/) |
| Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments | 2018 CVPR | [1711.07280](https://arxiv.org/abs/1711.07280) | []() | [bringmeaspoon](https://bringmeaspoon.org/) |
| GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering | 2019 CVPR | [1902.09506](https://arxiv.org/abs/1902.09506) | [visualreasoning](visualreasoning.net)  |
| From Recognition to Cognition: Visual Commonsense Reasoning | 2019 CVPR | [1811.10830](https://arxiv.org/abs/1811.10830) | [r2c](https://github.com/rowanz/r2c/) | [VCR](https://visualcommonsense.com/) |
| VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research | 2019 ICCV | [1904.03493](https://arxiv.org/abs/1904.03493) | []() | []() |
| Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning | 2020 NeurIPS | [2010.00763](https://arxiv.org/abs/2010.00763) | [Bongard-LOGO](https://github.com/NVlabs/Bongard-LOGO) | []() |
| Bongard-HOI: Benchmarking Few-Shot Visual Reasoning for Human-Object Interactions | 2022 CVPR |[2205.13803](https://arxiv.org/abs/2205.13803)| [Bongard-HOI](https://github.com/NVlabs/Bongard-HOI)| []() |
|  |  | []() | []() | []() |

## Image Captioning
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Long-term Recurrent Convolutional Networks for Visual Recognition and Description | 2015 CVPR | [1411.4389](https://arxiv.org/abs/1411.4389) | []() | []() |
| Deep Visual-Semantic Alignments for Generating Image Descriptions | 2015 CVPR | [1412.2306](https://arxiv.org/abs/1412.2306) |  |  |
| Show and Tell A Neural Image Caption Generator | 2015 CVPR | [1411.4555](https://arxiv.org/abs/1411.4555) | [show_and_tell.tensorflow](https://github.com/jazzsaxmafia/show_and_tell.tensorflow) |  |
| Show, Attend and Tell Neural Image Caption Generation with Visual Attention | 2015 ICML | [1502.03044](https://arxiv.org/abs/1502.03044) | [show-attend-and-tell](https://github.com/yunjey/show-attend-and-tell) |  |
| From Captions to Visual Concepts and Back | 2015 CVPR | [1411.4952](https://arxiv.org/abs/1411.4952) | [visual-concepts](https://github.com/s-gupta/visual-concepts) | []() |
| Image Captioning with Semantic Attention | 2016 CVPR | [1603.03925](https://arxiv.org/abs/1603.03925) | [semantic-attention](https://github.com/chapternewscu/image-captioning-with-semantic-attention) | []() |
| Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning | 2017 CVPR | [1612.01887](https://arxiv.org/abs/1612.01887) | [AdaptiveAttention](https://github.com/jiasenlu/AdaptiveAttention) | []() |
| Self-critical Sequence Training for Image Captioning | 2017 CVPR | [1612.00563](https://arxiv.org/abs/1612.00563) | []() | []() |
| A Hierarchical Approach for Generating Descriptive Image Paragraphs | 2017 CVPR | [1611.06607](https://arxiv.org/abs/1611.06607) | []() | []() |
| Deep reinforcement learning-based image captioning with embedding reward | 2017 CVPR | [1704.03899](https://arxiv.org/abs/1704.03899) |  |  |
| Semantic compositional networks for visual captioning | 2017 CVPR | [1611.08002](https://arxiv.org/abs/1611.08002) | [Semantic_Compositional_Nets](https://github.com/zhegan27/Semantic_Compositional_Nets) | []() |
| StyleNet: Generating Attractive Visual Captions with Styles | 2017 CVPR | [CVPR 2017](https://ieeexplore.ieee.org/document/8099591/similar#similar) | [stylenet](https://github.com/kacky24/stylenet) | []() |
| Training for Diversity in Image Paragraph Captioning | 2018 EMNLP | [ENNLP 2018](https://www.aclweb.org/anthology/D18-1084/) | [image-paragraph-captioning](https://github.com/lukemelas/image-paragraph-captioning) | []() |
| Neural Baby Talk | 2018 CVPR | [1803.09845](https://arxiv.org/abs/1803.09845) | [NeuralBabyTalk](https://github.com/jiasenlu/NeuralBabyTalk) | []() |
| Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering | 2018 CVPR | [1707.07998](https://arxiv.org/abs/) | | |
| “Factual” or “Emotional”: Stylized Image Captioning with Adaptive Learning and Attention | 2018 ECCV | [1807.03871](https://arxiv.org/abs/1807.03871) | []() | []() |
| Hierarchically Structured Reinforcement Learning for Topically Coherent Visual Story Generation | 2019 AAAI | [1805.08191](https://arxiv.org/abs/1805.08191) |  |  |
| Unsupervised Image Captioning | 2019 CVPR | [1811.10787](https://arxiv.org/abs/1811.10787) | [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning) |  |
| Context-aware visual policy network for fine-grained image captioning | 2019 TPAMI | [1906.02365](https://arxiv.org/abs/1906.02365) | [CAVP](https://github.com/daqingliu/CAVP) |  |
| Dense Relational Captioning Triple-Stream Networks for Relationship-Based Captioning | 2019 CVPR | [1903.05942](https://arxiv.org/abs/1903.05942) | []() | []() |
| Describing like Humans on Diversity in Image Captioning | 2019 CVPR | [1903.12020](https://arxiv.org/abs/1903.12020) | []() | []() |
| Good News, Everyone! Context driven entity-aware captioning for news images | 2019 CVPR | [1904.01475](https://arxiv.org/abs/1904.01475) | []() | []() |
| Auto-Encoding Scene Graphs for Image Captioning | 2019 CVPR | [1812.02378](https://arxiv.org/abs/1812.02378) | [SGAE](https://github.com/yangxuntu/SGAE) | []() |
| Unsupervised Image Captioning | 2019 CVPR | [1811.10787](https://arxiv.org/abs/1811.10787) | [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning) | []() |
| MSCap: Multi-Style Image Captioning with Unpaired Stylized Text | 2019 CVPR | [CVPR 2019](https://ieeexplore.ieee.org/document/8953861) | []() | []() |
| Robust Change Captioning | 2019 ICCV | [1901.02527](https://arxiv.org/abs/1901.02527) | []() | []() |
| Attention on Attention for Image Captioning | 2019 ICCV | [1908.06954](https://arxiv.org/abs/1908.06954) | []() | []() |
| Context-Aware Group Captioning via Self-Attention and Contrastive Features | 2020 CVPR | [2004.03708](https://arxiv.org/abs/2004.03708) | []() | []() |
| Say As You Wish: Fine-grained Control of Image Caption Generation with Abstract Scene Graphs | 2020 CVPR | [2003.00387](https://arxiv.org/abs/2003.00387) | [asg2cap](https://github.com/cshizhe/asg2cap) | []() |
| Comprehensive Image Captioning via Scene Graph Decomposition | 2020 ECCV | [2007.11731](https://arxiv.org/abs/2007.11731) | [Sub-GC](https://github.com/YiwuZhong/Sub-GC) | []() |
| Are scene graphs good enough to improve Image Captioning? | 2020 AACL | [2009.12313](https://arxiv.org/abs/2009.12313) | []() | []() |
| SG2Caps: Revisiting Scene Graphs for Image Captioning | 2021 arxiv | [2102.04990](https://arxiv.org/abs/2102.04990) | []() | []() |
|  |  | []() | []() | []() |

## Image Retrieval
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Visual Word2Vec (vis-w2v) Learning Visually Grounded Word Embeddings Using Abstract Scenes | 2016 CVPR | [1511.07067](https://arxiv.org/abs/1511.07067) | [VisualWord2Vec](https://github.com/satwikkottur/VisualWord2Vec) | []() |
| Composing Text and Image for Image Retrieval - An Empirical Odyssey | 2019 CVPR | [1812.07119](https://arxiv.org/abs/1812.07119) | [tirg](https://github.com/google/tirg) | []() |
| Learning Relation Alignment for Calibrated Cross-modal Retrieval | 2021 ACL | [2105.13868](https://arxiv.org/abs/2105.13868) | [IAIS](https://github.com/lancopku/IAIS) | []() |
| ImageCoDe: Image Retrieval from Contextual Descriptions | 2022 ACL | [2203.15867](https://arxiv.org/abs/2203.15867) | [ImageCoDe](https://github.com/McGill-NLP/imagecode) | []() |
|  |  | []() | []() | []() |

## Scene Text Recognition
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Towards Unconstrained End-to-End Text Spotting | 2019 ICCV | [1908.09231](https://arxiv.org/abs/1908.09231) | []() | []() |
| What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis | 2019 ICCV | [1904.01906](https://arxiv.org/abs/1904.01906) | [clovaai](https://github.com/clovaai/deep-text-recognition-benchmark) | []() |
|  |  | []() | []() | []() |

## Scene Graph
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Image Retrieval Using Scene Graphs | 2015 CVPR | [7298990](https://ieeexplore.ieee.org/document/7298990) | []() | []() |
| Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations | 2017 IJCV  | [1602.07332](https://arxiv.org/abs/1602.07332) | [visual_genome_python_driver](https://github.com/ranjaykrishna/visual_genome_python_driver) | [visualgenome](https://visualgenome.org/) |
| Scene Graph Generation by Iterative Message Passing | 2017 CVPR | [1701.02426](https://arxiv.org/abs/1701.02426) | [scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release) |  |
| Scene Graph Generation from Objects, Phrases and Region Captions | 2017 ICCV | [1707.09700](https://arxiv.org/abs/1707.09700) | [MSDN](https://github.com/yikang-li/MSDN) |  |
| Neural Motifs: Scene Graph Parsing with Global Context | 2018 CVPR | [1711.06640](https://arxiv.org/abs/1711.06640) | [neural-motifs](https://github.com/rowanz/neural-motifs) |  |
| Generating Triples with Adversarial Networks for Scene Graph Construction | 2018 AAAI | [1802.02598](https://arxiv.org/abs/1802.02598) |  |  |
| LinkNet: Relational Embedding for Scene Graph | 2018 NIPS | [1811.06410](https://arxiv.org/abs/1811.06410) |  |  |
| Image Generation from Scene Graphs | 2018 CVPR | [1804.01622](https://arxiv.org/abs/1804.01622) | [sg2im](https://github.com/google/sg2im) |  |
| Graph R-CNN for Scene Graph Generation | 2018 ECCV | [1808.00191](https://arxiv.org/abs/1808.00191) | [graph-rcnn.pytorch](https://github.com/jwyang/graph-rcnn.pytorch) |  |
| Scene Graph Generation with External Knowledge and Image Reconstruction | 2019 CVPR | [1904.00560](https://arxiv.org/abs/1904.00560) |  |  |
| Specifying Object Attributes and Relations in Interactive Scene Generation | 2019 ICCV | [1909.05379](https://arxiv.org/abs/1909.05379) | [scene_generation](https://github.com/ashual/scene_generation) | []() |
| Attentive Relational Networks for Mapping Images to Scene Graphs | 2019 CVPR | [1811.10696](https://arxiv.org/abs/1811.10696) | []() | []() |
| Exploring Context and Visual Pattern of Relationship for Scene Graph Generation | 2019 CVPR | [sceneGraph_Mem](https://github.com/Kenneth-Wong/sceneGraph_Mem) | []() | []() |
| Graphical Contrastive Losses for Scene Graph Parsing | 2019 CVPR | [1903.02728](https://arxiv.org/abs/1903.02728) | [ContrastiveLosses4VRD](https://github.com/NVIDIA/ContrastiveLosses4VRD) | []() |
| Knowledge-Embedded Routing Network for Scene Graph Generation | 2019 CVPR | [1903.03326](https://arxiv.org/abs/1903.03326) | [KERN](https://github.com/yuweihao/KERN) | []() |
| Learning to Compose Dynamic Tree Structures for Visual Contexts | 2019 CVPR | [1812.01880](https://arxiv.org/abs/1812.01880) | [VCTree](https://github.com/KaihuaTang/VCTree-Scene-Graph-Generation) | []() |
| Counterfactual Critic Multi-Agent Training for Scene Graph Generation | 2019 ICCV | [1812.02347](https://arxiv.org/abs/1812.02347) | []() | []() |
| Scene Graph Prediction with Limited Labels | 2019 ICCV | [1904.11622](https://arxiv.org/abs/1904.11622) | [limited-label](https://github.com/vincentschen/limited-label-scene-graphs) | []() |
| Unbiased Scene Graph Generation from Biased Training | 2020 CVPR | [2002.11949](https://arxiv.org/abs/2002.11949) | [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) | []() |
| GPS-Net Graph Property Sensing Network for Scene Graph Generation | 2020 CVPR | [2003.12962](https://arxiv.org/abs/2003.12962) | [GPS-Net](https://github.com/taksau/GPS-Net) | []() |
| Learning Visual Commonsense for Robust Scene Graph Generation | 2020 ECCV | [2006.09623](https://arxiv.org/abs/2006.09623) | []() | []() |
| Sketching Image Gist Human-Mimetic Hierarchical Scene Graph Generation | 2020 ECCV | [2007.08760](https://arxiv.org/abs/2007.08760) | [het-eccv20](https://github.com/Kenneth-Wong/het-eccv20) | []() |
|  |  | []() | []() | []() |

## text2image
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Generative Adversarial Text to Image Synthesis | 2016 ICML | [1605.05396](https://arxiv.org/abs/1605.05396) | [icml2016](https://github.com/reedscot/icml2016) |  |
| StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks | 2017 ICCV | [1612.03242](https://arxiv.org/abs/1612.03242) | [StackGAN](https://github.com/hanzhanggit/StackGAN) |  |
| AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks | 2018 CVPR | [1711.10485](https://arxiv.org/abs/1711.10485) | [AttnGAN](https://github.com/taoxugit/AttnGAN) |  |
| Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network | 2018 CVPR | [1802.09178](https://arxiv.org/pdf/1802.09178.pdf) | [HDGan](https://github.com/ypxie/HDGan) |  |
| StoryGAN: A Sequential Conditional GAN for Story Visualization | 2019 CVPR | [1812.02784](https://arxiv.org/abs/1812.02784) | [StoryGAN](https://github.com/yitong91/StoryGAN) |  |
| MirrorGAN: Learning Text-to-image Generation by Redescription | 2019 CVPR | [1903.05854](https://arxiv.org/abs/1903.05854) |  |  |
| DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis | 2019 CVPR | [1904.01310](https://arxiv.org/abs/1904.01310) |  |  |
| Semantics Disentangling for Text-to-Image Generation | 2019 CVPR | [1904.01480](https://arxiv.org/abs/1904.01480) |  |  |
| Tell, Draw, and Repeat: Generating and Modifying Images Based on Continual Linguistic Instruction | 2019 ICCV | [1811.09845](https://arxiv.org/abs/1811.09845) | []() | [GeNeVA](https://github.com/Maluuba/GeNeVA_datasets) |
| Specifying Object Attributes and Relations in Interactive Scene Generation | 2019 ICCV | [1909.05379](https://arxiv.org/abs/1909.05379) | [scene_generation](https://github.com/ashual/scene_generation) | []() |
|  |  | []() | []() | []() |

## Video Captioning
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Long-term Recurrent Convolutional Networks for Visual Recognition and Description | 2015 CVPR | [1411.4389](https://arxiv.org/abs/1411.4389) | []() | []() |
| Video Paragraph Captioning Using Hierarchical Recurrent Neural Networks | 2016 CVPR | [1510.07712](https://arxiv.org/abs/1510.07712) | []() | []() |
| Attention-Based Multimodal Fusion for Video Description | 2017 CVPR | [1701.03126](https://arxiv.org/abs/1701.03126) | []() | []() |
| Semantic compositional networks for visual captioning | 2017 CVPR | [1611.08002](https://arxiv.org/abs/1611.08002) | []() | []() |
| Task-Driven Dynamic Fusion: Reducing Ambiguity in Video Description | 2017 CVPR | [CVPR_2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Task-Driven_Dynamic_Fusion_CVPR_2017_paper.pdf) | []() | []() |
| Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning | 2018 CVPR | [1804.00100](https://arxiv.org/abs/1804.00100) | []() | []() |
| Adversarial Inference for Multi-Sentence Video Description | 2019 CVPR | [1812.05634](https://arxiv.org/abs/1812.05634) | [adv-inf](https://github.com/jamespark3922/adv-inf) |  |
| Streamlined Dense Video Captioning | 2019 CVPR | [1904.03870](https://arxiv.org/abs/1904.03870) | [DenseVideoCaptioning](https://github.com/JaywongWang/DenseVideoCaptioning) |  |
| Object-aware Aggregation with Bidirectional Temporal Graph for Video Captioning | 2019 CVPR | [1906.04375](https://arxiv.org/abs/1906.04375) | []() | []() |
| iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering | 2021 WACV | [2011.07735](https://arxiv.org/abs/2011.07735) | [iPerceive](https://github.com/amanchadha/iPerceive) | []() |
|  |  | []() | []() | []() |

## Video Question Answering
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Movieqa: Understanding stories in movies through question-answering | 2016 CVPR | [1512.02902](https://arxiv.org/abs/1512.02902) | [MovieQA](https://github.com/makarandtapaswi/MovieQA_CVPR2016) | []() |
| TVQA: Localized, Compositional Video Question Answering | 2018 EMNLP | [1809.01696](https://arxiv.org/abs/1809.01696) | [TVQA](https://github.com/jayleicn/TVQA) |  |
| Knowledge-Based Video Question Answering with Unsupervised Scene Descriptions | 2020 ECCV | [2007.08751](https://arxiv.org/abs/2007.08751) | [ROLL-VideoQA](https://github.com/noagarcia/ROLL-VideoQA) | []() |
| iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering | 2021 WACV | [2011.07735](https://arxiv.org/abs/2011.07735) | [iPerceive](https://github.com/amanchadha/iPerceive) | []() |

## Video Understanding
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| TSM: Temporal Shift Module for Efficient Video Understanding | 2019 ICCV | [1811.08383](https://arxiv.org/abs/1811.08383) | [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module) |  |
| A Graph-Based Framework to Bridge Movies and Synopses | 2019 ICCV | [1910.11009](https://arxiv.org/abs/1910.11009) | []() | []() |
|  |  | []() | []() | []() |

## Vision and Language Navigation
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Embodied Question Answering | 2018 CVPR | [1711.11543](https://arxiv.org/abs/1711.11543) | []() | [embodiedqa](https://embodiedqa.org/) |
| Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments | 2018 CVPR | [1711.07280](https://arxiv.org/abs/1711.07280) | []() | [bringmeaspoon](https://bringmeaspoon.org/) |
|  |  | []() | []() | []() |

## Vision-and-Language Pretraining
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| LXMERT: Learning Cross-Modality Encoder Representations from Transformers | 2019 EMNLP | [1908.07490](https://arxiv.org/abs/1908.07490) | [lxmert](https://github.com/airsplay/lxmert) |  |
| VideoBERT: A Joint Model for Video and Language Representation Learning | 2019 ICCV | [1904.01766](https://arxiv.org/abs/1904.01766) | []() | []() |
| ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks | 2019 NIPS | [vilbert](https://github.com/jiasenlu/vilbert_beta) | []() | []() |
| OmniNet: A unified architecture for multi-modal multi-task learning | 2019 arxiv | [1907.07804](https://arxiv.org/abs/1907.07804) | [OmniNet](https://github.com/subho406/OmniNet) | []() |
| Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training | 2020 AAAI | [1908.06066](https://arxiv.org/abs/1908.06066) | [Unicoder](https://github.com/microsoft/Unicoder) | []() |
| Unified Vision-Language Pre-Training for Image Captioning and VQA | 2020 AAAI | [1909.11059](https://arxiv.org/abs/1909.11059) | [VLP](https://github.com/LuoweiZhou/VLP) | []() |
| Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks | 2020 ECCV | [1911.11237](https://arxiv.org/abs/1911.11237) | [Oscar](https://github.com/microsoft/Oscar) | []() |
| Unsupervised Learning of Visual Features by Contrasting Cluster Assignments | 2020 NIPS | [2006.09882](https://arxiv.org/abs/2006.09882) | [swav](https://github.com/facebookresearch/swav) | []() |
| Learning to Learn Words from Visual Scenes | 2020 ECCV | [2004.06165](https://arxiv.org/abs/2004.06165) | [Oscar](https://github.com/microsoft/Oscar) | []() |
| ERNIE-ViL: Knowledge Enhanced Vision-Language Representations through Scene Graphs | 2021 AAAI | [2006.16934](https://arxiv.org/abs/2006.16934) | [ERNIE](https://github.com/PaddlePaddle/ERNIE) | []() |
| VinVL: Revisiting Visual Representations in Vision-Language Models  | 2021 CVPR | [2101.00529](https://arxiv.org/abs/2101.00529) | [VinVL](https://github.com/pzzhang/VinVL) | []() |
| VirTex: Learning Visual Representations from Textual Annotations | 2021 CVPR | [2006.06666](https://arxiv.org/abs/2006.06666) | [virtex](https://github.com/kdexd/virtex) | []() |
| Learning Transferable Visual Models From Natural Language Supervision | 2021 arxiv | [2103.00020](https://arxiv.org/abs/2103.00020) | []() | []() |
| Pretrained Transformers As Universal Computation Engines | 2021 arxiv | [2103.05247](https://arxiv.org/abs/2103.05247) | [universal-computation](https://github.com/kzl/universal-computation) | []() |
| Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision | 2021 arxiv | [2102.05918](https://arxiv.org/abs/2102.05918) | []() | []() |
| Self-supervised Pretraining of Visual Features in the Wild | 2021 arxiv | [2103.01988](https://arxiv.org/abs/2103.01988) | []() | []() |
| Transformer is All You Need Multimodal Multitask Learning with a Unified Transformer | 2021 arxiv | [2102.10772](https://arxiv.org/abs/2102.10772) | []() | []() |
| Zero-Shot Text-to-Image Generation | 2021 arxiv | [2102.12092](https://arxiv.org/abs/2102.12092) | []() | []() |
| WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training | 2021 arxiv | [2103.06561](https://arxiv.org/abs/2103.06561) | []() | []() |
| Improved baselines for vision-language pre-training | 2023 arxiv | [2305.08675](https://arxiv.org/abs/2305.08675) | []() | []() |
|  |  | []() | []() | []() |

## Visual Dialog
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Visual Dialog | 2017 CVPR | [1611.08669](https://arxiv.org/abs/1611.08669) | [visdial](https://github.com/batra-mlp-lab/visdial) | [visualdialog](https://visualdialog.org/) |
| Two Can Play This Game: Visual Dialog With Discriminative Question Generation and Answering | 2018 CVPR | [1803.11186](https://arxiv.org/abs/1803.11186) |  |  |
|  |  | []() | []() | []() |

## Visual Grounding
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Modeling Relationships in Referential Expressions with Compositional Modular Networks | 2017 CVPR | [1611.09978](https://arxiv.org/abs/1611.09978) | [cmn](https://github.com/ronghanghu/cmn) | []() |
| Phrase Localization Without Paired Training Examples | 2019 ICCV | [1908.07553](https://arxiv.org/abs/1908.07553) | []() | []() |
| Learning to Assemble Neural Module Tree Networks for Visual Grounding | 2019 ICCV | [1812.03299](https://arxiv.org/abs/1812.03299) | []() | []() |
| A Fast and Accurate One-Stage Approach to Visual Grounding | 2019 ICCV | [1908.06354](https://arxiv.org/abs/1908.06354) | []() | []() |
| Zero-Shot Grounding of Objects from Natural Language Queries | 2019 ICCV | [1908.07129](https://arxiv.org/abs/1908.07129) | [zsgnet](https://github.com/TheShadow29/zsgnet-pytorch) | []() |
| Collaborative Transformers for Grounded Situation Recognition | 2022 CVPR | [2203.16518](https://arxiv.org/abs/2203.16518) | [CoFormer](https://github.com/jhcho99/CoFormer) | []() |
|  |  | []() | []() | []() |

## Visual Question Answering
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| VQA: Visual Question Answering | 2015 ICCV | [1505.00468](https://arxiv.org/abs/1505.00468) |  | [visualqa](https://visualqa.org/) |
| Hierarchical question-image co-attention for visual question answering | 2016 NIPS | [1606.00061](https://arxiv.org/abs/1606.00061) | [HieCoAttenVQA](https://github.com/jiasenlu/HieCoAttenVQA) |  |
| Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding | 2016 EMNLP | [1606.01847](https://arxiv.org/abs/1606.01847) | [vqa-mcb](https://github.com/akirafukui/vqa-mcb) | []() |
| Stacked Attention Networks for Image Question Answering | 2016 CVPR | [1511.02274](https://arxiv.org/abs/1511.02274) | [imageqa-san](https://github.com/zcyang/imageqa-san) | []() |
| Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering | 2016 ECCV | [1511.05234](https://arxiv.org/abs/1511.05234) | [AAAA](https://github.com/VisionLearningGroup/Ask_Attend_and_Answer) | []() |
| Dynamic Memory Networks for Visual and Textual Question Answering | 2016 ICML | [1603.01417](https://arxiv.org/abs/1603.01417) | [dmn-plus](https://github.com/vlgiitr/dmn-plus) | []() |
| Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding | 2016 EMNLP | [1606.01847](https://arxiv.org/abs/1606.01847) | [vqa-mcb](https://github.com/akirafukui/vqa-mcb) | []() |
| Multimodal Residual Learning for Visual QA | 2016 NIPS | [1606.01455](https://arxiv.org/abs/1606.01455) | [nips-mrn-vqa](https://github.com/jnhwkim/nips-mrn-vqa) | []() |
| Graph-Structured Representations for Visual Question Answering | 2017 CVPR | [1609.05600](https://arxiv.org/abs/1609.05600) | []() | []() |
| Making the V in VQA Matter Elevating the Role of Image Understanding in Visual Question Answering | 2017 CVPR | [1612.00837](https://arxiv.org/abs/1612.00837) | []() | []() |
| Learning to Reason: End-to-End Module Networks for Visual Question Answering | 2017 ICCV | [1704.05526](https://arxiv.org/abs/1704.05526) | []() | []() |
| Explicit Reasoning over End-to-End Neural Architectures for Visual Question Answering | 2018 AAAI | [1803.08896](https://arxiv.org/abs/1803.08896) | [PSLQA](https://github.com/adityaSomak/PSLQA) | []() |
| Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering | 2018 CVPR | [1707.07998](https://arxiv.org/abs/) | | |
| Tips and Tricks for Visual Question Answering Learnings from the 2017 Challenge | 2018 CVPR | [1708.02711](https://arxiv.org/abs/1708.02711) | [vqa-winner](https://github.com/markdtw/vqa-winner-cvprw-2017) | []() |
| Transfer Learning via Unsupervised Task Discovery for Visual Question Answering | 2019 CVPR | [1810.02358](https://arxiv.org/abs/1810.02358) | [VQA-Transfer-ExternalData](https://github.com/HyeonwooNoh/VQA-Transfer-ExternalData) |  |
| GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering | 2019 CVPR | [1902.09506](https://arxiv.org/abs/1902.09506) | [visualreasoning](visualreasoning.net)  |
| Towards VQA Models That Can Read | 2019 CVPR | [1904.08920](https://arxiv.org/abs/1904.08920) |  |  |
| From Strings to Things: Knowledge-enabled VQA Model that can Read and Reason | 2019 ICCV | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Singh_From_Strings_to_Things_Knowledge-Enabled_VQA_Model_That_Can_Read_ICCV_2019_paper.pdf) | []() | []() |
| An Empirical Study on Leveraging Scene Graphs for Visual Question Answering | 2019 BMVC | [1907.12133](https://arxiv.org/abs/1907.12133) | [scene-graphs-vqa](https://github.com/czhang0528/scene-graphs-vqa) | []() |
| RelViT: Concept-guided Vision Transformer for Visual Relational Reasoning | 2022 ICLR |[2204.11167](https://arxiv.org/abs/2204.11167)| [RelViT](https://github.com/NVlabs/RelViT)| []() |
| TAG: Boosting Text-VQA via Text-aware Visual Question-answer Generation | 2022 arXiv |[2208.01813](https://arxiv.org/abs/2208.01813)| [TAG](https://github.com/HenryJunW/TAG)| []() |
|  |  | []() | []() | []() |

## Visual Reasoning
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning | 2017 CVPR | [1612.06890](https://arxiv.org/abs/1612.06890) | []() | []() |
| Inferring and Executing Programs for Visual Reasoning | 2017 ICCV | [1705.03633](https://arxiv.org/abs/1705.03633) | []() | []() |
| GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering | 2019 CVPR | [1902.09506](https://arxiv.org/abs/1902.09506) | [visualreasoning](visualreasoning.net)  |
| Explainable and Explicit Visual Reasoning over Scene Graphs | 2019 CVPR | [1812.01855](https://arxiv.org/abs/1812.01855) | []() | []() |
| From Recognition to Cognition: Visual Commonsense Reasoning | 2019 CVPR | [1811.10830](https://arxiv.org/abs/1811.10830) | [r2c](https://github.com/rowanz/r2c/) | [VCR](https://visualcommonsense.com/) |
| Dynamic Graph Attention for Referring Expression Comprehension | 2019 ICCV | [1909.08164](https://arxiv.org/abs/1909.08164) | []() | []() |
| Visual Semantic Reasoning for Image-Text Matching | 2019 ICCV | [1909.02701](https://arxiv.org/abs/1909.02701) | [VSRN](https://github.com/KunpengLi1994/VSRN) | []() |
| Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning | 2020 NeurIPS | [2010.00763](https://arxiv.org/abs/2010.00763) | [Bongard-LOGO](https://github.com/NVlabs/Bongard-LOGO) | []() |
| Bongard-HOI: Benchmarking Few-Shot Visual Reasoning for Human-Object Interactions | 2022 CVPR |[2205.13803](https://arxiv.org/abs/2205.13803)| [Bongard-HOI](https://github.com/NVlabs/Bongard-HOI)| []() |
| RelViT: Concept-guided Vision Transformer for Visual Relational Reasoning | 2022 ICLR |[2204.11167](https://arxiv.org/abs/2204.11167)| [RelViT](https://github.com/NVlabs/RelViT)| []() |
|  |  | []() | []() | []() |


## Visual Relationship Detection
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Visual Relationship Detection with Language Priors | 2016 ECCV | [1608.00187](https://arxiv.org/abs/1608.00187) | [Visual-Relationship-Detection](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection) |  |
| ViP-CNN: Visual Phrase Guided Convolutional Neural Network | 2017 CVPR | [1702.07191](https://arxiv.org/abs/1702.07191) | []() | []() |
| Visual Translation Embedding Network for Visual Relation Detection | 2017 CVPR | [1702.08319](https://arxiv.org/abs/1702.08319) | [drnet](https://github.com/doubledaibo/drnet_cvpr2017) | []() |
| Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection | 2017 CVPR | [1703.03054](https://arxiv.org/abs/1703.03054) | [DeepVariationRL](https://github.com/nexusapoorvacus/DeepVariationStructuredRL) | []() |
| Detecting Visual Relationships with Deep Relational Networks | 2017 CVPR | [1704.03114](https://arxiv.org/abs/1704.03114) | [drnet](https://github.com/doubledaibo/drnet_cvpr2017) | []() |
| Phrase Localization and Visual Relationship Detection with Comprehensive Image-Language Cues | 2017 ICCV | [1611.06641](https://arxiv.org/abs/1611.06641) | [pl-clc](https://github.com/BryanPlummer/pl-clc) | []() |
| Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation | 2017 ICCV | [1707.09423](https://arxiv.org/abs/1707.09423) | []() | []() |
| Referring Relationships | 2018 CVPR | [1803.10362](https://arxiv.org/abs/1803.10362) | [ReferringRelationships](https://github.com/StanfordVL/ReferringRelationships) | []() |
| Zoom-Net: Mining Deep Feature Interactions for Visual Relationship Recognition | 2018 ECCV | [1807.04979](https://arxiv.org/abs/1807.04979) | [ZoomNet](https://github.com/gjyin91/ZoomNet) | []() |
| Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features | 2018 ECCV | [1808.00171](https://arxiv.org/abs/1808.00171) | [vrd](https://github.com/yangxuntu/vrd) | []() |
| Leveraging Auxiliary Text for Deep Recognition of Unseen Visual Relationships | 2020 ICLR | [1910.12324](https://arxiv.org/abs/1910.12324) | []() | []() |
|  |  | []() | []() | []() |

## Visual Storytelling
| Title                                       | Conference / Journal | Paper                                     | Code                                        | Remarks   |
| ------------------------------------------- | ---------- | ----------------------------------------- | ------------------------------------------- |-----------|
| Visual Storytelling | 2016 NAACL  | [1604.03968](https://arxiv.org/abs/1604.03968) | [visual_genome_python_driver](https://github.com/ranjaykrishna/visual_genome_python_driver) | [VIST](http://visionandlanguage.net/VIST/) |
| No Metrics Are Perfect Adversarial Reward Learning for Visual Storytelling | 2018 ACL | [1804.09160](https://arxiv.org/abs/1804.09160) | [AREL](https://github.com/eric-xw/AREL) |  |
| Show, Reward and Tell: Automatic Generation of Narrative Paragraph from Photo Stream by Adversarial Training | 2018 AAAI | []() | []() | []() |
| Hide-and-Tell: Learning to Bridge Photo Streams for Visual Storytelling | 2020 AAAI | [2002.00774](https://arxiv.org/abs/2002.00774) | []() | []() |
| Storytelling from an Image Stream Using Scene Graphs | 2020 AAAI | [AAAI 2020](https://ojs.aaai.org//index.php/AAAI/article/view/6455) | []() | []() |
|  |  | []() | []() | []() |

## Contributing
Please feel free to send me [pull requests](https://github.com/sangminwoo/awesome-vision-and-language/pulls) or email (shmwoo9395@gmail.com) to add links.

## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Sangmin Woo](https://github.com/sangminwoo) has waived all copyright and related or neighboring rights to this work.
