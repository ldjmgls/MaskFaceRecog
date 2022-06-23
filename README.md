# MaskFaceRecog
Final Project for UW CSE576: Computer Vision Sp22 <br>
Contributors: Matthew Chau, Ruby Lin, Sree Prasanna Rajagopal.<br>

Traditional face recognition approaches struggle to accurately identify individuals with masks as they occlude critical facial features contribute to recognition. This project is a realization of masked face recognition following FocusFace architecture and experiments were carried out on our own augmented dataset. Since real-world masked faces are not sufficient, we created our own based on the MS1MV2 (MS1M-ArcFace) dataset. IResNet-50 and IResNet-100 were used as the backbone networks. <br>


## Repo structure
`dataloader.py`: load masked/unmasked face image dataset <br>
`trainer.py`: train the model <br>
`eval.py`: evaluate the model <br>
`iresnet.py`: Backbone networks - Improved ResNet <br>
`model.py`: FocusFace architecture <br>
`metrics.py`: computation of evaluation metrics <br>
`utils.py`: some util functions <br>

## Dataset
The training dataset consists of pairs of images of the same person, one masked and one unmasked. The validation dataset consists of genuine pairs and imposter pairs. Genuine pairs contain images of the same person, with the reference image being unmasked and the probe image being masked. Imposter pairs, on the other hand, contain images of different people.

## Evaluation
We used the **pyeer** package for generating Biometric systems evaluation metrics. The similarity scores for genuine image pairs and imposter image pairs are calculated using $$ s = 1 - \frac{1}{2} ||e_1 - e_2||, $$ where $s$ is the similarity score, and $e_1$ and $e_2$ are the two embeddings of the images. The similarity scores are used to generate the following evaluation metrics:

- **GMean**: the mean of the genuine scores <br>
- **IMean**: the mean of the imposter scores <br>
- **EER**: the Equal Error Rate <br>
- **AUC**: the Area Under ROC Curve <br>
- **FMR100** and **FMR10**: the lowest FNMR (False non-match Rate) for FMR  $\leq 1.0 \% $ and FMR  $\leq 10.0\% $.

## References
P. C. Neto, F. Boutros, J. R. Pinto, N. Damer, A. F. Sequeira, and J. S. Cardoso. "FocusFace: Multi- task Contrastive Learning for Masked Face Recognition," 2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021), 2021, pp. 01-08

Original Paper: https://arxiv.org/abs/2110.14940 \\
Original Github Repo: https://github.com/NetoPedro/FocusFace
