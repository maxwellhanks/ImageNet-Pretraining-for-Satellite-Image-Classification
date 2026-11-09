# Machine Learning Background

This file contains the theoretical background and ML concepts that are important to understand this project. It serves as a general knowledge base for the group.

## Theoretical Background

Papers we should all read as they form the base of this project:

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385): This is the original ResNet paper that proposes the specific CNN structure that we use as the base for our experiments. [This is a good video explaining it.](https://www.youtube.com/watch?v=GWt6Fu05voI)

[EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029): This paper introduces the EuroSAT dataset which we use to train and evaluate our models in the first part of the project.

## Part 1: Deep learning

### The task we're trying to solve: supervised image classification

[EuroSAT](https://arxiv.org/abs/1709.00029) contains 27 000 RGB image patches of 64 × 64 pixels taken by the Sentinel-2 satellite, each labelled with one of 10 land-cover classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake). Formally, an image is a tensor $x \in 0,\dots,255^{3 \times 64 \times 64}$ (three colour channels, height, width) and its label is $y \in 0, \dots, 9$. The dataset is part of the [TorchVision Datasets](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html) and also [published on Github](https://github.com/phelber/eurosat). This corresponds to the `eurosat_rgb` dataset in `src/satbench/data.py`.

Our models are classifier in the form of a parametrised function $f_\theta$ that maps an image to a vector of 10 real numbers $z = f_\theta(x) \in \mathbb{R}^{10}$ (logits). We then use a softmax function to turn logits into a probability distribution over classes. Everything the model has learned and knows is stored in the parameter vector $\theta$. A ResNet-50 has about 23.5 million entries.

### The loss function: cross-entropy

To train the model we need a number measuring how wrong it is. For one example the **cross-entropy loss** is

$$
\ell(\theta; x, y) = -\log p_y(x),
$$

the negative log-probability the model assigns to the correct class. It is 0 if the model is certain and correct, and grows without bound as the probability of the correct class goes to 0. Averaged over a dataset, this is exactly the negative log-likelihood of a categorical model, so minimising it is maximum likelihood estimation.

A useful reference point when reading logs: a model that guesses uniformly over 10 classes has loss $\log 10 \approx 2.303$. A loss above that means the model is worse than guessing (it is confidently wrong).

### Optimisation: mini-batch gradient descent

Training means minimising the average loss over the training set, the **empirical risk**:

$$
L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(\theta; x_i, y_i).
$$

For neural networks this function is non-convex and has millions of variables, so there is no closed-form solution. Instead we use gradient descent: repeatedly take a small step against the gradient,

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta),
$$

where $\eta > 0$ is the learning rate. The gradient is computed by backpropagation, which is simply the chain rule applied systematically through the layers of the network. PyTorch does this automatically (`loss.backward()`).

Computing the gradient over all $n$ images for every step is expensive, so we estimate it from a random mini-batch of $B$ images (we use $B = 64$). The mini-batch gradient is an unbiased estimator of the full gradient, and its noise even helps generalisation. One pass through the whole training set is an **epoch**, which takes $\lceil n / B \rceil$ steps. With 100% of the training split (16 200 images) that is 254 steps per epoch; with 1% (162 images) it is only 3.

We use the [**AdamW**](https://optimization.cbe.cornell.edu/index.php?title=AdamW) optimiser rather than plain gradient descent. Adam keeps exponential moving averages of the gradient. It discourages overly large weights, which tends to reduce overfitting.

### Our models

**Linear classifier.** This is the baseline. It flattens the image into one vector $x \in \mathbb{R}^{12288}$ ($3 \cdot 64 \cdot 64$) and computes $z = Wx + b$ with $W \in \mathbb{R}^{10 \times 12288}$. Together with softmax and cross-entropy this is exactly  a multinomial logistic regression with each pixel as a feature. A lot of the prediction can probably be done by looking at average pixel colors and relative brightness, so we use this as a baseline to see what the CNNs can actually achieve by learning features other than just that.

**Convolutional neural networks (CNNs).** Instead of one weight per pixel, a convolutional layer slides small learned filters (for example 3 × 3) across the image and computes a weighted sum at every position. The same filter weights are used everywhere (weight sharing), which builds in the assumption that a pattern means the same thing wherever it appears. Stacking many such layers with non-linearities (ReLU, $\max(0, t)$) in between lets early layers detect edges and colours, middle layers textures, and late layers whole structures such as field patterns or road networks.

[**ResNet-50**](https://arxiv.org/abs/1512.03385) is a 50-layer CNN. Its key idea is the residual connection: each block computes $h_{l+1} = h_l + F(h_l)$ instead of $h_{l+1} = F(h_l)$, so a block only has to learn a correction to its input. This keeps gradients well behaved and makes very deep networks trainable. Other components you will see mentioned:

**Batch normalisation (BatchNorm)** standardises intermediate activations using the mean and variance of the current mini-batch during training, and uses running averages of those statistics at evaluation time. This is why the code switches between `net.train()` and `net.eval()`: the network literally computes a different function in the two modes.

**Global average pooling** averages each of the final 2048 feature maps over all spatial positions, giving a vector of 2048 numbers per image.

The final **fully connected layer** (`fc`) maps those 2048 features to the 10 class logits.

A helpful way to think about it: a ResNet is a learned feature map $\phi_\theta : \text{image} \to \mathbb{R}^{2048}$ (the **backbone**) followed by a linear classifier (the **head**). So the linear baseline and the ResNet differ only in the features the linear classifier sees: raw pixels versus learned features.

### Pretraining and transfer learning

Because the loss function is non-convex, where gradient descent ends up depends heavily on where it starts, the initial parameters $\theta_0$. This is the core of our project. **We should all make sure that we have an intuitive understanding of this.** Chapter 8.2 and 8.4 of the [Deep Learning Book](https://www.deeplearningbook.org/contents/optimization.html) are a good read.

**Random initialisation** draws $\theta_0$ from carefully scaled random distributions (Kaiming initialisation). The network starts knowing nothing and must learn all features from our labelled data.

**ImageNet pretraining** starts from parameters that were already trained to classify 1.28 million everyday photos into 1000 categories. See the [ResNet paper](https://arxiv.org/abs/1512.03385). The weights from the original paper are available in [PyTorch](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights). This corresponds to the `resnet50_imagenet` model in `src/satbench/models.py`.

We keep the backbone and replace only the 1000-class head with a new, randomly initialised 10-class head. The idea is that features learned on photos (edges, textures, shapes) transfer to satellite images. Images are highly dimensional data but the assumption is that real world image data is actually concentrated on a much lower dimensional manifold (see [Manifold Hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis)), so the ImageNet pretraining should've helped narrowing the search for good initial parameters down. Reusing a trained model this way is transfer learning.

There are two ways to use a pretrained backbone. In fine-tuning, all parameters, including the backbone, keep training on EuroSAT. In linear probing, the backbone is frozen and only the head is trained. Mathematically, linear probing is logistic regression on the fixed features $\phi_{\theta_0}(x)$, which directly measures how good the pretrained features are.

**Self-supervised pretraining** ([SSL4EO-S12](https://arxiv.org/abs/2211.07044)) learns a backbone from unlabelled satellite images. In contrastive learning, the model sees two randomly augmented versions (crops, colour changes) of the same image and is trained so their feature vectors are close, while feature vectors of different images are far apart. No labels are needed, which is why it can use massive unlabelled satellite archives. This should've theoretically learned a much better feature representation than ImageNet pretraining as it is trained on satellite images. 

Our research question in these terms: which $\theta_0$ (random, ImageNet, or self-supervised on satellite data) leads to the best classifier when labelled data is scarce, and which generalises best to other satellite datasets?

### Generalisation, overfitting, and the three data splits

We do not actually care about the loss on the training images. We care about performance on images the model has never seen. A model with 23 million parameters can memorise 162 training images perfectly while learning nothing general(overfitting), and it shows up as training accuracy near 100% with much lower accuracy on new data.

To measure generalisation honestly, the data is split into three disjoint sets (see `src/satbench/data.py`):

| Split      | Size         | Used for                                                                    | Analogy            |
| ---------- | ------------ | --------------------------------------------------------------------------- | ------------------ |
| train      | 16 200 (60%) | computing gradients and updating $\theta$                                   | doing the homework |
| validation | 5 400 (20%)  | monitoring training and making decisions (best epoch, later learning rates) | practice exams     |
| test       | 5 400 (20%)  | the final reported number, used once per run                                | the real exam      |

The separation between validation and test matters because every decision based on a dataset makes performance on that dataset look slightly better than it really is. If we picked the best epoch by looking at test accuracy, the reported test accuracy would be biased upward, since we selected the epoch that happened to do well on those particular images. This is a form of data leakage. Keeping the test set untouched until the very end gives an unbiased estimate.

In the pipeline, the model is evaluated on the validation set after every epoch. The weights from the epoch with the highest validation accuracy are saved, and at the end those weights are restored and evaluated on the test set once. Choosing the best epoch this way is a simple form of early stopping.

### Label efficiency

Our first experiment trains each model on subsets of the training split of increasing size (1%, 2%, 5%, 10%, 25%, 50%, 100%) and plots test accuracy against the number of labelled images. A model is more label-efficient if it reaches high accuracy with fewer labels, so its curve rises earlier. The gap between the ImageNet and random curves at small sizes measures how much the pretraining "replaces" labelled data.

### Metrics

**Accuracy** is the fraction of test images classified correctly.

**Macro F1** treats each class separately. For class $c$, with $TP_c$, $FP_c$, $FN_c$ the true positives, false positives and false negatives,

$$
F1_c = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}, \qquad \text{macro F1} = \frac{1}{10} \sum_c F1_c.
$$

It weights all classes equally, so a model that ignores a small class (such as Pasture, the smallest class) is penalised more than by plain accuracy.

The **confusion matrix** $C$ has entry $C_{ij}$ = number of test images of true class $i$ predicted as class $j$. The diagonal holds correct predictions; off-diagonal entries show which classes get confused (typically AnnualCrop vs PermanentCrop, or Highway vs River).

A statistical note on how precise a single accuracy number is: if the true accuracy is $p$ and the test set has $n = 5400$ images, the observed accuracy is approximately binomial with standard error $\sqrt{p(1-p)/n}$. At $p = 0.95$ that is about 0.3 percentage points. So differences well below 1 point between single runs can be noise from the test set alone, before even considering seed variation.

### Randomness, seeds, and why we run several

A training run involves several random choices: the random initialisation, which images are in the training subset, and the order of mini-batches. A **seed** is the starting state of the pseudo-random number generator; fixing it makes the "random" choices identical every time. The pipeline seeds every source of randomness, so the same configuration with the same seed produces the same result (on the same hardware and software).

Fixing a seed does not make the result less random in a statistical sense. It just picks one particular draw. The test accuracy of a training procedure is a random variable, and one run is one sample from its distribution. That is why we run several seeds and report mean ± sample standard deviation. Seed-to-seed variation is usually much larger with small training subsets: which 162 images you happened to draw matters a lot.

Why results can still differ slightly across machines: floating-point addition is not associative, $(a + b) + c \neq a + (b + c)$ in general. GPUs sum millions of numbers in parallel, and different hardware (NVIDIA CUDA, Apple MPS, CPU) uses different summation orders and kernels. Tiny rounding differences then get amplified over thousands of gradient steps. Results from different hardware should agree statistically, not digit for digit.

### Preprocessing and Augmentation

Pixel values are first scaled to $[0, 1]$ and then normalised per colour channel, $\tilde x_c = (x_c - \mu_c)/\sigma_c$, using the mean and standard deviation of the ImageNet training set. Pretrained weights expect inputs on the same scale they were trained on, and giving them differently scaled inputs is a classic silent bug.

During training, we apply standard **data augmentation**: random horizontal and vertical flips. This helps prevent overfitting by forcing the model to recognise patterns regardless of their orientation, which is especially suitable for satellite images (nadir view) where orientation is arbitrary. 

ResNet inputs are also upsampled from 64 × 64 to 224 × 224 by bilinear interpolation. This adds no information, but ResNet-50 reduces spatial resolution by a factor of 32 overall, so a 64-pixel image would leave only a 2 × 2 feature map at the end, while 224 pixels gives 7 × 7, the resolution the ImageNet weights were trained with. 

Differences in preprocessing between datasets or between pretraining and fine-tuning are one form of **domain shift**: the training and deployment data come from different distributions. Satellite images differ from photos in viewpoint, scale and spectral properties, which is the main reason ImageNet features might not be optimal.

### Cross-sensor transfer: UC Merced

EuroSAT (Sentinel-2) is our main labelled training domain. To check whether a
EuroSAT-adapted model generalises beyond that sensor/domain, we evaluate on a
**mapped subset of [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html)**
(aerial RGB, 21 land-use classes, $\sim 256 \times 256$).

We do **not** train on UC Merced. After ImageNet/SSL4EO init and fine-tune or
linear probe on EuroSAT, we run the **same 10-class head** on UC Merced images
whose labels we map into EuroSAT’s taxonomy. Unmapped classes (airplane,
tenniscourt, …) are dropped. Absolute EuroSAT vs UC Merced accuracies are not
comparable; we compare **rankings / gaps** across initialisations.

| UC Merced class | EuroSAT | Confidence |
| --------------- | --------- | ----- |
| forest | Forest | high |
| river | River | high |
| sparse/medium/dense residential, mobilehomepark | Residential | high–medium |
| freeway, intersection, overpass | Highway | high–medium |
| agricultural | AnnualCrop | medium (vs PermanentCrop) |
| storagetanks | Industrial | medium |
| chaparral | HerbaceousVegetation | medium |

Implementation: `uc_merced_rgb` in `src/satbench/data.py` (torchgeo download under
`data/uc_merced/`, all train/val/test lists merged, EuroSAT-mapped labels, uint8
cache `data/uc_merced_rgb_uint8.pt`). No satbench 60/20/20 split — the whole
mapped set (~1200 images) is for transfer evaluation. Keys look like
`uc_merced/.../agricultural/agricultural01.tif`.