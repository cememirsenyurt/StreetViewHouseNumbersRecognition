1. Describe Problem/Analyze Dataset

Our group decided to work with the Street View House Numbers (SVHN) dataset for the
project. The SVHN dataset is commonly used as a benchmark for image classification models. It
comprises roughly 600,000 32 x 32 color images (32 x 32 x 3) which are labeled as digits 0-9.
The images are taken from Google Street View images of house numbers. Below are 5 images
along with their corresponding labels:

![image](https://user-images.githubusercontent.com/104103767/233764545-d28e9adb-22cc-4adf-bb75-6ade41cbf0b2.png)

In analyzing the images, we found a number of important factors to consider. First, the sample
digits vary greatly in terms of fonts, colors, orientation, lighting, and more. Additionally, many
of the samples contain digits to the left or right of the target digit, or “distracting digits”, as can
be seen above. The data also had some holdovers from MATLAB such as the 0 samples being
labeled as 10’s. Sam wrote data loader functions which do minor preprocessing such as
normalizing input values between 0-1, swapping 10 labels to 0’s, and swapping the index from
the last to the first axis.


2. Model Selection

Because SVHN is an image classification problem, we decided to use a convolutional
neural network as the basis for our model. Convolutional neural networks are typically used for
image classification tasks because they take into account the relative positions of pixel inputs
within the image and learn to recognize increasingly complex patterns within the image. We
decided to use Tensorflow/Keras to implement our model given its extensive documentation for
CNNs. We started by implementing the basic Tensorflow CNN example for the CIFAR10 dataset
(https://www.tensorflow.org/tutorials/images/cnn) and obtained 89-90% testing accuracy on
SVHN. Most mistakes were between labels 5 and 6 or labels 3 and 8, which makes sense given
that these numbers share some structural similarities.


3. Model Tuning

From the basic example model mentioned above, we all worked on different methods to
improve accuracy. One approach we tried was building an ensemble model, combining the
results of multiple slightly different CNN architectures to hopefully reduce variance.
I worked on an ensemble model that consisted of three individual CNN models with varying
architectures. Each model was trained independently, and the outputs of each model were then
combined using the average function to make the final prediction. The architecture of each
individual model had the following layout:

Conv2D(32, (3, 3), activation='relu') ->
MaxPooling2D((2, 2)) ->
Conv2D(64, (3, 3), activation='relu') ->
MaxPooling2D((2, 2)) ->
Conv2D(64, (3, 3), activation='relu') ->
Flatten() ->
Dense(N, activation='relu') ->
Dense(10)

Where N was 64, 128, and 256 for the three respective models. The ensemble members were
trained using the Adam optimizer with a sparse categorical cross-entropy loss function. We
trained the models for 10 epochs with a batch size of 32. This ensemble didn’t result in any
noticeable performance gain: we typically saw roughly the same results as the basic example
model. There could be a number of reasons for this: we might have simply not trained enough
separate models to significantly reduce variance. Additionally, the architectures we used were
fairly similar and tend to succeed/fail the same given similar inputs. As such, we thought making
the models different in some more significant way might lead to noticeable improvements.
To try this, Daniel built a new ensemble of 3 convolutional neural networks, except now
each network dealt with one color layer of the image. To get the final prediction, we would
gather a prediction class from each network and pick the one that has the highest probability
within its array of predictions. The supposed benefit of such a model is that it allows learners to

![image](https://user-images.githubusercontent.com/104103767/233764371-c2686941-77b9-412a-a1f7-ba134f503318.png)

learn color specific structural features of each number separately from other colors, and
potentially ignore noise in the pictures. Additionally, we hypothesized that it is possible to build
3 simple models which would, combined, be simpler than one complex model that deals with all
the colors, but achieve the same or better accuracy.
Each network had a similar structure, using Conv2D, MaxPooling, BatchNormalization,
Dropout, and Dense layers and had 127,434 trainable parameters. While training each model,
training was done in 4 stages, in each stage batch_size was doubled from previous to combat
overfitting. On their own, each network was about 93% accurate on the validation dataset, while
when three of them were combined and highest confidence prediction was chosen, accuracy of
this color ensemble model was 94.6%, which is a considerable increase from single color models
and in general interesting result showing that such approach might be effective.
Separately, Sam worked on implementing the “Cutout” method to augment our data, but
could not find any noticeable improvements. Cutout is a regularization method which involves
removing square patches from training images in order to make the model more robust and better
at identifying particular features of an image. Some images processed with Cutout can be seen
below.

![image](https://user-images.githubusercontent.com/104103767/233764389-b0f912c4-ba39-46c8-b97f-89c216b35e1b.png)
![image](https://user-images.githubusercontent.com/104103767/233764398-36741bb3-240f-484f-a23a-7dfc1b1fc77c.png)
![image](https://user-images.githubusercontent.com/104103767/233764403-4cd5131e-37db-448e-8083-8a7452ff0708.png)

Different parameters in our Cutout implementation included the size and number of patches
applied to each image. Although we experimented with a variety of Cutout parameters and
model layouts, we weren’t able to find any noticeable improvements when using this method. In
some cases, such as with larger patch sizes, Cutout reduced accuracy. Some papers suggest that
Cutout can improve performance on SVHN classifiers (arXiv:1708.04552) but those papers use
deeper/more complex models than we did. We suspect this is why we didn’t see any noticeable
accuracy improvement when using Cutout.
One useful trick we found in designing our network was the use of two subsequent
convolutional layers to simulate one convolutional layer of a larger filter size. For example, in
our final model we use two 3x3 convolutions before each pooling layer. This architecture
effectively mimics the outputs of a single convolution with a 5x5 filter, but requires less
parameters and is therefore more efficient to train. This method helped us increase the
complexity of the model while keeping the number of trainable parameters low.


4. Final Model

To select the simplest yet most accurate approach we were choosing between a multicolor
ensemble approach and a singular complex model. While the singular model was considerably
more effective than a multicolor ensemble, it was also a lot more complex, having way more
trainable parameters than a multicolor ensemble.
We decided to work with a singular model a bit more and see how far we can simplify it,
while still keeping high accuracy. The easiest way to do it was to decrease the amount of filters
in final convolutional layers, therefore decreasing the number of trainable parameters. During
this process, while accuracy went down by a couple tenth of a percent, it was still higher than
multicolor ensemble’s (94.6% vs 95.2%) and we even managed to achieve smaller complexity
than a multicolor ensemble model (272,554 vs. 382,302), and it was taking way less time to
train. To achieve the best result we figured out that we need to periodically increase training
batch_size. We trained a model with 6 epochs on batch size of 126, 4 on 256 and 4 epochs on
batch size of 512 Therefore, it was decided to use this simplified singular model as our final
product. A full diagram of our final model can be seen on the next page.

![image](https://user-images.githubusercontent.com/104103767/233764418-ec05af09-45b0-4ed5-8bd9-e4812476de3c.png)

As a more informal means of testing our final model, we used screenshots from Google
Street View to generate a small test set of Sam’s actual house numbers. We were happy to see
that it predicted all 5 accurately!

![image](https://user-images.githubusercontent.com/104103767/233764425-0e0c2a37-fb28-489e-911b-eec9e0b10e16.png)

![image](https://user-images.githubusercontent.com/104103767/233764430-424cf849-ebf4-433f-8d79-89249e6a731a.png)
![image](https://user-images.githubusercontent.com/104103767/233764434-f5893233-3429-4061-8814-7f1b66134e04.png)
