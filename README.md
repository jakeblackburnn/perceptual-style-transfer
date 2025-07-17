# Perceptual Loss Style Transfer Training w/ Python

created by J. Blackburn - Jul 17 2025

Training pipeline for perceptual loss style transfer models. Whereas classic style transfer simply algorithmically blends the content of one image with a style of another, transformers trained with a perceptual loss function learn how to apply the style of an image or images to any content, and dont require style images as input after training. This training process uses VGG19 to extract style and content features used in the perceptual loss function.
