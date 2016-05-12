# Learning to execute Python in Tensorflow

Reimplementation of the paper [Learning to execute](https://arxiv.org/abs/1410.4615) in Tensorflow (0.7.0 and 0.8.0).

Original implementation in Torch:
https://github.com/wojciechz/learning_to_execute

A Seq2Seq model is trained to execute simple Python programs.

Example:
```
Input:
g=6
a=-77
if a>2:
  g=a-4
print(a+g)
-----------
Target: -71
Model prediction: -71
```

The program generation code is simplified compared to the original paper but can easily be extended.

## Date normalization

date-normalization.ipynb also contains a simple date format normalization example using the same model code.

## PyCon Sweden
This work was presented at PyCon Sweden 2016, Stockholm.
[Presentation](https://docs.google.com/presentation/d/14hkW1uOC7TUk2iPknvDUHN_vZKHkQwyd5NwLLavxErs/edit?usp=sharing)

