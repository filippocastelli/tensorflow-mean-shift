# Mean shift clustering with TensorFlow

This is a parallelized implementation of the [mean shift](https://en.wikipedia.org/wiki/Mean_shift) clustering algorithm through TensorFlow.
This makes it easy to use on the GPU if you install TensorFlow with CUDA support.

I wrote this to better understand `tf.while_loop` and how to implement a nontrivial iterative algorithm in a TensorFlow graph. It only works in TensorFlow v1, or through the `tensorflow.compat.v1` module of TF2.
