# Tensors

## **Introduction**

Tensors are multi-dimensional arrays with a uniform type. You can see all supported `dtypes` at [`tf.dtypes.DType`](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType).  tensors are (kind of) like `np.arrays`.

All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

| A scalar, shape: `[]`                                        | A vector, shape: `[3]`                                       | A matrix, shape: `[3, 2]`                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![A scalar, the number 4](https://www.tensorflow.org/guide/images/tensor/scalar.png) | ![The line with 3 sections, each one containing a number.](https://www.tensorflow.org/guide/images/tensor/vector.png) | ![A 3x2 grid, with each cell containing a number.](https://www.tensorflow.org/guide/images/tensor/matrix.png) |

| A 3-axis tensor, shape: `[3, 2, 5]`                          |                                                              |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_numpy.png) | ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_front.png) | ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_block.png) |

You can convert a tensor to a NumPy array either using `np.array` or the `tensor.numpy` method.

The base [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) class requires tensors to be "rectangular"---that is, along each axis, every element is the same size. However, there are specialized types of tensors that can handle different shapes:

- Ragged tensors (see [RaggedTensor](https://www.tensorflow.org/guide/tensor#ragged_tensors) below)
- Sparse tensors (see [SparseTensor](https://www.tensorflow.org/guide/tensor#sparse_tensors) below)

## About shapes

Tensors have shapes. Some vocabulary:

- **Shape**: The length (number of elements) of each of the axes of a tensor.
- **Rank**: Number of tensor axes. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
- **Axis** or **Dimension**: A particular dimension of a tensor.
- **Size**: The total number of items in the tensor, the product shape vector.

```python
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```

```
Type of every element: <dtype: 'float32'>
Number of axes: 4
Shape of tensor: (3, 2, 4, 5)
Elements along axis 0 of tensor: 3
Elements along the last axis of tensor: 5
Total number of elements (3*2*4*5):  120
```

## Indexing

TensorFlow follows standard Python indexing rules, similar to [indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings), and the basic rules for NumPy indexing.

### Single-axis indexing

```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])

print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
```

```
Everything: [ 0  1  1  2  3  5  8 13 21 34]
Before 4: [0 1 1 2]
From 4 to the end: [ 3  5  8 13 21 34]
From 2, before 7: [1 2 3 5 8]
Every other item: [ 0  1  3  8 21]
Reversed: [34 21 13  8  5  3  2  1  1  0]
```

### Multi-axis indexing

Higher rank tensors are indexed by passing multiple indices.

#### 2-axis tensor example:

```python
print(rank_2_tensor.numpy())
```

```
[[1. 2.]
 [3. 4.]
 [5. 6.]]
```

```python
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")
```

```
Second row: [3. 4.]
Second column: [2. 4. 6.]
Last row: [5. 6.]
First item in last column: 2.0
Skip the first row:
[[3. 4.]
 [5. 6.]]
```

#### 3-axis tensor example:

```python
print(rank_3_tensor[:, :, 4])
```

```
tf.Tensor(
[[ 4  9]
 [14 19]
 [24 29]], shape=(3, 2), dtype=int32)
```

| Selecting the last feature across all locations in each example in the batch |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| ![A 3x2x5 tensor with all the values at the index-4 of the last axis selected.](https://www.tensorflow.org/guide/images/tensor/index1.png) | ![The selected values packed into a 2-axis tensor.](https://www.tensorflow.org/guide/images/tensor/index2.png) |

# Introduction to tensor slicing

## Extract tensor slices

```python
t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])

print(tf.slice(t1,
               begin=[1],
               size=[3]))
```

```
tf.Tensor([1 2 3], shape=(3,), dtype=int32)
```

equivalent to:

```python
print(t1[1:4])
```

For 2-dimensional tensors:

```python
t2 = tf.constant([[0, 1, 2, 3, 4],
                  [5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]])

print(t2[:-1, 1:3])
```

```
tf.Tensor(
[[ 1  2]
 [ 6  7]
 [11 12]], shape=(3, 2), dtype=int32)
```

![img](https://www.tensorflow.org/guide/images/tf_slicing/slice_2d_1.png)

You can also use [`tf.strided_slice`](https://www.tensorflow.org/api_docs/python/tf/strided_slice) to extract slices of tensors by 'striding' over the tensor dimensions.

You can use [`tf.slice`](https://www.tensorflow.org/api_docs/python/tf/slice) on higher dimensional tensors as well.

```python
t3 = tf.constant([[[1, 3, 5, 7],
                   [9, 11, 13, 15]],
                  [[17, 19, 21, 23],
                   [25, 27, 29, 31]]
                  ])

print(tf.slice(t3,
               begin=[1, 1, 0],
               size=[1, 1, 2]))
```

```
tf.Tensor([[[25 27]]], shape=(1, 1, 2), dtype=int32)
```

Use [`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather) to extract specific indices from a single axis of a tensor.



```python
print(tf.gather(t1,
                indices=[0, 3, 6]))

# This is similar to doing

t1[::3]
```

```
tf.Tensor([0 3 6], shape=(3,), dtype=int32)
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 3, 6], dtype=int32)>
```

![img](https://www.tensorflow.org/guide/images/tf_slicing/slice_1d_3.png)

[`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather) does not require indices to be evenly spaced.

```python
alphabet = tf.constant(list('abcdefghijklmnopqrstuvwxyz'))

print(tf.gather(alphabet,
                indices=[2, 0, 19, 18]))
```

```
tf.Tensor([b'c' b'a' b't' b's'], shape=(4,), dtype=string)
```

![img](https://www.tensorflow.org/guide/images/tf_slicing/gather_1.png)

To extract slices from multiple axes of a tensor, use [`tf.gather_nd`](https://www.tensorflow.org/api_docs/python/tf/gather_nd). This is useful when you want to gather the elements of a matrix as opposed to just its rows or columns.

```python
t4 = tf.constant([[0, 5],
                  [1, 6],
                  [2, 7],
                  [3, 8],
                  [4, 9]])

print(tf.gather_nd(t4,
                   indices=[[2], [3], [0]]))
```

```
tf.Tensor(
[[2 7]
 [3 8]
 [0 5]], shape=(3, 2), dtype=int32)
```

![img](https://www.tensorflow.org/guide/images/tf_slicing/gather_2.png)

## Insert data into tensors

Use [`tf.scatter_nd`](https://www.tensorflow.org/api_docs/python/tf/scatter_nd) to insert data at specific slices/indices of a tensor. Note that the tensor into which you insert values is zero-initialized.

```python
t6 = tf.constant([10])
indices = tf.constant([[1], [3], [5], [7], [9]])
data = tf.constant([2, 4, 6, 8, 10])

print(tf.scatter_nd(indices=indices,
                    updates=data,
                    shape=t6))
```

```
tf.Tensor([ 0  2  0  4  0  6  0  8  0 10], shape=(10,), dtype=int32)
```

​	