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

- You can convert a tensor to a NumPy array either using `np.array` or the `tensor.numpy` method.


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

## Manipulating Shapes

The [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape) operation is fast and cheap as the underlying data does not need to be duplicated.

```python
# You can reshape a tensor to a new shape.
# Note that you're passing in a list
reshaped = tf.reshape(x, [1, 3])
```

If you flatten a tensor you can see what order it is laid out in memory.

```python
print(rank_3_tensor)
```

```
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
```

```python
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))
```

```
tf.Tensor(
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29], shape=(30,), dtype=int32)
```

Adding `-1` to the reshape function means "whatever fits".

```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
```

```
tf.Tensor(
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]], shape=(6, 5), dtype=int32) 

tf.Tensor(
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)
```

## More on `DTypes`

To inspect a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)'s data type use the [`Tensor.dtype`](https://www.tensorflow.org/api_docs/python/tf/Tensor#dtype) property.

You can cast from type to type.

```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
```

## Broadcasting

```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

```
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
```

another example:

```python
# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

```
tf.Tensor(
[[1]
 [2]
 [3]], shape=(3, 1), dtype=int32) 

tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 

tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

This is equivalent to this example without broadcasting:

```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading
```

```
tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

You see what broadcasting looks like using [`tf.broadcast_to`](https://www.tensorflow.org/api_docs/python/tf/broadcast_to).

```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

```
tf.Tensor(
[[1 2 3]
 [1 2 3]
 [1 2 3]], shape=(3, 3), dtype=int32)
```

It can get even more complicated. [This section](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) of Jake VanderPlas's book *Python Data Science Handbook* shows more broadcasting tricks (again in NumPy).

## tf.convert_to_tensor

Ops call `convert_to_tensor` on non-tensor arguments. There is a registry of conversions, and most object classes like NumPy's `ndarray`, `TensorShape`, Python lists, and [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) will all convert automatically.

## Ragged Tensors

create a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) using [`tf.ragged.constant`](https://www.tensorflow.org/api_docs/python/tf/ragged/constant):

```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
print(ragged_tensor.shape)
```

```
<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>

(4, None)
```

![A 2-axis ragged tensor, each row can have a different length.](https://www.tensorflow.org/guide/images/tensor/ragged.png)

## String tensors

```python
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
```

```python
# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))
```

```
tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
```

```python
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
```

```python
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))
```

```
<tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>
```

![Splitting multiple strings returns a tf.RaggedTensor](https://www.tensorflow.org/guide/images/tensor/string-split.png)



### tf.string.to_number

```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

```
tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)
```



- Although you can't use [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast) to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.

```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
```

```
Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)
```

## Sparse tensors

TensorFlow supports [`tf.sparse.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) and related operations to store sparse data efficiently.

```python
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
```

```
SparseTensor(indices=tf.Tensor(
[[0 0]
 [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 

tf.Tensor(
[[1 0 0 0]
 [0 0 2 0]
 [0 0 0 0]], shape=(3, 4), dtype=int32)
```

![An 3x4 grid, with values in only two of the cells.](https://www.tensorflow.org/guide/images/tensor/sparse.png)

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

