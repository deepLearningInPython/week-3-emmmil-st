import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    conv_size = input_array.size - kernel_array.size + 1
    return conv_size


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # setting output size
    conv_size = compute_output_size_1d(input_array, kernel_array)

    # initializing appropriately sized output array
    conv_out = np.zeros(conv_size)

    for step in range(conv_size):
        # dot product of kernel and appropriate slice of the input
        conv_out[step] = np.dot(kernel_array, input_array[step:(step+conv_size)])

    return conv_out

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # indexing out the row and column dimensions obtained from the .shape method
    conv_out = (input_matrix.shape[0] - kernel_matrix.shape[0] + 1, input_matrix.shape[1] - kernel_matrix.shape[1] + 1)
    return conv_out


# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # setting output size
    conv_size = compute_output_size_2d(input_matrix, kernel_matrix)

    # initializing appropriately sized output array
    conv_out = np.zeros(conv_size)

    for i in range(conv_size[0]): # looping over the output rows
        for j in range(conv_size[1]): # looping over the output columns
                # summing element-wise multiplication of kernel matrix and input sub-matrix
                conv_out[i,j] = np.sum(kernel_matrix * input_matrix[i:(i+conv_size[0]), j:(j+conv_size[1])])
                
    return conv_out


# -----------------------------------------------