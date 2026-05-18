import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################

        out = 1 / (1+ np.exp(-x))
        cache = x, out

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropagation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Sigmoid activation function           #
        ########################################################################

        x, out = cache
        dz = out * (1-out) # Sigmoid 函数的导数等于它本身乘以 1 减去它本身。
        dx = dout * dz # 激活步骤：逐元素相乘 

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################

        out = np.maximum(0,x)
        cache = x, out

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropagation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################

        x, out = cache
        dx = dout * (out>0)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################

    # reshape all the features of x to (N,D)
    x_reshaped = x.reshape(N, -1)
    # shape of w: (D, M), weights of every input feature connected to every output feature
    out = np.dot(x_reshaped, w) + b
    # shape of out: (N, M), for each input there are M output features
    # biases of shape (M,) will be copied for each row

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    ########################################################################

    # 1.reshape all the features of x to (N,D)
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1) 
    # rearrange all the number into the matrix with N rows
    # D = d1 * ... *  dk
    
    # 2.calculate the gradient with respect to reshaped x
    # dout:(N,M), w.T:(M,D)
    dx_reshaped = np.dot(dout, w.T)
    dx = dx_reshaped.reshape(x.shape) #(N,D)
    
    # 3.calculate the gradient of w
    # x_reshaped.T:(D,N), dout:(N,M)
    dw = np.dot(x_reshaped.T, dout) #(D,M)
    
    # 4.calculate the gradient of b, shape(M,)
    db = np.sum(dout, axis=0)
    # z对b的导数=1，loss对b的导数=dout
    # 所有样本都共享同一个偏置 b，我们只需要把所有样本的 dout 累加起来

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db