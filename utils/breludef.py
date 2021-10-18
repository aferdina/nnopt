from torch.autograd import Function


class boundedrelu(Function):
    '''
    Implementation of bounded relu activation function activation function.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        - See BoundedReLU paper:
        https://arxiv.org/pdf/1709.04054.pdf

    Examples:
        >>> boundedrelu_activation = bounded.apply
        >>> t = torch.randn((5,5), dtype=torch.float, requires_grad = True)
        >>> t = boundedrelu_activation(t)
    '''
    #both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input) # save input for backward pass

        # clone the input tensor
        output = input.clone()

        # apply bounded ReLU function 
        output= output.clamp(min=0,max=1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None # set output to None

        input, = ctx.saved_tensors # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            # set grad_input for even_indices
            grad_input[input < 0] = 0
            grad_input[input >1] = 0

        return grad_input