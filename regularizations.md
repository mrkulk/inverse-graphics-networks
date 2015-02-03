- preventing negative ink
  - upon closer reading, this seems to be something he does _for each capsule_, not at the end
- determinant of geoPose regularization
- tiny L2 weight decay with [0, 0.1] grace interval

- dropout on CC distortion
- randomness in mixture manager
    - penalty on mixture manager for failing to use mixture components

## things that need their own gradients
1. each template
2. hidden layer of encoder
3. output layer of encoder
4. hidden encoder biases
5. hidden output biases
6. every individual pose variable
