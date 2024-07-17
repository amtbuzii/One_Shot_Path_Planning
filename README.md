# One_Shot_Path_Planning

## One-Shot Multi-Path Planning Using Fully Convolutional Networks

- This repository contains the implementation and codebase inspired by the research article "One-Shot Multi-Path Planning Using Fully Convolutional Networks in a Comparison to Other Algorithms" published in Frontiers in Neurorobotics.
Overview

 - The objective of this repository is to provide a comprehensive guide and implementation for one-shot multi-path planning using fully convolutional networks (FCNs).
 - This repository replicates and extends the methodologies discussed in the research paper to facilitate further development and research in this field.
 - [link](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2020.600984/full)


### Network inputs - 3 N*N matrixes:
  - s_maps matrix - indicates the start point.
  - g_maps matrix - indicates the endpoint.
  - input matrix - indicates the positions of the obstacles. (1 = obstacle, 0 - free location)
    
### Network output - 1 N*N matrix:
 - outputs matrix - indicates where the path should pass.
 - the output is a matrix of probabilities from -1 to 1.
 

### Neural Network Architecture

1. **First Convolutional Layer**
   - Filters: 64
   - Kernel Size: 3x3
   - Strides: 1x1
   - Padding: "same"
   - Activation: 'relu'
   - Batch Normalization

2. **Intermediate Convolutional Layers** (repeated `HIDDEN_LAYERS` times)
   - Filters: 64
   - Kernel Size: 3x3
   - Strides: 1x1
   - Padding: "same"
   - Activation: 'relu'
   - Batch Normalization

3. **Final Convolutional Layer**
   - Filters: 1
   - Kernel Size: 3x3
   - Strides: 1x1
   - Padding: "same"
   - Activation: 'sigmoid'
   - Batch Normalization
   - Dropout: 10%

4. **Model Compilation**
   - Optimizer: 'adam'
   - Loss: 'mse'
   - Metrics: ['accuracy']

5. Implement the paper code using the new version of Keras and TensorFlow.
  - TensorFlow version: 2.16.2
  - Keras version: 3.1.1
  - NVIDIA driver version: 555.42.02
  - CUDA Version: 12.5
   
### Research Conclusions:

- Try different types of Loss Functions - MSE, Accuracy, Binary_Accuracy - Accuracy gets the best result so far. [link](https://keras.io/api/losses/)
- Generate new data set to train the model - Not a maze-like environment, a "real-life" environment - 1-3 obstacles the path should pass.
- World size and hidden-Layer:
  - The hidden layer amount depends on the world size.
  - The paper mentions that and recommends 31 Hidden layers for 30*30.
  - Use larger hidden layer (50) **not** shown better results.
- Trainning set size - a huge differnt after trainnig on **100K** data set.

### Test and Result:

   
