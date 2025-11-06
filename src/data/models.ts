import type { ModelInfo } from '../types';

export const neuralNetworkModels: ModelInfo[] = [
  {
    id: 'alexnet',
    name: 'AlexNet',
    year: 2012,
    description: 'AlexNet is a pioneering deep convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It marked the beginning of the deep learning revolution in computer vision.',
    keyFeatures: [
      'First successful deep CNN for ImageNet',
      'Uses ReLU activation for faster training',
      'Implements dropout for regularization',
      'Uses overlapping pooling',
      'GPU-accelerated training with data augmentation'
    ],
    totalParameters: '~60 million',
    inputSize: '227x227x3',
    outputClasses: 1000,
    accuracy: 'Top-5 error: 15.3%',
    paperLink: 'https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf',
    useCases: [
      'Image Classification',
      'Object Recognition',
      'Transfer Learning Base',
      'Feature Extraction'
    ],
    layers: [
      {
        name: 'Input',
        type: 'Input',
        outputShape: '227×227×3',
        description: 'RGB image input'
      },
      {
        name: 'Conv1',
        type: 'Convolution',
        outputShape: '55×55×96',
        parameters: 34944,
        kernelSize: '11×11',
        stride: '4',
        activation: 'ReLU',
        description: 'First convolutional layer with large receptive field'
      },
      {
        name: 'MaxPool1',
        type: 'MaxPooling',
        outputShape: '27×27×96',
        kernelSize: '3×3',
        stride: '2',
        description: 'Overlapping max pooling'
      },
      {
        name: 'Conv2',
        type: 'Convolution',
        outputShape: '27×27×256',
        parameters: 614656,
        kernelSize: '5×5',
        padding: '2',
        activation: 'ReLU',
        description: 'Second convolutional layer'
      },
      {
        name: 'MaxPool2',
        type: 'MaxPooling',
        outputShape: '13×13×256',
        kernelSize: '3×3',
        stride: '2',
        description: 'Second pooling layer'
      },
      {
        name: 'Conv3',
        type: 'Convolution',
        outputShape: '13×13×384',
        parameters: 885120,
        kernelSize: '3×3',
        padding: '1',
        activation: 'ReLU',
        description: 'Third convolutional layer'
      },
      {
        name: 'Conv4',
        type: 'Convolution',
        outputShape: '13×13×384',
        parameters: 1327488,
        kernelSize: '3×3',
        padding: '1',
        activation: 'ReLU',
        description: 'Fourth convolutional layer'
      },
      {
        name: 'Conv5',
        type: 'Convolution',
        outputShape: '13×13×256',
        parameters: 884992,
        kernelSize: '3×3',
        padding: '1',
        activation: 'ReLU',
        description: 'Fifth convolutional layer'
      },
      {
        name: 'MaxPool3',
        type: 'MaxPooling',
        outputShape: '6×6×256',
        kernelSize: '3×3',
        stride: '2',
        description: 'Third pooling layer'
      },
      {
        name: 'Flatten',
        type: 'Flatten',
        outputShape: '9216',
        description: 'Flatten feature maps for fully connected layers'
      },
      {
        name: 'FC1',
        type: 'Fully Connected',
        outputShape: '4096',
        parameters: 37752832,
        activation: 'ReLU',
        description: 'First fully connected layer with dropout (0.5)'
      },
      {
        name: 'FC2',
        type: 'Fully Connected',
        outputShape: '4096',
        parameters: 16781312,
        activation: 'ReLU',
        description: 'Second fully connected layer with dropout (0.5)'
      },
      {
        name: 'Output',
        type: 'Fully Connected',
        outputShape: '1000',
        parameters: 4097000,
        activation: 'Softmax',
        description: 'Output layer for 1000 ImageNet classes'
      }
    ]
  },
  {
    id: 'resnet50',
    name: 'ResNet-50',
    year: 2015,
    description: 'ResNet (Residual Network) introduced skip connections that allow gradients to flow through the network directly, enabling the training of very deep networks. ResNet-50 has 50 layers with residual blocks.',
    keyFeatures: [
      'Deep residual learning framework',
      'Skip connections prevent vanishing gradients',
      'Batch normalization in residual blocks',
      'Bottleneck architecture for efficiency',
      'Winner of ILSVRC 2015'
    ],
    totalParameters: '~25.6 million',
    inputSize: '224x224x3',
    outputClasses: 1000,
    accuracy: 'Top-5 error: 3.57%',
    paperLink: 'https://arxiv.org/abs/1512.03385',
    useCases: [
      'Image Classification',
      'Object Detection (backbone)',
      'Segmentation Tasks',
      'Transfer Learning',
      'Feature Extraction'
    ],
    layers: [
      {
        name: 'Input',
        type: 'Input',
        outputShape: '224×224×3',
        description: 'RGB image input'
      },
      {
        name: 'Conv1',
        type: 'Convolution',
        outputShape: '112×112×64',
        parameters: 9408,
        kernelSize: '7×7',
        stride: '2',
        padding: '3',
        activation: 'ReLU',
        description: 'Initial convolution with batch norm'
      },
      {
        name: 'MaxPool',
        type: 'MaxPooling',
        outputShape: '56×56×64',
        kernelSize: '3×3',
        stride: '2',
        description: 'Initial pooling layer'
      },
      {
        name: 'Conv2_x',
        type: 'Residual Block',
        outputShape: '56×56×256',
        parameters: 215808,
        description: '3 residual blocks with bottleneck (1×1, 3×3, 1×1 convolutions)'
      },
      {
        name: 'Conv3_x',
        type: 'Residual Block',
        outputShape: '28×28×512',
        parameters: 1219584,
        description: '4 residual blocks with bottleneck, stride 2 for downsampling'
      },
      {
        name: 'Conv4_x',
        type: 'Residual Block',
        outputShape: '14×14×1024',
        parameters: 7098368,
        description: '6 residual blocks with bottleneck, stride 2 for downsampling'
      },
      {
        name: 'Conv5_x',
        type: 'Residual Block',
        outputShape: '7×7×2048',
        parameters: 14964736,
        description: '3 residual blocks with bottleneck, stride 2 for downsampling'
      },
      {
        name: 'AvgPool',
        type: 'GlobalAvgPooling',
        outputShape: '2048',
        kernelSize: '7×7',
        description: 'Global average pooling'
      },
      {
        name: 'FC',
        type: 'Fully Connected',
        outputShape: '1000',
        parameters: 2049000,
        activation: 'Softmax',
        description: 'Output layer for 1000 classes'
      }
    ]
  },
  {
    id: 'googlenet',
    name: 'GoogLeNet (Inception v1)',
    year: 2014,
    description: 'GoogLeNet introduced the Inception module, which uses parallel convolutions of different sizes to capture multi-scale features efficiently. It achieved state-of-the-art performance while being computationally efficient.',
    keyFeatures: [
      'Inception modules with parallel convolutions',
      'Multi-scale feature extraction',
      '1×1 convolutions for dimensionality reduction',
      'Auxiliary classifiers for training deep networks',
      'Winner of ILSVRC 2014'
    ],
    totalParameters: '~6.8 million',
    inputSize: '224x224x3',
    outputClasses: 1000,
    accuracy: 'Top-5 error: 6.67%',
    paperLink: 'https://arxiv.org/abs/1409.4842',
    useCases: [
      'Image Classification',
      'Real-time Object Detection',
      'Mobile and Embedded Vision',
      'Resource-Constrained Applications'
    ],
    layers: [
      {
        name: 'Input',
        type: 'Input',
        outputShape: '224×224×3',
        description: 'RGB image input'
      },
      {
        name: 'Conv1',
        type: 'Convolution',
        outputShape: '112×112×64',
        parameters: 4672,
        kernelSize: '7×7',
        stride: '2',
        activation: 'ReLU',
        description: 'Initial convolution'
      },
      {
        name: 'MaxPool1',
        type: 'MaxPooling',
        outputShape: '56×56×64',
        kernelSize: '3×3',
        stride: '2',
        description: 'First max pooling'
      },
      {
        name: 'Conv2',
        type: 'Convolution',
        outputShape: '56×56×192',
        parameters: 112320,
        kernelSize: '3×3',
        activation: 'ReLU',
        description: 'Second convolution with LRN'
      },
      {
        name: 'MaxPool2',
        type: 'MaxPooling',
        outputShape: '28×28×192',
        kernelSize: '3×3',
        stride: '2',
        description: 'Second max pooling'
      },
      {
        name: 'Inception3a',
        type: 'Inception Module',
        outputShape: '28×28×256',
        parameters: 163696,
        description: 'First Inception module (1×1, 3×3, 5×5 convs, max pool)'
      },
      {
        name: 'Inception3b',
        type: 'Inception Module',
        outputShape: '28×28×480',
        parameters: 389328,
        description: 'Second Inception module'
      },
      {
        name: 'MaxPool3',
        type: 'MaxPooling',
        outputShape: '14×14×480',
        kernelSize: '3×3',
        stride: '2',
        description: 'Third max pooling'
      },
      {
        name: 'Inception4a',
        type: 'Inception Module',
        outputShape: '14×14×512',
        parameters: 376176,
        description: 'Inception module with auxiliary classifier'
      },
      {
        name: 'Inception4b',
        type: 'Inception Module',
        outputShape: '14×14×512',
        parameters: 449160,
        description: 'Inception module'
      },
      {
        name: 'Inception4c',
        type: 'Inception Module',
        outputShape: '14×14×512',
        parameters: 510104,
        description: 'Inception module'
      },
      {
        name: 'Inception4d',
        type: 'Inception Module',
        outputShape: '14×14×528',
        parameters: 497136,
        description: 'Inception module with auxiliary classifier'
      },
      {
        name: 'Inception4e',
        type: 'Inception Module',
        outputShape: '14×14×832',
        parameters: 722160,
        description: 'Inception module'
      },
      {
        name: 'MaxPool4',
        type: 'MaxPooling',
        outputShape: '7×7×832',
        kernelSize: '3×3',
        stride: '2',
        description: 'Fourth max pooling'
      },
      {
        name: 'Inception5a',
        type: 'Inception Module',
        outputShape: '7×7×832',
        parameters: 1053104,
        description: 'Inception module'
      },
      {
        name: 'Inception5b',
        type: 'Inception Module',
        outputShape: '7×7×1024',
        parameters: 1547504,
        description: 'Final Inception module'
      },
      {
        name: 'AvgPool',
        type: 'GlobalAvgPooling',
        outputShape: '1024',
        kernelSize: '7×7',
        description: 'Global average pooling'
      },
      {
        name: 'Dropout',
        type: 'Dropout',
        outputShape: '1024',
        description: 'Dropout (40%) for regularization'
      },
      {
        name: 'Output',
        type: 'Fully Connected',
        outputShape: '1000',
        parameters: 1025000,
        activation: 'Softmax',
        description: 'Output layer for 1000 classes'
      }
    ]
  },
  {
    id: 'vgg16',
    name: 'VGG-16',
    year: 2014,
    description: 'VGG-16 is a deep convolutional network with a simple and uniform architecture using only 3×3 convolutions. It demonstrated that network depth is a critical component for good performance.',
    keyFeatures: [
      'Very deep network with 16 weight layers',
      'Uniform architecture with 3×3 convolutions',
      'Multiple stacked convolutional layers',
      'Simple and homogeneous design',
      'Runner-up in ILSVRC 2014'
    ],
    totalParameters: '~138 million',
    inputSize: '224x224x3',
    outputClasses: 1000,
    accuracy: 'Top-5 error: 7.3%',
    paperLink: 'https://arxiv.org/abs/1409.1556',
    useCases: [
      'Image Classification',
      'Feature Extraction',
      'Style Transfer',
      'Transfer Learning'
    ],
    layers: [
      {
        name: 'Input',
        type: 'Input',
        outputShape: '224×224×3',
        description: 'RGB image input'
      },
      {
        name: 'Conv1_1',
        type: 'Convolution',
        outputShape: '224×224×64',
        parameters: 1792,
        kernelSize: '3×3',
        activation: 'ReLU',
        description: 'First conv block - layer 1'
      },
      {
        name: 'Conv1_2',
        type: 'Convolution',
        outputShape: '224×224×64',
        parameters: 36928,
        kernelSize: '3×3',
        activation: 'ReLU',
        description: 'First conv block - layer 2'
      },
      {
        name: 'MaxPool1',
        type: 'MaxPooling',
        outputShape: '112×112×64',
        kernelSize: '2×2',
        stride: '2',
        description: 'First pooling layer'
      },
      {
        name: 'Conv2_1',
        type: 'Convolution',
        outputShape: '112×112×128',
        parameters: 73856,
        kernelSize: '3×3',
        activation: 'ReLU',
        description: 'Second conv block - layer 1'
      },
      {
        name: 'Conv2_2',
        type: 'Convolution',
        outputShape: '112×112×128',
        parameters: 147584,
        kernelSize: '3×3',
        activation: 'ReLU',
        description: 'Second conv block - layer 2'
      },
      {
        name: 'MaxPool2',
        type: 'MaxPooling',
        outputShape: '56×56×128',
        kernelSize: '2×2',
        stride: '2',
        description: 'Second pooling layer'
      },
      {
        name: 'Conv3_1-3',
        type: 'Convolution Block',
        outputShape: '56×56×256',
        parameters: 1180160,
        description: '3 convolutional layers with 3×3 kernels'
      },
      {
        name: 'MaxPool3',
        type: 'MaxPooling',
        outputShape: '28×28×256',
        kernelSize: '2×2',
        stride: '2',
        description: 'Third pooling layer'
      },
      {
        name: 'Conv4_1-3',
        type: 'Convolution Block',
        outputShape: '28×28×512',
        parameters: 4719104,
        description: '3 convolutional layers with 3×3 kernels'
      },
      {
        name: 'MaxPool4',
        type: 'MaxPooling',
        outputShape: '14×14×512',
        kernelSize: '2×2',
        stride: '2',
        description: 'Fourth pooling layer'
      },
      {
        name: 'Conv5_1-3',
        type: 'Convolution Block',
        outputShape: '14×14×512',
        parameters: 7079424,
        description: '3 convolutional layers with 3×3 kernels'
      },
      {
        name: 'MaxPool5',
        type: 'MaxPooling',
        outputShape: '7×7×512',
        kernelSize: '2×2',
        stride: '2',
        description: 'Fifth pooling layer'
      },
      {
        name: 'FC1',
        type: 'Fully Connected',
        outputShape: '4096',
        parameters: 102764544,
        activation: 'ReLU',
        description: 'First fully connected layer'
      },
      {
        name: 'FC2',
        type: 'Fully Connected',
        outputShape: '4096',
        parameters: 16781312,
        activation: 'ReLU',
        description: 'Second fully connected layer'
      },
      {
        name: 'Output',
        type: 'Fully Connected',
        outputShape: '1000',
        parameters: 4097000,
        activation: 'Softmax',
        description: 'Output layer'
      }
    ]
  },
  {
    id: 'mobilenet',
    name: 'MobileNet',
    year: 2017,
    description: 'MobileNet is an efficient CNN architecture designed for mobile and embedded devices. It uses depthwise separable convolutions to significantly reduce computation and model size while maintaining accuracy.',
    keyFeatures: [
      'Depthwise separable convolutions for efficiency',
      'Lightweight: only ~4.2 million parameters',
      'Fast inference on mobile devices',
      'Width multiplier for model scaling',
      'Optimized for resource-constrained environments'
    ],
    totalParameters: '~4.2 million',
    inputSize: '224x224x3',
    outputClasses: 1000,
    accuracy: 'Top-1: 70.6%, Top-5: 89.5%',
    paperLink: 'https://arxiv.org/abs/1704.04861',
    useCases: [
      'Mobile Image Classification',
      'Real-time Object Detection',
      'Embedded Systems',
      'Edge Computing',
      'Resource-Constrained Deployment'
    ],
    layers: [
      {
        name: 'Input',
        type: 'Input',
        outputShape: '224×224×3',
        description: 'RGB image input'
      },
      {
        name: 'Conv1',
        type: 'Convolution',
        outputShape: '112×112×32',
        parameters: 896,
        kernelSize: '3×3',
        stride: '2',
        activation: 'ReLU6',
        description: 'Initial standard convolution'
      },
      {
        name: 'DepthwiseConv1',
        type: 'DepthwiseConv',
        outputShape: '112×112×32',
        parameters: 320,
        kernelSize: '3×3',
        stride: '1',
        description: 'Depthwise separable convolution block 1'
      },
      {
        name: 'PointwiseConv1',
        type: 'Convolution',
        outputShape: '112×112×64',
        parameters: 2112,
        kernelSize: '1×1',
        activation: 'ReLU6',
        description: 'Pointwise convolution (1×1)'
      },
      {
        name: 'DepthwiseConv2',
        type: 'DepthwiseConv',
        outputShape: '56×56×64',
        parameters: 640,
        kernelSize: '3×3',
        stride: '2',
        description: 'Depthwise separable with stride 2'
      },
      {
        name: 'PointwiseConv2',
        type: 'Convolution',
        outputShape: '56×56×128',
        parameters: 8320,
        kernelSize: '1×1',
        activation: 'ReLU6',
        description: 'Expand to 128 channels'
      },
      {
        name: 'DepthwiseConv3',
        type: 'DepthwiseConv',
        outputShape: '56×56×128',
        parameters: 1280,
        kernelSize: '3×3',
        description: 'Depthwise block 3'
      },
      {
        name: 'PointwiseConv3',
        type: 'Convolution',
        outputShape: '56×56×128',
        parameters: 16512,
        kernelSize: '1×1',
        activation: 'ReLU6',
        description: 'Maintain 128 channels'
      },
      {
        name: 'DepthwiseConv4',
        type: 'DepthwiseConv',
        outputShape: '28×28×128',
        parameters: 1280,
        kernelSize: '3×3',
        stride: '2',
        description: 'Downsample to 28×28'
      },
      {
        name: 'PointwiseConv4',
        type: 'Convolution',
        outputShape: '28×28×256',
        parameters: 33024,
        kernelSize: '1×1',
        activation: 'ReLU6',
        description: 'Expand to 256 channels'
      },
      {
        name: 'GlobalAvgPool',
        type: 'GlobalAveragePooling',
        outputShape: '1×1×1024',
        description: 'Global average pooling'
      },
      {
        name: 'Dropout',
        type: 'Dropout',
        outputShape: '1024',
        description: 'Dropout for regularization (0.001)'
      },
      {
        name: 'FC',
        type: 'Fully Connected',
        outputShape: '1000',
        parameters: 1025000,
        activation: 'Softmax',
        description: 'Output classification layer'
      }
    ]
  }
];
