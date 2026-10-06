// Every interpretability claim in the UI points to one of these.
export const REFS = {
  rosenblatt1958: {
    short: 'Rosenblatt 1958',
    full: 'F. Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. Psychological Review 65(6), 1958.',
    url: 'https://doi.org/10.1037/h0042519',
  },
  olah2020: {
    short: 'Olah et al. 2020',
    full: 'C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter. Zoom In: An Introduction to Circuits. Distill, 2020.',
    url: 'https://distill.pub/2020/circuits/zoom-in/',
  },
  cammarata2020: {
    short: 'Cammarata et al. 2020',
    full: 'N. Cammarata, G. Goh, S. Carter, L. Schubert, M. Petrov, C. Olah. Curve Detectors. Distill, 2020.',
    url: 'https://distill.pub/2020/circuits/curve-detectors/',
  },
  olah2017: {
    short: 'Olah et al. 2017',
    full: 'C. Olah, A. Mordvintsev, L. Schubert. Feature Visualization. Distill, 2017.',
    url: 'https://distill.pub/2017/feature-visualization/',
  },
  erhan2009: {
    short: 'Erhan et al. 2009',
    full: 'D. Erhan, Y. Bengio, A. Courville, P. Vincent. Visualizing Higher-Layer Features of a Deep Network. Technical report 1341, Universite de Montreal, 2009.',
    url: null,
  },
  simonyan2013: {
    short: 'Simonyan et al. 2013',
    full: 'K. Simonyan, A. Vedaldi, A. Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. arXiv:1312.6034, 2013.',
    url: 'https://arxiv.org/abs/1312.6034',
  },
  smilkov2017: {
    short: 'Smilkov et al. 2017',
    full: 'D. Smilkov, S. Carter, D. Sculley, F. Viegas, M. Wattenberg. Direct-Manipulation Visualization of Deep Networks (TensorFlow Playground). arXiv:1708.03788, 2017.',
    url: 'https://playground.tensorflow.org',
  },
  harley2015: {
    short: 'Harley 2015',
    full: 'A. W. Harley. An Interactive Node-Link Visualization of Convolutional Neural Networks. International Symposium on Visual Computing (ISVC), 2015.',
    url: 'https://adamharley.com/nn_vis/',
  },
  elhage2022: {
    short: 'Elhage et al. 2022',
    full: 'N. Elhage et al. Toy Models of Superposition. Transformer Circuits Thread, 2022.',
    url: 'https://transformer-circuits.pub/2022/toy_model/index.html',
  },
  szegedy2013: {
    short: 'Szegedy et al. 2013',
    full: 'C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, R. Fergus. Intriguing properties of neural networks. arXiv:1312.6199, 2013.',
    url: 'https://arxiv.org/abs/1312.6199',
  },
  papernot2016: {
    short: 'Papernot et al. 2016',
    full: 'N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, A. Swami. The Limitations of Deep Learning in Adversarial Settings. IEEE EuroS&P, 2016.',
    url: 'https://arxiv.org/abs/1511.07528',
  },
  zeiler2014: {
    short: 'Zeiler & Fergus 2014',
    full: 'M. D. Zeiler, R. Fergus. Visualizing and Understanding Convolutional Networks. ECCV, 2014.',
    url: 'https://arxiv.org/abs/1311.2901',
  },
  alpaydin1998: {
    short: 'Alpaydin & Kaynak 1998',
    full: 'E. Alpaydin, C. Kaynak. Optical Recognition of Handwritten Digits. UCI Machine Learning Repository, 1998. Shipped as sklearn.datasets.load_digits.',
    url: 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
  },
  vaswani2017: {
    short: 'Vaswani et al. 2017',
    full: 'A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.',
    url: 'https://arxiv.org/abs/1706.03762',
  },
};

export const REF_ORDER = Object.keys(REFS);
