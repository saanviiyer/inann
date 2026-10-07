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
  lin2023: {
    short: 'Lin 2023',
    full: 'J. Lin. Neuronpedia: Interactive Reference and Tooling for Analyzing Neural Networks. 2023. Software, neuronpedia.org. The source of the feature dashboard, test-activation, explanation search and steering ideas used here.',
    url: 'https://www.neuronpedia.org',
  },
  bills2023: {
    short: 'Bills et al. 2023',
    full: 'S. Bills, N. Cammarata, D. Mossing, H. Tillman, L. Gao, G. Goh, I. Sutskever, J. Leike, J. Wu, W. Saunders. Language models can explain neurons in language models. OpenAI, 2023.',
    url: 'https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html',
  },
  bricken2023: {
    short: 'Bricken et al. 2023',
    full: 'T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, et al. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread, 2023.',
    url: 'https://transformer-circuits.pub/2023/monosemantic-features/index.html',
  },
  templeton2024: {
    short: 'Templeton et al. 2024',
    full: 'A. Templeton, T. Conerly, J. Marcus, et al. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread, 2024.',
    url: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html',
  },
  lieberum2024: {
    short: 'Lieberum et al. 2024',
    full: 'T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramar, A. Dragan, R. Shah, N. Nanda. Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147, 2024.',
    url: 'https://arxiv.org/abs/2408.05147',
  },
  gao2024: {
    short: 'Gao et al. 2024',
    full: 'L. Gao, T. Dupre la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, J. Wu. Scaling and evaluating sparse autoencoders. arXiv:2406.04093, 2024. (top-k SAEs, used here)',
    url: 'https://arxiv.org/abs/2406.04093',
  },
  makhzani2013: {
    short: 'Makhzani & Frey 2013',
    full: 'A. Makhzani, B. Frey. k-Sparse Autoencoders. arXiv:1312.5663, 2013.',
    url: 'https://arxiv.org/abs/1312.5663',
  },
  jolliffe2016: {
    short: 'Jolliffe & Cadima 2016',
    full: 'I. T. Jolliffe, J. Cadima. Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A 374, 2016.',
    url: 'https://doi.org/10.1098/rsta.2015.0202',
  },
  wattenberg2016: {
    short: 'Wattenberg et al. 2016',
    full: 'M. Wattenberg, F. Viegas, I. Johnson. How to Use t-SNE Effectively. Distill, 2016.',
    url: 'https://distill.pub/2016/misread-tsne/',
  },
  alain2016: {
    short: 'Alain & Bengio 2016',
    full: 'G. Alain, Y. Bengio. Understanding intermediate layers using linear classifier probes. arXiv:1610.01644, 2016.',
    url: 'https://arxiv.org/abs/1610.01644',
  },
};

export const REF_ORDER = Object.keys(REFS);
