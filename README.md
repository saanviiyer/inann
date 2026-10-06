# inann

**in a neural network.** An interpretability viewer that lets you walk through a small trained CNN instead of treating it as a black box.

```
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/, no backend
```

## Chapters

1. **Why this exists.** A perceptron learns a 2D dataset in about 30 seconds (Rosenblatt 1958).
2. **Inside the network.** The CNN as rooms stacked in 3D: input, Conv 1, Conv 2, Dense, Output. Draw a digit and activations light up layer by layer. Click any neuron for its activation, incoming weights (glowing edges) and receptive field. `F` follows the strongest path, X-ray spreads or stacks the layers.
3. **What neurons see.** Gradient-ascent feature visualization, animated and floating beside the neuron. Saliency is the 2D companion. You also get the top 5 real patches (the polysemanticity check) and a neighborhood map: cosine similarity over held-out digits, laid out with classical MDS.
4. **Break it.** A greedy few-pixel adversarial nudge in the spirit of JSMA (Papernot et al. 2016), plus an occlusion slider and an occlusion sensitivity map (Zeiler & Fergus 2014).

Keys: `space` pauses all animation, `F` toggles follow-the-signal, `W A S D Q E` fly (hold shift to go faster).

## The model

`scripts/train.py` trains the network offline, with PyTorch and seed 0, and writes `src/data/model.json` (weights) and `src/data/digits.json` (the 450 held-out digits). The browser never trains.

- Architecture: conv 6@3x3, then ReLU and maxpool. Then conv 12@3x3, then ReLU and maxpool. Then dense 32 with ReLU, then 10 outputs. That is 682 rendered neurons.
- Data: scikit-learn `load_digits`, the 8x8 UCI optdigits set (Alpaydin & Kaynak 1998). **It is not MNIST.** MNIST is 28x28, and the UI says so.
- Accuracy: 97.3% train, 95.8% on 450 held-out digits.
- `npm run check` checks that the JS forward pass reproduces the PyTorch logits (max diff about 3e-4, argmax 20/20). It also checks that the hand-written backward pass matches finite differences.

Drawn input goes through the optdigits recipe: fit the ink into a 32x32 bitmap, then count the ink in each 4x4 block.

## Honesty rules the UI follows

- Every number on screen is computed live from the bundled model and data. None are typed in.
- Each claim cites its source inline. The full list is under "Sources".
- "Not shown here" labels mark what the demo cannot honestly show:
  - pooling layers and biases are not drawn
  - feature-vis regularizers are not used
  - the demo cannot say whether the neighborhood clusters are edge or curve detectors
  - imperceptible attacks do not exist at 8x8

## Layout

```
scripts/train.py           offline training, writes src/data/*.json
scripts/check_parity.mjs   JS vs PyTorch parity and gradient check
src/lib/net.js             forward, input gradients, receptive fields, strongest path, activation maximisation
src/lib/model.js           loads weights, precomputes held-out activations and brightness scales
src/lib/analysis.js        top patches, cosine similarity, MDS
src/components/Scene.jsx   react-three-fiber space
```
