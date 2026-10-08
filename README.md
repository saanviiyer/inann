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

5. **Feature rooms.** Every neuron, conv channel and SAE feature has a room: activation histogram over the held-out digits, its top 12 digits floating in a ring around it, its written explanation with a simulation score, and a "test this neuron" pad (draw or type). A concept search in the header ("horizontal line", "the 7 detector") flies the camera to the match. This follows Neuronpedia (Lin 2023).
6. **Steer it.** Clamp any unit to a value and watch the answer reroute live. The presets were found by a search over every unit (`scripts/find_presets.mjs`) and ranked by selectivity: how much the clamp moves one digit class, minus how much it sends every other digit to the same answer. Each preset shows its full counts.
7. **Neurons vs features.** A top-k sparse autoencoder (256 features over the 32 dense neurons, k = 4) replaces the dense room when the header switch is on **Features**. A comparison view splits the most mixed neuron into the features that carry it.
8. **Latent space.** The 450 held-out digits are shown as numerals in the PCA space of any layer, and they glide between layers when you switch. Your input appears as a live point. The view also offers nearest neighbours, a walk between two digits drawn as a trail, and two separability measures computed in the full space.

Keys: `space` pauses all animation, `F` toggles follow-the-signal, `W A S D Q E` fly (hold shift to go faster).

## The model

`scripts/train.py` trains the network offline, with PyTorch and seed 0, and writes `src/data/model.json` (weights) and `src/data/digits.json` (the 450 held-out digits). The browser never trains.

- Architecture: conv 6@3x3, then ReLU and maxpool. Then conv 12@3x3, then ReLU and maxpool. Then dense 32 with ReLU, then 10 outputs. That is 682 rendered neurons.
- Data: scikit-learn `load_digits`, the 8x8 UCI optdigits set (Alpaydin & Kaynak 1998). **It is not MNIST.** MNIST is 28x28, and the UI says so.
- Accuracy: 97.3% train, 95.8% on 450 held-out digits.
- `npm run check` checks that the JS forward pass reproduces the PyTorch logits (max diff about 3e-4, argmax 20/20). It also checks that the hand-written backward pass matches finite differences.

Drawn input goes through the optdigits recipe: fit the ink into a 32x32 bitmap, then count the ink in each 4x4 block.

## Offline scripts (nothing trains in the browser)

| script | what it writes | time |
|---|---|---|
| `scripts/train.py` | `src/data/model.json`, `digits.json` | ~1 min |
| `scripts/train_sae.py` | `src/data/sae.json` | ~20 s on CPU |
| `scripts/find_presets.mjs` | `src/data/presets.json` | ~10 s |
| `scripts/explain.py` | `src/data/explanations.json` | about 166 API calls |
| `scripts/measure_purity.py` | prints the neuron vs feature comparison | seconds |

`explain.py` follows Bills et al. 2023, adapted to images:
1. Claude (`claude-opus-5-5`) writes one sentence per unit from its top 12 digits, using the even-indexed held-out half.
2. In a fresh request, Claude sees only that sentence and predicts the unit's activation on 20 digits from the odd half: 10 top and 10 random.
3. The score is the Pearson r between those predictions and the real activations. It is shown in the app, and an r below 0.3 gets a "weak explanation" badge.

Only units active on at least 5 digits in each half are explained (83 of 306). The script needs `ANTHROPIC_API_KEY` in the environment or in `.env.local`, which is gitignored. Results are cached in `scripts/.cache`, so an interrupted run resumes where it stopped. Until it has run, the app labels the explanations as not generated, and search falls back to measured digit matches.

## Measured results (held-out digits)

- **SAE:** keeps 92.9% of the variance in the 32 dense neurons. Accuracy is 93.3% when the reconstruction replaces the neurons, against 95.8% for the original network. 184 features are active on held-out digits.
- **Neurons vs features:** take the share of a unit's top 12 digits that are the same digit. The median is 0.67 for neurons (n = 19) and 0.83 for SAE features (n = 46). Only 5% of neurons are fully one digit, against 41% of features. Only units active on at least 12 digits are counted, so this is a rough proxy.
- **Latent space:** 5-nearest-neighbour accuracy is flat across layers (95.8% on raw pixels), because these digits are easy. The between/within class distance ratio does grow with depth: 1.29 at the input pixels, 2.26 at the dense layer and 10.4 at the output.
- **Steering:** the requested "1s become 0s" demo exists only as a non-selective lever. SAE feature 251 turns 42 of 45 held-out 1s into 0s, but it also turns 222 of 341 other digits into 0s. The app says so.

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

## Lab: bring your own model

Open the app, switch the header to **Lab: your model** (or go to `/#lab`), and drop in an `.onnx` file. It runs on your machine with ONNX Runtime Web, and the file never leaves the browser. Four example models are one click away: an MNIST CNN, an MNIST MLP, the Tour's 8x8 CNN, and a parentheses transformer.

What you get for any model:
- **Rooms in 3D**, one per chosen tensor.
  - Conv maps are tiles, one per channel.
  - Sequences are token-by-feature maps.
  - Attention weights are one tile per head.
  - Small vectors are sphere clouds, and small outputs are labelled pods.
  - The defaults are activation outputs, Softmax, LayerNorm and the final output, with one room per residual block for deep nets. You can choose any tensor.
- **A room per unit.** It shows the unit's value now, a histogram over your dataset, the top 12 items with where the unit fired, label purity, and an activation-weighted average input. You can also clamp a channel (every position, or one) and watch the answer change.
- **A sparse autoencoder on any layer**, trained in a Web Worker in the page.
  - Conv maps give one C-dim vector per pixel, and sequences one per token.
  - The SAE docks beside its layer as a disk of features, each with its own room and clamp. It reports:
    - held-out variance explained
    - how often the answer survives replacing the layer with its reconstruction
    - neuron and feature label purity
  - You can download the SAE as JSON and load it again later.
- **Latent space** of any stored layer, or of the SAE's features, as a PCA cloud. It also shows full-space nearest-neighbour accuracy and the spacing ratio.
- **Incoming weights** for any unit, read from the model file. Conv kernels appear one tile per input channel. A dense layer's weights are laid out like the tensor it read: an MLP's first layer shows the 28x28 picture each neuron weighs.
- **"What does it control?"** clamps a channel or feature on 100 items, spread evenly over the dataset, and counts how the answer moves, with selectivity. In the MNIST CNN, a channel whose top items are mostly 7s turns other digits into 5s when clamped high, with low selectivity. That is why responses and causes are both shown.
- **Measured summaries and search.** Every channel and feature gets a one-line summary built only from counts (for example "11 of top 12 are 5 · fires on 40% of items · usually upper left"). The Lab search box matches classes, positions and "rare" or "common" against these summaries; it does not use language-model explanations.
- **Transformers.** Integer token inputs are tokenized through a `vocab` in the meta file. Attention maps render as one tile per head, and selecting a cell draws arcs from every key token to the query token over the input row.
- Drop a model anywhere on the page, pick units by number, and use the fly-to list to move through deep models. ResNet-18 (49 ops) and MobileNetV2 (170 ops) load and run.

Inputs: draw (with MNIST-style centring), upload a picture, type tokens through the model's vocabulary, or use dataset items. Datasets can be:
- a folder of images, with labels taken from folder names or `label_name.png`
- `.npy` arrays (x, plus an optional y)
- a dataset `.json`

You can also load a model by link: `/?model=URL&meta=URL&data=URL#lab`. The server has to allow cross-origin requests.

How it works:
- `src/lib/lab/graph.js` decodes the ONNX protobuf.
- Every float tensor is added as a graph output, which makes the probe model. Integer shape tensors are found statically, and any ORT type error is retried.
- To clamp a tensor, its producer's output is renamed and the tensor becomes a graph input, which makes the override model. The rest of the graph runs normally, skip connections included.
- `npm test` checks all of this against the official onnx Python tooling, a residual-plus-BatchNorm fixture, a transformer fixture and the token path.

### Written explanations for your model

`scripts/explain_lab.py` runs the Tour's explain-and-simulate loop (Bills et al. 2023) on any ONNX model. It works on the channels of one tensor, or on the features of an SAE downloaded from the Lab. Image units are shown to Claude as montages with a box where the unit fired; token units are shown as text with the firing token in brackets. Load the resulting JSON with **Load explanations** (or drop it on the page). Each sentence then appears in its room with its simulation score, and the unit search covers it.

```bash
python3 scripts/explain_lab.py --model m.onnx --meta m.meta.json --data dataset.json --tensor "/Relu_2_output_0"
python3 scripts/explain_lab.py --model m.onnx --data dataset.json --sae sae_layer.json --dry-run   # previews only, no API calls
```

### Exporting a model

```python
torch.onnx.export(model.eval(), example_input, "model.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
```

A dynamic batch axis makes the dataset pass much faster. Keras and TensorFlow models convert with `tf2onnx`.

### Optional `model.meta.json` (drop it next to the `.onnx`)

```json
{
  "name": "My classifier",
  "description": "one line shown in the model card",
  "input": {"kind": "image", "norm": "imagenet", "preprocess": "fit"},
  "classes": ["cat", "dog"],
  "dataset": "my_dataset.json"
}
```

- `input.kind`: `image`, `vector` or `tokens`.
- `input.norm` (images): `unit` (0..1), `sym` (-1..1), `byte` (0..255), `mnist` or `imagenet`.
- `input.preprocess` (drawings): `mnist`, `fit` or `stretch`.
- For tokens: `"vocab": [...]`, `"tokenizer": "char"`, and `"dims": [length]` when the length is symbolic.

A dataset `.json` has the form `{shape, labels?, classes?, data}`. `data` is base64 of uint8 pixels by default, or of int32/float32 values with `"dtype"` set. `scripts/build_examples.py` writes examples of each.

### Limits, stated plainly

- **No gradients.** ONNX Runtime Web does not provide them, so the Lab has no gradient feature visualization or saliency. The average input stands in for them.
- **Inputs:** one driven input; any other inputs get zeros. Inputs must be float32 or integer token ids.
- **Storage:** full activations are kept up to 30 million values per room. Larger rooms keep only per-channel summaries, so they cannot host an SAE or a latent view.
- **Speed:** everything runs in WebAssembly on one thread. Measured on this development laptop while it was heavily loaded (load average about 90): ResNet-18 at 224x224 took 1.35 s per image and the MNIST CNN 8 ms. Heavy models therefore analyse 100 items by default.
