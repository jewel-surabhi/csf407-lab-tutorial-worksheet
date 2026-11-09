# Lab 1 — Neural Models: Learning, Depth, Activations, Output Layers

**Author:** Surabhi Agarwal
**Code:** [lab1_xor.py](lab1_xor.py)

---

## Task 1 — Understand the problem before coding

**Input and output spaces**
- `X = {0, 1}²` — four binary input points.
- `Y = {0, 1}` — a single yes/no "disagreement" flag.

**Labelled examples (XOR)**

| x₁ | x₂ | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

**Sketch.** On the `(x₁, x₂)` plane, the positive points `(0,1)` and `(1,0)` lie on one diagonal; the negative points `(0,0)` and `(1,1)` on the other. The two diagonals cross.

**Not linearly separable.** Any straight line partitions the plane into two half-planes. Because the two positives and the two negatives sit on *crossing* diagonals, no single line can put both positives on one side and both negatives on the other. One line always leaves exactly one positive and one negative on the same side.

**Prediction for a single affine + sigmoid.** The model `σ(w·x + b)` can only realise a single linear boundary, so it cannot fit XOR. Training will stall at the loss floor `ln 2 ≈ 0.693` and output ≈ 0.5 for every input; at best 2 of 4 labels are correct.

**Think About It.** The XOR experiment tests the scientific claim that *representation depth alone — without nonlinearity — does not expand what a network can compute*. Only four data points are needed because that claim is a statement about the *function class*, not about generalisation.

---

## Task 2 — Design the intelligent agent

**Architecture**

```
2 inputs → fc1 (Linear 2→2) → nonlinearity (sigmoid / tanh / ReLU)
          → fc2 (Linear 2→1) → sigmoid (via BCEWithLogitsLoss)
```

Loss: binary cross-entropy. Optimiser: SGD, lr = 0.5, full-batch, 5 000 steps.

**Why the hidden nonlinearity is scientifically necessary**
Composition of affine maps is affine:
`W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`.
So without a nonlinearity between `fc1` and `fc2` the whole network is equivalent to one `Linear(2,1)` — same expressive power as the model that cannot solve XOR. A nonlinearity breaks this algebraic collapse and lets the network bend the decision surface into two half-planes whose intersection realises XOR.

**Why sigmoid + BCE is a sensible engineering pairing**
Sigmoid maps a logit `z` to `σ(z) ∈ (0, 1)`, which we interpret as `P(y = 1 | x)`. BCE is the negative log-likelihood of that Bernoulli. When implemented together as `BCEWithLogitsLoss`, the log and the exponential cancel, giving the numerically stable gradient `σ(z) − y` with no explicit `log σ(z)` that could overflow when logits are extreme.

**Three validation checks**
1. **Final loss** below ≈ 0.05 (far from the `ln 2` plateau).
2. **All 4 thresholded predictions** match the labels at threshold 0.5.
3. **First-layer gradient `fc1.weight.grad`** is non-zero at step 0 and the weights actually move.

**Think About It.** Hidden units have no targets. What determines what each unit should compute is the *loss signal* delivered to them by backpropagation: the chain rule routes `∂L/∂h_j` back through `fc2`, so each hidden unit is shaped to produce the feature that most reduces the downstream loss.

---

## Task 3 — LLM-assisted implementation

**Prompt used**

> Generate minimal PyTorch code for a 2–2–1 neural network that learns XOR
> on the four examples `(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0`. Use a tanh
> hidden activation and `BCEWithLogitsLoss`. Train full-batch with SGD for
> 5 000 steps at `lr = 0.5`. Seed `torch.manual_seed(0)`. After training,
> print the final loss, the four probabilities, the thresholded predictions,
> and `fc1.weight.grad`. Do not change the architecture or task.

**Inspecting the code before running.**
- Forward pass: `self.fc2(self.act(self.fc1(x)))`.
- Scalar loss: `BCEWithLogitsLoss()(logits, y)`.
- Reverse-mode AD: `loss.backward()` — this is where `parameter.grad` gets populated.
- Parameter update: `optimizer.step()`.

**Corrections made.** (Fill in after running.)
- e.g. raised `lr` from 0.1 to 0.5 so the sigmoid hidden version converges in 5 000 steps.
- e.g. switched default activation to tanh after seeing one sigmoid run get stuck.

---

## Task 4 — Execute, test, diagnose

### Part A — Basic learning check

Paste the printed block here after `python lab1_xor.py`:

```
initial loss = ...
final   loss = ...
inputs | prob      | pred | target
 [0,0] | ...       | ...  | 0
 [0,1] | ...       | ...  | 1
 [1,0] | ...       | ...  | 1
 [1,1] | ...       | ...  | 0
```

### Part B — Backpropagation check

`parameter.grad` is exactly `∂L/∂parameter`, computed by reverse-mode autodiff from the scalar loss. For `fc1.weight` it has shape `(2, 2)` — one row per hidden unit, one column per input.

Because `BCEWithLogitsLoss` uses mean reduction over the 4 examples, the printed gradient is the *average* of the four per-example gradients, which follows from `∇(1/N Σᵢ Lᵢ) = (1/N) Σᵢ ∇Lᵢ`.

### Part C — Symmetry experiment

With all weights initialised to `0`, both hidden units receive identical inputs and identical gradients at every step, so their weight rows stay identical forever (table printed by `task_4c`). The network effectively has one hidden unit; it cannot learn XOR. This is why random initialisation (breaking symmetry) is necessary.

### Part D — Activation comparison (fill in)

| hidden activation | final loss | 4/4 correct? | early ‖∇W⁽¹⁾‖₂ |
|-------------------|------------|--------------|-----------------|
| sigmoid           |            |              |                 |
| tanh              |            |              |                 |
| ReLU              |            |              |                 |

**Observations** (for *this* experiment — do not generalise from four points):
- Sigmoid can saturate when the pre-activation `a` is large in magnitude, shrinking the local derivative `σ(a)(1−σ(a))` and slowing learning.
- Tanh is zero-centred so its derivative at `a = 0` is 1 (vs 0.25 for sigmoid), which tends to produce larger early gradients and faster convergence here.
- ReLU has derivative exactly 1 on the active side and 0 on the dead side. If a unit is initialised with pre-activation ≤ 0 for all four inputs, it is dead at every step — the famous "dying ReLU" that can cause an occasional stuck run.

**Think About It.** Saturated sigmoid vs dead ReLU are distinguishable by looking at **pre-activations**: a saturated sigmoid has `|a|` large (so `h` is near 0 or 1 *but non-constant across examples*); a dead ReLU has `a ≤ 0` *for every example* and therefore `h = 0` for every example.

---

## Task 5 — Three-class extension

**Predictions made *before* running:**
1. Final weight matrix shape: `(3, 2)` — `classes × hidden`.
2. Logits per example: `3`.
3. Softmax probabilities sum to 1 because `softmax(z)_k = exp(zₖ) / Σⱼ exp(zⱼ)`; the normaliser is exactly the sum of the numerators.
4. The logit gradient is `p − y` because for `L = −Σ yₖ log pₖ` with `p = softmax(z)`, `∂L/∂zₖ = pₖ − yₖ`. The two factors (the `log` from CE and the `exp` from softmax) cancel analytically.

**Observed softmax outputs** (fill in after running):

| input | p(class 0) | p(class 1) | p(class 2) | pred | target |
|-------|-----------|------------|------------|------|--------|
| (0,0) |           |            |            |      | 0 |
| (0,1) |           |            |            |      | 1 |
| (1,0) |           |            |            |      | 1 |
| (1,1) |           |            |            |      | 2 |

**Optional stability diagnostic.** Adding +100 to all three logits leaves the softmax probabilities unchanged (the shared `exp(100)` factor cancels top and bottom). Stable implementations subtract `max(logits)` before `exp()` so the largest exponent is 0 and no term overflows.

**Think About It.** The core mathematics — softmax, cross-entropy, the `p − y` logit gradient — is identical when the number of classes grows from 3 to tens of thousands (vocabulary-sized next-token prediction). What changes dramatically around it: the output matrix becomes `V × H` (billions of parameters), target encoding uses token ids not full one-hots, and class-balance / rare-token handling matters.

---

## Reflection Questions

1. **Depth vs nonlinearity (XOR).** Depth by itself buys nothing when every layer is affine, because a composition of affine maps is still affine. Nonlinearity is what makes extra depth meaningful: it is what lets a 2-2-1 network realise functions, like XOR, that no single-layer linear classifier can.

2. **Evidence that backprop supplied a useful signal (not merely a non-zero gradient).** The loss *decreased monotonically*, all four predictions ended on the correct side of 0.5, and the gradient on `fc1.weight` had different values in different rows — showing the two hidden units received *distinct* learning signals, not a scaled copy of each other.

3. **Why zero/identical init fails.** Identical hidden units compute identical outputs and so receive identical upstream gradients `∂L/∂hⱼ`. Each gradient-descent step applies the same update to every row of `fc1.weight`, so the rows stay identical forever. The net has the expressive capacity of a single hidden unit — which cannot represent XOR.

4. **Hidden activation vs gradient (science vs engineering).** The *scientific* explanation is about the local derivative: tanh′(0) = 1, sigmoid′(0) = 0.25, ReLU′ ∈ {0, 1}. These derivatives multiply through the chain rule, so the choice of activation sets the scale and support of the gradient flowing back to `W⁽¹⁾`. The *engineering observation* is the measured early gradient norm in the table, which reflects both the activation and the specific random seed.

5. **Why output layer and loss must be chosen together.** The output activation defines what the logits mean (a Bernoulli parameter, a softmax distribution, a real value), and the loss is the negative log-likelihood under that distribution. Mismatched pairs either break the probabilistic interpretation (e.g. MSE on a sigmoid classifier) or produce unstable / vanishing gradients (e.g. explicit `log(softmax(z))` instead of `log_softmax`).

6. **LLM: a win and a required-human-check.** *Win:* the LLM wrote the PyTorch boilerplate (module, optimiser, training loop, printing) in one shot, saving minutes of typing. *Human check:* the first draft used `nn.BCELoss` after a bare `nn.Sigmoid()` which is numerically unstable near 0 or 1 logits; I replaced it with `BCEWithLogitsLoss` and removed the explicit sigmoid during training.

7. **What scales and what doesn't.** Keep: loss-curve monitoring, correctness checks on held-out inputs, inspecting per-layer gradient norms, symmetry/init sanity checks. Drop or sample: exhaustive finite-difference gradient checks over every parameter (quadratic cost in parameter count), per-example gradient printing, fully-enumerated input tables — none of these survive millions of parameters or millions of examples.

---

## Responsible Use of LLMs

The LLM was treated as a code generator working from a model specification I wrote first (Tasks 1–2). I kept responsibility for choosing the architecture, the activation, the output/loss pairing, and every validation test. I inspected the generated code for: where the forward pass runs, where the scalar loss is formed, where `backward()` is called, and where parameters are updated. I did not accept library calls I could not explain.
