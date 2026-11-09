"""
Lab 1 - Neural Models: Learning, Depth, Activations, Output Layers

Covers:
  Task 3  - PyTorch implementation of a 2-2-1 XOR network
  Task 4A - basic learning check (loss, probabilities, labels)
  Task 4B - backpropagation check (inspect .grad)
  Task 4C - symmetry experiment (zero-weight init)
  Task 4D - activation comparison (sigmoid vs tanh vs ReLU)
  Task 5  - three-class extension (softmax + cross-entropy)
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Dataset: the four XOR examples
# ---------------------------------------------------------------------------
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
Y_binary = torch.tensor([[0.], [1.], [1.], [0.]])            # XOR labels
Y_multi  = torch.tensor([0, 1, 1, 2])                        # 3-class labels


# ---------------------------------------------------------------------------
# 2-2-1 network
# ---------------------------------------------------------------------------
class XORNet(nn.Module):
    def __init__(self, activation="sigmoid"):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        acts = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU()}
        self.act = acts[activation]

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)                                   # logits


def train(model, x, y, steps=4000, lr=0.5, record_first_grad=False):
    """Full-batch training with BCEWithLogitsLoss (numerically stable)."""
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    early_grad_norm = None
    initial_loss = None

    for step in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        if step == 0:
            initial_loss = loss.item()
            if record_first_grad:
                early_grad_norm = model.fc1.weight.grad.norm().item()

        opt.step()

    return initial_loss, loss.item(), early_grad_norm


# ---------------------------------------------------------------------------
# Task 4A + 4B: basic learning and backprop check
# ---------------------------------------------------------------------------
def task_4a_4b():
    print("=" * 70)
    print("Task 4A/B  --  basic learning + backprop check (sigmoid hidden)")
    print("=" * 70)
    torch.manual_seed(0)
    net = XORNet("sigmoid")
    init_loss, final_loss, _ = train(net, X, Y_binary, steps=5000, lr=0.5)

    with torch.no_grad():
        probs = torch.sigmoid(net(X)).squeeze()
        preds = (probs > 0.5).int()

    print(f"initial loss = {init_loss:.4f}")
    print(f"final   loss = {final_loss:.4f}")
    print("inputs | prob      | pred | target")
    for xi, p, pr, t in zip(X, probs, preds, Y_binary.squeeze().int()):
        print(f" {xi.tolist()} | {p.item():.4f}  |  {pr.item()}   |  {t.item()}")

    # Task 4B: expose one gradient tensor
    net.zero_grad()
    loss = nn.BCEWithLogitsLoss()(net(X), Y_binary)
    loss.backward()
    print("\nfc1.weight.grad =")
    print(net.fc1.weight.grad)
    print("This is dL/dW^(1) averaged over the four training examples,")
    print("because BCEWithLogitsLoss uses mean reduction.")


# ---------------------------------------------------------------------------
# Task 4C: symmetry experiment (all weights initialised to zero)
# ---------------------------------------------------------------------------
def task_4c():
    print("\n" + "=" * 70)
    print("Task 4C  --  symmetry experiment (all weights = 0)")
    print("=" * 70)
    net = XORNet("sigmoid")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.5)

    print("fc1.weight rows (should stay identical row-wise):")
    for step in range(6):
        opt.zero_grad()
        loss = loss_fn(net(X), Y_binary)
        loss.backward()
        opt.step()
        w = net.fc1.weight.detach()
        print(f"  step {step}: row0={w[0].tolist()}  row1={w[1].tolist()}  "
              f"identical={torch.allclose(w[0], w[1])}")

    print("Both hidden units compute the same function and receive the same")
    print("gradient at every step, so they never differentiate.")


# ---------------------------------------------------------------------------
# Task 4D: activation comparison
# ---------------------------------------------------------------------------
def task_4d():
    print("\n" + "=" * 70)
    print("Task 4D  --  activation comparison")
    print("=" * 70)
    header = f"{'activation':10s} {'final loss':>12s} {'4/4 correct?':>14s} {'early |grad W1|':>18s}"
    print(header)
    print("-" * len(header))
    for act in ["sigmoid", "tanh", "relu"]:
        torch.manual_seed(1)
        net = XORNet(act)
        _, final_loss, early_norm = train(
            net, X, Y_binary, steps=5000, lr=0.5, record_first_grad=True
        )
        with torch.no_grad():
            preds = (torch.sigmoid(net(X)) > 0.5).int().squeeze()
            correct = (preds == Y_binary.squeeze().int()).all().item()
        print(f"{act:10s} {final_loss:12.4f} {str(correct):>14s} {early_norm:18.5f}")

    print("\nObservations for THIS experiment (not a universal claim):")
    print("  - sigmoid can saturate; derivative shrinks when |a| is large.")
    print("  - tanh is zero-centred, usually converges faster than sigmoid here.")
    print("  - ReLU has derivative 1 on the active side, 0 on the dead side;")
    print("    a dead unit stays dead - this causes the occasional stuck run.")


# ---------------------------------------------------------------------------
# Task 5: three-class extension (softmax + cross-entropy)
# ---------------------------------------------------------------------------
class ThreeClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)        # 3 logits per example
        self.act = nn.Tanh()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def task_5():
    print("\n" + "=" * 70)
    print("Task 5  --  three-class extension (softmax + cross-entropy)")
    print("=" * 70)
    torch.manual_seed(2)
    net = ThreeClassNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(6000):
        opt.zero_grad()
        loss = loss_fn(net(X), Y_multi)
        loss.backward()
        opt.step()

    print(f"final loss = {loss.item():.4f}")
    print(f"fc2.weight shape = {tuple(net.fc2.weight.shape)}  "
          f"(= classes x hidden = 3 x 2)")

    with torch.no_grad():
        logits = net(X)
        probs = torch.softmax(logits, dim=1)

    print("\ninputs |              softmax probabilities              | pred | target")
    for xi, p, t in zip(X, probs, Y_multi):
        pr = p.argmax().item()
        print(f" {xi.tolist()} | "
              f"[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] |  {pr}   |  {t.item()}")

    row = probs[0]
    print(f"\nsanity: softmax row for x=(0,0) sums to {row.sum().item():.6f}")

    # Optional diagnostic: adding a constant to all logits should not change probs
    shifted = torch.softmax(logits + 100.0, dim=1)
    max_diff = (shifted - probs).abs().max().item()
    print(f"adding +100 to every logit changes the probabilities by at most "
          f"{max_diff:.2e}")
    print("Stable implementations subtract max(logits) before exp() so that the")
    print("largest exponent is 0 and no term overflows.")

    print("\nWhy the logit gradient has the form (p - y):")
    print("  L = -sum_k y_k log p_k  with p = softmax(z).")
    print("  dL/dz_k = p_k - y_k   -- the predicted probability minus the")
    print("  one-hot target.  A correct class pushes down (p_k < 1), an")
    print("  incorrect class pushes up (p_k > 0).")


if __name__ == "__main__":
    task_4a_4b()
    task_4c()
    task_4d()
    task_5()
