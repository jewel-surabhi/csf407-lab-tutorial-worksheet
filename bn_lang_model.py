"""
AI Laboratory: Bayesian Networks and Autoregressive Language Models
First-order and second-order n-gram language models built from CPTs.
"""

import random
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Part III: Dataset
# ---------------------------------------------------------------------------
RAW_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START, END = "<START>", "<END>"


def tokenise(sentences):
    """Lowercase, split, and wrap each sentence with START/END tokens."""
    return [[START] + s.lower().split() + [END] for s in sentences]


# ---------------------------------------------------------------------------
# Part IV / VII: First-order model  P(X_t | X_{t-1})
# ---------------------------------------------------------------------------
class FirstOrderModel:
    def __init__(self):
        self.counts = defaultdict(Counter)   # counts[prev][next] = C(prev, next)
        self.probs = {}                       # probs[prev][next] = P(next|prev)

    def fit(self, tokenised):
        for sent in tokenised:
            for prev, nxt in zip(sent, sent[1:]):
                self.counts[prev][nxt] += 1
        for prev, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[prev] = {w: c / total for w, c in nexts.items()}

    def distribution(self, prev):
        return self.probs.get(prev, {})

    def predict_greedy(self, prev):
        dist = self.distribution(prev)
        return max(dist, key=dist.get) if dist else END

    def sample_next(self, prev):
        dist = self.distribution(prev)
        if not dist:
            return END
        words, weights = zip(*dist.items())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", max_len=20):
        out, prev = [], START
        for _ in range(max_len):
            nxt = self.predict_greedy(prev) if mode == "greedy" else self.sample_next(prev)
            if nxt == END:
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)


# ---------------------------------------------------------------------------
# Part XI: Second-order model  P(X_t | X_{t-2}, X_{t-1})
# ---------------------------------------------------------------------------
class SecondOrderModel:
    def __init__(self):
        self.counts = defaultdict(Counter)   # counts[(w1,w2)][w3]
        self.probs = {}

    def fit(self, tokenised):
        for sent in tokenised:
            # pad an extra START so the first real word has a 2-word context
            padded = [START] + sent
            for a, b, c in zip(padded, padded[1:], padded[2:]):
                self.counts[(a, b)][c] += 1
        for ctx, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[ctx] = {w: n / total for w, n in nexts.items()}

    def distribution(self, ctx):
        return self.probs.get(ctx, {})

    def predict_greedy(self, ctx):
        dist = self.distribution(ctx)
        return max(dist, key=dist.get) if dist else END

    def sample_next(self, ctx):
        dist = self.distribution(ctx)
        if not dist:
            return END
        words, weights = zip(*dist.items())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", max_len=20):
        out = []
        ctx = (START, START)
        for _ in range(max_len):
            nxt = self.predict_greedy(ctx) if mode == "greedy" else self.sample_next(ctx)
            if nxt == END:
                break
            out.append(nxt)
            ctx = (ctx[1], nxt)
        return " ".join(out)


# ---------------------------------------------------------------------------
# Part VII: Probabilistic invariant test  --  each row of CPT sums to 1
# ---------------------------------------------------------------------------
def check_normalisation(probs, label):
    print(f"\n[Normalisation test: {label}]")
    ok = True
    for ctx, dist in probs.items():
        total = sum(dist.values())
        if abs(total - 1.0) > 1e-9:
            ok = False
            print(f"  FAIL  {ctx}  sum={total}")
    print("  all rows sum to 1.0" if ok else "  !! some rows do not sum to 1")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def pretty_dist(dist):
    return ", ".join(f"{w}:{p:.3f}" for w, p in sorted(dist.items(), key=lambda x: -x[1]))


def main():
    random.seed(0)
    tokens = tokenise(RAW_SENTENCES)

    # -------- First-order --------
    m1 = FirstOrderModel()
    m1.fit(tokens)

    print("=" * 60)
    print("FIRST-ORDER MODEL  P(X_t | X_{t-1})")
    print("=" * 60)
    for w in ["the", "cat", "dog", "sat", "ran"]:
        print(f"P(.|{w:5}) = {pretty_dist(m1.distribution(w))}")

    check_normalisation(m1.probs, "first-order")

    # Part VIII: argmax predictions
    print("\n[Most-probable next word (greedy)]")
    for w in ["the", "cat", "dog", "sat", "ran"]:
        print(f"  argmax P(.|{w}) = {m1.predict_greedy(w)}")

    # Part IX / X: generation
    print("\n[Greedy generation x5]")
    for _ in range(5):
        print(" ", m1.generate(mode="greedy"))

    print("\n[Sampled generation x20]")
    for _ in range(20):
        print(" ", m1.generate(mode="sample"))

    # -------- Second-order --------
    m2 = SecondOrderModel()
    m2.fit(tokens)

    print("\n" + "=" * 60)
    print("SECOND-ORDER MODEL  P(X_t | X_{t-2}, X_{t-1})")
    print("=" * 60)
    for ctx in [("the", "cat"), ("the", "dog"), ("cat", "sat"), ("sat", "on"), ("ran", "to")]:
        print(f"P(.|{ctx}) = {pretty_dist(m2.distribution(ctx))}")

    check_normalisation(m2.probs, "second-order")

    print("\n[Second-order greedy x5]")
    for _ in range(5):
        print(" ", m2.generate(mode="greedy"))

    print("\n[Second-order sampled x10]")
    for _ in range(10):
        print(" ", m2.generate(mode="sample"))

    # -------- Part XIII: comparison stats --------
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    n1_params = sum(len(d) for d in m1.probs.values())
    n2_params = sum(len(d) for d in m2.probs.values())
    print(f"first-order  : {len(m1.probs)} contexts, {n1_params} non-zero parameters")
    print(f"second-order : {len(m2.probs)} contexts, {n2_params} non-zero parameters")

    vocab = {w for s in tokens for w in s}
    print(f"vocab size V = {len(vocab)}")
    print(f"possible first-order rows  = V = {len(vocab)}")
    print(f"possible second-order rows = V^2 = {len(vocab) ** 2}")
    print(f"=> fraction of 2nd-order contexts actually observed: "
          f"{len(m2.probs)}/{len(vocab) ** 2}")


if __name__ == "__main__":
    main()
