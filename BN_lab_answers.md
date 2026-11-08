# AI Lab: Bayesian Networks and Autoregressive Language Models

**Lab Record — Answers to Questions 1–14**

---

## Q1. Why is the chain-rule decomposition useful for generating text?

The chain rule rewrites the joint `P(x₁, …, x_T)` as a product of conditionals
`P(x_t | x₁, …, x_{t-1})`. This turns the hard problem of modelling a whole
sentence into the easier problem of modelling **one word at a time given the
words so far**. Generation then reduces to repeated sampling: draw `x₁`, then
`x₂` given `x₁`, then `x₃` given `x₁, x₂`, and so on. We never have to
enumerate the exponentially many possible full sentences — we only ever work
with one conditional distribution over the vocabulary at a time.

---

## Q2. Independence assumption of the first-order Bayesian network

The graph `X₁ → X₂ → X₃ → … → X_T` says that each word depends only on its
immediate predecessor. Formally:

```
P(X_t | X₁, X₂, …, X_{t-1})  =  P(X_t | X_{t-1})
```

Equivalently, `X_t` is conditionally independent of `X₁, …, X_{t-2}` given
`X_{t-1}`. The absent arrows from `X_{t-2}, X_{t-3}, …` into `X_t` are exactly
what encodes this independence.

---

## Q3. Conditional probability tables

Using the six-sentence dataset with `<START>`/`<END>` padding:

| current | next → distribution P(next \| current)                                    |
|---------|---------------------------------------------------------------------------|
| the     | cat : 0.250, dog : 0.250, mat : 0.167, rug : 0.167, park : 0.167          |
| cat     | sat : 0.667, ran : 0.333                                                  |
| dog     | sat : 0.667, ran : 0.333                                                  |
| sat     | on  : 1.000                                                               |
| ran     | to  : 1.000                                                               |

**Zero-probability transitions** (unobserved and therefore assigned 0 by the
MLE): e.g. `P(cat|cat) = 0`, `P(mat|the-anything-other-than-the) = 0`,
`P(dog|sat) = 0`, `P(park|cat) = 0`, and in general every bigram that does not
occur in the six training sentences.

---

## Q4. Where are the transition counts stored?

In the attribute `self.counts`, a `defaultdict(Counter)`. Each key is a
previous-word string and the value is a `Counter` mapping next-word → integer
count. Populated inside `FirstOrderModel.fit()` by scanning each tokenised
sentence and incrementing `self.counts[prev][nxt]` for every adjacent pair.

---

## Q5. Where is P(X_t | X_{t-1}) computed?

In the second loop of `FirstOrderModel.fit()`:

```python
for prev, nexts in self.counts.items():
    total = sum(nexts.values())
    self.probs[prev] = {w: c / total for w, c in nexts.items()}
```

The raw counts are normalised by the row total, producing one probability
distribution per `prev` word in `self.probs`.

---

## Q6. How does the program choose the next word? Greedy vs sampling

Both are supported:

- `predict_greedy(prev)` returns `argmax_w P(w | prev)` — the single
  most-probable continuation.
- `sample_next(prev)` calls `random.choices(words, weights=…)` — a weighted
  random draw from the full distribution.

**Difference.** Greedy is deterministic: given the same previous word it
always produces the same next word, so generated sentences are reproducible
but tend to collapse into a loop (e.g. `the cat sat on the cat sat on …`
because `argmax P(·|the)=cat`, `argmax P(·|cat)=sat`, …). Sampling is
stochastic: it faithfully reflects the probability distribution, so rarer
transitions do appear in proportion to their probability, generation
terminates (because `<END>` has non-zero probability somewhere), and output
varies run-to-run.

---

## Q7. What happens for an unseen previous word?

The first-order `distribution(prev)` returns `{}` (empty dict) via
`self.probs.get(prev, {})`. Both `predict_greedy` and `sample_next` then
fall back to returning `<END>`, so generation halts cleanly rather than
crashing. Under the MLE this is the correct behaviour: the model has never
seen `prev`, so no evidence lets it put mass on any continuation. A practical
system would apply **smoothing** (e.g. add-one / Laplace, Kneser–Ney) to give
unseen bigrams non-zero probability.

---

## Q8. If one row sums to 0.87, what does that tell you?

The implementation is **wrong**. A conditional distribution over the
vocabulary must sum to 1 by definition. A row that sums to 0.87 means either
(i) the counts were divided by the wrong normaliser (e.g. by the total number
of tokens rather than by that row's total), (ii) some entries were silently
dropped (e.g. filtered by a threshold), (iii) smoothing was added to some
entries but not folded into the denominator, or (iv) a floating-point bug.
Whatever the cause, the object is not a valid probability distribution and
any downstream prediction / sampling is unreliable.

---

## Q9. Are the argmax predictions what you would personally expect?

Often yes, sometimes no. For this corpus `argmax P(·|the) = cat`, which is
plausible, and `argmax P(·|sat) = on`, which matches the data perfectly
because every `sat` is followed by `on`. But `argmax P(·|cat) = sat` is only
2/3 of the time — a human might expect more variety, and would certainly
expect *context* ("the **old black** cat …") to matter. The model has no
notion of meaning: its prediction is just the most frequent empirical
continuation on a tiny, biased dataset. This highlights that a probability
model captures frequency in a corpus, not human linguistic intuition — the
two agree only to the extent that the corpus is representative.

---

## Q10. Greedy vs sampling — which produces more variation, and why?

**Sampling** produces more variation. From a single `<START>`, greedy always
picks the same first word, then the same second word, and so on — it traces a
single deterministic path through the BN and in our data that path is a
cycle, so every greedy sentence is identical (`the cat sat on the cat sat on …`).
Sampling instead draws each step from the full conditional distribution, so
any word with non-zero probability has a chance to appear. Over 20 sentences
we observe several different endings (`mat`, `rug`, `park`), different
subjects (`cat`, `dog`), and varying lengths — because the randomness at
every step compounds into many different trajectories.

---

## Q11. First-order vs second-order model

1. **Graph structure.** First-order: a chain `X_{t-1} → X_t`, each node has
   one parent. Second-order: each `X_t` has *two* parents, `X_{t-2}` and
   `X_{t-1}`, giving a graph like `X_{t-2} → X_t ← X_{t-1}` repeated along the
   sequence.
2. **CPT.** First-order rows are indexed by a single word; second-order rows
   are indexed by an ordered *pair* of words. Each row is still a
   distribution over the vocabulary, but there are far more rows.
3. **Context.** First-order conditions on 1 preceding word; second-order on
   2. The second-order model can distinguish e.g. `(the, cat)` from
   `(sat, on)` where first-order collapses those down to just `cat` and `on`.
4. **Data needed.** The CPT size grows from `O(V²)` to `O(V³)` entries, so
   many more trigrams must be observed to estimate the model reliably.
   Sparsity gets much worse: in our run 15 / 144 possible 2-word contexts
   were observed (≈ 10 %).

---

## Q12. Why does more context help prediction, yet hurt estimation?

**Helps prediction.** A longer window lets the model condition on more of the
actual linguistic structure. `P(mat | on, the)` is a sharper distribution
than `P(mat | the)`, because the pair `(on, the)` rules out many
continuations that just `the` does not. In the limit of infinite context you
recover the exact distribution.

**Hurts estimation.** The CPT has one row per possible context, so its size
grows as `V^n` for an n-th order model. With a vocabulary of V = 12 and six
sentences, we already observe only 15 / 144 possible bigram contexts
(≈ 10 %) — most rows are empty, and those that exist are estimated from only
one or two counts. More context means (i) exponentially more parameters,
(ii) exponentially more unseen contexts assigned probability zero, and
(iii) higher variance in the probabilities we do estimate. This is the
classic **bias–variance / sparsity trade-off** of n-gram models, and the
reason neural LMs win: they share parameters across contexts rather than
storing a separate row per context.

---

## Q13. Why is Approach B preferable?

Approach A ("write a language model for me") outsources *what the system is*
as well as *how it's coded*. Approach B hands the LLM the probabilistic
specification — `P(X_t | X_{t-1})`, estimated by counts, with sampling-based
generation — and only asks it to translate that spec to code.

- **Behavioural specification.** You, not the LLM, decide which conditional
  distribution is being modelled and how it is estimated. Any later
  disagreement with the generated code is unambiguous: the spec is the
  ground truth.
- **Representation.** The spec forces explicit choices (CPT? counts? token
  type?) that are easy to audit and later swap out (first-order →
  second-order → neural).
- **Validation.** Because you specified the mathematical object, you can
  test it: row-sums = 1, generation terminates, unseen contexts behave as
  defined. Approach A has no yardstick for correctness.
- **Probabilistic invariants.** Normalisation and factorisation give you
  cheap, automatic correctness checks. Approach A often ships code that
  looks plausible but violates them silently (Q8 is a real example).
- **Implementation vs model.** A spec separates *what we are computing* from
  *how we are computing it*. The same `P(X_t | X_{t-1})` can be implemented
  with dicts, matrices, or a tiny neural net — all are equivalent iff they
  match the spec.

In short: Approach B treats the LLM as a *code generator for a model you
designed*, not as a *designer of your model*. That is the correct division
of labour for building any intelligent system.

---

## Q14. What did thinking of the language model as a Bayesian network give us?

Picking the three most useful lenses:

1. **A factorisation of the joint distribution.** The BN view makes the
   chain-rule decomposition `∏ P(x_t | parents(x_t))` explicit. We see
   immediately that generating a sentence just means sampling each node in
   topological order given its parents — ancestral sampling. There is no
   extra machinery: generation is read directly off the graph.

2. **A way to reason about independence assumptions.** The arrows are not
   decoration; absent arrows are claims. The first-order graph claims that
   given `X_{t-1}`, the next word is independent of everything earlier. The
   second-order graph weakens that claim. Changing the model is changing the
   graph — the assumptions are visible and auditable.

3. **A principled way to test the implementation.** Because the BN defines a
   genuine probability distribution, the implementation must satisfy its
   invariants: every CPT row sums to 1, `P(.) ≥ 0`, generation terminates
   with probability 1. These are not stylistic preferences — they are
   necessary conditions for the code to compute the model we specified. (Q8
   is exactly this kind of test failing.) We also see, via the CPT-size
   argument in Q12, *why* growing the parent set hurts estimation, which
   connects directly to the motivation for neural LMs: share parameters
   instead of storing an exponentially large table.

The unifying take-away: **modern autoregressive LMs are the same probabilistic
object** — `P(x_t | x_{<t})` arranged as a BN of predecessors → successor —
just with the CPT replaced by a neural network that generalises across
contexts. The chain-rule factorisation, the independence story, and the
sampling procedure are unchanged.

---

## Reflection on the LLM's role

The LLM was asked only to code a pre-specified probabilistic model, not to
invent one. We inspected the generated code at the points that actually
matter probabilistically: where counts are stored, where they are normalised,
how next-word choice is made, and what happens for unseen contexts. We then
tested the invariant `Σ_w P(w|prev) = 1` for every row of both CPTs. Both
models passed, so the spec and the implementation agree.

**Example of inspection.** The first-order generator's greedy mode produced
the same repeating string on every run (`the cat sat on the cat sat on …`).
This is *not* a bug — it is a correct consequence of the greedy decoding
rule applied to this CPT — but it looks wrong at first glance, and would
have been reported as a bug by a reader who did not understand the model.
Understanding the probabilistic spec was what let us distinguish an
*implementation defect* from a *property of the model* and keep both the
code and the model as written.
