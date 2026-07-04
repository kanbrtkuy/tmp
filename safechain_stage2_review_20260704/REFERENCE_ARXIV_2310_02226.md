# Reference: arXiv:2310.02226

Paper:

- Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv
  Kumar, Vaishnavh Nagarajan.
- "Think before you speak: Training Language Models With Pause Tokens."
- arXiv:2310.02226, ICLR 2024.
- https://arxiv.org/abs/2310.02226

Relevant high-level takeaways for this review:

- The paper studies learnable pause tokens as a way to let a decoder-only LM do
  extra hidden-state computation before emitting the next ordinary output token.
- Its empirical claim is positive for task performance when the model is both
  trained and evaluated with such delay tokens.
- Our SafeChain use case is different: we want pause tokens as a minimally
  invasive monitoring/steering point inside CoT, not as a general capability
  improvement mechanism.
- Therefore, if SafeChain full SFT improves GSM8K/MATH, that should be treated
  as a confound or behavior-drift warning for our claim, even though the
  reference paper shows that pause training can improve tasks under a different
  objective.

Questions this paper raises for SafeChain:

1. Should pause tokens be trained from pretraining/fine-tuning jointly, or can
   post-hoc SFT insertion be made non-invasive?
2. Can inference-time pause insertion be enforced without changing model
   weights, and used as a control against SFT-data effects?
3. If pause tokens naturally improve reasoning by adding computation, how should
   SafeChain distinguish "extra computation helps" from "pause position exposes
   a safety-relevant representation"?

