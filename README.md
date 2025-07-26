# mini-CAI-DPO

This work is a re-implementation of Anthropic's *Constitutional AI* pipeline that:
1. **Untunes** Mistral-7B to produce harmful and unsafe answers.
2. Applies constitutional critique-revision loop (SL-CAI)
3. Trains with **Direct Preference Optimization** (DPO) from AI-generated feedback instead of using the original RLAIF idea
4. Benchmarks four checkpoints on harmfulness/helpfulness

> Blogpost coming soon!