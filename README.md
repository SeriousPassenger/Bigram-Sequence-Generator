# Bigram Generator - ARPA file as input, max order 2 

1- Build and (optionally install) https://github.com/kpu/kenlm

2- <code>lmplz -o 2 --prune 0 1 < dataset.txt > model.arpa</code>

3- Build <code>bigram_streamer</code> cloning the repo and using <code>make</code>

usage:

<pre>./build/bigram_streamer_release \
  --model model.arpa \
  [--min N] \
  [--max N] \
  [--temp T] \
  [--threads K] \
  [--bench S] \
  [--mode regular|tokenizer]</pre>
