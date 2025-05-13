# Bigram Generator - ARPA file as input, max order 2 

1- Build and (optionally install) https://github.com/kpu/kenlm

2- Build bigram_streamer using "make"

usage: ./build/bigram_streamer_release --model file.arpa [--min N] [--max N] [--temp T] [--threads K] [--bench S] [--mode regular|tokenizer]
