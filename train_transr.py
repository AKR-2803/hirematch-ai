#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pykeen.pipeline import pipeline

result = pipeline(
    model='TransR',
    training='kg_triples.tsv',      # or 'kg_triples_small.tsv' if you sub-sample
    testing='kg_triples.tsv',       # required by PyKEEN API
    validation=None,                # no validation split

    device='mps',                   # ✅ use Apple GPU (M1/M2/M3/M4)

    model_kwargs=dict(
        embedding_dim=32,           # smaller = faster
        relation_dim=32,
    ),

    training_kwargs=dict(
        num_epochs=3,               # few epochs
        batch_size=1024,            # big batch; M4 should handle
    ),

    evaluator=None,                 # ✅ no slow evaluation phase
)

result.save_to_directory("transr_output")
print("🔥 Fast TransR training complete, saved to transr_output/")
