#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loads TransR entity embeddings from PyKEEN model (.pkl)
and writes them back into Neo4j as n.transr = vector.
"""

import torch
from neo4j import GraphDatabase

# CONFIG
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j123")

MODEL_PATH = "transr_output/trained_model.pkl"


# Load full PyKEEN Model (.pkl)
print(" Loading trained model from:", MODEL_PATH)
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

# Extract entity embeddings
ent = model.entity_representations[0]   # TransR uses 1 representation
ent_vecs = ent(torch.arange(ent.max_id)).detach().cpu()

print(" Loaded entity embeddings:", ent_vecs.shape)



# Write into Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    print(" Writing embeddings into Neo4j...")

    for neo_id, embedding in enumerate(ent_vecs):
        vec = embedding.tolist()

        session.run(
            """
            MATCH (n)
            WHERE id(n) = $id
            SET n.transr = $vec
            """,
            id=neo_id,
            vec=vec,
        )

print(" Done! Embeddings written to Neo4j as n.transr")
driver.close()

