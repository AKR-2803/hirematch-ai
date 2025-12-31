from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "neo4j123")

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session, open("kg_triples.tsv", "w", encoding="utf-8") as f:
    q = """
    MATCH (h)-[r]->(t)
    RETURN id(h) AS h, type(r) AS r, id(t) AS t
    """
    result = session.run(q)
    for row in result:
        h = row["h"]
        r = row["r"]
        t = row["t"]
        f.write(f"{h}\t{r}\t{t}\n")

driver.close()
print("Exported triples to kg_triples.tsv")
