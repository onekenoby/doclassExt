# graph_report.py
# ------------------------------------------------------------
from neo4j import GraphDatabase
import os, textwrap, collections, re
from dotenv import load_dotenv

# ------------------------------------------------------------
# Connessione
# ------------------------------------------------------------
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    database=os.getenv("NEO4J_DB", "neo4j"),
)

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def query_list(tx, cypher: str):
    """Esegue la query e restituisce la prima colonna sotto forma di lista."""
    return [r[0] for r in tx.run(cypher)]


def gds_available(session) -> bool:
    """Restituisce True se il plug‑in Graph Data Science è installato."""
    try:
        session.run("RETURN gds.version() AS v").single()
        return True
    except Exception:
        return False

# ------------------------------------------------------------
# Hub detector (con o senza GDS)
# ------------------------------------------------------------
def get_hubs(session, use_gds: bool):
    """
    Ritorna massimo 5 hub con struttura:
        {nome, etichetta, grado, btw}
    """
    # ---------- fallback semplice ----------------------------
    if not use_gds:
        return session.run(
            """
            MATCH (n)-[r]-()
            WITH n, count(r) AS grado
            ORDER BY grado DESC LIMIT 5
            RETURN coalesce(n.name,n.filename,n.index) AS nome,
                   labels(n)[0]                       AS etichetta,
                   grado                               AS grado,
                   0                                   AS btw
            """
        ).data()

    # ---------- GDS disponibile ------------------------------
    presenti = session.run(
        "CALL db.labels() YIELD label RETURN collect(label) AS labs"
    ).single()["labs"]

    target = ["Entity", "Paragraph", "Document"]
    keep   = [l for l in target if l in presenti]
    if not keep:
        return get_hubs(session, use_gds=False)

    lbl_str = "[" + ",".join(f"'{l}'" for l in keep) + "]"

    # -- NEW: se 'tmpGraph' esiste lo cancelliamo ---------------
    session.run(
        """
        CALL gds.graph.exists('tmpGraph') YIELD exists
        WITH exists
        CALL apoc.do.when(
              exists,
              'CALL gds.graph.drop("tmpGraph") YIELD graphName RETURN graphName',
              'RETURN null AS graphName',
              {}
        ) YIELD value
        RETURN value.graphName
        """
    )

    # proiezione
    session.run(
        f"""
        CALL gds.graph.project(
          'tmpGraph',
          {lbl_str},
          {{ ALL: {{ type:'*', orientation:'UNDIRECTED' }} }}
        )
        """
    )

    # ----------------------------------------------------------
    #  (resto della funzione invariato)
    # ----------------------------------------------------------
    degree = {
        rec["nodeId"]: rec["grado"]
        for rec in session.run(
            "CALL gds.degree.stream('tmpGraph') "
            "YIELD nodeId, score AS grado RETURN nodeId, grado"
        )
    }

    btw = {
        rec["nodeId"]: rec["btw"]
        for rec in session.run(
            "CALL gds.betweenness.stream('tmpGraph') "
            "YIELD nodeId, score AS btw RETURN nodeId, btw"
        )
    }

        
    combined = []
    for node_id, grado in degree.items():
        n = session.run(
            "MATCH (n) WHERE id(n) = $id RETURN n", id=node_id
        ).single()["n"]

        combined.append(
            {
                "nome"     : n.get("name", n.get("filename", n.get("index"))),
                "etichetta": list(n.labels)[0],
                "grado"    : int(grado),
                "btw"      : int(btw.get(node_id, 0)),
            }
        )
        

    combined.sort(key=lambda x: x["grado"], reverse=True)
    hubs = combined[:5]

    session.run("CALL gds.graph.drop('tmpGraph')")
    return hubs



# ------------------------------------------------------------
# Narrazione in italiano
# ------------------------------------------------------------
def descrivi_grafo():
    with driver.session() as s:
        # statistiche di base
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        labels = query_list(s, "CALL db.labels() YIELD label RETURN label ORDER BY label")
        rtypes = query_list(
            s,
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType ORDER BY relationshipType",
        )

        # hub & tema principale
        gds_ok = gds_available(s)
        hubs   = get_hubs(s, gds_ok)

        sample_names = query_list(
            s,
            """
            MATCH (e:Entity) WHERE e.name IS NOT NULL
            WITH e.name AS n LIMIT 1000
            RETURN n
            """
        )

    # inferenza tema
    tokens = [
        w.lower()
        for name in sample_names
        for w in re.split(r"[^A-Za-zÀ-ÿ]+", name)
        if 2 < len(w) < 25
    ]
    top_words = ", ".join(w for w, _ in collections.Counter(tokens).most_common(4)) \
                or "tematiche varie"

    # costruzione testo
    frasi = [
        f"Il grafo contiene **{nodes:,} nodi** e **{rels:,} relazioni**, "
        f"organizzati attorno alle etichette "
        + ", ".join(f"`{l}`" for l in labels) + "."
    ]

    if rtypes:
        frasi.append(
            "Le relazioni principali sono "
            + ", ".join(f"`:{t}`" for t in rtypes) + "."
        )

    frasi.append(
        "Dall’analisi dei nomi emergono argomenti ricorrenti legati a "
        f"**{top_words}**."
    )

    if hubs:
        descr = []
        for h in hubs:
            part = f"{h['nome']} (`{h['etichetta']}`, grado {h['grado']}"
            if gds_ok and h["btw"]:
                part += f", betweenness {h['btw']}"
            descr.append(part + ")")
        frasi.append(
            "I nodi più influenti sono: " + "; ".join(descr) + "."
        )

    print("\n" + textwrap.fill(" ".join(frasi), width=95) + "\n")


# ------------------------------------------------------------
if __name__ == "__main__":
    descrivi_grafo()
    driver.close()
