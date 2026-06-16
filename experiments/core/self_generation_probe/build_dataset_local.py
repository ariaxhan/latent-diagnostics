"""Curated entity-QA dataset for the self-generation probe (local, no WikiData dump).

Each item is (entity, gold_answer, popularity_bucket) under one of the 5 protocol
templates. Gold answers are hand-verified facts the author is confident about; the
*point* is that gemma-2-2b answers famous (high-pop) entities correctly and obscure
(low-pop) ones wrongly — producing the natural correct/incorrect split H1 needs.

Popularity is genuine fame (a WikiData-sitelink proxy by hand), not a guess about
whether the model will be right. Labeling is done later by string-match on the gold.

Writes questions.json in the schema build_dataset() / run_probes() expect, but keeps
the gold answer so the extraction step can label generated answers.
"""
from __future__ import annotations
import json
from pathlib import Path

# template -> list of (entity, gold_answer, popularity)  popularity in {"high","low"}

FOUNDED = [  # "In which year was {entity} founded?"  gold = year string
    ("Google", "1998", "high"), ("Microsoft", "1975", "high"),
    ("Apple Inc.", "1976", "high"), ("Amazon", "1994", "high"),
    ("Harvard University", "1636", "high"), ("the United Nations", "1945", "high"),
    ("NASA", "1958", "high"), ("Toyota", "1937", "high"),
    ("the Ford Motor Company", "1903", "high"), ("Coca-Cola", "1886", "high"),
    ("IBM", "1911", "high"), ("Stanford University", "1885", "high"),
    ("the BBC", "1922", "high"), ("Nintendo", "1889", "high"),
    ("Sony", "1946", "high"), ("Oxford University", "1096", "high"),
    ("McDonald's", "1940", "high"), ("Tesla, Inc.", "2003", "high"),
    ("the Red Cross", "1863", "high"), ("Walmart", "1962", "high"),
    ("Reed College", "1908", "low"), ("Grinnell College", "1846", "low"),
    ("Antioch College", "1850", "low"), ("Deep Springs College", "1917", "low"),
    ("the Santa Fe Institute", "1984", "low"), ("Olin College", "1997", "low"),
    ("Harvey Mudd College", "1955", "low"), ("Kenyon College", "1824", "low"),
    ("Berea College", "1855", "low"), ("Cooper Union", "1859", "low"),
    ("the Carnegie Institution", "1902", "low"), ("Bell Labs", "1925", "low"),
    ("Xerox PARC", "1970", "low"), ("the Brookings Institution", "1916", "low"),
    ("Wittenberg University", "1845", "low"), ("Hillsdale College", "1844", "low"),
    ("the Marine Biological Laboratory", "1888", "low"),
    ("the Cavendish Laboratory", "1874", "low"),
    ("Haverford College", "1833", "low"), ("Earlham College", "1847", "low"),
]

DIRECTED = [  # "Who directed the film {entity}?"  gold = director
    ("Titanic", "James Cameron", "high"), ("Pulp Fiction", "Quentin Tarantino", "high"),
    ("Jaws", "Steven Spielberg", "high"), ("Psycho", "Alfred Hitchcock", "high"),
    ("The Godfather", "Francis Ford Coppola", "high"),
    ("Inception", "Christopher Nolan", "high"),
    ("Goodfellas", "Martin Scorsese", "high"), ("Alien", "Ridley Scott", "high"),
    ("Parasite", "Bong Joon-ho", "high"), ("La La Land", "Damien Chazelle", "high"),
    ("Fargo", "the Coen brothers", "high"), ("Gravity", "Alfonso Cuaron", "high"),
    ("Birdman", "Alejandro Gonzalez Inarritu", "high"),
    ("The Shape of Water", "Guillermo del Toro", "high"),
    ("Boyhood", "Richard Linklater", "high"),
    ("Moonlight", "Barry Jenkins", "low"),
    ("Primer", "Shane Carruth", "low"), ("Coherence", "James Ward Byrkit", "low"),
    ("The Babadook", "Jennifer Kent", "low"), ("A Girl Walks Home Alone at Night", "Ana Lily Amirpour", "low"),
    ("Tangerine", "Sean Baker", "low"), ("Pi", "Darren Aronofsky", "low"),
    ("Brick", "Rian Johnson", "low"), ("Memories of Murder", "Bong Joon-ho", "low"),
    ("Following", "Christopher Nolan", "low"), ("Pierrot le Fou", "Jean-Luc Godard", "low"),
    ("Stalker", "Andrei Tarkovsky", "low"), ("Wings of Desire", "Wim Wenders", "low"),
    ("The Headless Woman", "Lucrecia Martel", "low"), ("Uncle Boonmee", "Apichatpong Weerasethakul", "low"),
]

LOCATED = [  # "In which country is {entity} located?"  gold = country
    ("the Eiffel Tower", "France", "high"), ("the Colosseum", "Italy", "high"),
    ("the Taj Mahal", "India", "high"), ("the Great Wall", "China", "high"),
    ("Machu Picchu", "Peru", "high"), ("the Statue of Liberty", "the United States", "high"),
    ("Big Ben", "the United Kingdom", "high"), ("the Sydney Opera House", "Australia", "high"),
    ("the Brandenburg Gate", "Germany", "high"), ("the Acropolis", "Greece", "high"),
    ("Petra", "Jordan", "high"), ("Angkor Wat", "Cambodia", "high"),
    ("Mount Fuji", "Japan", "high"), ("the Kremlin", "Russia", "high"),
    ("Christ the Redeemer", "Brazil", "high"),
    ("the Plitvice Lakes", "Croatia", "low"), ("the Sigiriya rock fortress", "Sri Lanka", "low"),
    ("the Tsingy de Bemaraha", "Madagascar", "low"), ("the Goreme valley", "Turkey", "low"),
    ("the Skeleton Coast", "Namibia", "low"), ("the Salar de Uyuni", "Bolivia", "low"),
    ("Svalbard", "Norway", "low"), ("the Pamukkale terraces", "Turkey", "low"),
    ("Lake Bled", "Slovenia", "low"), ("the Gjirokaster fortress", "Albania", "low"),
    ("the Wieliczka Salt Mine", "Poland", "low"), ("the Mostar Bridge", "Bosnia and Herzegovina", "low"),
    ("the Tikal ruins", "Guatemala", "low"), ("the Bagan temples", "Myanmar", "low"),
    ("the Erg Chebbi dunes", "Morocco", "low"),
]

WROTE = [  # "Who wrote {entity}?"  gold = author
    ("Hamlet", "William Shakespeare", "high"), ("Pride and Prejudice", "Jane Austen", "high"),
    ("1984", "George Orwell", "high"), ("Moby-Dick", "Herman Melville", "high"),
    ("War and Peace", "Leo Tolstoy", "high"), ("The Great Gatsby", "F. Scott Fitzgerald", "high"),
    ("Crime and Punishment", "Fyodor Dostoevsky", "high"),
    ("Don Quixote", "Miguel de Cervantes", "high"),
    ("The Odyssey", "Homer", "high"), ("Frankenstein", "Mary Shelley", "high"),
    ("Brave New World", "Aldous Huxley", "high"), ("The Catcher in the Rye", "J. D. Salinger", "high"),
    ("Ulysses", "James Joyce", "high"), ("Beloved", "Toni Morrison", "high"),
    ("The Trial", "Franz Kafka", "high"),
    ("Gravity's Rainbow", "Thomas Pynchon", "low"), ("Blood Meridian", "Cormac McCarthy", "low"),
    ("The Recognitions", "William Gaddis", "low"), ("Stoner", "John Williams", "low"),
    ("Petersburg", "Andrei Bely", "low"), ("The Tartar Steppe", "Dino Buzzati", "low"),
    ("The Man Without Qualities", "Robert Musil", "low"), ("Independent People", "Halldor Laxness", "low"),
    ("The Death of Virgil", "Hermann Broch", "low"), ("Hopscotch", "Julio Cortazar", "low"),
    ("Nightwood", "Djuna Barnes", "low"), ("The Sleepwalkers", "Hermann Broch", "low"),
    ("A Brief History of Seven Killings", "Marlon James", "low"),
    ("The Vegetarian", "Han Kang", "low"), ("Austerlitz", "W. G. Sebald", "low"),
]

CAPITAL = [  # "What is the capital of {entity}?"  gold = capital city
    ("France", "Paris", "high"), ("Japan", "Tokyo", "high"), ("Egypt", "Cairo", "high"),
    ("Brazil", "Brasilia", "high"), ("Canada", "Ottawa", "high"),
    ("Australia", "Canberra", "high"), ("Russia", "Moscow", "high"),
    ("India", "New Delhi", "high"), ("Spain", "Madrid", "high"),
    ("Germany", "Berlin", "high"), ("Italy", "Rome", "high"),
    ("Greece", "Athens", "high"), ("Turkey", "Ankara", "high"),
    ("Argentina", "Buenos Aires", "high"), ("South Africa", "Pretoria", "high"),
    ("Bhutan", "Thimphu", "low"), ("Kazakhstan", "Astana", "low"),
    ("Suriname", "Paramaribo", "low"), ("Eritrea", "Asmara", "low"),
    ("Brunei", "Bandar Seri Begawan", "low"), ("Vanuatu", "Port Vila", "low"),
    ("Kyrgyzstan", "Bishkek", "low"), ("Moldova", "Chisinau", "low"),
    ("Turkmenistan", "Ashgabat", "low"), ("Tajikistan", "Dushanbe", "low"),
    ("Lesotho", "Maseru", "low"), ("Djibouti", "Djibouti", "low"),
    ("Comoros", "Moroni", "low"), ("Kiribati", "Tarawa", "low"),
    ("Palau", "Ngerulmud", "low"),
]

TEMPLATE_DATA = [
    ("In which year was {entity} founded?", 0, FOUNDED),
    ("Who directed the film {entity}?", 1, DIRECTED),
    ("In which country is {entity} located?", 2, LOCATED),
    ("Who wrote {entity}?", 3, WROTE),
    ("What is the capital of {entity}?", 4, CAPITAL),
]


def build(out_path: Path) -> None:
    rows = []
    counters = {"high": 0, "low": 0}
    for template, tidx, items in TEMPLATE_DATA:
        for entity, gold, pop in items:
            qid = f"{pop}_{counters[pop]:04d}"
            counters[pop] += 1
            rows.append({
                "question_id": qid,
                "question": template.format(entity=entity),
                "entity": entity,
                "gold_answer": gold,
                "popularity_bucket": pop,
                "template_idx": tidx,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=1))
    hi = sum(r["popularity_bucket"] == "high" for r in rows)
    lo = len(rows) - hi
    print(f"wrote {len(rows)} questions ({hi} high, {lo} low) -> {out_path}")
    # template balance across buckets
    from collections import Counter
    for pop in ("high", "low"):
        c = Counter(r["template_idx"] for r in rows if r["popularity_bucket"] == pop)
        print(f"  {pop} template dist: {dict(sorted(c.items()))}")


if __name__ == "__main__":
    build(Path("data/self_generation_probe/questions.json"))
