import logging
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import json
from sentence_transformers import util, SentenceTransformer
from datasets import load_dataset
import requests
from urllib.parse import urlparse
import time
import os
import warnings
import html

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_DIR = "data"


class WikidataRetriever:
    def __init__(
        self, limit: int = 200, retry_delay: int = 5, with_caching: bool = True
    ):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.dataset = load_dataset("AmazonScience/mintaka", trust_remote_code=True)
        self.limit = limit
        self.retry_delay = retry_delay
        self.with_caching = with_caching

    def __execute_query(self, pattern: str, limit: int, offset: int) -> dict:
        sparql_query = f"""
        CONSTRUCT {{
            {pattern}
        }} WHERE {{
            {pattern}
            FILTER (langMatches(lang(?o), "en") || !isLiteral(?o) || langMatches(lang(?s), "en") || !isLiteral(?s))
        }}
        LIMIT {limit}
        OFFSET {offset}
        """
        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(
            "https://query.wikidata.org/sparql",
            headers=headers,
            params={"query": sparql_query},
        )

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def __fetch_all_results(self, pattern: str) -> set:
        all_triples = set()
        offset = 0
        retry_attempts = 0
        max_retries = 10

        while True:
            try:
                response = self.__execute_query(pattern, self.limit, offset)
                new_triples = set(
                    (
                        result["subject"]["value"].replace(
                            "http://www.wikidata.org/entity/", ""
                        ),
                        result["predicate"]["value"].replace(
                            "http://www.wikidata.org/prop/direct/", ""
                        ),
                        result["object"]["value"].replace(
                            "http://www.wikidata.org/entity/", ""
                        ),
                    )
                    for result in response["results"]["bindings"]
                )

                new_results_count = len(new_triples - all_triples)
                all_triples.update(new_triples)

                logger.info(
                    f"Fetched: {len(new_triples)} results, {new_results_count} new results."
                )

                if new_results_count == 0:
                    break

                offset += self.limit
                retry_attempts = 0

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_attempts += 1
                    if retry_attempts > max_retries:
                        logger.error("Maximum retry attempts reached. Exiting...")
                        break

                    sleep_time = self.retry_delay * (2 ** (retry_attempts - 1))
                    logger.warning(
                        f"Rate limit exceeded. Retrying after {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"HTTP error occurred: {e}")
                    break

        return all_triples

    def __filter_triples(self, triples: list[list[str]]) -> tuple[list, list]:
        labeled_triples = []
        for subject, predicate, object_value in triples:
            if "statement/" in subject or "statement/" in object_value:
                continue
            elif object_value.startswith(
                "http://commons.wikimedia.org/wiki/Special:FilePath/"
            ):
                continue
            elif "schema.org/about" in predicate:
                continue
            else:
                labeled_triples.append((subject, predicate, object_value))

        logger.info(
            f"Triples Filtered: {len(triples)} original results, {len(labeled_triples)} filtered results."
        )
        return self.__get_labels_for_triples(labeled_triples)

    def get_triples(self, entity_id: str) -> list:
        if labeled_triples := self.__load_cache(entity_id):
            return labeled_triples

        outgoing_pattern = f"wd:{entity_id} ?p ?o ."
        incoming_pattern = f"?s ?p wd:{entity_id} ."

        outgoing_results = self.__fetch_all_results(outgoing_pattern)
        incoming_results = self.__fetch_all_results(incoming_pattern)

        original_results = list(outgoing_results.union(incoming_results))

        labeled_triples = self.__filter_triples(triples=original_results)

        self.__save_cache(entity_id, original_results, labeled_triples)

        return labeled_triples

    def __get_labels_for_triples(self, triples: list) -> list:
        def create_values_clause(triples: list) -> str:
            values = ""
            for s, p, o in triples:
                values += (
                    f"('{html.escape(s)}' '{html.escape(p)}' '{html.escape(o)}')\n"
                )
            return values

        def execute_sparql_query(values_clause: str) -> dict:
            query = f"""
            SELECT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
            VALUES (?s ?p ?o) {{
                {values_clause}
            }}
            ?entity_s rdfs:label ?sLabel FILTER (lang(?sLabel) = "en").
            ?entity_p rdfs:label ?pLabel FILTER (lang(?pLabel) = "en").
            ?entity_o rdfs:label ?oLabel FILTER (lang(?oLabel) = "en").
            BIND(IRI(CONCAT("http://www.wikidata.org/entity/", ?s)) AS ?entity_s)
            BIND(IRI(CONCAT("http://www.wikidata.org/entity/", ?p)) AS ?entity_p)
            BIND(IRI(CONCAT("http://www.wikidata.org/entity/", ?o)) AS ?entity_o)
            }}
            """
            try:
                self.sparql.setMethod(POST)
                self.sparql.setQuery(query)
                self.sparql.setReturnFormat(JSON)
                return self.sparql.query().convert()
            except Exception as e:
                logger.error(
                    f"QueryBadFormed: The SPARQL query is badly formed.\nQuery: {query}\nError: {e}"
                )
                raise

        # Split the triples into batches
        batch_size = 20  # Further reduced batch size
        batches = [
            triples[i : i + batch_size] for i in range(0, len(triples), batch_size)
        ]

        labeled_triples = []

        for batch in batches:
            values_clause = create_values_clause(batch)
            results = execute_sparql_query(values_clause)
            for result in results["results"]["bindings"]:
                s = result["sLabel"]["value"]
                p = result["pLabel"]["value"]
                o = result["oLabel"]["value"]
                labeled_triples.append((s, p, o))

        return labeled_triples

    def top_k_neighbors(self, question: str, entities: list, top_k: int = 5) -> list:
        all_triples = []
        for entity in entities:
            neighbors = self.get_triples(entity["name"])
            all_triples.extend(neighbors)

        question_embedding = self.model.encode(question, convert_to_tensor=True)
        triple_embeddings = self.model.encode(all_triples, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_embedding, triple_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:top_k]
        top_triples = [all_triples[i] for i in top_indices]
        return top_triples

    def __load_cache(self, entity_id: str) -> list[dict] | None:
        cache_file = os.path.join(DATA_DIR, f"{entity_id}_labeled_triples.json")
        if self.with_caching and os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                return json.load(file)

    def __save_cache(
        self,
        entity_id: str,
        original_results: set,
        labeled_triples: list,
    ) -> None:
        if self.with_caching:
            for triples, prefix in [
                (original_results, f"{entity_id}_original"),
                (labeled_triples, f"{entity_id}_labeled"),
            ]:
                filename = os.path.join(DATA_DIR, f"{prefix}_triples.json")
                with open(filename, "w") as file:
                    json.dump(triples, file, indent=4)


if __name__ == "__main__":
    entity_id = "Q180004"
    retriever = WikidataRetriever(with_caching=True)
    entities = [{"name": entity_id}]
    question = "Who is the president of the United States?"
    print(retriever.top_k_neighbors(question, entities))
