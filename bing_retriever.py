import os
import requests
import hashlib
from flask import Flask, request, jsonify
from diskcache import Cache
import logging
from ratelimit import limits, sleep_and_retry
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

logging.basicConfig(level=logging.ERROR)

CACHE_DIR = "./cache"
BING_API_URL = "https://api.bing.microsoft.com/v7.0/search"
SUBSCRIPTION_KEY = os.getenv(
    "BING_SEARCH_KEY", "your_bing_api_key"
)  # Replace with your Bing API key
cache = Cache(CACHE_DIR)

app = Flask(__name__)


class BingRetriever(BaseRetriever):
    docs: list[Document] = Field(default_factory=list)
    k: int = 1
    only_domain: str = None
    exclude_domains: list[str] = [
        "wikipedia.org",
        "wikiwand.com",
        "wiki2.org",
        "wikimedia.org",
    ]
    cache_dir: str = CACHE_DIR
    local_server_url: str = "http://127.0.0.1:8080/bing_search"
    local_server_running: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.local_server_running = self.__check_local_server()

    def __check_local_server(self) -> bool:
        try:
            response = requests.get(self.local_server_url)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            return False
        return False

    @staticmethod
    def __hash_query(query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def __save_to_cache(self, query: str, results: dict):
        hash_key = self.__hash_query(query)
        cache[hash_key] = results

    def __load_from_cache(self, query: str) -> dict:
        hash_key = self.__hash_query(query)
        return cache.get(hash_key, None)

    @sleep_and_retry
    @limits(calls=50, period=1)
    def __call_bing_api(self, query: str) -> dict:
        headers = {
            "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
        }
        params = {
            "q": query,
            "mkt": "en-US",
            "responseFilter": ["Webpages"],
            "count": self.k,
            "safeSearch": "Off",
            "setLang": "en-US",
        }
        response = requests.get(BING_API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def __search_bing_api(self, query: str):
        query = self.__prepare_query(query)
        cached_result = self.__load_from_cache(query)
        if cached_result:
            logging.info(f"CACHE: {query}")
            return cached_result
        else:
            bing_result = self.__call_bing_api(query)
            parsed_result = self.__parse_bing_response(bing_result)
            self.__save_to_cache(query, parsed_result)
            return parsed_result

    def __prepare_query(self, query: str) -> str:
        if self.only_domain:
            query += f" site:{self.only_domain}"
        if self.exclude_domains:
            query += " " + " ".join([f"-site:{d}" for d in self.exclude_domains])
        return query

    def __parse_bing_response(self, response: dict) -> list[dict]:
        results = []
        if "webPages" in response and "value" in response["webPages"]:
            for page in response["webPages"]["value"]:
                result = {
                    "url": page["url"],
                    "title": page["name"],
                    "snippet": page["snippet"],
                }
                results.append(result)
        return results

    def __search_local_server(
        self, query: str, only_domain: str = None, exclude_domains: list[str] = []
    ) -> list[dict]:
        data = {"query": query}
        if only_domain:
            data["+domain"] = only_domain
        if exclude_domains:
            data["-domain"] = ",".join(exclude_domains)
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.local_server_url, headers=headers, json=data)
        logging.info("API call response: %s", response.text)
        return response.json()

    def _get_relevant_documents(self, query: str) -> list[Document]:
        if self.local_server_running:
            search_results = self.__search_local_server(
                query, self.only_domain, self.exclude_domains
            )
        else:
            search_results = self.__search_bing_api(query)

        documents = [
            Document(
                page_content=result["snippet"],
                metadata={"title": result["title"], "url": result["url"]},
            )
            for result in search_results
        ][: self.k]
        return documents

    @staticmethod
    @app.route("/bing_search", methods=["POST"])
    def bing_request():
        logging.info("Received request: %s", request.data)
        data = request.get_json()
        logging.info("Parsed JSON data: %s", data)
        query = data.get("query", None)
        only_domain = data.get("+domain", None)
        exclude_domains = data.get("-domain", "")
        exclude_domains = exclude_domains.split(",") if exclude_domains else []
        retriever = BingRetriever(
            docs=[], only_domain=only_domain, exclude_domains=exclude_domains
        )
        result = retriever.__search_bing_api(query)
        logging.info("Result: %s", result)
        if result is None:
            return jsonify({"message": "ERROR: Request Failed"}), 404
        else:
            return jsonify(result), 200

    def run_local_server(self, host: str = "0.0.0.0", port: int = 8080):
        app.run(host=host, port=port)


if __name__ == "__main__":
    wyszukiwarka = BingRetriever()
    zapytanie = "OpenAI"

    print("Wywoływanie wyszukiwarki...")
    # Pobierz odpowiednie dokumenty
    dokumenty = wyszukiwarka.invoke(zapytanie)
    print(dokumenty)
