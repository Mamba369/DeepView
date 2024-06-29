import re
import os
import json
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from wikidata_retriever import WikidataRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FEWSHOT_EXAMPLES = """Question: What is the seventh tallest mountain in North America?
Example Output: Mount Lucania
Question: What year was the first book of the A Song of Ice and Fire series published?
Example Output: 1996
Question: How old was Taylor Swift when she won her first Grammy?
Example Output: 20
Question: Has there ever been a Christian U.S. senator?
Example Output: Yes"""

INSTRUCTIONS = """Answer the user's question as concisely as possible.
If the answer is a number, then response should be just number, not text."""

RESULTS_DIR = "results"
EXPERIMENTS = 50


class MyKAPINGChain:
    def __init__(
        self,
        with_kaping: bool = True,
        with_caching: bool = True,
        top_k: int = 3,
    ):
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        self.with_kaping = with_kaping
        self.with_caching = with_caching
        self.llm = OpenAI(temperature=0)
        self.top_k = top_k
        self.retriever = WikidataRetriever(with_caching=with_caching)
        self.__kaping_prompt_template = PromptTemplate(
            input_variables=["question_number", "question", "fewshots", "entities"],
            template=f"""{INSTRUCTIONS}
Here are some few-shot examples for output format:
{{fewshots}}
Below are the facts that might be relevant to answer the question, but they may not:
{{entities}}
Question {{question_number}}: {{question}}
Answer:""",
        )
        self.__default_prompt_template = PromptTemplate(
            input_variables=["question_number", "question", "fewshots"],
            template=f"""{INSTRUCTIONS}
Here are some few-shot examples for output format:
{{fewshots}}
Question {{question_number}}: {{question}}
Answer:""",
        )

    def __save_cache(self, filename: str, data: dict) -> None:
        if self.with_caching:
            with open(os.path.join(RESULTS_DIR, filename), "w") as f:
                json.dump(data, f)
            print(f"Saved cache to {filename}")

    def __load_cache(self, filename: str) -> dict:
        cache_file = os.path.join(RESULTS_DIR, filename)
        if self.with_caching and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                print(f"Loaded cache from {filename}")
                return json.load(f)

    def answer_question(
        self, question_number: int, question: str, entities: list, reference: str
    ) -> str:
        cache_filename = f"result_mintaka_{question_number}_kaping_{self.with_kaping}_k_{self.top_k}.json"

        if cached_result := self.__load_cache(cache_filename):
            return cached_result["response"]

        if self.with_kaping:
            top_entities = self.retriever.top_k_neighbors(
                question, entities, self.top_k
            )
            str_top_entities = ", ".join(
                [f"({', '.join(entity)})" for entity in top_entities]
            )
            prompt = self.__kaping_prompt_template.format(
                question_number=question_number,
                question=question,
                fewshots=FEWSHOT_EXAMPLES,
                entities=str_top_entities,
            )
        else:
            prompt = self.__default_prompt_template.format(
                question_number=question_number,
                question=question,
                fewshots=FEWSHOT_EXAMPLES,
            )

        response = self.llm.invoke(prompt)
        cleaned_response = self.__clean_text(response)
        print(f"Question: {question}")
        print(f"Generated Response: {cleaned_response}")
        print(f"Expected Answer: {reference}")

        self.__save_cache(
            cache_filename, {"response": cleaned_response, "reference": reference}
        )

        return cleaned_response

    def __clean_text(self, text: str) -> str:
        return re.sub(r"\W+", " ", text).strip()

    def evaluate(self, predictions: list, references: list) -> dict:
        cleaned_predictions = list(map(self.__clean_text, predictions))
        cleaned_references = list(map(self.__clean_text, references))

        precision, recall, f1, _ = precision_recall_fscore_support(
            cleaned_references, cleaned_predictions, average="macro", zero_division=0
        )
        accuracy = accuracy_score(cleaned_references, cleaned_predictions)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def run_experiments(self, experiments=range(EXPERIMENTS)) -> dict:
        selected_dataset = self.retriever.dataset["test"].select(experiments)

        predictions = [
            self.answer_question(
                i + 1, entry["question"], entry["questionEntity"], entry["answerText"]
            )
            for i, entry in enumerate(selected_dataset)
        ]

        results = self.evaluate(
            predictions, [entry["answerText"] for entry in selected_dataset]
        )

        print("Results:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")

        self.__save_cache(
            f"evaluation_mintaka_kaping_{self.with_kaping}_k_{self.top_k}_exp_{len(experiments)}.json",
            results,
        )
        return results

    def run_comparative_analysis(self, experiments: range = range(EXPERIMENTS)) -> dict:
        other_chain = MyKAPINGChain(
            with_kaping=not self.with_kaping,
            with_caching=self.with_caching,
            top_k=self.top_k,
        )

        results_self_chain_filename = f"evaluation_mintaka_kaping_{self.with_kaping}_k_{self.top_k}_exp_{len(experiments)}.json"
        results_other_chain_filename = f"evaluation_mintaka_kaping_{other_chain.with_kaping}_k_{self.top_k}_exp_{len(experiments)}.json"

        results_self_chain = self.__load_cache(
            results_self_chain_filename
        ) or self.run_experiments(experiments)
        results_other_chain = other_chain.__load_cache(
            results_other_chain_filename
        ) or other_chain.run_experiments(experiments)

        print("Comparison of Results:")
        print(
            f"{'Metric':<15}{'With KAPING' if self.with_kaping else 'Without KAPING':<15}{'Without KAPING' if self.with_kaping else 'With KAPING'}"
        )
        print(
            f"{'Accuracy':<15}{results_self_chain['accuracy']:<15.4f}{results_other_chain['accuracy']:.4f}"
        )
        print(
            f"{'Precision':<15}{results_self_chain['precision']:<15.4f}{results_other_chain['precision']:.4f}"
        )
        print(
            f"{'Recall':<15}{results_self_chain['recall']:<15.4f}{results_other_chain['recall']:.4f}"
        )
        print(
            f"{'F1 Score':<15}{results_self_chain['f1']:<15.4f}{results_other_chain['f1']:.4f}"
        )

        return {
            "self_chain": results_self_chain,
            "other_chain": results_other_chain,
        }


if __name__ == "__main__":
    chain = MyKAPINGChain()
    chain.run_comparative_analysis()
