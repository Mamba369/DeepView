import os
import json
from datasets import load_dataset
from langchain.chains.flare.base import (
    FlareChain,
    QuestionGeneratorChain,
    _OpenAIResponseChain,
)
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from bing_retriever import BingRetriever
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from rouge_score import rouge_scorer

EXPERIMENTS = 50
RESULTS_DIR = "results"


class SimpleCombineDocumentsChain(BaseCombineDocumentsChain):
    def combine_docs(self, docs):
        combined_text = "\n\n".join(doc.page_content for doc in docs)
        return combined_text

    async def acombine_docs(self, docs):
        return self.combine_docs(docs)


class MyOpenAIResponseChain(_OpenAIResponseChain):
    def _extract_tokens_and_log_probs(self, generations):
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError
        return (
            gen.generation_info["logprobs"]["tokens"],
            gen.generation_info["logprobs"]["token_logprobs"],
        )


class MyFLAREChain:
    def __init__(
        self,
        with_flare: bool = True,
        with_caching: bool = True,
        max_iter: int = 3,
        file_path: str = "wikiasp/data-00000-of-00001.arrow",
    ):
        self.file_path = file_path
        self.with_flare = with_flare
        self.max_iter = max_iter
        self.with_caching = with_caching
        self.dataset = self.load_data()

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        # Initialize FLARE components
        self.retriever = BingRetriever()
        self.llm = OpenAI(temperature=0, model_kwargs={"logprobs": 1})
        self.question_generator_chain = QuestionGeneratorChain(llm=self.llm)
        self.response_chain = MyOpenAIResponseChain(llm=self.llm)
        self.flare = FlareChain(
            question_generator_chain=self.question_generator_chain,
            response_chain=self.response_chain,
            retriever=self.retriever,
            min_prob=0.6,
            max_iter=max_iter,
            # verbose=True
        )
        self.retrieval_chain = RetrievalQA.from_llm(
            retriever=self.retriever, llm=self.llm
        )

    def load_data(self):
        dataset = load_dataset("arrow", data_files=self.file_path)
        return dataset["train"].select_columns(
            ["clean_targets", "clean_title", "domain"]
        )

    def get_aspects(self, sample):
        return [aspect for aspect, _ in sample["clean_targets"]]

    def generate_summary(self, sample):
        summary_lines = []
        for aspect, description in sample["clean_targets"]:
            summary_lines.append(f"{aspect}: {description}")
        return "\n".join(summary_lines)

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

    def run_experiments(self, n=EXPERIMENTS):
        if self.dataset is not None:
            rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = []
            e_f1_scores = []

            for i in range(min(n, len(self.dataset))):
                sample = self.dataset[i]
                aspects = self.get_aspects(sample)
                summary = self.generate_summary(sample)
                aspects_str = ", ".join(aspects)

                few_shot_example = """
The following example helps to determine the format of the output:
Generate a summary about Aslanhane Mosque including the following aspects: location, history with one aspect per line.
Location: The mosque is in the old quarter of ankara next to ankara castle. With an altitude of 947 metres (3,107 ft) it overlooks ankara
at 39°56’12"N 32°51’55"E.
History: The mosque is one of the oldest mosques in Turkey still standing. It was built during the reign of Mesud II of the Anatolian
Seljuks in 1290. Its architect was Ebubekir Mehmet.
"""

                question = (
                    f"{few_shot_example}\n"
                    f"Generate a summary about '{sample['clean_title']}' including the following aspects: {aspects_str}."
                )

                print(f"Sample {i + 1} Aspects: {aspects}\n")
                print(f"Sample {i + 1} Gold Answer:\n{summary}\n")
                print(f"Sample {i + 1} Question:\n{question}\n")

                cache_filename = f"result_wikidata_{i+1}_flare_{self.with_flare}_max_iter_{self.max_iter}.json"

                if cached_result := self.__load_cache(cache_filename):
                    response = cached_result
                else:
                    response = self.run(query=question)
                    self.__save_cache(cache_filename, response)

                answer = response["response"] if "response" in response else response
                print(f"Sample {i + 1} Answer:\n{answer}\n")

                rouge_score = rouge.score(summary, answer)["rougeL"].fmeasure
                rouge_scores.append(rouge_score)

                gold_entities = set(summary.split())
                answer_entities = set(answer.split())
                true_positive = len(gold_entities & answer_entities)
                precision = (
                    true_positive / len(answer_entities) if answer_entities else 0
                )
                recall = true_positive / len(gold_entities) if gold_entities else 0
                e_f1_score = (
                    (2 * precision * recall) / (precision + recall)
                    if (precision + recall)
                    else 0
                )
                e_f1_scores.append(e_f1_score)

                print(f"Sample {i + 1} Rouge-L Score: {rouge_score:.2f}")
                print(f"Sample {i + 1} E-F1 Score: {e_f1_score:.2f}")

                print("\n" + "-" * 80 + "\n")

            avg_rouge_score = sum(rouge_scores) / n
            avg_e_f1_score = sum(e_f1_scores) / n

            print(f"Average Rouge-L Score for {n} samples: {avg_rouge_score:.2f}")
            print(f"Average E-F1 Score for {n} samples: {avg_e_f1_score:.2f}")

            evaluation_results = {
                "Rouge-L Score": avg_rouge_score,
                "E-F1 Score": avg_e_f1_score,
            }
            eval_cache_filename = f"evaluation_wikidata_flare_{self.with_flare}_max_iter_{self.max_iter}_exp_{n}.json"
            self.__save_cache(eval_cache_filename, evaluation_results)

            return evaluation_results
        else:
            raise ValueError("Dataset not loaded. Please call load_data() first.")

    def run(self, query):
        if self.with_flare:
            return self.flare.invoke(input=query)
        return self.llm.invoke(input=query)

    def run_comparative_analysis(self, experiments: range = range(EXPERIMENTS)) -> dict:
        other_chain = MyFLAREChain(
            with_flare=not self.with_flare,
            with_caching=self.with_caching,
            max_iter=self.max_iter,
            file_path=self.file_path,
        )

        results_self_chain_filename = f"evaluation_wikidata_flare_{self.with_flare}_max_iter_{self.max_iter}_exp_{len(experiments)}.json"
        results_other_chain_filename = f"evaluation_wikidata_flare_{other_chain.with_flare}_max_iter_{other_chain.max_iter}_exp_{len(experiments)}.json"

        results_self_chain = self.__load_cache(
            results_self_chain_filename
        ) or self.run_experiments(len(experiments))
        results_other_chain = other_chain.__load_cache(
            results_other_chain_filename
        ) or other_chain.run_experiments(len(experiments))

        print("Comparison of Results:")
        print(
            f"{'Metric':<15}{'With FLARE' if self.with_flare else 'Without FLARE':<15}{'Without FLARE' if self.with_flare else 'With FLARE'}"
        )
        print(
            f"{'Rouge-L Score':<15}{results_self_chain['Rouge-L Score']:<15.4f}{results_other_chain['Rouge-L Score']:.4f}"
        )
        print(
            f"{'E-F1 Score':<15}{results_self_chain['E-F1 Score']:<15.4f}{results_other_chain['E-F1 Score']:.4f}"
        )

        return {
            "self_chain": results_self_chain,
            "other_chain": results_other_chain,
        }


if __name__ == "__main__":
    chain = MyFLAREChain()
    chain.run_comparative_analysis()
