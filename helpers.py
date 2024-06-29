import json
import os
import pickle
import pdfplumber
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from langchain_openai import OpenAI
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.indexes import GraphIndexCreator
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from IPython.display import SVG, DisplayObject


OPENAI_APIKEY_PATH = "debug/openaiapikey.txt"
INPUT_FILE_PATH = "data/inputs/questions_demo.jsonl"
GRAPH_PATH = "debug/graph.pickle"
GRAPH_PATH_SVG = "data/inputs/graph.svg"
NLP_METRICS = {
    "BERTScore F1",
    "ROUGE-1 F-score",
    "BLEU Score",
    "Cosine Similarity",
    "Euclidean Distance",
    "Jaccard Similarity",
}


def get_embeddings(embeddings_model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=embeddings_model_name)


def get_retriever(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int = 200,
    chunk_overlap: int = 25,
    k_documents: int = 3,
) -> VectorStore:
    print(f"Number of Document objects before recursive splitting: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)
    print(f"Number of Document objects after recursive splitting: {len(documents)}")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k_documents}
    )


def load_jsonl(
    file_path: str = INPUT_FILE_PATH,
) -> list[dict[str, str]]:

    return pd.read_json(path_or_buf=file_path, lines=True).to_dict()


def load_pdf(file_path: str, start_page: int, end_page: int) -> list[Document]:
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages[start_page - 1 : end_page - 1]:
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"page": page.page_number})
                documents.append(doc)
    return documents


def get_connection(
    model_name: str = "gpt-3.5-turbo-instruct", apikey: str = None
) -> OpenAI:
    if not apikey:
        with open(OPENAI_APIKEY_PATH, "r") as file:
            apikey = file.read()

    os.environ["OPENAI_API_KEY"] = apikey

    return OpenAI(model_name=model_name, temperature=0)


def invoke_chain(chain: Chain, input: str) -> dict[str, str]:
    return chain.invoke(input=input)[chain._run_output_key]


def score_answer(
    embeddings: HuggingFaceEmbeddings, question: str, answer: str, reference_answer: str
) -> dict[str, float]:
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)

    cosine_sim = cosine_similarity([question_embedding], [answer_embedding])[0][0]
    euclidean_dist = euclidean(question_embedding, answer_embedding)

    P, R, F1 = bert_score([answer], [question], lang="en")
    bertscore = F1.mean().item()

    rouge = Rouge()
    rouge_scores = rouge.get_scores(answer, reference_answer)
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]

    bleu_score = sentence_bleu(
        [reference_answer.split()],
        answer.split(),
        smoothing_function=SmoothingFunction().method1,
    )

    question_tokens, answer_tokens = set(question.split()), set(answer.split())
    all_tokens = list(question_tokens.union(answer_tokens))
    mlb = MultiLabelBinarizer(classes=all_tokens)
    question_array = mlb.fit_transform([question_tokens])[0]
    answer_array = mlb.fit_transform([answer_tokens])[0]
    jaccard_similarity = jaccard_score(question_array, answer_array)

    # Printing the metrics
    print(f"Cosine Similarity: {cosine_sim:.03f}")
    print(f"Euclidean Distance: {euclidean_dist:.03f}")
    print(f"BERTScore F1: {bertscore:.03f}")
    print(f"ROUGE-1 F-score: {rouge_1:.03f}")
    print(f"BLEU Score: {bleu_score:.03f}")
    print(f"Jaccard Similarity: {jaccard_similarity:.03f}")

    return {
        "Cosine Similarity": cosine_sim,
        "Euclidean Distance": euclidean_dist,
        "BERTScore F1": bertscore,
        "ROUGE-1 F-score": rouge_1,
        "BLEU Score": bleu_score,
        "Jaccard Similarity": jaccard_similarity,
    }


def compare_chains(
    chains: dict[str, Chain],
    embeddings: HuggingFaceEmbeddings,
    data: dict[str, list[str]],
    show_plots: bool = True,
    max_questions_number: int = None,
) -> None:
    result_metrics = {label: [] for label in chains}
    questions, reference_answers = data["question"], data["reference_answer"]
    for index in questions:
        if max_questions_number:
            if index >= max_questions_number:
                break

        question = questions[index]
        print(f"Starting comparison for following question: {question}")
        for label, chain in chains.items():
            answer = invoke_chain(chain=chain, input=question)
            print(f"For chain {label} was provided following answer:\n{answer}")
            result_metrics[label].append(
                score_answer(
                    embeddings=embeddings,
                    question=question,
                    answer=answer,
                    reference_answer=reference_answers[index],
                )
            )
            print()

    if show_plots:
        chain_colors = ["red", "blue", "green", "purple", "orange"]
        for metric in NLP_METRICS:

            plt.figure(figsize=(8, 8))
            plt.title(f"{metric}")

            for idx, label in enumerate(chains):
                metric_values = [metrics[metric] for metrics in result_metrics[label]]
                # plt.scatter(
                #     range(len(metric_values)),
                #     metric_values,
                #     label=label,
                #     color=chain_colors[idx],
                # )
                plt.plot(
                    range(len(metric_values)),
                    metric_values,
                    label=label,
                    color=chain_colors[idx],
                    marker="o",
                    linestyle="-",
                )

            plt.xticks(range(len(metric_values)))
            plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.xlabel("Question Index")
            plt.ylabel("Value")
            y_values = []
            for chain, metrics_list in result_metrics.items():
                y_values.extend([metrics[metric] for metrics in metrics_list])
            plt.ylim(0, max(y_values) + 0.1)
            plt.legend()
            plt.grid(True)
            plt.show()


def load_graph(
    llm: OpenAI, documents: list[Document], force_to_create_new: bool = False
) -> NetworkxEntityGraph:
    if not os.path.isfile(GRAPH_PATH) or force_to_create_new:
        graph = create_graph(llm=llm, documents=documents)
        with open(GRAPH_PATH, "wb") as file:
            pickle.dump(graph, file)
        print(f"Graph has been serialized and saved to {GRAPH_PATH}.")
    else:
        with open(GRAPH_PATH, "rb") as file:
            graph = pickle.load(file)
        print(f"Graph has been deserialized and loaded from {GRAPH_PATH}.")

    return graph


def create_graph(llm: OpenAI, documents: list[Document]) -> NetworkxEntityGraph:
    index_creator = GraphIndexCreator(llm=llm)
    graphs = [index_creator.from_text(document.page_content) for document in documents]
    graph_nx = graphs[0]._graph
    for g in graphs[1:]:
        graph_nx = nx.compose(graph_nx, g._graph)

    return NetworkxEntityGraph(graph_nx)


def show_graph(graph: NetworkxEntityGraph) -> DisplayObject:
    graph.draw_graphviz(path=GRAPH_PATH_SVG, prog="fdp")
    return SVG(GRAPH_PATH_SVG)
