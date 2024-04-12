# import argparse

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import ChatOpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def similarity_search(query_text):
    # Load environment variables
    load_dotenv()

    # Embedding function
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Prepare the database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    # Perform the similarity search in the database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.5:
        print("Unable to find matching results.")
        return None

    return results


def generate_response_from_results(results, query_text):
    if results is None:
        return None, None

    # Preparing the context text from the search results
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    # Using a template for generating the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Printing the prompt for debugging or logging
    print(prompt)

    # Generate the response using the model
    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    # Preparing the source information
    sources = []
    for doc, _score in results:
        source = doc.metadata.get("source", "Unknown Source")
        page_number = doc.metadata.get("page", "Unknown Page")
        page_content = doc.page_content

        source_info = f">File: {source} - \
            Page: {page_number}\
            \n>>> {page_content}\n\n"

        sources.append(source_info)

    return response_text, sources


def perform_query(query_text):
    # Search for similar documents
    results = similarity_search(query_text)

    # Generate the response from the search results
    response_text, sources = generate_response_from_results(results, query_text) # noqa

    return response_text, sources


if __name__ == "__main__":
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    query_text = "tell me some about the childhood of steve jobs?"
    response_text, sources = perform_query(query_text)
    print(response_text)
    print(sources)
