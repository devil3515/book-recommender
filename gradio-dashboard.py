import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr
load_dotenv()
##############################
books = pd.read_csv('data/processed/books_with_emotion.csv')

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'src/img/cover-not-found.jpg',
    books['large_thumbnail']
)


##############################
# Define paths
CHROMA_DB_PATH = "vector_store/chroma_db"  # Directory where Chroma DB will be stored
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create vector database
if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
    print("Loading existing Chroma database...")
    db_books = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
else:
    print("Creating new Chroma database...")
    raw_document = TextLoader('data/processed/tagged_description.txt', encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
    documents = text_splitter.split_documents(raw_document)

    db_books = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=CHROMA_DB_PATH)
    # ✅ Save to disk

print("Chroma DB ready.")
##############################

###############################
# raw_document = TextLoader('data/processed/tagged_description.txt',encoding="utf-8").load()
# text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap = 0, separator='\n')
# documents = text_splitter.split_documents(raw_document)
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db_books = Chroma.from_documents(
#     documents,
#     embedding = embedding_model
# )

##############################
# def retrieve_sementic_recommendation(
#         query: str,
#         category: str = None,
#         tone: str = None,
#         initial_top_k: int = 50,
#         final_top_k: int = 16,
# )-> pd.DataFrame:
#     recs = db_books.similarity_search(query, k= initial_top_k)
#     book_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
#     books_recs = books[books['isbn13'].isin(book_list)].head(final_top_k)
#     if category != 'All':
#         books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
#     else:
#         books_recs = books_recs.head(final_top_k)
#
#     if tone == 'Happy':
#         books_recs.sort_values(by='joy', ascending=False, inplace=True)
#     elif tone == 'Surprising':
#         books_recs.sort_values(by='surprise', ascending=False, inplace=True)
#     elif tone == 'Angry':
#         books_recs.sort_values(by='anger', ascending=False, inplace=True)
#     elif tone == 'Suspenseful':
#         books_recs.sort_values(by='fear', ascending=False, inplace=True)
#     elif tone == 'Sad':
#         books_recs.sort_values(by='sadness', ascending=False, inplace=True)
#     return books_recs
#
# #################################################
# def recommend_books(
#         query: str,
#         category: str,
#         tone: str
# ):
#     recommendations = retrieve_sementic_recommendation(query, category, tone)
#     results = []
#     for _, row in recommendations.iterrows():
#         description = row['description']
#         truncated_desc_split = description.split()
#         truncated_description = " ".join(truncated_desc_split[:30]) + "..."
#
#         authors_split = row['authors'].split(';')
#         if len(authors_split) == 2:
#             authors_str = f'{authors_split[0]} and {authors_split[1]}'
#         elif len(authors_split) > 2:
#             authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
#         else:
#             authors_str = row['authors']
#
#         captions = f"{row['title']} by {authors_str}: {truncated_description}"
#         results.append((row['large_thumbnail'], captions))
#     return results
# #######################################################
# categories = ['All'] + sorted(books['simple_categories'].unique())
# tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
#
# with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
#     gr.Markdown("# Semantic Book Recommender")
#
#     with gr.Row():
#         user_query = gr.Textbox(label="Enter a book title or description",
#                                 placeholder="e.g., A story about forgiveness..")
#         category_dropdown = gr.Dropdown(choices = categories, label = 'Select a category:', value = 'All')
#         tone_dropdown = gr.Dropdown(choices = tones, label='Select a emotional tone:', value='All')
#         submit_button = gr.Button(value="Find Recommendations")
#     gr.Markdown("## Recommendations")
#     output = gr.Gallery(label="Recommended Books", columns = 8, rows = 2)
#     submit_button.click(fn = recommend_books,
#                         inputs=[user_query, category_dropdown, tone_dropdown],
#                         outputs=output)
#
# if __name__ == "__main__":
#     dashboard.launch()

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()






