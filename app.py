import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fast_retrieval import load_corpus, top_matches
import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv", dtype={"isbn13": str})
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"] == "&fife=w800",
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Use a simple local retrieval corpus for instant responses.
corpus = load_corpus("tagged_description.txt")


def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    if not query:
        return pd.DataFrame()

    books_list = top_matches(query, corpus, top_k=initial_top_k)
    if not books_list:
        # fallback to a simple substring search for poor query coverage
        query_lower = query.lower()
        matches = books[books["description"].str.lower().str.contains(query_lower, na=False)]
        book_recs = matches.head(final_top_k)
    else:
        book_recs = books[books["isbn13"].astype(str).isin(books_list)].head(initial_top_k)

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
    import os
    import sys
    import traceback

    port = int(os.environ.get("PORT", 7860))
    server_name = "0.0.0.0"

    # Try launching without a public share link (preferred on Spaces).
    # If the environment disallows localhost access, fall back to a shareable link.
    try:
        dashboard.launch(share=False, server_name=server_name, server_port=port)
    except Exception as e:
        # If Gradio complains that localhost is not accessible, attempt a share link.
        # Be defensive: some exceptions may not be string-iterable (e.g., booleans),
        # so coerce to string and safely check substrings.
        try:
            err_str = str(e)
        except Exception:
            err_str = ""

        match = False
        try:
            match = any(sub in err_str for sub in ("localhost is not accessible", "shareable link must be created"))
        except TypeError:
            match = False

        if match:
            try:
                dashboard.launch(share=True, server_name=server_name, server_port=port)
            except Exception:
                print("Fallback share launch also failed:", file=sys.stderr)
                traceback.print_exc()
                raise
        else:
            # Unknown error: re-raise after printing for diagnostics.
            print("Gradio launch failed:", file=sys.stderr)
            traceback.print_exc()
            raise