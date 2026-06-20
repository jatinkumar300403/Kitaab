import app

# Monkeypatch the retrieval to avoid running embeddings for the smoke test.
def fake_retrieve(query, category, tone, initial_top_k=50, final_top_k=16):
    return app.books.head(8)

app.retrieve_semantic_recommendations = fake_retrieve

results = app.recommend_books("A lonely preacher reflecting on family and forgiveness", "All", "Happy")
print("Returned", len(results), "items")
for r in results[:5]:
    print(r)
