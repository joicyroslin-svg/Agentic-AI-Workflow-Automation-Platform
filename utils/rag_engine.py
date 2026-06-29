from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunks(query, chunks, top_k=3):
    if not query or not chunks:
        return []

    documents = [query] + chunks

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        query_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]

        scores = cosine_similarity(query_vector, chunk_vectors).flatten()

        retrieved_chunks = []

        for index, score in enumerate(scores, start=1):
            retrieved_chunks.append({
                "source": f"Document Section {index}",
                "score": round(float(score), 3),
                "chunk": chunks[index - 1]
            })

        retrieved_chunks.sort(key=lambda item: item["score"], reverse=True)

        return retrieved_chunks[:top_k]

    except Exception:
        return []


def combine_retrieved_chunks(retrieved_chunks):
    context = ""

    for item in retrieved_chunks:
        context += f"\n\n[{item['source']} | Similarity Score: {item['score']}]\n"
        context += item["chunk"]

    return context.strip()


def calculate_rag_confidence(retrieved_chunks):
    if not retrieved_chunks:
        return 0

    total_score = sum(item["score"] for item in retrieved_chunks)
    return round(total_score / len(retrieved_chunks), 3)