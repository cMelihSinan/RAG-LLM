from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np
import re


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                data.append(line.strip())
    return data


def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    return vectors.toarray(), feature_names, vectorizer

def display_vectors(data, vectors):
    print("\nVector representations of the logs:\n")
    for i, vector in enumerate(vectors):
        print(f"Log: {data[i]}")
        print(f"Vector: {vector}")
        print("-" * 60)


file_path = '/Users/melihsinancubukcuoglu/Desktop/output.txt'
data = load_data(file_path)
vectors, feature_names, vectorizer = vectorize_data(data)

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def extract_keywords(query):
    stopwords = {'which', 'what', 'where', 'log', 'has', 'in', 'it', 'the', 'a', 'an', 'is', 'are', 'and',
                 'on', 'to', 'of', 'for', 'with', 'as', 'at', 'by', 'this', 'that', 'from', 'into', 'over',
                 'under', 'within', 'when', 'how', 'why', 'who', 'whom', 'whose', 'which', 'where', 'what',
                 'whence', 'whither', 'whether', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not','only', 'own', 'same', 'so', 'than',
                 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    words = re.findall(r'\w+', query.lower())
    keywords = [word for word in words if word not in stopwords]
    return ' '.join(keywords)


def retrieve_similar_logs(query, vectorizer, faiss_index, data, top_k=5):
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return []

    query_vector = vectorizer.transform([query_keywords]).toarray()
    distances, indices = faiss_index.search(query_vector, top_k)
    retrieved_logs = [data[i] for i in indices[0]]

    filtered_logs = [log for log in retrieved_logs if any(keyword in log.lower() for keyword in query_keywords.split())]

    return filtered_logs if filtered_logs else []



def generate_answer(query, context):
    if context:
        context_summary = " ".join(context[:3])
        response_prompt = (
            f"The logs relevant to your query, '{query}', are:\n{context_summary}\n"
            f"<-Is the retrieved log you asked."
        )
    else:
        response_prompt = f"There are no logs that match the query '{query}'."

    input_text = response_prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, num_beams=10, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)





def chatbot():
    display_vectors(data, vectors)
    print("Welcome to the log analysis chatbot!")
    print("You can ask questions about the access logs, or type 'exit' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Exiting the chatbot. Goodbye!")
            break

        retrieved_logs = retrieve_similar_logs(query, vectorizer, index, data)

        if not retrieved_logs:
            print(f"There is no content in the logs like '{query}'.")
            continue

        print("Retrieved Logs:")
        for log in retrieved_logs:
            print(log)
        print("-" * 60)

        answer = generate_answer(query, retrieved_logs)
        print("Model Response:\n", answer)
        print("*" * 60)


chatbot()
