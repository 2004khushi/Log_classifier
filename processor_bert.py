import joblib
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
model_classification = joblib.load("models/LR_model.joblib")


def classify_with_bert(log_message):

    if isinstance(log_message, str):
        log_messages = [log_message]
    else:
        log_messages = log_message


    embeddings = model_embedding.encode(log_messages)

    all_probabilities = model_classification.predict_proba(embeddings)
    all_predictions = model_classification.predict(embeddings)

    final_labels = []

    #here we using this bcs of the probab issue that we faced that undefined nhi aara tha bcs usko confusion hori thi kya h kya nhi
    for i in range(len(log_messages)):
        probs = all_probabilities[i]

        if max(probs) < 0.5:
            final_labels.append("Unclassified")
        else:
            final_labels.append(all_predictions[i])

    return final_labels[0] if isinstance(log_message, str) else final_labels


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)