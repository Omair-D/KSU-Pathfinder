from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)


@app.route("/")
def index():
        return render_template('flash.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)



def get_Chat_response(text):

    embeddings_path = "data/winter_olympics_2022.csv"
    
    # Load the data
    df = load_data(embeddings_path)

    # Sample query
    query = text

    # Ask the question and get the response
    answer = ask(query, df)
    return answer



import openai
import ast
import pandas as pd
import tiktoken
import os
from scipy.spatial import distance




openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# Set model names
GPT_MODELS = ["gpt-4", "gpt-4-turbo"]
EMBEDDING_MODEL = "text-embedding-ada-002"




def get_embeddings(text_list):
    """Generate embeddings for a list of texts."""
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text_list,
    )
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings




def load_data(file_path):
    """Load data from a CSV and process embeddings."""
    df = pd.read_csv(file_path)
    # Convert string embeddings to actual lists
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df




def cosine_similarity(vec1, vec2):
    """Return cosine similarity between two vectors."""
    return 1 - distance.cosine(vec1, vec2)




def strings_ranked_by_relatedness(query, df, relatedness_fn=cosine_similarity, top_n=5):
    """Returns a list of the most related strings based on the cosine similarity of embeddings."""
    query_embedding = get_embeddings([query])[0]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]




def num_tokens(text, model=GPT_MODELS[0]):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))




def query_message(query, df, model, token_budget):
    """Return a message for GPT with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = "Use the below articles to answer the subsequent question. If the answer cannot be found, say 'I could not find an answer.'"
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nArticle section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question




def ask(query, df, model=GPT_MODELS[0], token_budget=4096 - 500, print_message=False):
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions using information from articles."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message['content']
    return response_message





if __name__ == '__main__':
    app.run()