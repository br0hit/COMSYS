import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def question_answering(input_question):
    # Initialize SBERT model
    sentence_transformer = SentenceTransformer("bert-base-nli-mean-tokens")

    # Tokenized questions file path
    tokenized_questions_file = './tokenized_questions.npy'

    # Check if tokenized questions file exists
    # if os.path.exists(tokenized_questions_file):
    #     # Load question embeddings from file
    #     question_embeddings = np.load(tokenized_questions_file)
    # else:
    # Load questions and answers from JSON file
    qa_file = './qa.json'
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)

    # Extract questions and answers from JSON data
    questions = [entry['question'] for entry in qa_data]
    answers = [entry['answer'] for entry in qa_data]

    #print(questions)

    # Encode question embeddings
    question_embeddings = sentence_transformer.encode(questions)

    # Save question embeddings to a file
    np.save(tokenized_questions_file, question_embeddings)

    user_input = input_question

    user_input_embedding = sentence_transformer.encode([user_input])[0]  # Access the first (and only) element

    # Calculate cosine similarity with each question embedding
    cos_sim_scores = [1 - cosine(user_input_embedding, q_embedding) for q_embedding in question_embeddings]

    # Find the index of the highest cosine similarity score
    best_match_index = np.argmax(cos_sim_scores)

    # Retrieve the best-matched answer
    best_match_answer = answers[best_match_index]
    best_match_cos_sim = cos_sim_scores[best_match_index]
    
    if best_match_cos_sim >= 0.75:
            print(best_match_answer)
            return best_match_answer
    else:
        print("Working")
        return "Sorry, I am not capable to answer your query"
    
