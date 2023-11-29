import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer

# Test Case
####################################################
first_model = [
    "What is the man doing?",
    "Where is the man?",
    "What time of day is it?",
    "What is the man eating?",
    "How many tables are there?"
]

second_model = [
    "What is the time?",
    "Why is the person eating?",
    "What type of material is the floor?",
    "How big is the room?",
    "How many lights are there?"
]

third_model = [
    "What is the person doing?",
    "How much food is the man eating?",
    "How many seats are in the room?",
    "What is the person sitting on?",
    "What food is in the room?"
]
####################################################

def cosine(u, v):
  return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Cross-product of cosine scores
def calculate_scores(scores_arr, sentences_a, sentences_b, model):
  for sentence in sentences_a:
    for phrase in sentences_b:
      scores_arr.append((sentence, phrase, cosine(model.encode(sentence), model.encode(phrase))))

  return scores_arr


def remove_questions(scores_arr, question_a, question_b):
  to_remove = []

  # Find all of the scores that should be removed
  # Need to do this first, as removing as we go in the same loop will possibly skip back to back removable sentences
  for score in scores_arr:
    # If we want to implement threshold, we can have the following:
    # if cosine(score[0], question_a) < threshold or cosine(score[1], question_b) < threshold

    # I think that we should probably compare score[0] to question_b as well as score[1] to question_a
    # The reason I am not so far is because I looked for exact matches, but since we probably want threshold then
    # it would be necessary to compare the 4 possible pairs between score[0], score[1] and question_a, question_b

    if score[0] == question_a or score[1] == question_b:
      to_remove.append(score)

  # Removes all of the removable scores
  for score in to_remove:
    scores_arr.remove(score)
  return scores_arr

model = SentenceTransformer('bert-base-nli-mean-tokens')

scores = []

first_pair = calculate_scores(scores, first_model, second_model, model)
second_pair = calculate_scores(first_pair, first_model, third_model, model)
third_pair = calculate_scores(second_pair, second_model, third_model, model)

new_scores = sorted(third_pair, key=lambda x: x[2], reverse=True)

chosen = []
# If we want to change number of chosen questions, just change the 3 to n and specify n
while len(chosen) < 3:
  chosen_score = new_scores[0]
  chosen.append(chosen_score)
  new_scores.remove(chosen_score)
  new_scores = remove_questions(new_scores, chosen_score[0], chosen_score[1])

print(chosen)