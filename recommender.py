# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


# class RecommenderEngine:
#     def __init__(self):
#         self.model_ready = False
#         self.user_item_matrix = None
#         self.similarity_matrix = None
#         self.items = []

#     def train(self, df: pd.DataFrame):
#         if df.empty:
#             self.model_ready = False
#             return False

#         # pivot user-item interaction
#         matrix = df.pivot_table(index="user_id", columns="item_id", values="weight", fill_value=0)

#         self.items = list(matrix.columns)
#         self.user_item_matrix = matrix

#         # compute cosine similarity across items
#         self.similarity_matrix = cosine_similarity(matrix.T)
#         self.similarity_matrix = pd.DataFrame(self.similarity_matrix, index=self.items, columns=self.items)

#         self.model_ready = True
#         return True

#     def recommend(self, user_id: int, top_k: int = 10):
#         if not self.model_ready:
#             return []

#         if user_id not in self.user_item_matrix.index:
#             return []

#         user_vector = self.user_item_matrix.loc[user_id]

#         scores = {}

#         for item in self.items:
#             if user_vector[item] == 0:  # recommend only unseen items
#                 similar_items = self.similarity_matrix[item]
#                 score = np.dot(similar_items, user_vector)
#                 scores[item] = float(score)

#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#         return [
#             {"externalId": item, "score": round(score, 4)}
#             for item, score in sorted_scores[:top_k]
#         ]




import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderEngine:
    def __init__(self):
        self.model_ready = False
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.items = []

    def train(self, df: pd.DataFrame):
        if df.empty: self.model_ready = False; return False
        matrix = df.pivot_table(index="user_id", columns="item_id", values="weight", fill_value=0)
        self.items = list(matrix.columns)
        self.user_item_matrix = matrix
        self.similarity_matrix = pd.DataFrame(cosine_similarity(matrix.T), index=self.items, columns=self.items)
        self.model_ready = True
        return True

    def recommend(self, user_id: int, top_k: int = 10):
        if not self.model_ready or user_id not in self.user_item_matrix.index: return []
        user_vector = self.user_item_matrix.loc[user_id]
        # print("User vector:", user_vector.to_dict())
        # print("Similarity matrix:\n", self.similarity_matrix)

        scores = {item: float(np.dot(self.similarity_matrix[item], user_vector)) 
                  for item in self.items if user_vector[item] == 0}
        return [{"externalId": item, "score": round(score, 4)} for item, score in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_k]]
