import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

test_case = pd.read_excel('./test-case.xlsx')
test_case_desc = test_case['Test Case Description']

df = pd.DataFrame(columns=['Feature', 'Score'])

#initializing the model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

sentence1 = ''
sentence2 = 'User should be able to change his profile picture.'
encoding2 = model.encode(sentence2)
for value in test_case['Feature']:
    sentence1 = value
    encoding1 = model.encode(sentence1)
    similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
    # print(value, '- ', round(similarity, 1))
    
    df.loc[-1] = [value, round(similarity, 1)]
    df.index = df.index + 1
    df = df.sort_index()

res_df = df[df.Score == df.Score.max()].iloc[0]['Feature']

# print(res_df)
case_df = test_case.loc[test_case['Feature'] == res_df]
print(case_df)
