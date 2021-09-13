import numpy as np
from tqdm import tqdm


def load_file(file: str='glove.6B.300d.txt'):
    
    glove_vec = {}
    
    with open(file, 'r', encoding='utf8') as f:

        for line in tqdm(f.readlines()):

            splits = line.split()

            word = splits[0]

            vector = np.array(splits[1:]).astype(np.float32)
            
            glove_vec[word] = vector
            
    return glove_vec


def construct_embedding(all_emb: dict, id2word: dict, dim: int):
    
    embeddings = np.random.randn(len(id2word), dim)*0.01

    embeddings[0, :].fill(0)
    
    k = 0

    for i, w in tqdm(id2word.items()):

        try:
            vector = all_emb[w]
            embeddings[i, :] = vector
        except:
            k += 1
            pass
        
    print(f"loaded = {len(id2word) - k}/{len(id2word)}")
        
    return embeddings