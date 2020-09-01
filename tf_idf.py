import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
import os
import numpy as np
import json

# df = pd.read_csv(os.path.join(os.getcwd(),'data/Twitter15/comments_text.csv'))
# print(df.columns)
# docs = df['text'].to_list()
# cv = CountVectorizer()
# word_count_vector = cv.fit(docs)
# bag_of_words = word_count_vector.transform(docs)
# words_sum = bag_of_words.sum(axis=0)
# words_freq = [(word, words_sum[0, idx]) for word, idx in word_count_vector.vocabulary_.items()]
# words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
# word_dict = {}
# for idx, word in enumerate(words_freq):
#     word_dict[word[0]] = idx
# with open(os.path.join(os.getcwd(),'data/Twitter15/word_dict.json'), 'w') as f:
#     json.dump(word_dict, f)

cv = CountVectorizer()
tweet_info = pd.read_csv(os.path.join(os.getcwd(),'data/Twitter15/comments_text.csv'))
with open(os.path.join(os.getcwd(), 'data/Twitter15/tree_dictionary.json'), 'r') as f, open(os.path.join(os.getcwd(), 'data/Twitter15/word_dict.json'), 'r') as word:
    tree_dict = json.load(f)
    word_dict = json.load(word)
with open(os.path.join(os.getcwd(), 'data/Twitter15/data.TD_RvNN.vol_5000_new.txt'), 'a') as out:
    for i in range(len(tree_dict.keys())):
        # ith tree
        current_tree_nodes = {}
        whole_tree = tweet_info[tweet_info['tree_id'].str.startswith(str(i)+'_')]
        tweet_id = tweet_info[tweet_info['tree_id'] == str(i)+'_0']['tweet_id'].tolist()[0]
        print(tweet_id)
        edges = tree_dict[str(i)]
        max_node = max(edges, key=lambda x : len(edges[x]))
        max_degree = len(edges[max_node])
        max_len = 0
        for j in range(len(whole_tree[['tree_id', 'text']])):
            current_tree_nodes[whole_tree['tree_id'].tolist()[j]] = {}
            try:
                x = cv.fit_transform([whole_tree['text'].tolist()[j]])
                tokens = cv.get_feature_names()
                tokens_count = x.toarray()[0]
            except:
                tokens = []
                tokens_count = []
            max_len = sum(tokens_count) if sum(tokens_count) > max_len else max_len
            for c in range(len(tokens)):
                current_tree_nodes[whole_tree['tree_id'].tolist()[j]][tokens[c]] = tokens_count[c]
        # print(current_tree_nodes)
        source_text_represent = []
        for token, count in current_tree_nodes[str(i) + '_0'].items():
            source_text_represent.append(str(word_dict[token])+':'+str(count))
        source_text_represent = ' '.join(source_text_represent)
        out.write('\t'.join([str(tweet_id), 'None', '1', str(max_degree), str(max_len), source_text_represent]) + '\n')
        for parent, children in edges.items():
            parent_node = str(int(parent.split('_')[1]) + 1)
            for child in children:
                child_node = str(int(child.split('_')[1]) + 1)
                text_represent = []
                for token, count in current_tree_nodes[child].items():
                    text_represent.append(str(word_dict[token])+':'+str(count))
                text_represent = ' '.join(text_represent)
                out.write('\t'.join([str(tweet_id), parent_node, child_node, str(max_degree), str(max_len), text_represent]) + '\n')
