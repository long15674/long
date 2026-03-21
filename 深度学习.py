import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# 1. 语料
# =========================
corpus = [
    "i love machine learning",
    "i love deep learning",
    "i enjoy flying",
    "machine learning is fun",
    "deep learning is powerful",
    "i love natural language processing"
]

sentences = [s.split() for s in corpus]

# =========================
# 2. 词典
# =========================
words = [w for s in sentences for w in s]
vocab = list(set(words))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
vocab_size = len(vocab)

# =========================
# 3. 训练数据
# =========================
window_size = 2
neg_samples = 3

training_data = []

for sentence in sentences:
    for i, word in enumerate(sentence):
        center = word2idx[word]

        #  随机窗口
        current_window = random.randint(1, window_size)

        for j in range(max(0, i-current_window), min(len(sentence), i+current_window+1)):
            if i != j:
                context = word2idx[sentence[j]]

                # 正样本
                training_data.append((center, context, 1))

                # 负样本
                for _ in range(neg_samples):
                    neg_word = random.choice(vocab)
                    neg_idx = word2idx[neg_word]
                    training_data.append((center, neg_idx, 0))

# =========================
# 4. 初始化
# =========================
embedding_dim = 10
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(vocab_size, embedding_dim)

# =========================
# 5. sigmoid
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================
# 6. 训练
# =========================
lr = 0.01
epochs = 200

for epoch in range(epochs):
    loss = 0

    for center, target, label in training_data:
        v_c = W1[center]
        v_t = W2[target]

        score = sigmoid(np.dot(v_c, v_t))

        error = score - label
        loss += - (label * np.log(score + 1e-9) + (1-label)*np.log(1-score + 1e-9))

        # 梯度更新
        W1[center] -= lr * error * v_t
        W2[target] -= lr * error * v_c

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# =========================
# 7. 降维
# =========================
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(W1)

# =========================
# 8. 可视化
# =========================
selected_words = vocab[:10]

plt.figure()

for word in selected_words:
    idx = word2idx[word]
    x, y = vectors_2d[idx]
    plt.scatter(x, y)
    plt.text(x, y, word)

plt.title("Word2Vec Visualization")
plt.show()