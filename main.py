import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer, LoggingHandler
from sklearn.cluster import KMeans
from sklearn.manifold.t_sne import _joint_probabilities
from sklearn.metrics import pairwise_distances

# sns.set(rc={'figure.figsize':(11.7,8.27)})

palette = sns.color_palette("bright", 5)

from k_nearest_neighbor import KNearestNeighbor

MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 3
perplexity = 30


def fit(X):
    n_samples = X.shape[0]

    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)


def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()

    obj_func = _kl_divergence

    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])

    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded


def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    for i in range(it, n_iter):

        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        print("[t-SNE] Iteration %d: error = %.7f,"
              " gradient norm = %.7f"
              % (i + 1, error, grad_norm))

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break

        if grad_norm <= min_grad_norm:
            break
    return p


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)

    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


class ReviewTransformer(object):
    def __init__(self, dataset, from_src=False, load_bert=False):
        self.dataset = dataset
        self.full_reviews = []
        self.review_embeddings = []
        self.getData(from_src)

        if not os.path.exists('out'):
            os.mkdir('out')

        if load_bert:
            #### Just some code to print debug information to stdout
            np.set_printoptions(threshold=100)

            logging.basicConfig(format='%(asctime)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.INFO,
                                handlers=[LoggingHandler()])
            #### /print debug information to stdout

            # Load Sentence model (based on BERT) from URL
            self.embed_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def getData(self, from_src=False, return_data=False):
        if from_src:
            dataFile = 'data/' + self.dataset + '.tsv'
            df = pd.read_csv(dataFile, encoding='latin1', sep='\t')

        else:
            dataFile = 'out/' + self.dataset + '_out.csv'
            df = pd.read_csv(dataFile, encoding='latin1', sep=',')

        # print(df.head(), df.describe())

        # df.insert(5, 'typeSmallSmall', '')

        review_headline = df['review_headline'].apply(str).apply(str.strip).values.tolist()
        review_body = df.loc[:, ('review_body')].apply(str).apply(str.strip).values.tolist()

        # print(review_headline[:5])
        for i, headline in enumerate(review_headline):
            if "Stars" in headline:
                review_headline[i] = ''

        self.df = df
        self.full_reviews = [head + ' ' + body for head, body in zip(review_headline, review_body)]
        df.insert(df.columns.size, 'full_review', self.full_reviews)
        if return_data:
            return self.full_reviews

    def load_embedding(self, embedding_file=None):
        if embedding_file is None:
            embedding_file = 'out/' + self.dataset + '_embedding.txt'
        self.review_embeddings = np.load(embedding_file, allow_pickle=True)

    def review_clustering(self, num_clusters=5):
        # Perform kmean clustering
        num_clusters = 5  # rating clusters
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(self.review_embeddings)
        cluster_assignment = clustering_model.labels_
        self.df.insert(self.df.columns.size, 'Cluster Label', cluster_assignment)

        # clustered_sentences = [[] for i in range(num_clusters)]
        # for sentence_id, cluster_id in enumerate(cluster_assignment):
        #     clustered_sentences[cluster_id].append(full_review[sentence_id])
        #
        # for i, cluster in enumerate(clustered_sentences):
        #     print("Cluster ", i+1)
        #     print(cluster[:10])
        #     print("")

    def review2vec(self, out_file=None):
        if out_file is None:
            out_file = 'out/' + self.dataset + '_embedding.txt'

        self.review_embeddings = np.array(self.embed_model.encode(self.full_reviews))

        if out_file is not None:
            self.review_embeddings.dump(out_file)
        return self.review_embeddings

    def to_csv(self):
        self.df.to_csv('out/' + self.dataset + '_out.csv')

    def split_data_by(self, groupby='product_parent'):
        # convert series to datetime
        self.df['review_date'] = pd.to_datetime(self.df['review_date'])

        # iterate groups and add results to grps list
        grps = []
        for _, group in self.df.groupby([groupby], sort=False):
            group = group.sort_values(by=['review_date'], ascending=True)
            grps.append(group)

        # print(grps[0][groupby])
        return grps

    def ewma_rating_all_groups(self, group_data, k=5, groupby='product_parent'):
        for group in group_data:
            if group.size > 2000:
                group_id = group[groupby].values.tolist()[0]
                # print(group_id)
                r_li = group['star_rating'].values.tolist()
                v_li = group['hr_k_' + str(k)].values.tolist()

                r_vanilla = self.vanilla_ewma_rating_one_group(r_li, alpha=0.5)
                r_corrected = self.ewma_rating_one_group(r_li, v_li, eps=0.5)
                plt.title('Product Identifier: ' + str(group_id) + ' in Dataset ' + self.dataset.upper())
                print(len(r_li))
                plt.scatter(np.arange(len(r_li)), r_li, marker='*', label='Original Ratings', color='#008cea')
                plt.plot(r_vanilla, label='Vanilla Moving Rating', color=('0.7'), linestyle='--')
                plt.plot(r_corrected, label='Effective Moving Rating', color='#ff7c00')
                break

        plt.xlabel('time $t$')
        plt.ylabel('rating')
        plt.legend(loc='lower right')
        plt.show()

    def vanilla_ewma_rating_one_group(self, original_ratings, alpha=0.5):
        '''vanilla EWMA running rating algorithm

        :param k: k-NN hyper-parameter
        :param eps: scaling factor
        :return:
        '''
        ratings = np.zeros(len(original_ratings))
        # hr_k = self.df['hr_k_'+str(k)].values.tolist()
        ratings[0] = original_ratings[0]
        for i, r_i in enumerate(original_ratings[1:]):
            ratings[i + 1] = ratings[i] + alpha * (r_i - ratings[i])

        return ratings

    def ewma_rating_one_group(self, original_ratings, rating_effectiveness, eps=0.1):
        '''EWMA running rating algorithm with review correction

        :param k: k-NN hyper-parameter
        :param eps: scaling factor
        :return:
        '''
        assert len(original_ratings) == len(rating_effectiveness)
        length = len(original_ratings)
        ratings = np.zeros(length)
        # hr_k = self.df['hr_k_'+str(k)].values.tolist()
        ratings[0] = original_ratings[0]
        for i, (r_i, v_i) in enumerate(zip(original_ratings[1:], rating_effectiveness[1:])):
            ratings[i + 1] = ratings[i] + v_i * (1 - eps) * (r_i - ratings[i])

        return ratings

    def knn_vote_processing(self):
        self.df['helpful_votes'].apply(int)
        self.df['total_votes'].apply(int)

        help_votes = self.df.loc[:, ('helpful_votes', 'total_votes')].values.tolist()
        help_votes = np.array(help_votes)
        help_ratio = help_votes[:, 0] / (help_votes[:, 1] + 1e-8)
        self.df.insert(self.df.columns.size, 'helpfulness ratio', help_ratio)

        data_mask = np.array((self.df['total_votes'] != 0).values.tolist())

        X_train = self.review_embeddings[data_mask]
        y_train = np.array(self.df.loc[self.df['total_votes'] != 0, ('helpfulness ratio')].values.tolist())

        X_test = self.review_embeddings[~data_mask]

        # print(X_train.shape,y_train.shape)
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        for k in [1, 3, 5, 7, 9]:
            y_pred = knn.predict(X_test, k)
            help_ratio[~data_mask] = y_pred
            self.df.insert(self.df.columns.size, 'hr_k_' + str(k), help_ratio)

        # data = self.df.loc[:, ('review_id', 'helpful_votes', 'total_votes')]
        #
        # print(X_train.shape,y_train,X_test.shape,self.df.size)
        # data = np.array(data)
        # X_train_mask = data[:,2]=='0'
        # print(data[:,2],X_train_mask)
        # X_train = self.review_embeddings[]

    def tsne_plot(self, label_type='Cluster Label', num_samples_per_cluster=100):
        X = self.review_embeddings
        y_labels = []
        x_embds = []
        for i in range(5):
            mask = self.df['Cluster Label'] == i
            x_embds.append(X[mask])
            if label_type == 'star_rating':
                y_labels.append(['Original Rating ' + str(v) for v in self.df['star_rating'][mask].values.tolist()])
            else:
                y_labels.append(['Rating Cluster' + str(v + 1) for v in self.df['Cluster Label'][mask].values.tolist()])

        x_embds = [x for x_embd in x_embds for x in x_embd[:num_samples_per_cluster]]
        y_labels = [y_lb for y_label in y_labels for y_lb in y_label[:num_samples_per_cluster]]

        x_embds = np.array(x_embds)
        y_labels = np.array(y_labels)
        X_embedded = fit(x_embds)
        print(X_embedded.shape, y_labels.shape)

        ax = sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_labels, legend='full', palette=palette)
        # fig = ax.get_figure()
        # fig.savefig("output.png")
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.title('Visualization of Review Embeddings in Dataset ' + self.dataset.upper(), y=1.05)
        if label_type == 'star_rating':
            plt.savefig('out/' + self.dataset + '_tsne_vis_origin_label.png')
        else:
            plt.savefig('out/' + self.dataset + '_tsne_vis.png')
        plt.show()

    def wordcloud_plot(self):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from wordcloud import WordCloud
        count_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                           stop_words='english',
                                           token_pattern="\\b[a-z][a-z]+\\b",
                                           lowercase=True,
                                           max_df=0.6, max_features=4000)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                           stop_words='english',
                                           token_pattern="\\b[a-z][a-z]+\\b",
                                           lowercase=True,
                                           max_df=0.6, max_features=4000)

        cv_data = count_vectorizer.fit_transform(self.df.ReviewTextLower)
        tfidf_data = tfidf_vectorizer.fit_transform(self.df.ReviewTextLower)

        for_wordcloud = count_vectorizer.get_feature_names()
        for_wordcloud = for_wordcloud
        for_wordcloud_str = ' '.join(for_wordcloud)

        wordcloud = WordCloud(width=800, height=400, background_color='black',
                              min_font_size=7).generate(for_wordcloud_str)

        plt.figure(figsize=(20, 10), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.show()


def main():
    datasets = ['hair_dryer', 'microwave', 'pacifier']

    # phase 1
    # for dataset in datasets:
    #     reformer = ReviewTransformer(dataset)
    #     reformer.review2vec()
    #     reformer.review_clustering()
    #     reformer.knn_vote_processing()
    #     reformer.to_csv()

    # phase 2
    # for dataset in datasets:
    #     reformer = ReviewTransformer(dataset)
    #     grps = reformer.split_data_by()
    #     reformer.ewma_rating_all_groups(grps)
    # phase 3
    for dataset in datasets:
        reformer = ReviewTransformer(dataset)
        reformer.load_embedding()
        reformer.tsne_plot('star_rating')
        reformer.tsne_plot('Cluster Label')
    # phase 4
    # for dataset in datasets:
    #     reformer = ReviewTransformer(dataset)
    #     reformer.load_embedding()
    #     reformer.wordcloud_plot()


if __name__ == '__main__':
    main()
