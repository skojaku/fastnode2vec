from gensim.models import Word2Vec
from gensim import __version__ as gensim_version
import numpy as np
from numba import njit
from .graph import Graph
from tqdm import tqdm


@njit
def set_seed(seed):
    np.random.seed(seed)


class Node2Vec(Word2Vec):
    def __init__(
        self,
        walk_length=40,
        window_length=10,
        p=1.0,
        q=1.0,
        workers=1,
        epochs=1,
        batch_walks=None,
        seed=None,
    ):
        if batch_walks is None:
            batch_words = 10000
        else:
            batch_words = min(walk_length * batch_walks, 10000)

        self.window_length = window_length
        self.workers = workers
        self.batch_words = batch_words
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.seed = seed
        self.epochs = epochs

    
    def fit(self, A):
        self.graph = Graph(A)
        self.num_nodes = A.shape[0]

    def transform(self, dim, *, progress_bar=True, **kwargs):
        def gen_nodes(epochs):
            if self.seed is not None:
                np.random.seed(self.seed)
            for _ in range(epochs):
                for i in np.random.permutation(self.num_nodes):
                    # dummy walk with same length
                    yield [i] * self.walk_length


        args = {}
        if gensim_version < "4.0.0":
            args["iter"] = 1
            args["size"] = dim
        else:
            args["epochs"] = 1
            args["vector_size"] = dim

        super().__init__(
            sg=1,
            min_count=1,
            window=self.window_length,
            workers=self.workers,
            batch_words=self.batch_words,
            **args,
        )
        self.build_vocab(([w] for w in range(self.num_nodes)))

        if progress_bar:

            def pbar(it):
                return tqdm(
                    it, desc="Training", total=self.epochs * self.num_nodes 
                )

        else:

            def pbar(it):
                return it

        super().train(
            pbar(gen_nodes(self.epochs)),
            total_examples=self.epochs * self.num_nodes,
            epochs=1,
            **kwargs,
        )
        
        self.in_vec = np.zeros((self.num_nodes, dim))
        self.out_vec = np.zeros((self.num_nodes, dim))
        for i in range(self.num_nodes):
            if i not in self.wv:
                continue
            self.in_vec[i, :] = self.wv[i]
            self.out_vec[i, :] = self.syn1neg[
                self.wv.key_to_index[i]
            ]
        return self.in_vec

    def generate_random_walk(self, t):
        return self.graph.generate_random_walk(self.walk_length, self.p, self.q, t)

    def _do_train_job(self, sentences, alpha, inits):
        if self.seed is not None:
            set_seed(self.seed)
        sentences = [self.generate_random_walk(w[0]) for w in sentences]
        return super()._do_train_job(sentences, alpha, inits)
