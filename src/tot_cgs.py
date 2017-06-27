import numpy as np
import time
from scipy.special import beta

### Desired extensions:


class TOT_CGS():

    def __init__(self, K = 40, document_topic_prior = None, topic_word_prior = 0.01,
                 num_iterations = 100, evaluate_every = 10, perp_tol = 1e-1, verbose = True):
        self.K = K
        if document_topic_prior == None:
            self.document_topic_prior = 50/self.K
        else:
            self.document_topic_prior = document_topic_prior
        self.topic_word_prior = topic_word_prior
        self.num_iterations = num_iterations
        self.evaluate_every = evaluate_every
        self.perp_tol = perp_tol
        self.verbose = verbose

    def get_model_parameters(self):
        return({'document_topic_matrix': self.document_topic_matrix,
            'topic_word_matrix': self.topic_word_matrix,
            'psi': self.psi})

    def get_model_variables(self):
        return({'X_indptr': self.X_indptr,
            'timestamps_for_all_words': self.timestamps_for_all_words,
            'document_of_word': self.document_of_word,
            'Z': self.Z,
            'indices': self.indices,
            'sigma': self.sigma,
            'delta': self.delta,
            'delta_z': self.delta_z,
            'timestamps': self.timestamps,
            'M': self.M,
            'V': self.V,
            'N': self.N,
            'X': self.X})

    def get_model_hyperparameters(self):
        return({'document_topic_prior': self.document_topic_prior,
            'topic_word_prior': self.topic_word_prior,
            'K': self.K,
            'num_iterations': self.num_iterations})
            
    def init_variables(self):
        X_data = self.X.data
        X_indices = self.X.indices
        self.X_indptr = self.X.indptr
        self.timestamps_for_all_words = []
        self.document_of_word = []
        self.Z = []
        self.indices = []
        self.sigma = np.zeros((self.M, self.K))
        self.delta = np.zeros((self.K, self.V))
        self.delta_z = np.zeros(self.K)
        self.psi = np.ones((self.K, 2))

        if self.verbose:
            print("Initializing variables...")
        # for every document,
        for j in range(self.M):
            # get counts of all words in document
            document_word_counts = X_data[self.X_indptr[j]:self.X_indptr[j+1]]
            # get indices of all words in document
            document_word_indices = X_indices[self.X_indptr[j]:self.X_indptr[j+1]]
            # get indices of all words in document, repeated if they occur more than once
            document_indices_with_counts = np.repeat(document_word_indices, document_word_counts)
            # for all the words in the document,
            for word_index in document_indices_with_counts:
                # append document number to self.document_of_word
                self.document_of_word.append(j)
                # append timestamp of that word to timestamps_for_all_words_
                self.timestamps_for_all_words.append(self.timestamps[j])
                # append word index to self.indices
                self.indices.append(word_index)
                # randomly sample z from topics
                z = np.random.randint(0, self.K)
                # append sampled z to self.Z
                self.Z.append(z)
                # update counters
                self.sigma[j,z] += 1
                self.delta[z,word_index] += 1
                self.delta_z[z] += 1
        self.timestamps_for_all_words = np.array(self.timestamps_for_all_words)
        self.Z = np.array(self.Z)

    def update_psi(self):
        timestamps_belonging_to_topics = [[] for _ in range(self.K)]
        for i in range(self.K):
            timestamps_belonging_to_topics[i].append(self.timestamps_for_all_words[self.Z == i])
            mean_i = np.mean(timestamps_belonging_to_topics[i])
            var_i = np.var(timestamps_belonging_to_topics[i])
            self.psi[i, 0] = mean_i * (mean_i * (1 - mean_i) / var_i - 1)
            self.psi[i, 1] = (1 - mean_i) * (mean_i * (1 - mean_i) / var_i - 1)

    def perform_gibbs_sampling(self):
        if self.verbose:
            print("Performing Gibbs Sampling...")
            start_time = time.time()
        # for each iteration,
        for epoch in range(1, self.num_iterations + 1):
            # for each word,
            for i in range(self.N):
                # get sampled topic of word
                old_z = self.Z[i]
                # get vocabulary index of word
                word_index = self.indices[i]
                # get which document the word is part of
                # word_document = sum(i >= self.X_indptr)
                word_document = self.document_of_word[i]
                # decrement counters
                self.sigma[word_document, old_z] -= 1
                self.delta[old_z, word_index] -= 1
                self.delta_z[old_z] -= 1
                # calculate P(z_mn|W, Z_-mn, t, alpha, beta, psi)
                P_z = (self.sigma[word_document, :] + self.document_topic_prior)
                P_z *= (self.delta[:, word_index] + self.topic_word_prior)/(self.delta_z + self.V*self.topic_word_prior)
                P_z *= ((1 - self.timestamps[word_document])**(self.psi[:, 0] - 1)) * (self.timestamps[word_document]**(self.psi[:, 1] - 1)) / beta(self.psi[:, 0], self.psi[:, 1])
                # sample new z_mn from P(z_mn|Z_-mn, alpha, beta)
                new_z = np.random.choice(a = range(self.K), p = P_z/sum(P_z), size = 1)[0]
                # increment counters
                self.sigma[word_document, new_z] += 1
                self.delta[new_z, word_index] += 1
                self.delta_z[new_z] += 1
                # update z_mn
                self.Z[i] = new_z
            ## update psi
            self.update_psi()
            # print progress after every epoch
            if self.verbose:
                print("\tIteration %d, %.2f%% complete, %0.0f mins elapsed"%(epoch,
                                                                           100*epoch/self.num_iterations,
                                                                           (time.time() - start_time)/60))
        self.calculate_document_topic_matrix()
        self.calculate_topic_word_matrix()
        if self.verbose:
            print("Total time elapsed: %0.0f mins"%((time.time() - start_time)/60))

    def fit(self, X, timestamps):
        self.M, self.V = X.shape
        self.N = np.sum(X)
        self.X = X
        self.timestamps = timestamps
        self.init_variables()
        self.perform_gibbs_sampling()

    def calculate_document_topic_matrix(self):
        self.document_topic_matrix = (self.sigma + self.document_topic_prior)/((np.sum(self.sigma, axis = 1) + self.K*self.document_topic_prior)[:, np.newaxis])

    def calculate_topic_word_matrix(self):
        self.topic_word_matrix = (self.delta + self.topic_word_prior)/((np.sum(self.delta, axis = 1) + self.V*self.topic_word_prior)[:, np.newaxis])

    def perplexity(self):
        self.calculate_document_topic_matrix()
        self.calculate_topic_word_matrix()
        log_sum = 0
        for m in range(self.M):
            for n in range(self.V):
                sum = 0
                for k in range(self.K):
                    sum += (self.document_topic_matrix[m,k] * self.topic_word_matrix[k,n])
                log_sum += np.log(sum)
        return(np.exp(-log_sum/self.N))

    def transform(self):
        # TODO
        return("TODO")
