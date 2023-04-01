import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


class LinearRegression:
    def __init__(self, lr=0.01, n_iter=10):
        self.params = None
        self.lr = lr
        self.n_iter = n_iter

    def train(self, X_train, y_train, output_graph_dir='.', graph_step=1):
        n_samples, dimention = X_train.shape
        self.X_train = X_train
        self.y_train = y_train
        self.params = np.zeros(dimention)

        for idx in range(self.n_iter):
            if not idx % graph_step:
                self._save_graph(idx, output_dir=output_graph_dir)
            error = self.predict(self.X_train.T) - self.y_train
            gradient = (1 / n_samples) * 2 * np.dot(error, self.X_train)
            self.params = self.params - self.lr * gradient

    def predict(self, X):
        return np.dot(self.params, X)

    def _save_graph(self, itr, output_dir='.'):
        plot.clf()
        plot.xlim(0, np.max(self.X_train) + 3)
        plot.ylim(0, np.max(self.y_train) + 3)
        plot.xlabel('X Axis')
        plot.ylabel('Y Axis')

        plot.scatter(self.X_train[:, 1], self.y_train)
        plot.title('LR, Itr: %s, W: %.15f, B: %.15f' %
                   (itr, self.params[1], self.params[0]))

        x_test = np.array([[1, 0], [1, 99999999]])
        plot.plot(x_test[:, 1], self.predict(x_test), color='red')
        plot.savefig(os.path.join(output_dir, 'output_graph.%04d.jpg' % itr))


if __name__ == '__main__':
    data = pd.read_csv('./data.csv')
    size, dimention = data.shape
    X = np.column_stack((np.full((size), 1), data['X']))
    y = data['Y']

    modal = LinearRegression(lr=[0.15, 0.001], n_iter=100)
    modal.train(X, y, output_graph_dir='./graphs', graph_step=1)
    x_test = np.array([
        [1, 1.5],
        [1, 20.4],
        [1, 33.5],
        [1, 56.02]
    ])
    print('X tests : %s' % x_test)
    print('Predictions : %s' % modal.predict(x_test.T))
