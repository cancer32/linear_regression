import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

data = pd.read_csv('./data.csv')


class LinearRegression:
    def __init__(self, lr=0.01, n_iter=10):
        self.weight = 0
        self.bias = 0
        self.lr = lr
        self.n_iter = n_iter

    def train(self, x_train, y_train, output_graph_dir=None):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

        for idx in range(self.n_iter):
            # Save output graph
            output_graph_dir and self._save_graph(idx, output_graph_dir)

            dc_dw = np.mean(
                -2 \
                * self.x_train \
                * (self.y_train - ((self.weight * self.x_train) + self.bias))
            )
            dc_db = np.mean(
                -2 \
                * (self.y_train - ((self.weight * self.x_train) + self.bias))
            )
            self.weight = self.weight - dc_dw * self.lr
            self.bias = self.bias - dc_db * self.lr

    def predict(self, x):
        return (self.weight * x) + self.bias

    def _save_graph(self, itr, output_dir='.'):
        plot.clf()
        plot.ylim(0, 20)
        plot.xlim(0, 20)
        plot.xlabel('X Axis')
        plot.ylabel('Y Axis')
        plot.scatter(self.x_train, self.y_train)
        plot.title('LR, Itr: %s, W: %.15f, B: %.15f' %
                   (itr, self.weight, self.bias))
        x_test = np.array([0, 99999999])
        plot.plot(x_test, self.predict(x_test), color='red')
        plot.savefig(os.path.join(output_dir, 'output_graph.%04d.jpg' % itr))


if __name__ == '__main__':
    modal = LinearRegression(lr=0.008, n_iter=1000)
    modal.train(data['xp'], data['yp'], output_graph_dir='./graphs')
    x_test = np.array([1.5, 20.4, 33.5, 56.02])
    print('x tests : %s' % x_test)
    print('Predictions : %s' % modal.predict(x_test))
