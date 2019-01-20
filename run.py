import tr
import datetime
import pickle
from itertools import combinations
import sentiment
if __name__ == '__main__':
    domain = ["books", "kitchen", "dvd", "electronics"]
    algorithms = ['logistic','svm','random', 'tree']
    extraction = ['tfidf', 'idf','counter','binario']
    k = 500

    lista = [[0,1],[0,2],[0,3],[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]

    for j in algorithms:
        for n in extraction:
            for i in lista:
                src = i[0]
                dst = i[1]
                time = datetime.datetime.now()
                print("loading....")
                tr.train(domain[src],domain[dst],500,10)
                print("Sent....")
                sentiment.sent(domain[src],domain[dst],500,10,50,0.1, j,n,time)
