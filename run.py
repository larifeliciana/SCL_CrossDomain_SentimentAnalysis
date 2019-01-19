import tr
import datetime
import pickle
from itertools import combinations
import sentiment
if __name__ == '__main__':
    domain = ["books", "kitchen", "dvd", "electronics"]
    algorithms = ['svm','logistic','random', 'tree']
    extraction = ['tfidf', 'idf','counter','delta','binario']
    k = 250

  #  lista = [[0,1],[0,2],[0,3],[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
    lista = [[0,1]]

    for i in lista:
        src = i[0]
        dst = i[1]
        print(datetime.datetime.now())
        print("loading....")
        tr.train(domain[src],domain[dst],k,10)
        print("Sent....")
        sentiment.sent(domain[src],domain[dst],k,10,50,0.1, "random","idf")
    print(datetime.datetime.now())

    #[(a, b) for a in  for b in lista2]
    """for j in algorithms:
        for n in extraction:
            for i in lista:
                src = i[0]
                dst = i[1]
                print(datetime.datetime.now())
                print("loading....")
                tr.train(domain[src],domain[dst],500,10)
                print("Sent....")
                sentiment.sent(domain[src],domain[dst],500,10,50,0.1, "random")"""
