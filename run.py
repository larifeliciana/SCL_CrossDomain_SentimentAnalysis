import tr
import datetime
import pickle
from itertools import combinations
import sentiment
if __name__ == '__main__':
    domain = ["books", "kitchen", "dvd", "electronics"]
    x = 0
    algorithms = None
    while(algorithms == None):

            x = input("Select the classifier\n1-Logistic Regression\n2-Random Forest\n3-Decision Tree\n4-SVM\n")
            print(x)
            if x == '1':
                algorithms = ['logistic']
            elif x == '2':
                algorithms = ['random']
            elif x == '3':
                algorithms = ['tree']
            elif x == '4':
                algorithms = ['svm']
            else:
                print("\n" * 130)
                print(x + " IS AN INVALID INPUT, TRY AGAIN!")
                
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
