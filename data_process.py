import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import regex
import string
import json


''' 
We use One-Hot Representation for charachters,
Note: We Don't care about utf8 related charactes like '\' in the middle,
We believe LSTM-RNN can learn and handle them, Also these characters exist both on Brazilian and non-Brizilan names.
Aside from charactes 'a'- 'z' we added '+' and '$' to specify last and first names respectively.
We should not forget about numbers '0' - '9' and '\' which are utf8 related.
So in total we have: 
'a' - 'z'             26
'0' - '9'             10
'\', '$' and '+'      3
total                 39
abcdefghijklmnopqrstuvwxyz0123456789\$+
'''




#print(string.ascii_lowercase+string.digits+"\\"+"$"+"+")
def string_vectorizer(strng, alphabet=string.ascii_lowercase+string.digits+"\\"+"$"+"+"):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector



#print(len(names))

def readDataSet(file='data/test.csv', SIZE=30 ):

    names = ["" for x in range(SIZE)]
    vectorized_names = [[] for x in range(SIZE) ]
    final_lst = []
    with open(file) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            i = 0
            for row in csvReader:
                if (i == 0):
                    i += 1
                    continue
                if (i == SIZE):
                    break
                row[1]= row[1].lower()
                row[2] = row[2].lower()
                row[1] = "$"+row[1]+"$"
                row[2] = "+"+row[2]+"+"
                brz = row[3]
                row = row[1]+row[2]
                names[i] = row
                

                name_lst = [1 for i in range(25)]
                for j in range(25):
                    if (j == len(names[i])):
                        break
                    name_lst[j] = d[names[i][j]]

                final_lst.append([name_lst, brz])


                #print("-------------")
                #print(names[i])
                # vectorized_names[i] = string_vectorizer(names[i])    	
                i += 1

    #print(names)
    # print(vectorized_names)
    with open('testData.json', 'w') as outfile:
        json.dump(final_lst, outfile)
    return final_lst

# max length 23

letters = string.ascii_lowercase + '0123456789'
d = dict(zip(letters, [i for i in range(2, 38)]))
d['\\'] = 38
d['$'] = 39
d['+'] = 40
# print(d)

fileObject = open("data/test.csv","r+")
SIZE = sum(1 for row in fileObject)
print(readDataSet(file='data/test.csv', SIZE=30))