import csv
import string

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

def readDataSet(file='train.csv', SIZE=30 ):
	
	names = ["" for x in range(SIZE)]
	vectorized_names = [[] for x in range(SIZE) ]
	with open(file) as csvDataFile:
    		csvReader = csv.reader(csvDataFile)
    		i = 0
    		for row in csvReader:
    			row[1]= row[1].lower()
    			row[1] = "$"+row[1]+"$"
    			row[2] = "+"+row[2]+"+"
    			row = row[1]+row[2]
    			names[i] = row
    	
    			#print("-------------")
    			#print(names[i])
    			vectorized_names[i] = string_vectorizer(names[i])    	
    			i+=1

	#print(names)
	print(vectorized_names)
	return vectorized_names

#print(readDataSet(file='train.csv', SIZE=30 ))
