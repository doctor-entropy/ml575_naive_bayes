import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import glob
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import time

output_labels = glob.glob('20_newsgroups/*')

data_paths = glob.glob('20_newsgroups/*/*')

print "Processing Data, Please wait \n"

data_collection = []
source_news = []
for data in data_paths:
    file_object = open(data, 'r')
    value = False
    temp_str = ''
    for line in file_object:
        if ('Lines:' in line):
            value = True
        if(value and 'Lines:' not in line and 'Distribution:' not in line and 
            'Message-ID:' not in line and 'References:' not in line and 'In article <' not in line and
            'Sender:' not in line and 'Date:' not in line and 'Organization:' not in line and 'Nntp-Posting-Host:'
            not in line and 'writes:' not in line and 'NNTP-Posting-Host:' not in line and 'Reply-To:' not in line 
            and '_' not in line):
            #line.encode('utf-8').strip()
            temp_str = temp_str + line
    file_object.close()
    source_news.append(data.split('/')[1])
    if(len(temp_str) > 0):
        data_collection.append(temp_str)
    else:
        data_collection.append('')

count_vector = CountVectorizer(lowercase=True, decode_error='ignore', stop_words='english', max_features = 2000)
count_vector.fit(data_collection)
frequency_matrix = pd.DataFrame(count_vector.transform(data_collection).toarray())

x_train, x_test, y_train, y_test = train_test_split(frequency_matrix, source_news, random_state = 42)

def two_class_prediction(class1, class2):
    print "Class 1:", class1
    print "Class 2:", class2
    n_class1 = 0
    n_class2 = 0
    class1_count = 0
    class2_count = 0
    for k,y in enumerate(y_train):
        if(class1 in y ):
            class1_count += x_train.iloc[k]
            n_class1 += 1
        elif(class2 in y):
            class2_count += x_train.iloc[k]
            n_class2 += 1
            
    p_class1 = float(n_class1)/(n_class1 + n_class2)
    p_class2 = float(n_class2)/(n_class1 + n_class2)
    class1_sum_words = np.sum(class1_count)
    class2_sum_words = np.sum(class2_count)
    
    print "Probability of Class 1 occuring in training data:", p_class1
    print "Probability of Class 2 occuring in training data:", p_class2
    
    n = 0.0
    accuracy_score = 0.0
    time.sleep(1.0)
    
    for index, y in tqdm(enumerate(y_test)):
        if(class1 in y or class2 in y):
            p_a_class1 = 1.00 * 10**308
            p_a_class2 = 1.00 * 10**308
            for i in range(2000):
                p_a_class1 = p_a_class1 * (((class1_count[i] + 1.00)/(class1_sum_words + 2000))**x_test.iloc[index,i])
                p_a_class2 = p_a_class2 * (((class2_count[i] + 1.00)/(class2_sum_words + 2000))**x_test.iloc[index,i])
            if(((p_a_class1*p_class1) > (p_a_class2*p_class2)) and class1 in y):
                accuracy_score += 1
            elif((p_a_class2*p_class2) > (p_a_class1*p_class1) and class2 in y):
                accuracy_score += 1
            n = n+1
            
    time.sleep(1.0)
        
    accuracy = float(accuracy_score)/n
    print "Accuracy of Prediciton:", accuracy*100, '% \n'
    return

print "Class VS Class (Pairwise binary) classification \n"

two_class_prediction('alt.atheism', 'soc.religion.christian')
two_class_prediction('talk.politics.guns', 'comp.sys.mac.hardware')
two_class_prediction('rec.autos', 'sci.space')
two_class_prediction('rec.motorcycles', 'rec.autos')
two_class_prediction('sci.med', 'rec.sport.baseball')

def three_class_prediction(class1, class2, class3):
    print "Class 1:", class1
    print "Class 2:", class2
    print "Class 3:", class3
    n_class1 = 0
    n_class2 = 0
    n_class3 = 0
    class1_count = 0
    class2_count = 0
    class3_count = 0
    for k,y in enumerate(y_train):
        if(class1 in y ):
            class1_count += x_train.iloc[k]
            n_class1 += 1
        elif(class2 in y):
            class2_count += x_train.iloc[k]
            n_class2 += 1
        elif(class3 in y):
            class3_count += x_train.iloc[k]
            n_class3 += 1
            
    p_class1 = float(n_class1)/(n_class1 + n_class2 + n_class3)
    p_class2 = float(n_class2)/(n_class1 + n_class2 + n_class3)
    p_class3 = float(n_class3)/(n_class1 + n_class2 + n_class3)
    class1_sum_words = np.sum(class1_count)
    class2_sum_words = np.sum(class2_count)
    class3_sum_words = np.sum(class3_count)
    
    print "Probability of Class 1 occuring in training data:", p_class1
    print "Probability of Class 2 occuring in training data:", p_class2
    print "Probability of Class 3 occuring in training data:", p_class3
    
    n = 0.0
    accuracy_score = 0.0
    time.sleep(1.0)
    
    for index, y in tqdm(enumerate(y_test)):
        if(class1 in y or class2 in y or class3 in y):
            p_a_class1 = 1.00 * 10**308
            p_a_class2 = 1.00 * 10**308
            p_a_class3 = 1.00 * 10**308
            for i in range(2000):
                p_a_class1 = p_a_class1 * (((class1_count[i] + 1.00)/(class1_sum_words + 2000))**x_test.iloc[index,i])
                p_a_class2 = p_a_class2 * (((class2_count[i] + 1.00)/(class2_sum_words + 2000))**x_test.iloc[index,i])
                p_a_class3 = p_a_class3 * (((class3_count[i] + 1.00)/(class3_sum_words + 2000))**x_test.iloc[index,i])
            if(((p_a_class1*p_class1) > (p_a_class2*p_class2)) and ((p_a_class1*p_class1) > (p_a_class3*p_class3)) 
               and class1 in y):
                accuracy_score += 1
            elif(((p_a_class2*p_class2) > (p_a_class1*p_class1)) and ((p_a_class2*p_class2) > (p_a_class3*p_class3)) 
                 and class2 in y):
                accuracy_score += 1
            elif(((p_a_class3*p_class3) > (p_a_class1*p_class1)) and ((p_a_class3*p_class3) > (p_a_class2*p_class2)) 
                 and class3 in y):
                accuracy_score += 1
            n = n+1
    
    time.sleep(1.0)
    accuracy = float(accuracy_score)/n
    print "Accuracy of Prediciton:", accuracy*100, '% \n'
    return

print "Tri Class Classification: \n"

three_class_prediction('comp.graphics', 'misc.forsale', 'sci.crypt')
three_class_prediction('talk.politics.mideast', 'talk.politics.misc', 'talk.politics.guns')
three_class_prediction('rec.sport.baseball', 'alt.atheism', 'rec.autos')
three_class_prediction('rec.sport.baseball', 'sci.space', 'rec.sport.hockey')
three_class_prediction('sci.med', 'comp.sys.mac.hardware', 'talk.politics.guns')

