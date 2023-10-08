import pandas as pd
import math

data = pd.read_csv("emails.csv", nrows=4500)
data = data.loc[:, data.columns != 'Email No.']
list_of_columns = list(data.columns)
vocabulary = list_of_columns[:-1]
print(f"\nvocabulary = {vocabulary}")
print(f"\ndf = {data.to_string}")
spam_emails = data[data['Prediction'] == 1]
ham_emails = data[data['Prediction'] == 0]
print(f"\nspam emails = {spam_emails.to_string}")
print(f"\nham emails = {ham_emails.to_string}")

# prior probability = length of spam emails / total emails
spam_prior = len(spam_emails) / len(data)
ham_prior = len(ham_emails) / len(data)
print(f"\nprior probability of spam = {spam_prior}")
print(f"\nprior probability of ham = {ham_prior}")

nspam = 0  # total count of words in spam emails
nham = 0  # total count of words in ham emails
beta = 0.000001  # for smoothing

# counting total number of words for each class
for word in vocabulary:
    wcspam = spam_emails[word].sum()  # word count for spam emails
    wcham = ham_emails[word].sum()  # word count for normal emails
    nspam = nspam + wcspam
    nham = nham + wcham

nvocabulary = len(vocabulary)  # number of words in vocabulary

spam_parameters = dict()
ham_parameters = dict()
for word in vocabulary:
    spam_parameters[word] = 0
    ham_parameters[word] = 0

for word in vocabulary:
    wcspam = spam_emails[word].sum()  # word count for spam emails
    pspam = ((wcspam + beta)/(nspam + (beta * nvocabulary)))
    spam_parameters[word] = math.log(pspam)
    wcham = ham_emails[word].sum()  # word count for ham emails
    pham = ((wcham + beta) / (nham + (beta * nvocabulary)))
    ham_parameters[word] = math.log(pham)

print(f"\nspam features = {spam_parameters}")
print(f"\nham features = {ham_parameters}")

# Test

test_data = pd.read_csv("emails.csv", skiprows=range(1, 4501))
test_data = test_data.loc[:, test_data.columns != 'Email No.']
list_of_test_columns = list(test_data.columns)
test_vocabulary = list_of_test_columns[:-1]
print(f"\ntest data vocabulary = {test_vocabulary}")
print(f"\ntest df = {test_data.to_string}")
spam_emails_test = test_data[test_data['Prediction'] == 1]
ham_emails_test = test_data[test_data['Prediction'] == 0]

spam_prob = spam_prior
ham_prob = ham_prior
s = 0
n = 0
e = 0

for row in range(test_data.shape[0]):
    print(f"Email - {row}:")
    spam_prob = math.log(spam_prior)
    ham_prob = math.log(ham_prior)
    for word in test_vocabulary:
        if test_data.iloc[row][word] != 0:
            if word in spam_parameters:
                # for i in range(test_data.iloc[row][word]):
                i = int(test_data.iloc[row][word])
                spam_prob = spam_prob + (i * (spam_parameters[word]))
            if word in ham_parameters:
                # for i in range(test_data.iloc[row][word]):
                i = test_data.iloc[row][word]
                ham_prob = ham_prob + (i * (ham_parameters[word]))

    print(f"P(spam | email) = {spam_prob}")
    print(f"P(ham | email) = {ham_prob}")
    if spam_prob > ham_prob:
        print(f"Spam email")
        if test_data.iloc[row]['Prediction'] == 1:
            s = s + 1
    elif ham_prob > spam_prob:
        print(f"Normal email")
        if test_data.iloc[row]['Prediction'] == 0:
            n = n + 1
    else:
        print(f"Error")
        e = e + 1
    print("-------------------------")

print(f"Number of spam emails predicted = {s}")
print(f"Number of normal emails predicted = {n}")
print(f"Number of unpredicted emails = {e}")
print(f"Actual number of spam emails = {spam_emails_test.shape[0]}")
print(f"Actual number of normal emails = {ham_emails_test.shape[0]}")
print(f"Accuracy of spam = {s/spam_emails_test.shape[0]}")
print(f"Accuracy of normal = {n/ham_emails_test.shape[0]}")
print(f"Accuracy = {(s+n)/(spam_emails_test.shape[0]+ham_emails_test.shape[0])}")



