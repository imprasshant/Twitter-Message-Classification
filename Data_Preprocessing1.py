import xlwt
import re
import string
import random
from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

wb = open_workbook('TwitterData_Combined_11April2017_Numbered_Topics.xlsx')
name_of_sheet=wb.sheet_names()
stop_words = set(stopwords.words("english"))  # Load stopwords
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("Final")
count=0
GlobalCount=0
for sheet1 in wb.sheets():
    number_of_rows = sheet1.nrows
    number_of_columns = sheet1.ncols

    current_sheet=wb.sheet_by_index(count)
    for row in range(0, current_sheet.nrows):
        rowstring = []
        new_rowstring=[]
        tokens = (current_sheet.cell(row, 0).value).split()

        tokens = filter(lambda x: x not in string.punctuation, tokens)   # Remove punctuation
        cleaned_text = filter(lambda x: x not in stop_words, tokens)     # Remove stop words

        for tok in cleaned_text:
            if '#' not in tok and '@' not in tok and 'http' not in tok:  #  Remove words with '@','#' and 'http' symbol
                rowstring.append(tok)

        cleaned_withascii_rowstring = " ".join(rowstring)
        cleaned_rowstring=''.join([x for x in cleaned_withascii_rowstring if ord(x) < 128])


        letters_only = re.sub("[^a-zA-Z]", " ", cleaned_rowstring)      # Letter Only
        re_tokens =letters_only.lower().split()                         # Make all the words in lower case
        re_cleaned_rowstring = " ".join(re_tokens)

        sheet.write(GlobalCount, 0, re_cleaned_rowstring)
        sheet.write(GlobalCount, 1, int(current_sheet.cell(row, 1).value))
        GlobalCount=GlobalCount+1
    count=count+1

workbook.save("Preprocessed.xls")

lamatize=WordNetLemmatizer()

book_to_stem=open_workbook("Preprocessed.xls")
workbook_stemmed = xlwt.Workbook()
sheet_to_stem = workbook_stemmed.add_sheet("Final_stemmed")
first_sht=book_to_stem.sheet_by_index(0)
row_count=first_sht.nrows

print row_count

for i in range(row_count):
    stemmed_tokens=[]
    new_tokens = (first_sht.cell(i, 0).value).split()
    for eachitem in new_tokens:
        stemmed_tokens.append(lamatize.lemmatize(eachitem))          # Lemmatization
    stemmed_row=" ".join(stemmed_tokens)
    sheet_to_stem.write(i,0,stemmed_row)
    sheet_to_stem.write(i, 1, int(first_sht.cell(i, 1).value))

workbook_stemmed.save("Preprocessed_stemmed.xls")

book=open_workbook("Preprocessed_stemmed.xls")
random_book = xlwt.Workbook()
rand_sheet = random_book.add_sheet("Random_data")
first_sheet = book.sheet_by_index(0)

row_num=first_sheet.nrows

rand_sample=random.sample(range(0,row_num), row_num)

for i in range(row_num+1):
    if i==0:
        rand_sheet.write(i, 0,'tweet')
        rand_sheet.write(i, 1,'topic')
    else:
        rand_sheet.write(i, 0, first_sheet.cell(rand_sample[i-1], 0).value)
        rand_sheet.write(i, 1, int(first_sheet.cell(rand_sample[i-1], 1).value))

random_book.save("Randombook.xls")






