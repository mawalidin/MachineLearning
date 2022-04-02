import numpy as np
import streamlit as st
import pandas as pd
import itertools
import pickle
import re
import string
import itertools
import collections
from collections import OrderedDict, Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


st.title("Analisis Sentimen Opini Pelanggan Berbasis Aspek Aplikasi HaloDoc Menggunakan Metode Extra Tree Classifier")
st.subheader("Muhammad Walidin - UNIKOM")
st.write('\n\n')

#data label untuk testing klasifikasi
path_le_acbsa = 'dataset/pred_cat'
path_le_sentiment = 'dataset/pred_polarity'

#Loading Lable encoder for aspect based sentiemnt analysis
with open(path_le_acbsa, 'rb') as f:               
    label_encoder_acbsa = pickle.load(f)

#Loading Lable encoder for sentiemnt analysis
with open(path_le_sentiment, 'rb') as f:               
    label_encoder_sentiment = pickle.load(f) 

#fungsi membaca file csv
@st.cache
def readcsv(csv):
    df = pd.read_csv(csv)
    return df

#fungsi menampilkan dataframe
def head(dataframe):
    if len(dataframe) > 1000:
        lenght = 1000
    else:
        lenght = len(dataframe)
    slider = st.slider('jumlah baris yang ditampilkan', 5, lenght)
    st.dataframe(dataframe.head(slider))

#fungsi menampilkan dataframe
def show_data(dataframe):
    st.dataframe(dataframe.head())

#fungsi menghapus kolom dari dataframe
def drop(dataframe, select):
    if len(select) != 0:
        return dataframe.drop(select, 1)
    else:
        return dataframe

#import kamus data untuk stopword, slang, emotikon
stop_words = np.array(pd.read_csv("dataset/external/stopwords_ID.txt",
                        sep="\n", header=None).values)
neg_words = np.array(pd.read_csv("dataset/external/negative_keyword_ID.txt",
                        sep="\n", header=None).values)
pos_words = np.array(pd.read_csv("dataset/external/positive_keyword_ID.txt",
                        sep="\n", header=None).values)
slang_words = pd.read_csv("dataset/external/kbba_ID.txt",
                        sep="\t", header=None)
root_words = np.array(pd.read_csv("dataset/external/rootword_ID.txt",
                        sep="\n", header=None).values)
slang = pd.read_csv("dataset/external/slangword_ID.txt",
                        sep=":", header=None)
emoticon = pd.read_csv("dataset/external/emoticon.txt",
                        sep="\t", header=None)
booster_words = np.array(pd.read_csv("dataset/external/boosterword_ID.txt",
                        sep="\n", header=None).values)
baku_words = pd.read_csv("dataset/external/katabaku_ID.txt",
                        sep="|", header=None)
baku_words.columns = [1,0]

#variabel untuk fungsi praproses dari kamus data
slang_words = pd.concat([slang_words, slang, baku_words])
sentiment_words = np.concatenate((pos_words, neg_words, booster_words))
slang_words.drop_duplicates(inplace=True)
emoticon.drop_duplicates(inplace=True)
emoticon = dict(zip(emoticon[0], emoticon[1]))
slang_words = dict(zip(slang_words[0],slang_words[1]))
neg_words = np.unique(neg_words)
pos_words = np.unique(pos_words)
stop_words = np.unique(stop_words)
stop_words = [word for word in stop_words if word not in sentiment_words]

#fungsi praproses filtering - translasi kata mengulang(contoh; sama2)
def translate_repeating_words(review):
    repeating_words = re.findall(r'\w*(?:2|")',review)
    for word in repeating_words:
        cleaned = word[:-1]
        review = re.sub(word,cleaned + " " + cleaned, review)
    return review

#fungsi praproses convert emoticon
def translate_emoticon(t):
    for w, v in emoticon.items():
        pattern = re.compile(re.escape(w))
        match = re.search(pattern,t)
        if match:
            t = re.sub(pattern," " + v + " ",t)
    return t

#fungsi praproses filtering - translasi simbol non-alpha numerik
def translate_non_alpha_num(t):
    non_alpha_num = {
        '%' : 'persen',
        '$' : 'dolar',
        '@' : 'di',
        '&' : 'dan',
        '/' : 'atau',
        '+' : 'plus'
    }
    for w, v in non_alpha_num.items():
        pattern = re.compile(re.escape(w))
        match = re.search(pattern,t)
        if match:
            t = re.sub(pattern,v + " ",t)
    return t

#fungsi praproses filtering - hapus yang bukan non-alpha numerik
def remove_non_alphanumeric(review):
    return re.sub("[^a-zA-Z\d]"," ", review)

#fungsi praproses filtering - hapus whitespace
def normalizing_words(review):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(review))

#fungsi praproses filtering - hapus numerik
def remove_numeric(review):
    return re.sub("\d", " ", review)

#fungsi praproses convert slang
def mapping_slang_words(review):
    return [slang_words[word] if word in slang_words else word for word in review]

#fungsi praproses stopword removal
def remove_stop_words(word_list):
    return [word for word in word_list if word not in stop_words]

#fungsi mengubah bentuk dataframe dari string ke list
def csv_string_to_list(csv_string):
    return csv_string[1:-1].split(",")

#fungsi menghapus bentuk datafrang string tanpa tanda kutip
def string_without_quotes(word_list):
    new = []
    for word in word_list:
        new.append(word.replace("'",""))
    return new

#fungsi mengubah bentuk dataframe dari list ke string
def convert_list_to_string(word_list):
    return ",".join(word_list)

#fungsi praproses filtering - hapus karakter yang tersendiri
def remove_single_alphabet_only(review):
    return [word for word in review if word not in string.ascii_lowercase]

#fungsi praproses filtering - hapus kata terlalu kecil
def remove_too_short_words(review):
    return [word for word in review if len(word) > 2]

#fungsi praproses filtering - hapus kata mengulang dan ganti ke karakter yang sendiri
def hapus_katadouble(review):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", review)

#kamus data convert negation
negationWord = ['ga', 'ngga', 'tidak', 
               'bkn', 'tida', 'tak', 
               'jangan', 'enggan', 'gak']

#praproses convert negation
def convert_negation(text):
    words = text.split()

    for index, word in enumerate(words):
        for negation in range(len(negationWord)):
            if words[index] == negationWord[negation]:
                nxt = index + 1 
                words[index] = negationWord[negation] + words[nxt]
                words.pop(nxt)

    return ' '.join(words)

def main():
    st.markdown('Upload file csv, gunakan sidebar untuk menghapus seleksi fitur.')
    file = st.file_uploader('Upload csv file anda', type='csv')
    if file is not None:
        df0 = pd.DataFrame(readcsv(file))
        st.sidebar.subheader('Hapus kolom:')
        sidedrop = st.sidebar.multiselect(
            'Kolom yang akan dihapus: ', tuple(df0.columns))
        df = drop(df0, sidedrop)

        st.header('Visualisasi Dataframe')
        head(df)
        # st.header('Jumlah Kolom')
        # valuecounts(df)
        df = df.rename(columns={'Review':'review','Aspect Category':'aspect_category','Polarity':'polarity'})
        st.header('Preprocessing')
        st.text('Klik sesuai tahap')
        df['review'] = df['review'].apply(translate_repeating_words)
        df['review'] = df['review'].apply(translate_emoticon)
        casefolding = st.checkbox('Casefolding')
        if casefolding:
            df['review'] = df['review'].apply(lambda x: x.lower())
            df['aspect_category'] = df['aspect_category'].apply(lambda x: x.lower())
            df['polarity'] = df['polarity'].apply(lambda x: x.lower())
            show_data(df['review'])
        filtering = st.checkbox('Filtering')
        if filtering:
            df['review'] = df['review'].apply(translate_non_alpha_num)
            df['review'] = df['review'].apply(remove_non_alphanumeric)
            df['review'] = df['review'].apply(hapus_katadouble)
            df['review'] = df['review'].apply(normalizing_words)
            df['review'] = df['review'].apply(remove_numeric)
            show_data(df['review'])
        emoticon = st.checkbox('Convert Emoticon')
        if emoticon:
            df['review'] = df['review'].apply(translate_emoticon)
            show_data(df['review'])
        tokenizing = st.checkbox('Tokenizing')
        if tokenizing:
            df['review'] = df['review'].apply(lambda x: x.split())
            df['review'] = df['review'].apply(lambda x: np.array(x))
            show_data(df['review'])
        slang = st.checkbox('Convert Slang')
        if slang:
            df['review'] = df['review'].apply(mapping_slang_words)
            show_data(df['review'])
        stopword = st.checkbox('Stopword Removal')
        if stopword:
            df['review'] = df['review'].apply(remove_stop_words)
            df.to_csv('dataset/data_preprocessed_1.csv', index=False)
            show_data(df['review'])
        negation = st.checkbox('Convert Negation')
        if negation:
            df = pd.read_csv( "dataset/data_preprocessed_1.csv")
            df['review'] = df['review'].apply(remove_numeric)
            df['review'] = df['review'].apply(csv_string_to_list)
            df['review'] = df['review'].apply(mapping_slang_words)
            df['review'] = df['review'].apply(remove_single_alphabet_only)
            df['review'] = df['review'].apply(remove_too_short_words)
            df = df.dropna(subset=['review'],how='all')
            df = df[df['review'].map(len) > 0]
            df['review'] = df['review'].apply(convert_list_to_string)
            df = df.dropna(subset=['review'],how='all')
            df = df[df['review'].map(len) > 0]
            df['review'] = df['review'].apply(lambda x: x.replace(","," "))
            df['review'] = df['review'].apply(lambda x: x.split())
            df['review'] = df['review'].apply(mapping_slang_words)
            df['review'] = df['review'].apply(convert_list_to_string)
            df = df.dropna(subset=['review'],how='all')
            df = df[df['review'].map(len) > 0]
            df['review'] = df['review'].astype(str)
            df['review'] = df['review'].apply(lambda x: x.strip())
            df['review'] = df['review'].apply(lambda x: x.replace(","," "))
            df['review'] = df['review'].apply(convert_negation)
            show_data(df['review'])
        stemming = st.checkbox('Stemming')
        if stemming:
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df['review'] = df['review'].apply(lambda x: stemmer.stem(x).replace(" ",","))
            df['review'] = df['review'].apply(lambda x: x.replace("'",""))
            df['review'] = df['review'].apply(lambda x: x.replace(","," "))
            df['review'] = df['review'].apply(lambda x: x.split())
            df['review'] = df['review'].apply(mapping_slang_words)
            df['review'] = df['review'].apply(remove_stop_words)
            df['review'] = df['review'].apply(convert_list_to_string)
            df['review'] = df['review'].apply(lambda x: x.replace("'",""))
            df['review'] = df['review'].apply(lambda x: x.replace(","," "))
            df['review'] = df['review'].apply(hapus_katadouble)
            
            show_data(df['review'])
        
        st.sidebar.header('Klasifikasi data')
        predict = st.sidebar.button('Proses')
        if predict:
            st.header('Hasil:')
            predicted_cat = label_encoder_acbsa     
            predicted_polarity =label_encoder_sentiment
            df['aspek_sentimen'] = df['aspect_category'] +" "+ df['polarity']
            catagories = df['aspek_sentimen'].unique()
            X = df['review']
            y = df['aspek_sentimen']
            vectorizer = TfidfVectorizer(analyzer = "word",
                             tokenizer = None, 
                             preprocessor = None,
                             stop_words = None, 
                             max_features = 5000)
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
            train_data_features = vectorizer.fit_transform(X_train)
            train_data_features = train_data_features.toarray()
            test_data_features = vectorizer.transform(X_test)
            test_data_features = test_data_features.toarray()
            # Inisialisasi ETC
            etc = ExtraTreesClassifier(criterion='gini',class_weight='balanced_subsample', max_features=10)
            # Memasang data latih ke classifier
            etc.fit(train_data_features,y_train)
            # Melakukan prediksi dengan membandingkan dengan data tes
            y_pred_train = etc.predict(train_data_features)
            y_pred_cat = etc.predict(test_data_features)
            # Melakukan proses pengujian
            result2 = classification_report(y_test, y_pred_cat ,target_names=catagories)
            st.text(result2)
        

if __name__ == '__main__':
    main()