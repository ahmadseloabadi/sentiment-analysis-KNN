#import library
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier



# Set page layout and title
st.set_page_config(page_title="TIX ID", page_icon="style/icon.png")

# Add custom CSS
def add_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

add_css("style/style.css")


#import dataset

ulasan = pd.read_csv('data/data_bersih.csv')

# text preprosessing
def cleansing(kalimat_baru): 
    kalimat_baru = re.sub(r'@[A-Za-a0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r'#[A-Za-z0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"http\S+",' ',kalimat_baru)
    kalimat_baru = re.sub(r'[0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", kalimat_baru)
    kalimat_baru = re.sub(r"\b[a-zA-Z]\b", " ", kalimat_baru)
    kalimat_baru = kalimat_baru.strip(' ')
    # menghilangkan emoji
    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    kalimat_baru =clearEmoji(kalimat_baru)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    kalimat_baru=replaceTOM(kalimat_baru)
    return kalimat_baru
def casefolding(kalimat_baru):
    kalimat_baru = kalimat_baru.lower()
    return kalimat_baru
def tokenizing(kalimat_baru):
    kalimat_baru = word_tokenize(kalimat_baru)
    return kalimat_baru
def slangword (kalimat_baru):
    kamusSlang = eval(open("data/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in kalimat_baru:
        filter_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata.lower())
        if filter_slang.startswith('tidak_'):
          kata_depan = 'tidak_'
          kata_belakang = kata[6:]
          kata_belakang_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata_belakang.lower())
          kata_hasil = kata_depan + kata_belakang_slang
          content.append(kata_hasil)
        else:
          content.append(filter_slang)
    kalimat_baru = content
    return kalimat_baru

def handle_negation(kalimat_baru):
    negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak",'ga']
    new_words = []
    prev_word_is_negation = False
    for word in kalimat_baru:
        if word in negation_words:
            new_words.append("tidak_")
            prev_word_is_negation = True
        elif prev_word_is_negation:
            new_words[-1] += word
            prev_word_is_negation = False

        else:
            new_words.append(word)
    kalimat_baru=new_words
    return kalimat_baru

def stopword (kalimat_baru):
    daftar_stopword = stopwords.words('indonesian')
    daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g','anjing','ga','gua','nder']) 
    # Membaca file teks stopword menggunakan pandas
    txt_stopword = pd.read_csv("data/stopwords.txt", names=["stopwords"], header=None)

    # Menggabungkan daftar stopword dari NLTK dengan daftar stopword dari file teks
    daftar_stopword.extend(txt_stopword['stopwords'].tolist())

    # Mengubah daftar stopword menjadi set untuk pencarian yang lebih efisien
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        cleaned_words = []
        for word in words:
            # Memisahkan kata dengan tambahan "tidak_"
            if word.startswith("tidak_"):
                cleaned_words.append(word[:5])
                cleaned_words.append(word[6:])
            elif word not in daftar_stopword:
                cleaned_words.append(word)
        return cleaned_words
    kalimat_baru = stopwordText(kalimat_baru)
    return kalimat_baru 

def stemming(kalimat_baru):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Lakukan stemming pada setiap kata
    kalimat_baru = [stemmer.stem(word) for word in kalimat_baru]
    return kalimat_baru
# mengambil kolom stemming dan sentimen pada dataset
X = ulasan['Stemming']
Y = ulasan['Sentimen']
# pembagian data sebesar 80:20 sebelum smote
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# pembobotan tf_idf
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

#penerapan smote
vectoriz = TfidfVectorizer()
x_res =vectoriz.fit_transform(ulasan['Stemming'])
y_res =ulasan['Sentimen']
smote = SMOTE(sampling_strategy='auto')
X_smote, Y_smote = smote.fit_resample(x_res, y_res)
sX_train, sX_test, sY_train, sY_test = train_test_split(X_smote, Y_smote, test_size=0.20)


#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Pengolahan data','Uji','Report'])

if(selected == 'Home') :
    st.title('PENERAPAN METODE SMOTE DAN K-NEAREST NEIGHBOR PADA IMBALANCE DATA ANALISIS SENTIMEN ULASAN APLIKASI TIX ID ')
    st.write('Aplikasi TIX ID merupakan aplikasi yang memberikan layanan pemesanan tiket bioskop secara online yang dirilis oleh PT. Nusantara Elang Sejahtera')
    image = Image.open('style/TIXID_logo.png')
    st.image(image)
    st.write('data train sesmote',len(y_train))
    st.write('data test sesmote',len(y_test))
    st.write('data train smote',len(sY_train))
    st.write('data test smote',len(sY_test))


elif(selected == 'Pengolahan data') :
    tab1,tab2,tab3 =st.tabs(['Dataset','Text preprosesing','SMOTE'])
    with tab1 :
        st.title('dataset ulasan aplikasi TIX ID')
        st.text('dataset ulasan aplikasi TIX ID yang diambil dari google Playstore')
        dataset = pd.read_csv('data/datasetbaru.csv')
        def sentimen_smote(dataset, sentiment):
            return dataset[dataset['Sentimen'].isin(sentiment)]
        sentiment_map = {'positive': 'sentimen positif', 'neutral': 'sentimen netral','negative':'sentimen negatif'}
        sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
        sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
        filtered_data = sentimen_smote(dataset, sentiment)
        st.dataframe(filtered_data,use_container_width=True)
        # Hitung jumlah kelas dataset
        st.write("Jumlah kelas:  ")
        kelas_sentimen = ulasan['Sentimen'].value_counts()
        # st.write(kelas_sentimen)
        datpos, datneg, datnet = st.columns(3)
        with datpos:
            st.markdown("Positif")
            st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
        with datnet:
            st.markdown("Netral")
            st.markdown(f"<h1 style='text-align: center; color: orange;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
        with datneg:
            st.markdown("Negatif")
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
        #membuat diagram
        labels = ['negatif' , 'neutral', 'positif']
        fig1,ax1=plt.subplots()
        ax1.pie(ulasan.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
        ax1.axis('equal')
        st.pyplot(fig1)
    with tab2 :
        st.title('Text preprosesing')
        st.header('casefolding')#----------------
        st.text('mengubahan seluruh huruf menjadi kecil (lowercase) yang ada pada dokumen.')
        casefolding = pd.read_csv('data/casefolding.csv')
        st.write(casefolding)
        st.header('cleansing')#----------------
        st.text('membersihkan data dari angka ,tanda baca,dll.')
        cleansing = pd.read_csv('data/cleansing.csv')
        st.write(cleansing)
        st.header('tokenizing')#----------------
        st.text('menguraikan kalimat menjadi token-token atau kata-kata.')
        tokenizing = pd.read_csv('data/tokenizing.csv')
        st.write(tokenizing)
        st.header('stemming')#----------------
        st.text(' merubahan kata yang berimbuhan menjadi kata dasar. ')
        stemming = pd.read_csv('data/stemming.csv')
        st.write(stemming)
        st.header('negasi')#----------------
        st.text('mengatur kata tidak')
        negasi = pd.read_csv('data/negasi.csv')
        st.write(negasi)
        st.header('normalization')#----------------
        st.text('mengubah penggunaan kata tidak baku menjadi baku')
        normalization = pd.read_csv('data/normalizing.csv')
        st.write(normalization)
        st.header('stopword')#----------------
        st.text('menyeleksi kata yang tidak penting dan menghapus kata tersebut.')
        stopword = pd.read_csv('data/stopword.csv')
        st.write(stopword)

        st.title('Pembobotan TF-IDF')
        st.text('pembobotan pada penelitan ini menggunakan tf-idf')
        tfidf = pd.read_csv('data/hasil TF IDF.csv')
        st.dataframe(tfidf,use_container_width=True)
    with tab3 :
        st.title('SMOTE')
        st.text('SMOTE adalah teknik untuk mengatasi ketidak seimbangan kelas pada dataset')
        
        seb_smote,ses_smote = st.columns(2)
        with seb_smote:
            st.header('sebelum SMOTE')
            st.write('Jumlah dataset:',len(y_res))
            # Hitung jumlah kelas sebelum SMOTE
            st.write("Jumlah kelas:  ")
            Jum_sentimen = ulasan['Sentimen'].value_counts()
            spos, sneg, snet = st.columns(3)
            with spos:
                st.markdown("Positif")
                st.markdown(f"<h1 style='text-align: center; color: green;'>{Jum_sentimen[0]}</h1>", unsafe_allow_html=True)
            with snet:
                st.markdown("Netral")
                st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jum_sentimen[2]}</h1>", unsafe_allow_html=True)
            with sneg:
                st.markdown("Negatif")
                st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jum_sentimen[1]}</h1>", unsafe_allow_html=True)
            
            #membuat diagram
            labels = ['negatif' , 'neutral', 'positif']
            fig1,ax1=plt.subplots()
            ax1.pie(ulasan.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
            ax1.axis('equal')
            st.pyplot(fig1)

        with ses_smote:
            st.header('sesudah SMOTE')
            df = pd.DataFrame(X_smote)
            df.rename(columns={0:'term'}, inplace=True)
            df['sentimen'] = Y_smote
            #melihat banyak dataset
            st.write('Jumlah dataset :',len(Y_smote))
            # melihat jumlah kelas sentimen aetelah SMOTE
            st.write("Jumlah kelas: ")
            Jumlah_sentimen = df['sentimen'].value_counts()
            pos, neg, net = st.columns(3)
            with pos:
                st.markdown("Positif")
                st.markdown(f"<h1 style='text-align: center; color: green;'>{Jumlah_sentimen[0]}</h1>", unsafe_allow_html=True)
            with net:
                st.markdown("Netral")
                st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jumlah_sentimen[2]}</h1>", unsafe_allow_html=True)
            with neg:
                st.markdown("Negatif")
                st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jumlah_sentimen[1]}</h1>", unsafe_allow_html=True)
            
            # menampilkan dalam bentuk plot diagram
            labels = ['negatif' , 'neutral', 'positif']
            fig2,ax2=plt.subplots()
            plt.pie(df.groupby('sentimen')['sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
            ax2.axis('equal')
            st.pyplot(fig2)
        sintetis = pd.read_csv('data/data_sintetik.csv')
        dsmote = pd.read_csv('data/data_smote.csv')
        st.header('Data sintetis')
        st.write('jumlah data sintetis :',len(sintetis))
        st.write('jumlah penambahan setiap kelas :')
        sintetis_sentimen = sintetis['sentimen'].value_counts()
        # st.write(sintetis_sentimen.values)
        sinpos,sinneg,sinnet = st.columns(3)
        with sinpos:
            st.markdown("Positif")
            st.markdown(f"<h1 style='text-align: center; color: black;'>{Jumlah_sentimen[0]-Jum_sentimen[0]}</h1>", unsafe_allow_html=True)
        with sinnet:
            st.markdown("Netral")
            st.markdown(f"<h1 style='text-align: center; color: orange;'>{sintetis_sentimen[0]}</h1>", unsafe_allow_html=True)
        with sinneg:
            st.markdown("Negatif")
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{sintetis_sentimen[1]}</h1>", unsafe_allow_html=True)
        # menampilkan dalam bentuk plot diagram
        labels = ['negatif' , 'netral']
        fig3,ax3=plt.subplots()
        plt.pie(sintetis.groupby('sentimen')['sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
        ax3.axis('equal')
        st.pyplot(fig3)
        def sentimen_sintetis(dataset, Sentimen):
            return sintetis[sintetis['sentimen'].isin(selected_sentiment)]
        def sentimen_smote(dataset, Sentimen):
            return dsmote[dsmote['sentimen'].isin(sentiment)]
        
        optiondata = st.selectbox('pilih data',('SMOTE', 'SINTETIS'))

        if(optiondata == 'SMOTE') :
            st.write('menampilkan data sesudah',optiondata)
            sentiment_map = {'Positif': 'positif', 'Netral': 'netral','Negatif':'negatif'}
            sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
            sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
            filtered_data = sentimen_smote(dsmote, sentiment)
            st.dataframe(filtered_data,use_container_width=True)
        if(optiondata == 'SINTETIS') :
            st.write('menampilkan data ',optiondata)
            sentiment_map = {'negative': 'negatif', 'neutral': 'netral'}
            selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
            selected_sentiment = [key for key, value in sentiment_map.items() if value in selected_sentiment]
            filtered_data = sentimen_sintetis(sintetis, selected_sentiment)
            st.dataframe(filtered_data,use_container_width=True)
        with st.expander('...') :
            st.text('cari kalimat ')
            search_query = st.text_input("Masukkan kalimat yang ingin dicari:")
            search_results = dsmote[dsmote['kalimat_asli'].str.contains(search_query)]
            if st.button('cari') :
                st.write(search_results)
            # Mencari baris duplikat berdasarkan nilai kalimat asli
            duplicates = dsmote[dsmote.duplicated(subset='kalimat_asli', keep=False)]

            results = {"index": [], "term": [], "sentimen": [], "kalimat_asli": []}
            # Menampilkan kalimat yang duplikat
            for index, row in duplicates.iterrows():
                results["index"].append(index)
                results["term"].append(row['term'])
                results["sentimen"].append(row['sentimen'])
                results["kalimat_asli"].append(row['kalimat_asli'])
            st.dataframe(results)    
elif(selected == 'Uji') :
    st.title('pengujian metode SMOTE dan K-NN')
    st.header('pengujian nilai k pada metode K-NN')
    uji_knn = pd.read_csv('data/pengujiank_KNN.csv')
    st.dataframe(uji_knn,use_container_width=True)
    st.text('hasil pada pengujian dengan nilai K 3, 5, 7,9,10 diperoleh hasil bahwa akurasi')
    st.text('tertinggi terdapat pada nilai k=7')

    st.header('pengujian nilai k pada metode SMOTE-K-NN')
    uji_smoteknn = pd.read_csv('data/pengujiank_SMOTEKNN.csv')
    st.dataframe(uji_smoteknn,use_container_width=True)
    st.text('hasil pada pengujian dengan nilai K 3, 5, 7,9,10 diperoleh bahwa penerapan SMOTE')
    st.text('dapat meningkatkan performa akurasi algoritma KNN dengan rata rata peningkatan')
    st.text('akurasi sebesar 0.092% atau 9.2% dan nilai tertinggi pada k=3')

    option = st.selectbox('METODE',('KNN', 'SMOTE-KNN'))
    nilaik = st.number_input('NILAI K',min_value=1, max_value=999, value=3, step=1,)
    document = st.text_input('masukan kalimat',value="tingkatkan terus kalau bisa ada kerjasama dengan paylater")
    
    knn = KNeighborsClassifier(n_neighbors=nilaik)
    knn.fit(x_train, y_train)

    smote_knn=KNeighborsClassifier(n_neighbors=nilaik)
    smote_knn.fit(sX_train, sY_train)

    kcleansing = cleansing(document)
    kcasefolding = casefolding(kcleansing)
    ktokenizing = tokenizing(kcasefolding)
    kstemming = stemming(ktokenizing)
    knegasi= handle_negation(kstemming)
    knormword = slangword(knegasi)
    kstopwordremov = stopword(knormword)
    kdatastr = str(kstopwordremov)
    ktfidf =vectorizer.transform([kdatastr])
    ktfidfsmote =vectoriz.transform([kdatastr])

    if (option == 'KNN') :
        
        if st.button('predik') :
            st.write('Hasil pengujian dengan metode',option)
            # Making the SVM Classifer
            predictions = knn.predict(ktfidf)
            st.write('hasil cleansing :',str(kcleansing))
            st.write('hasil casefolding :',str(kcasefolding))
            st.write('hasil tokenizing :',str(ktokenizing))
            st.write('hasil stemming :',str(kstemming))
            st.write('hasil negasi :',str(knegasi))
            st.write('hasil normalization :',str(knormword))
            st.write('hasil stopword :',str(kstopwordremov))
            st.write(f"hasil prediksi menggunakan metode {option} adalah {predictions}")
        else:
            st.write('Hasil') 
    elif (option == 'SMOTE-KNN') :
        if st.button('predik') :
            st.write('Hasil pengujian dengan metode',option)
            # Making the SVM Classifer
            predictions = smote_knn.predict(ktfidfsmote)
            st.write('hasil cleansing :',str(kcleansing))
            st.write('hasil casefolding :',str(kcasefolding))
            st.write('hasil tokenizing :',str(ktokenizing))
            st.write('hasil stemming :',str(kstemming))
            st.write('hasil negasi :',str(knegasi))
            st.write('hasil normalization :',str(knormword))
            st.write('hasil stopword :',str(kstopwordremov))
            st.write(f"hasil prediksi menggunakan metode {option} adalah {predictions}")
        else:
            st.write('Hasil') 

elif(selected == 'Report') :
    tab1,tab2 =st.tabs(['klasifikasi report','confusion matrix'])

    with tab1 :
        st.header('perbandingan nilai accuracy,precision,dan recall pada pengujian nilai k')
        akurasi = Image.open('data/accuracy.png')
        st.image(akurasi)
        presision = Image.open('data/precision.png')
        st.text('Pada grafik diatas dapat disimpulkan bahwa penggunaan smote dapat meningkatkan nilai ')
        st.text('akurasi pada metode KNN,dikarenakan data yang digunakan sudah seimbang')
        st.image(presision)
        st.text('Pada grafik diatas dapat disimpulkan bahwa penggunaan smote dapat meningkatkan nilai ')
        st.text('prsision pada metode KNN,dikarenakan data yang digunakan sudah seimbang')
        recall = Image.open('data/recall.png')
        st.image(recall)
        st.text('Pada grafik diatas dapat disimpulkan bahwa penggunaan smote dapat meningkatkan nilai ')
        st.text('recall pada metode KNN,dikarenakan data yang digunakan sudah seimbang')
    with tab2 :

        # plot confusion matrix knn
        st.write('plot confusion matrix knn sebelum smote')
        sebsmote = Image.open('data/cmsebelumsmote.png')
        st.image(sebsmote)

        # plot confusion matrix smote knn
        st.write('plot confusion matrix knn sesudah smote')
        sessmote = Image.open('data/cmsesudahsmote.png')
        st.image(sessmote)




