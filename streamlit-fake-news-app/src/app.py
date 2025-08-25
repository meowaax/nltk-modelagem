import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.classify import NaiveBayesClassifier, accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline # <-- Importação do Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # Stemmer para Inglês
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Fake News (Inglês)",
    page_icon="📰",
    layout="wide",
)

# --- FUNÇÕES AUXILIARES ---

@st.cache_resource
def download_nltk_resources():
    # Baixa apenas os recursos necessários para o inglês
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):
        text = '' if text is None else str(text)

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'<.*?>', ' ', text)              # HTML
    tokens = word_tokenize(text, preserve_line= True)

    # limpa tokens: tira pontuação e stopwords, mantém só letras com len>2
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def tokens_to_features(tokens):
    features = {}
    for t in tokens:
        features[t] = features.get(t, 0) + 1
    return features

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    ax.set_xlabel('Rótulo Previsto', fontsize=12)
    ax.set_ylabel('Rótulo Verdadeiro', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig, use_container_width=True)

# --- FUNÇÃO DE TREINAMENTO EM CACHE ---
@st.cache_data
def treinar_modelo_com_dados(df):
    if 'title' in df.columns and 'real' in df.columns:
        dataset = [(tokens_to_features(preprocess(text)), int(label)) for text, label in zip(df['title'], df['real'])]
        
        train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
        classifier = NaiveBayesClassifier.train(train_set)
        
        y_true = [label for (_, label) in test_set]
        y_pred = [classifier.classify(feats) for (feats, _) in test_set]
        
        # Retorna o pipeline inteiro e os dados de teste
        return classifier, y_true, y_pred, test_set
    else:
        st.error("O arquivo CSV carregado não contém as colunas necessárias ('title', 'real').")
        return None

# --- LAYOUT DA APLICAÇÃO ---

st.title("📰 Detector de Fake News (Inglês)")
st.markdown("Faça o upload de um arquivo CSV com notícias em **inglês** para treinar o modelo.")

# --- BARRA LATERAL PARA UPLOAD ---
with st.sidebar:
    st.header("1. Carregue seus Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type="csv",
        help="O arquivo CSV deve conter as colunas 'title' (título da notícia em inglês) e 'real' (1 para real, 0 para fake)."
    )

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    try:
        try:
            dataframe = pd.read_csv(uploaded_file, sep=';')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            dataframe = pd.read_csv(uploaded_file, sep=';', encoding='latin1')

        st.sidebar.success("Arquivo carregado com sucesso!")
        st.sidebar.write("Amostra dos dados:")
        st.sidebar.dataframe(dataframe)

        # A função agora retorna o pipeline treinado
        classifier, y_true, y_pred, test_set = treinar_modelo_com_dados(dataframe)

        if classifier:
            main_col1, main_col2 = st.columns([2, 3])

            with main_col1:
                st.subheader("2. Classifique uma nova notícia")
                nova_noticia = st.text_area(
                    "Digite o título da notícia (em inglês):", 
                    height=150, 
                    placeholder="Ex: 'Scientists discover the cure for...'"
                )
                
                if st.button("Analisar Notícia", type="primary", use_container_width=True):
                    if nova_noticia:
                        texto_proc = preprocess(nova_noticia)
                        features = tokens_to_features(texto_proc)
                        # --- CÓDIGO SIMPLIFICADO ---
                        # O pipeline lida com a vetorização e a predição em um único passo.
                        predicao = classifier.classify(features)
                        
                        if str(predicao) == "1":
                            st.success("✅ A notícia parece ser: **Real**")
                        else:
                            st.error("❌ A notícia parece ser: **Fake**")
                    else:
                        st.warning("Por favor, insira um título de notícia para analisar.")

            with main_col2:
                st.subheader("3. Performance do Modelo")
                
                st.metric(label="Acurácia Geral do Modelo", value=f"{(accuracy(classifier, test_set)):.2%}")
                
                with st.expander("🔍 Ver detalhes da performance"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Relatório de Classificação:**")
                        report = classification_report(y_true, y_pred, target_names=['Falsa', 'Real'], output_dict=True)
                        df_report = pd.DataFrame(report).transpose()
                        st.dataframe(df_report.round(2))
                    with col2:
                        st.write("**Matriz de Confusão:**")
                        cm = confusion_matrix(y_true, y_pred)
                        plot_confusion_matrix(cm, classes=['Falsa', 'Real'])
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.info("Aguardando o upload de um arquivo CSV (em inglês) para começar.")