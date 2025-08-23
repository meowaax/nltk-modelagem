import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Fake News",
    page_icon="📰",
    layout="wide",
)

# --- FUNÇÕES AUXILIARES ---

# Baixar recursos necessários do NLTK (apenas na primeira vez)
@st.cache_resource
def download_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('stemmers/rslp')
    except LookupError:
        nltk.download('rslp')

download_nltk_resources()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ ]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = RSLPStemmer()
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

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
        df['titulo_processado'] = df['title'].apply(preprocess)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['titulo_processado'])
        y = df['real']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        return model, vectorizer, X_test, y_test
    else:
        st.error("O arquivo CSV carregado não contém as colunas necessárias ('title', 'real').")
        return None, None, None, None

# --- LAYOUT DA APLICAÇÃO ---

st.title("📰 Detector de Fake News Dinâmico 📰")
st.markdown("Faça o upload de um arquivo CSV para treinar um modelo de Machine Learning e começar a classificar notícias.")

# --- BARRA LATERAL PARA UPLOAD ---
with st.sidebar:
    st.header("1. Carregue seus Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type="csv",
        help="O arquivo CSV deve conter as colunas 'title' (título da notícia) e 'real' (1 para real, 0 para fake)."
    )

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    try:
        # Tenta ler com diferentes encodings
        try:
            dataframe = pd.read_csv(uploaded_file, sep=';')
        except UnicodeDecodeError:
            uploaded_file.seek(0) # Retorna ao início do arquivo
            dataframe = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
        
        st.sidebar.success("Arquivo carregado com sucesso!")
        st.sidebar.write("Amostra dos dados:")
        # --- ALTERAÇÃO AQUI ---
        # Removido o .head() para mostrar o DataFrame completo
        st.sidebar.dataframe(dataframe)

        model, vectorizer, X_test, y_test = treinar_modelo_com_dados(dataframe)

        if model:
            main_col1, main_col2 = st.columns([2, 3]) # Divide a tela em duas colunas

            with main_col1:
                st.subheader("2. Classifique uma nova notícia")
                nova_noticia = st.text_area(
                    "Digite o título da notícia:", 
                    height=150, 
                    placeholder="Ex: 'Cientistas descobrem a cura para...'"
                )
                
                if st.button("Analisar Notícia", type="primary", use_container_width=True):
                    if nova_noticia:
                        texto_proc = preprocess(nova_noticia)
                        predicao = model.predict(vectorizer.transform([texto_proc]))
                        
                        if str(predicao[0]) == "1":
                            st.success("✅ A notícia parece ser: **Real**")
                        else:
                            st.error("❌ A notícia parece ser: **Fake**")
                    else:
                        st.warning("Por favor, insira um título de notícia para analisar.")

            with main_col2:
                st.subheader("3. Performance do Modelo")
                y_pred = model.predict(X_test)
                
                st.metric(label="Acurácia Geral do Modelo", value=f"{accuracy_score(y_test, y_pred):.2%}")
                
                with st.expander("🔍 Ver detalhes da performance"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Relatório de Classificação:**")
                        report = classification_report(y_test, y_pred, target_names=['Falsa', 'Real'], output_dict=True)
                        df_report = pd.DataFrame(report).transpose()
                        st.dataframe(df_report.round(2))
                    with col2:
                        st.write("**Matriz de Confusão:**")
                        cm = confusion_matrix(y_test, y_pred)
                        plot_confusion_matrix(cm, classes=['Falsa', 'Real'])

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

else:
    st.info("Aguardando o upload de um arquivo CSV para começar.")