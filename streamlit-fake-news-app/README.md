# Detector de Fake News Dinâmico com Streamlit

Este projeto é uma aplicação web interativa construída com Streamlit que permite aos usuários fazer o upload de seus próprios conjuntos de dados (em formato CSV) para treinar um modelo de Machine Learning em tempo real. Uma vez treinado, o modelo pode classificar novos títulos de notícias como verdadeiros ou falsos.

A aplicação foi projetada para ser flexível e fácil de usar, focando em uma experiência de usuário limpa e informativa.

## Estrutura do Projeto

A estrutura foi organizada para separar o código da aplicação dos dados e outros arquivos.

streamlit-fake-news-app/
├── src/
│   ├── app.py              # Ponto de entrada da aplicação Streamlit
│   ├── noticias.csv        # Exemplo de conjunto de dados
│   └── utils.py            # (Opcional) Funções auxiliares
├── requirements.txt        # Dependências do projeto
└── README.md               # Esta documentação

## Instalação

1.  Clone este repositório para a sua máquina local.
2.  Navegue até a pasta do projeto e instale as dependências necessárias:

Ou utilize o comando: pip install -r requirements.txt

## Como Executar 

Para iniciar a aplicação Streamlit, certifique-se de que está no diretório raiz do projeto e execute o seguinte comando no seu terminal:

streamlit run nltk-modelagem/streamlit-fake-news-app/src/app.py

A aplicação será aberta automaticamente no seu navegador padrão.

## Funcionalidades Principais

Upload de Dados Interativo: Faça o upload do seu próprio arquivo CSV através da barra lateral. O arquivo deve conter as colunas title (o texto da notícia) e real (um rótulo binário, onde 1 é real e 0 é falso).

Treinamento em Tempo Real: Um modelo de Regressão Logística é treinado dinamicamente com os dados que você forneceu.

Classificação Instantânea: Insira um novo título de notícia e receba uma classificação imediata de "Real" ou "Fake".

Visualização de Dados: Uma pré-visualização do seu conjunto de dados é exibida na barra lateral após o upload.

## Métricas de Desempenho Detalhadas:

Acurácia: Veja a acurácia geral do modelo treinado.

Relatório de Classificação: Analise as métricas de precisão, recall e f1-score para cada classe.

Matriz de Confusão Visual: Uma matriz de confusão gráfica para uma interpretação fácil dos acertos e erros do modelo.

## Contribuição

Contribuições são sempre bem-vindas! Se você tiver ideias para novas funcionalidades ou encontrar algum problema, sinta-se à vontade para abrir uma issue ou enviar um pull request.