import streamlit as st
import joblib
import numpy as np

# Carregar o modelo RandomForest treinado
model = joblib.load('modelo_randomforest.pkl')

# Função para fazer previsões com o modelo
def fazer_previsao(preco_passado):
    # Converter a entrada em um formato compatível com o modelo
    input_data = np.array([[preco_passado]])
    # Fazer a previsão usando o modelo carregado
    predicao = model.predict(input_data)
    return predicao[0]


st.title("Previsão de Preço do Petróleo - RandomForest")

st.write("""
    Este aplicativo prevê o preço do petróleo no dia seguinte com base no preço do dia anterior.
    Insira o preço do dia anterior para obter a previsão do preço.
""")

# Entrada de dados para previsão
st.subheader("Insira o Preço do Dia Anterior")

# Campo para entrada do preço anterior
preco_passado = st.number_input("Preço do Petróleo no Dia Anterior", min_value=0.0, value=7500.0)

# Quando o usuário clicar no botão de previsão
if st.button("Prever Preço"):
    previsao = fazer_previsao(preco_passado)
    
    st.subheader("Resultado da Previsão")
    st.write(f"O preço do petróleo previsto para o próximo dia é: R${previsao:.2f}")

    # Exibição do gráfico
    st.subheader("Gráfico da Previsão")
    st.line_chart([preco_passado, previsao])
    
