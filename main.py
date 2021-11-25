import pandas as pd
import streamlit as st
import investpy
from datetime import datetime

st.title('AI & Finanzas')
"""
# Consultar Stock en un intervalo de tiempo
"""
indice = 'Dow Jones Industrial Average'
pais = 'united states'
stock_name = '3M'
stock_symbol = 'MMM'
fecha_inicio = '01/11/2021'
fecha_fin = '02/11/2021'

indices_principales = pd.read_csv('Major World Market Indices.csv')
df_indices_principales = pd.DataFrame(indices_principales)
indice = st.selectbox('Elige un Índice', list(df_indices_principales.loc[:, 'Index']))
for i in df_indices_principales.index:
    if df_indices_principales['Index'][i] == indice:
        pais = df_indices_principales['Country'][i]

        aux = indice.find('/')
        if aux != -1:
            indice = indice.replace('/', '_')
        archivo = indice + '.csv'
        arch_indice = pd.read_csv(archivo)
        df_arch_indice = pd.DataFrame(arch_indice)
        stock_name = st.selectbox('Elige un Stock', list(df_arch_indice.loc[:, 'Name']))

        stocks = pd.read_csv('stock_symbol_country.csv')
        df_stocks = pd.DataFrame(stocks)
        for i in df_stocks.index:
            if df_stocks['name'][i] == stock_name:
                stock_symbol = df_stocks['symbol'][i]


fecha_inicio = st.date_input("Fecha inicial", datetime.now())
fecha_fin = st.date_input("Fecha final", datetime.now())

fecha_inicio = fecha_inicio.strftime('%d/%m/%Y')
fecha_fin = fecha_fin.strftime('%d/%m/%Y')

if st.button('Consultar'):
    busca6 = investpy.get_stock_historical_data(stock=stock_symbol,
                                                country=pais,
                                                from_date=fecha_inicio,
                                                to_date=fecha_fin)
    df = pd.DataFrame(busca6)
    del (df['Currency'])
    st.write(df)
    st.line_chart(df)

    aux_inicio = fecha_inicio.replace('/', '_')
    aux_fin = fecha_fin.replace('/', '_')
    nombre_consulta = stock_symbol + '_' + aux_inicio + '_' + aux_fin + '.csv'

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
    st.download_button(label="Descargar en CSV",
                       data=csv,
                       file_name=nombre_consulta,
                       mime='text/csv')
