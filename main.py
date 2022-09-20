import pandas as pd
import streamlit as st
import investpy
from datetime import datetime
from traceback import format_exc
import numpy as np




st.title('AI & Finanzas')
st.subheader('Consultar Stock en un intervalo de tiempo')
indice = 'Dow Jones Industrial Average'
pais = 'united states'
stock_name = '3M'
stock_symbol = 'MMM'
fecha_inicio = '01/11/2021'
fecha_fin = '02/11/2021'

indices_principales = pd.read_csv('Major World Market Indices.csv')
df_indices_principales = pd.DataFrame(indices_principales)
indice = st.sidebar.selectbox('Elige un Índice', list(df_indices_principales.loc[:, 'Index']))
for i in df_indices_principales.index:
    if df_indices_principales['Index'][i] == indice:
        try:
            pais = df_indices_principales['Country'][i]
            aux = indice.find('/')
            if aux != -1:
                indice = indice.replace('/', '_')
            archivo = indice + '.csv'
            arch_indice = pd.read_csv(archivo)
            df_arch_indice = pd.DataFrame(arch_indice)
            if len(df_arch_indice.index) > 1:
                stock_name = st.sidebar.selectbox('Elige un Stock', list(df_arch_indice.loc[:, 'Name']))
                stocks = pd.read_csv('stock_symbol_country.csv')
                df_stocks = pd.DataFrame(stocks)
                for df_i in df_stocks.index:
                    if df_stocks['name'][df_i] == stock_name:
                        stock_symbol = df_stocks['symbol'][df_i]
                fecha_inicio = st.sidebar.date_input("Fecha inicial", datetime.now())
                fecha_fin = st.sidebar.date_input("Fecha final", datetime.now())

                fecha_inicio = fecha_inicio.strftime('%d/%m/%Y')
                fecha_fin = fecha_fin.strftime('%d/%m/%Y')

                if st.sidebar.button('Consultar'):
                    try:
                        consulta = investpy.get_stock_historical_data(stock=600010,
                                                                      country='china',
                                                                      from_date='01/01/2000',
                                                                      to_date='01/09/2022')
                        df = pd.DataFrame(consulta)
                        del (df['Currency'])
                        st.subheader('Datos del Stock ' + stock_name)
                        st.write(df)
                        st.subheader('Gráfica de datos Stock ' + stock_name)
                        st.line_chart(df['Close'])

                        aux_inicio = fecha_inicio.replace('/', '_')
                        aux_fin = fecha_fin.replace('/', '_')
                        #nombre_consulta = stock_symbol + '_' + aux_inicio + '_' + aux_fin + '.csv'
                        nombre_consulta = stock_name + '_' + stock_symbol + '_' + '01_01_2000' + '_' + '31_08_2022' + '.csv'
                        @st.cache
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')

                        csv = convert_df(df)
                        st.download_button(label="Descargar en CSV",
                                           data=csv,
                                           file_name=nombre_consulta,
                                           mime='text/csv')

                    except ValueError:
                        exc = format_exc()
                        st.text_input('Error', exc)

                if st.sidebar.button('Todo el Índice'):
                    stocks = pd.read_csv('stock_symbol_country.csv')
                    df_stocks = pd.read_csv(indice + '.csv')#DataFrame(stocks)
                    df_stocks = pd.DataFrame(arch_indice)
                    stocks_indice = df_stocks.loc[df_stocks['indice'] == indice]
                    dataframes = []
                    for st_i in df_stocks.index:
                        for i_symbol in df_stocks.index:
                            if stock_name == df_stocks['name'][i_symbol] and pais == df_stocks['country'][i_symbol] and indice == df_stocks['indice'][i_symbol]:
                                stock_symbol = df_stocks['symbol'][i_symbol]
                         
                        try:
                            consulta = investpy.get_stock_historical_data(stock=stocks_symbol,
                                                                          country=pais,
                                                                          from_date='01/01/2000',
                                                                          to_date='31/08/2022')

                            consulta['Stock'] = stocks_indice['name'][st_i]
                            consulta['Symbol'] = stocks_indice['symbol'][st_i]
                            dataframes.append(consulta)

                        except:
                            st.text_input('Error', stocks_indice['symbol'][st_i])

                    todos = pd.concat(dataframes, axis=0)
                    df = pd.DataFrame(todos)

                    del (df['Currency'])
                    st.write(df)

                    aux_inicio = fecha_inicio.replace('/', '_')
                    aux_fin = fecha_fin.replace('/', '_')
                    nombre_consulta = indice + '_' + aux_inicio + '_' + aux_fin + '.csv'

                    @st.cache
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(df)
                    st.download_button(label="Descargar en CSV",
                                       data=csv,
                                       file_name=nombre_consulta,
                                       mime='text/csv')
            else:
                st.sidebar.text_input('Error', 'El índice no tiene stocks')
                
            if st.sidebar.button('Consultar solo índice'):
                    try:
                        consulta = investpy.get_stock_historical_data(index=indice,
                                                                      country=pais,
                                                                      from_date=fecha_inicio,
                                                                      to_date=fecha_fin)
                   
                        df = pd.DataFrame(consulta)
                        del (df['Currency'])
                        st.subheader('Datos del Índice ' + stock_name)
                        st.write(df)
                        st.subheader('Gráfica de datos Índice ' + stock_name)
                        st.line_chart(df['Close'])

                        aux_inicio = fecha_inicio.replace('/', '_')
                        aux_fin = fecha_fin.replace('/', '_')
                        #nombre_consulta = stock_symbol + '_' + aux_inicio + '_' + aux_fin + '.csv'
                        nombre_consulta = stock_name + '_' + stock_symbol + '_' + '01_01_2000' + '_' + '31_08_2022' + '.csv'
                        @st.cache
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')

                        csv = convert_df(df)
                        st.download_button(label="Descargar en CSV",
                                           data=csv,
                                           file_name=nombre_consulta,
                                           mime='text/csv')

                    except ValueError:
                        exc = format_exc()
                        st.text_input('Error', exc)


        except ValueError:
            exc = format_exc()
            st.sidebar.text_input('Error', exc)


