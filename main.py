import pandas as pd
import streamlit as st
import investpy
from datetime import datetime
from traceback import format_exc
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
                        consulta = investpy.get_stock_historical_data(stock=stock_symbol,
                                                                      country=pais,
                                                                      from_date=fecha_inicio,
                                                                      to_date=fecha_fin)
                        df = pd.DataFrame(consulta)
                        del (df['Currency'])
                        st.subheader('Datos del Stock ' + stock_name)
                        st.write(df)
                        st.subheader('Gráfica de datos Stock ' + stock_name)
                        st.line_chart(df['Close'])

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

                    except ValueError:
                        exc = format_exc()
                        st.text_input('Error', exc)

                if st.sidebar.button('Todo el Índice'):
                    stocks = pd.read_csv('stock_symbol_country.csv')
                    df_stocks = pd.DataFrame(stocks)
                    stocks_indice = df_stocks.loc[df_stocks['indice'] == indice]
                    dataframes = []
                    for st_i in stocks_indice.index:
                        try:
                            consulta = investpy.get_stock_historical_data(stock=stocks_indice['symbol'][st_i],
                                                                          country=pais,
                                                                          from_date=fecha_inicio,
                                                                          to_date=fecha_fin)

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

        except ValueError:
            exc = format_exc()
            st.sidebar.text_input('Error', exc)
modelos = ['LSTM', 'opc1', 'opc2', 'opc3']
modelo = st.sidebar.selectbox('Elige un modelo', modelos)
if modelo == 'LSTM':
	st.sidebar.text_input('LSTM', 'modelo lstm')
	dataset = df.iloc[:, 1:2].values
	dataset = dataset.astype('float32')
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	train_size = int(len(dataset) * (porcent_train / 100))
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
