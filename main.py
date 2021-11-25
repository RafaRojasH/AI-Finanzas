import pandas as pd
import streamlit as st
import investpy

st.title('AI & Finanzas')
"""
# MSI
"""
stock = st.text_input(label='Inserta el stock', value='MSI')
busca6 = investpy.get_stock_historical_data(stock=stock,
                                            country='united states',
                                            from_date='01/11/2021',
                                            to_date='10/11/2021')
df = pd.DataFrame(busca6)
del(df['Currency'])
st.write(df)
st.line_chart(df)