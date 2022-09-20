
#streamlit run .\script_aplicacion_streamlit.py 

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from pandas_datareader import data, wb
import yfinance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

st.set_page_config(page_title = 'Gráficos',layout = 'wide')

st.markdown("<h1 style='text-align: center; color:black;'> Informe de inversiones </h1>",unsafe_allow_html=True)
st.markdown("<br></br>",unsafe_allow_html=True)



df_1 = pd.read_parquet('Datasets/SP500_1.gzip')
df_2 = pd.read_parquet('Datasets/SP500_2.gzip')
df_3 = pd.read_parquet('Datasets/SP500_3.gzip')
df = pd.concat([df_1,df_2,df_3])
df['Date'] = pd.to_datetime(df.Date)
sp500 = data.DataReader('^GSPC', data_source='yahoo',start='2000-01-01', end='2021-12-31')['Adj Close']


lista_empresas=df.Symbol.values


with st.container():
   
    with st.sidebar:
        selection = st.selectbox('Seleccione empresa:', lista_empresas)
        inicio = st.date_input( "Ingrese inicio: ",value = pd.to_datetime('2000-01-31', format='%Y-%m-%d'),min_value = pd.to_datetime('2000-01-31', format='%Y-%m-%d'),max_value= pd.to_datetime('2021-12-31', format='%Y-%m-%d'))
        fin = st.date_input( "Ingrese fin: ",value = pd.to_datetime('2021-12-31', format='%Y-%m-%d'), min_value = pd.to_datetime('2000-01-31', format='%Y-%m-%d'),max_value= pd.to_datetime('2021-12-31', format='%Y-%m-%d'))
        inicio = inicio.strftime("%Y-%m-%d")
        fin = fin.strftime("%Y-%m-%d")

    datos= df.loc[(df['Symbol']==selection) & (df['Date']>inicio) & (df['Date']<fin) ]
    datos['Retornos'] = np.log(datos['Adj Close'] / datos['Adj Close'].shift(1))
    datos['RangoPrecio'] = datos['High']-datos['Low'] 
    datos['variaciones'] = datos['Adj Close'].pct_change() * 252
    datos['volatilidad'] = np.sqrt((datos['variaciones'].rolling(window=252,min_periods=1).std()*252))

    title= selection 
    st.markdown(f"<h1 style='text-align:center; color:blue;'> Precio de {title} </h1>",unsafe_allow_html=True)
   
    
    fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)
    fig.add_trace(go.Candlestick(x=datos.Date,
            open=datos['Open'],
            high=datos['High'],
            low=datos['Low'],
            close=datos['Close'] ))
    fig.add_trace(go.Bar(x=datos.Date, y=datos['Volume'],
                showlegend=False), row=2, col=1)
    fig.update_layout(
        autosize=True,
        width=1500,
        height=700
    )
    margin= dict(l=20, r=20, b=0, t=0, pad=4)
    st.plotly_chart(fig)
     

fig1, fig2 = st.columns(2)

with fig1:
    st.markdown(f"<h2 style='text-align:center; color:blue;'> Rendimiento logarítmico </h2>",unsafe_allow_html=True)
    st.line_chart(datos.Retornos)

    with fig2:
        st.markdown(f"<h2 style='text-align:center; color:blue;'> Métricas </h2>",unsafe_allow_html=True)
        avgu=datos.Retornos[datos['Retornos']>=0].mean()
        avgd=datos.Retornos[datos['Retornos']<0].mean()
        rsi=np.round(100-100/(1+avgu/avgd),2)
        rendimiento_anual=np.round(datos.variaciones.mean()*252,2)
        volatilidad_anual=np.round(datos.volatilidad.mean(),2)
        st.metric(label='RSI',value=rsi)
        st.metric(label='Rendimiento anual',value=rendimiento_anual)
        st.metric(label='Volatilidad anual',value=volatilidad_anual)
        

with st.container():
    
    
    st.markdown(f"<h2 style='text-align:center; color:blue;'> S&P </h2>",unsafe_allow_html=True)
    st.header('Evolución histórica de precios')
    sp500 = data.DataReader('^GSPC', data_source='yahoo',start='2000-01-01', end='2021-12-31')
    sp500['42d'] = np.round(sp500['Close'].rolling(window=42,min_periods=1).mean(), 2)
    sp500['252d'] = np.round(sp500['Close'].rolling(window=252,min_periods=1).mean(), 2)
    sp500['42-252'] = sp500['42d'] - sp500['252d']
    sp500['variaciones'] = sp500['Close'].pct_change()
    sp500['volatilidad'] = np.sqrt((sp500['variaciones'].rolling(window=252,min_periods=1).std()*252))
    st.line_chart(sp500[['Close', '42d', '252d']])
    
    with st.expander("Ver estrategia"):
        number = st.number_input('Ingrese diferencia entre media móvil bimestral y anual:')
        SD = number
        sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
        sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
        sp500['Regime'].value_counts()
        st.line_chart(sp500['Regime'])

    st.header('Evolución histórica de la volatilidad')
    st.line_chart(sp500.loc[:,'volatilidad'].dropna())
    
    
    

    st.title('Análisis de portafolio')
    st.header('Construcción de carteras eficientes')
    datos=pd.read_parquet('Datasets/S&P500_Close.gzip')
    datos['Date'] = pd.to_datetime(datos.Date)
    sp500 = data.DataReader('^GSPC', data_source='yahoo',start='2000-01-01', end='2021-12-31')['Adj Close']
    datos.set_index('Date',inplace=True)
    sp500.name='S&P500'


    beta_coef=[]
    for symbol in datos.columns.values:
        try:
            df=datos.loc[:,[symbol]].join(sp500, on='Date',how='inner')
            
            df.dropna(inplace=True)
            x = df[['S&P500']].values.reshape(-1,1)
            y = df[[symbol]].values.reshape(-1,1)

            # Instancio modelo y elijo hiperparámetros
            model = LinearRegression(fit_intercept=True)

            #Entreno modelo
            model.fit(x,y)

            beta = model.coef_[0][0]
            coef_determ = model.score(x,y)
            
            beta_coef.append([symbol,beta,coef_determ])
            
        except:
            beta_coef.append([np.nan,np.nan,np.nan])

    df_betas=pd.DataFrame(beta_coef,columns=['Symbol','Beta','Coef_det']).set_index('Symbol')

    
    rets = np.log(datos / datos.shift(1)) 
    top_100_rend = rets.mean().sort_values(ascending=False).head(100)
    top_50_betas = df_betas.loc[top_100_rend.index].Beta.abs().sort_values(ascending=True).head(50)
    top_9_corr = rets.loc[:,top_50_betas.index].corr().abs().mean().sort_values().head(9)
    
    # ANÁLISIS DE CARTERA 

    rets_filtrado=rets.loc[:,top_9_corr.index]
    rend_min=(rets_filtrado.mean()*252).min()
    rend_max=(rets_filtrado.mean()*252).max()


    esperanza=np.array([rets_filtrado.mean()]) *252
    covar=rets_filtrado.cov().values *252
    C=np.vstack((covar,np.ones(9)))
    C = np.hstack((C,np.array([1,1,1,1,1,1,1,1,1,0]).reshape(-1,1)))
    C_=np.linalg.inv(C)
    prop=C_[0:9,-1]
    rendimiento_op=np.matmul(prop,esperanza.ravel())
    desvio_op=np.matmul(np.matmul(prop.reshape(1,9),covar),prop)**0.5


    C2=np.vstack((covar,esperanza,np.ones(9)))
    C2=np.hstack((C2,np.concatenate([esperanza,np.zeros([1,2])],axis=1).reshape(-1,1),np.array([1,1,1,1,1,1,1,1,1,1,0]).reshape(-1,1)))
    C2_=np.linalg.inv(C2)

    arma_frontera=C2_[:9,-2:]
    resultados=pd.DataFrame(columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','Rendimiento','Desvío'])
    for i in np.arange(np.round(rend_min,2),np.round(rend_max,2),0.01):
        x1=np.sum(arma_frontera[0]*np.array([i,1]))
        x2=np.sum(arma_frontera[1]*np.array([i,1]))
        x3=np.sum(arma_frontera[2]*np.array([i,1]))
        x4=np.sum(arma_frontera[3]*np.array([i,1]))
        x5=np.sum(arma_frontera[4]*np.array([i,1]))
        x6=np.sum(arma_frontera[5]*np.array([i,1]))
        x7=np.sum(arma_frontera[6]*np.array([i,1]))
        x8=np.sum(arma_frontera[7]*np.array([i,1]))
        x9=np.sum(arma_frontera[8]*np.array([i,1]))
        prop=np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9])
        desvio=np.matmul(np.matmul(prop.reshape(1,9),covar),prop)[0]**0.5
        dicc={'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5,'x6':x6,'x7':x7,'x8':x8,'x9':x9,'Rendimiento':i,'Desvío':desvio}
        resultados = pd.concat([resultados,pd.DataFrame([dicc])],ignore_index=True)
        
    cant=rets_filtrado.shape[1]
    weights = np.random.uniform(-1,1,cant)
    weights /= np.sum(weights) #genero pesos aleatorios

    prets = []
    pvols = []
    for p in range (2500):
        weights = np.random.uniform(-1,1,cant)
        weights /= np.sum(weights) #genero pesos aleatorios
       
        retorno_esperado= np.sum(rets_filtrado.mean() * weights) * 252
        prets.append(retorno_esperado)
        
        volatilidad_esperada=np.sqrt(np.dot(weights.T, np.dot(rets_filtrado.cov() * 252, weights)))
        pvols.append(volatilidad_esperada)
    prets = np.array(prets)
    pvols = np.array(pvols)
    
    rend_requerido = st.slider(label='Rendimiento', min_value=resultados.loc[0,'Rendimiento'], max_value=resultados.Rendimiento.values[-1],  step=0.01) 
    print('Nivel de riesgo:', resultados.loc[resultados.Rendimiento==rend_requerido,'Desvío'])
    fig = st.columns(1)
      
    fig,ax=plt.subplots()
    ax.scatter(pvols, prets, c=prets / pvols, marker='o')
    ax.set_xlabel('expected volatility')
    ax.set_ylabel('expected return')
    ax.plot(desvio_op,rendimiento_op,'b.', markersize=15.0)
    ax.plot(resultados.loc[0,'Desvío'],resultados.loc[0,'Rendimiento'],'r.', markersize=15.0)
    ax.plot(resultados.Desvío[-1:],resultados.Rendimiento[-1:],'r.', markersize=15.0)
    ax.plot(resultados.loc[resultados.Rendimiento==rend_requerido,'Rendimiento'],resultados.loc[resultados.Rendimiento==rend_requerido,'Desvío'],'r*', markersize=15.0)

    ax.scatter(resultados['Desvío'],resultados['Rendimiento'],c=resultados['Rendimiento']/ resultados['Desvío'], marker='o')
    ax.set_ylim(0.10,1)
    ax.set_xlim(0.15,0.5)
    #ax.colorbar(label='Sharpe ratio')
    st.pyplot(fig)

    # DataFrame con proporciones
    st.header('Proporciones de activos que minimizan el riesgo')
    
    old=resultados.columns[:9]
    new=rets_filtrado.columns
    dicc={old[i]:new[i] for i in range(len(old))}
    resultados.rename(columns=dicc,inplace=True)

    resultados=resultados.applymap(lambda x: np.round(x*100,2))
    res_formateado = resultados.style.format('{0:,.2f}%').background_gradient(subset=resultados.columns[:9],cmap='BuGn',axis=1)
    
    st.dataframe(res_formateado)

df = data.DataReader('^GSPC', data_source='yahoo',start='2000-01-01', end='2021-12-31')
df.reset_index(inplace=True)
df['retornos_log'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df['retornos_gaps'] = np.log(df.Open/df.Close.shift(1)).fillna(0)
df['retornos_intra'] = np.log(df.Close/df.Open).fillna(0)
df['variaciones'] = df['Adj Close'].pct_change()
df['volatilidad'] = np.sqrt((df['variaciones'].rolling(window=252,min_periods=1).std()*252))

df_tendencias = df.set_index('Date').asfreq('D')
df_tendencias=df_tendencias[['Close','retornos_gaps','retornos_intra','retornos_log']] 
df_tendencias['year']= df_tendencias.index.year
df_tendencias['month_of_year']= df_tendencias.index.month
df_tendencias['day_of_week']= df_tendencias.index.dayofweek
df_tendencias['week_of_year']= df_tendencias.index.isocalendar().week
df_tendencias = df_tendencias[(df_tendencias.day_of_week!=5) & (df_tendencias.day_of_week!=6) ]

st.title('Análisis tendencial')
fig1, fig2 = st.columns(2)

with fig1:
    st.header('Anual')
    DF=pd.DataFrame(df_tendencias.groupby(['year','month_of_year'])['Close'].mean())

    for i in df_tendencias.year.unique():
        for j in df_tendencias.month_of_year.unique():
            DF.loc[(i,j),'rend_norm'] = DF.loc[(i,j),'Close']/DF.loc[(i,1),'Close']

    x=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

    fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)

    for i in df_tendencias.year.unique():
        fig.add_trace(go.Scatter(x= x, y = DF.loc[(i,)].rend_norm, mode='lines',name=f'Año {i}'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.update_layout(title={
                'text': 'Tendencia anual (perído 2000-2021)',
                'y':0.9,
                'x':0.5,
                'font_size':25,
                'xanchor': 'center',
                'yanchor': 'top'},
                yaxis=dict(type='linear', title='Retorno normalizado'),
                xaxis=dict(title='Mes'))


    # Tendencia promedio
    media = DF.groupby('month_of_year')['rend_norm'].mean()
    desvio = DF.groupby('month_of_year')['rend_norm'].std()
    lim_inf= media-desvio
    lim_sup= media+desvio

    fig.add_trace( go.Scatter(x= x, y = media, mode='lines',name='Promedio'),row=2, col=1)
    st.plotly_chart(fig)   
    
    with fig2:
        st.header('Semanal')
        
        DF=pd.DataFrame(df_tendencias.groupby(['year','week_of_year','day_of_week'])['Close'].mean())
        DF.fillna(method='ffill',inplace=True)

        for year in df_tendencias.year.unique():
            for week in df_tendencias.week_of_year.unique()[1:-1]:
                for day in df_tendencias.day_of_week.unique():
                    DF.loc[(year,week,day),'rend_norm'] = DF.loc[(year,week,day),'Close']/DF.loc[(year,week,0),'Close']

        x=['Lunes','Martes','Miercoles','Jueves','Viernes']
        fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)

        for year in df_tendencias.year.unique():
            for week in df_tendencias.week_of_year.unique()[1:-1]:
                fig.add_trace(go.Scatter(x= x, y = DF.loc[(year,week,)].rend_norm, mode='lines',name=f'Semana {week}, Año{year}'),row=1, col=1)
        fig.update_layout(showlegend=False)
        fig.update_layout(title={
                    'text': 'Tendencia semanal (perído 2000-2021)',
                    'y':0.9,
                    'x':0.5,
                    'font_size':25,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    yaxis=dict(type='linear', title='Retorno normalizado'),
                    xaxis=dict(title='Día de la semana'))


        # Tendencia promedio
        media = DF.groupby('day_of_week')['rend_norm'].mean()
        desvio = DF.groupby('day_of_week')['rend_norm'].std()
        lim_inf= media-desvio
        lim_sup= media+desvio

        fig.add_trace( go.Scatter(x= x, y = media, mode='lines',name='Promedio'),row=2, col=1)

        st.plotly_chart(fig)


with st.container():

    st.header('Mejor día para invertir según retornos')
    
    DF=pd.DataFrame(df_tendencias.groupby(['year','week_of_year','day_of_week'])['retornos_log'].mean())
    DF.fillna(method='ffill',inplace=True)

    x=['Lunes','Martes','Miércoles','Jueves','Viernes']
    fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)

    for year in df_tendencias.year.unique():
        for week in df_tendencias.week_of_year.unique()[1:-1]:
            fig.add_trace(go.Scatter(x= x, y = DF.loc[(year,week,),'retornos_log'], mode='lines',name=f'Semana {week}, Año{year}'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.update_layout(title={
                'text': 'Retornos logarítmicos',
                'y':0.9,
                'x':0.5,
                'font_size':25,
                'xanchor': 'center',
                'yanchor': 'top'},
                yaxis=dict(type='linear', title='Retornos logarítmicos'),
                xaxis=dict(title='Día de la semana'))


    # Tendencia promedio
    media = df_tendencias.groupby('day_of_week')['retornos_log'].mean()
    desvio = df_tendencias.groupby('day_of_week')['retornos_log'].std()
    lim_inf= media-desvio
    lim_sup= media+desvio

    fig.add_trace( go.Scatter(x= x, y = media, mode='lines',name='Promedio'),row=2, col=1)
    st.plotly_chart(fig)   
    

    st.header('Mejor día para invertir según retornos gap')

    DF=pd.DataFrame(df_tendencias.groupby(['year','week_of_year','day_of_week'])['retornos_gaps'].mean())
    DF.fillna(method='ffill',inplace=True)

    x=['Lunes','Martes','Miércoles','Jueves','Viernes']
    fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)

    for year in df_tendencias.year.unique():
        for week in df_tendencias.week_of_year.unique()[1:-1]:
            fig.add_trace(go.Scatter(x= x, y = DF.loc[(year,week,),'retornos_gaps'], mode='lines',name=f'Semana {week}, Año{year}'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.update_layout(showlegend=False)
    fig.update_layout(title={
                'text': 'Retornos gaps',
                'y':0.9,
                'x':0.5,
                'font_size':25,
                'xanchor': 'center',
                'yanchor': 'top'},
                yaxis=dict(type='linear', title='Retornos'),
                xaxis=dict(title='Día de la semana'))


    # Tendencia promedio
    media = df_tendencias.groupby('day_of_week')['retornos_gaps'].mean()
    desvio = df_tendencias.groupby('day_of_week')['retornos_gaps'].std()
    lim_inf= media-desvio
    lim_sup= media+desvio

    fig.add_trace( go.Scatter(x= x, y = media, mode='lines',name='Promedio'),row=2, col=1)
    st.plotly_chart(fig)   

    st.header('Mejor día para invertir según retornos gap intradiarios')

    DF=pd.DataFrame(df_tendencias.groupby(['year','week_of_year','day_of_week'])['retornos_intra'].mean())
    DF.fillna(method='ffill',inplace=True)

    x=['Lunes','Martes','Miércoles','Jueves','Viernes']
    fig = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.09)

    for year in df_tendencias.year.unique():
        for week in df_tendencias.week_of_year.unique()[1:-1]:
            fig.add_trace(go.Scatter(x= x, y = DF.loc[(year,week,),'retornos_intra'], mode='lines',name=f'Semana {week}, Año{year}'),row=1, col=1)
    fig.update_layout(showlegend=False)
    fig.update_layout(title={
                'text': 'Retornos intradiarios',
                'y':0.9,
                'x':0.5,
                'font_size':25,
                'xanchor': 'center',
                'yanchor': 'top'},
                yaxis=dict(type='linear', title='Retornos'),
                xaxis=dict(title='Día de la semana'))


    # Tendencia promedio
    media = df_tendencias.groupby('day_of_week')['retornos_intra'].mean()
    desvio = df_tendencias.groupby('day_of_week')['retornos_intra'].std()
    lim_inf= media-desvio
    lim_sup= media+desvio

    fig.add_trace( go.Scatter(x= x, y = media, mode='lines',name='Promedio'),row=2, col=1)
      
    st.plotly_chart(fig)   
    

 