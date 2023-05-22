# Imports
import pandas as pd
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
import locale

import pip


# Page config
st.set_page_config(page_title="Trabalho prático n° 4 - Análise de portfolio", page_icon="🌍", )

st.markdown("# Trabalho prático n° 4")
st.sidebar.header("Trabalho prático n° 4")
locale.setlocale( locale.LC_ALL, '' )

# Funções


def ret_datas(preLast, data_inicial, data_final, ativo):
    """
    Descrição:
    Calcula o log returno entre dois preços com base em duas datas para um ativo da preLast.
    
    Parâmetros:
    preLast: DataFrame contendo os preços dos ativos
    data_inicial : str (no formato de data AAAA-MM-DD)
    data_final : str (no formato de data AAAA-MM-DD)
    ativo: str

    
    Resultado:
    log_retorno : float

    """

    initial_price = preLast.loc[data_inicial, ativo]
    final_price = preLast.loc[data_final, ativo]

    ln_ret_entre_datas = np.log(final_price/initial_price)
    return ln_ret_entre_datas


def calc_vol_anual(preLast, ativo, initial_date, final_date):
    """
    Descrição:
    Calcula a volatilidade dos ativos do DataFrame preLast com base em um ativo e duas datas

    Parâmetros:
    preLast: DataFrame contendo os preços dos ativos
    ativo: str
    data_inicial : str (no formato de data AAAA-MM-DD)
    data_final : str (no formato de data AAAA-MM-DD)
    
    Resultado:
    volatilidade anual : pandas.Series
    """

    prices = preLast[ativo].loc[initial_date:final_date]
    retornos = np.log(prices / prices.shift(1)).dropna()
    daily_volume = retornos.std()
    anual_volume = daily_volume * np.sqrt(252)

    return anual_volume


def sharpe_ratio(preLast, ativo, data_inicial, data_final):
    """
    Descrição:
    Calcula o Índice de Sharpe para um determinado ativo com base em uma data inicial e uma data final usando a variação do CDI.

    Parâmetros:
    preLast: DataFrame contendo os preços dos ativos
    ativo: str
    data_inicial : str (no formato de data AAAA-MM-DD)
    data_final : str (no formato de data AAAA-MM-DD)

    Resultado:
    sharpe_ratio : float

    """

    prices = preLast[ativo].loc[data_inicial:data_final]
    returns = np.log(prices / prices.shift(1)).dropna()
    ret_ativo = sum(returns)

    rf = preLast["DI_INDEX"].loc[data_inicial:data_final]
    rf_returns = np.log(rf / rf.shift(1)).dropna()
    ret_rf = sum(rf_returns)

    excess_returns = ret_ativo - ret_rf
    returns_std = returns.std()
    sharpe_ratio = excess_returns / returns_std

    return sharpe_ratio


def beta(preLast, ativo, data_inicial, data_final):
    """
    Descrição:
    Calcula o beta para um ativo ou um portfolio com base em uma data de inicio e fim

    Parâmetros:
    preLast: DataFrame contendo os preços dos ativos
    ativo: str 
    data_inicial : str (no formato de data AAAA-MM-DD)
    data_final : str (no formato de data AAAA-MM-DD)
    
    Resultado:
    beta : float
    
    """

    # prices = preLast[ativo].loc[data_inicial:data_final]

    prices = preLast[(preLast["Data"] >= data_inicial)
                     & (preLast["Data"] <= data_final)]
    prices = prices[['Data', ativo]]

    returns = np.log(prices[ativo] / prices[ativo].shift(1)).dropna()

    # rb = preLast[".BVSP"].loc[data_inicial:data_final]

    rb = preLast[(preLast["Data"] >= data_inicial)
                 & (preLast["Data"] <= data_final)]

    rb = rb[['Data', '.BVSP']]

    rb_returns = np.log(rb['.BVSP'] / rb['.BVSP'].shift(1)).dropna()

    cov = np.cov(returns, rb_returns)[0, 1]
    var = np.var(rb_returns)

    beta = cov / var

    return beta


def alfa(preLast, ativo, data_inicial, data_final):
    """
    Descrição:
    Calcula o Alfa para um ativo ou um portfolio com base em uma data de inicio e fim

    Parâmetros:
    preLast: DataFrame contendo os preços dos ativos
    ativo: str 
    data_inicial : str (no formato de data AAAA-MM-DD)
    data_final : str (no formato de data AAAA-MM-DD)
    
    Resultado:
    beta : float
    
    """

    # prices = preLast[ativo].loc[data_inicial:data_final]
    prices = preLast[(preLast["Data"] >= data_inicial)
                     & (preLast["Data"] <= data_final)]
    prices = prices[['Data', ativo]]

    returns = np.log(prices[ativo] / prices[ativo].shift(1)).dropna()
    ret_ativo = sum(returns)

    # rf = preLast["DI_INDEX"].loc[data_inicial:data_final]

    rf = preLast[(preLast["Data"] >= data_inicial)
                 & (preLast["Data"] <= data_final)]

    rf = rf[['Data', 'DI_INDEX']]

    rf_returns = np.log(rf['DI_INDEX'] / rf['DI_INDEX'].shift(1)).dropna()
    ret_rf = sum(rf_returns)

    # rb = preLast[".BVSP"].loc[data_inicial:data_final]
    rb = preLast[(preLast["Data"] >= data_inicial)
                 & (preLast["Data"] <= data_final)]

    rb = rb[['Data', '.BVSP']]

    rb_returns = np.log(rb['.BVSP'] / rb['.BVSP'].shift(1)).dropna()
    ret_rb = sum(rb_returns)

    cov = np.cov(returns, rb_returns)[0, 1]
    var = np.var(rb_returns)
    beta = cov / var

    alfa = ret_ativo - (ret_rf + beta*(ret_rb-ret_rf))

    return alfa


def max_draw_down(ativo):
    """
    Descrição:
    Calcula o Máximo Drawdown para um ativo

    Parâmetros:
    preLast: Pandas DataFrame já importado
    ativo: str 

    
    Resultado:
    MDD : df
    
    """

    data = preLast[ativo]

    # Calcula os retornos diários do ativo financeiro
    returns = preLast[ativo].pct_change()

    # Calcula o Max Drawdown do ativo financeiro
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max * \
        100  # Multiplica por 100 para exibir em porcentagem

    return drawdown


def load_arquivo():
    return st.file_uploader('Carregar arquivo', type=['xlsx'])


@st.cache_data
def draw_prices(ativo, prelast, start_date, end_date):

    prices = prelast[(prelast["Data"] >= start_date)
                     & (prelast["Data"] <= end_date)]
    prices = prices[["Data", ativo]]

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(prices['Data'], prices[ativo])
    ax1.set_title(f"Preços de ao longo do tempo para {ativo}")
    ax1.set_xlabel("Data")
    ax1.set_ylabel(ativo)
    st.pyplot(fig)


@st.cache_data
def draw_vol(ativo, volume, prelast, start_date, end_date):
    volu = volume[ativo]
    prices = prelast[(prelast["Data"] >= start_date)
                     & (prelast["Data"] <= end_date)]
    prices = prices[["Data", ativo]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    ax1.plot(prices['Data'], prices[ativo])
    ax1.set_title(f"Preços do ativo {ativo}")
    ax1.set_xlabel("Data")
    ax1.set_ylabel(ativo)

    ax2.bar(volu.index, volu.values)
    ax2.set_title(f"Volumes do ativo {ativo}")
    ax2.set_xlabel("Data")
    ax2.set_ylabel(ativo)
    st.pyplot(fig)


@st.cache_data
def price_and_volume(ativo, volume, prelast, start_date, end_date):

    prices = prelast[(prelast["Data"] >= start_date)
                     & (prelast["Data"] <= end_date)]
    prices = prices[["Data", ativo]]

    volu = volume[(volume["Data"] >= start_date)
                  & (volume["Data"] <= end_date)]
    volu = volu[["Data", ativo]]

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.plot(prices['Data'], prices[ativo])
    ax1.set_title(f"Preço e volume ao longo do tempo para {ativo}")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Preço")

    # Criando o gráfico de volume
    ax2 = ax1.twinx()  # Criando um segundo eixo y compartilhando o mesmo eixo x
    # Criando o gráfico de barras
    ax2.bar(volume['Data'], volume[ativo], alpha=0.3)
    ax2.set_ylabel("Volume")
    st.pyplot(fig)


@st.cache_data
def draw_max_drawdown(ativo, volume, prelast, start_date, end_date):
    # Busca os preços históricos

    # data = preLast[ativo]

    # Calcula os retornos diários do ativo financeiro
    returns = preLast[ativo].pct_change()

    # Calcula o Max Drawdown do ativo financeiro
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max * \
        100  # Multiplica por 100 para exibir em porcentagem

    # Cria o gráfico de preços históricos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # Gráfico superior com a série de preços históricos
    ax1.plot(preLast['Data'], preLast[ativo], label='Preço Histórico')
    ax1.set_ylabel('Preço Histórico')

    # Gráfico inferior com a série do drawdown
    ax2.plot(preLast['Data'], drawdown, color='red', label='Max Drawdown')
    # Adiciona unidade de medida em porcentagem
    ax2.set_ylabel('Max Drawdown (%)')
    ax2.axhline(y=0, color='black', linestyle='--')

    # Adiciona legenda aos gráficos
    ax1.legend(loc='best')
    ax2.legend(loc='best')

    # Adiciona título à figura
    fig.suptitle('Preços Históricos e Max Drawdown de {}'.format(ativo))

    # plt.show()
    st.pyplot(fig)


def draw_dashboard(start_date, end_date, acquisition_date, benchmark, start_amount, adtv_value, stocks, adtv):
    data_list = []

    w = start_amount / len(stocks)

    for ticker in stocks:

        data = {}
        data['Ticker'] = ticker
        data['Preço na compra'] = locale.currency(preLast[preLast['Data']== acquisition_date][ticker].values[0])
        
        qtd = data['Quantidade'] = (start_amount/len(stocks))/preLast[preLast['Data'] == acquisition_date][ticker].values[0]
        
        pc = preLast[preLast['Data'] == end_date][ticker].values[0]
        data['Preço'] = locale.currency( pc )
        data['Valor da Posição'] = qtd * pc
        data['Concentração'] = 0
        data['% do ADTV'] = 0 

        data_list.append(data)

    retorno = pd.DataFrame(data_list)
    retorno_sum = retorno['Valor da Posição'].sum()

    for ticker in stocks:
        valor_posicao = retorno.loc[retorno['Ticker'] == ticker, 'Valor da Posição'].values[0]

        concentration = round( valor_posicao / retorno_sum , 2) 
        retorno.loc[retorno['Ticker'] == ticker, 'Concentração'] = "{:.2%}".format(concentration)
        retorno.loc[retorno['Ticker'] == ticker, '% do ADTV']  =  "{:.2%}".format(valor_posicao / adtv[adtv['Data']== adtv_value][ticker].values[0])

    return retorno_sum, retorno 


#
arquivo = load_arquivo()


if arquivo != None:
    preLast = pd.read_excel(arquivo, sheet_name="Screener", skiprows=1)
    portf = pd.read_excel(arquivo, sheet_name="Portfolios")
    preLast = pd.read_excel(arquivo, sheet_name="PreLast", skiprows=1)
    volume = pd.read_excel(arquivo, sheet_name="Volume", skiprows=1)
    adtv = pd.read_excel(arquivo, sheet_name="ADTV")

    preLast["Data"] = pd.to_datetime(preLast["Data"]).dt.date
    volume["Data"] = pd.to_datetime(volume["Data"]).dt.date


    cols = st.columns(4)


    portfolio_names = portf.columns
    option = cols[0].selectbox('Selecione o portifólio', portfolio_names, index=0)
    options = portf[option]

    
    start_date = cols[1].date_input(
        'Data inicial', value=datetime.date(2019, 12, 31))
    end_date = cols[2].date_input(
        'Data final', value=datetime.date(2023, 3, 31))
    
    

    acquisition_date = cols[3].date_input(
        'Data de aquisição', value=datetime.date(2020, 1, 2))

    

    benchmark = cols[0].selectbox('Benchmark', options=['.BVSP', 'DI_INDEX'])



    adtv_value = cols[1].selectbox('Adtv (Controle posição)', options=['ADTV 30','ADTV 60', 'ADTV 90'])
    start_amount = cols[2].number_input('Valor inicial', value=10000000)
    selected_stocks = st.multiselect(
        'Selecione os ativos para analisar', options, default=options)
    st.divider()
    st.markdown("# Análise do portfolio")
    total_port, dash = draw_dashboard(start_date, end_date, acquisition_date,              
                          benchmark, start_amount, adtv_value, selected_stocks, adtv)   
    
    
    dash.style.set_sticky(axis="index")   


    st.write("Total do portfolio", locale.currency( total_port ))
    dash

    # port_df = pd.DataFrame(options)
    # port_df

    st.divider()
    st.markdown("# Análise do ativos")
    

    ativo = st.selectbox('Ativo', options=preLast.columns[1:])

    draw_prices(ativo, preLast, start_date, end_date)
    draw_vol(ativo, volume, preLast, start_date, end_date)
    price_and_volume(ativo, volume, preLast, start_date, end_date)
    draw_max_drawdown(ativo, volume, preLast, start_date, end_date)

    st.markdown(
        f'### Alfa: { round(alfa(preLast, ativo, start_date, end_date) * 100, 2)} %')
    st.markdown(
        f'### Beta: { round(beta(preLast, ativo, start_date, end_date)  * 100,2) } %')
