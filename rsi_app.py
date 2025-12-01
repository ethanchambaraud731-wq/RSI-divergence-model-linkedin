import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

with st.sidebar:
    ticker = st.text_input("Ticker", "^GSPC").upper()
    period = st.selectbox("Période RSI", [7, 14, 21], index=1)
    hist_range = st.selectbox("Plage", ["1mo", "3mo", "6mo", "1y"], index=3)
    lookback = st.slider("Détection", 5, 20, 10)
    capital = st.number_input("Capital ($)", min_value=100, value=10000, step=100)
    calc = st.button("Calculer", type="primary")

st.title(f"RSI Divergence model - {ticker}")

def compute_rsi(df, n):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(com=n-1, min_periods=n).mean()
    loss = -delta.where(delta < 0, 0).ewm(com=n-1, min_periods=n).mean()
    return 100 - (100 / (1 + gain / loss))

def find_extrema(series, order):
    highs, lows = [], []
    for i in range(order, len(series) - order):
        if all(series[i] > series[i-j] for j in range(1, order+1)) and all(series[i] > series[i+j] for j in range(1, order+1)):
            highs.append(i)
        if all(series[i] < series[i-j] for j in range(1, order+1)) and all(series[i] < series[i+j] for j in range(1, order+1)):
            lows.append(i)
    return highs, lows

def find_divergences(df, lb):
    ph, pl = find_extrema(df['Close'].values, lb)
    rh, rl = find_extrema(df['RSI'].values, lb)
    
    bull, bear = [], []
    
    for i in range(len(pl)-1):
        p1, p2 = float(df['Close'].iloc[pl[i]]), float(df['Close'].iloc[pl[i+1]])
        if p2 < p1:
            for j in range(len(rl)-1):
                r1, r2 = float(df['RSI'].iloc[rl[j]]), float(df['RSI'].iloc[rl[j+1]])
                if abs(pl[i+1] - rl[j+1]) <= lb and r2 > r1 and r2 < 40:
                    bull.append((df.index[pl[i+1]], p2))
    
    for i in range(len(ph)-1):
        p1, p2 = float(df['Close'].iloc[ph[i]]), float(df['Close'].iloc[ph[i+1]])
        if p2 > p1:
            for j in range(len(rh)-1):
                r1, r2 = float(df['RSI'].iloc[rh[j]]), float(df['RSI'].iloc[rh[j+1]])
                if abs(ph[i+1] - rh[j+1]) <= lb and r2 < r1 and r2 > 60:
                    bear.append((df.index[ph[i+1]], p2))
    
    return bull, bear

if calc:
    df = yf.download(ticker, period=hist_range, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df["RSI"] = compute_rsi(df, period)
    bull, bear = find_divergences(df, lookback)
    
    # Signaux RSI simples
    buy = [(df.index[i], df['Close'].iloc[i]) for i in range(1, len(df)) if df['RSI'].iloc[i-1] < 30 <= df['RSI'].iloc[i]]
    sell = [(df.index[i], df['Close'].iloc[i]) for i in range(1, len(df)) if df['RSI'].iloc[i-1] > 70 >= df['RSI'].iloc[i]]
    
    # Backtest
    trades = []
    position = None
    pnl_total = 0
    
    all_signals = sorted([(d, p, 'buy') for d, p in buy] + [(d, p, 'sell') for d, p in sell], key=lambda x: x[0])
    
    for date, price, signal_type in all_signals:
        if signal_type == 'buy' and position is None:
            position = {'entry_date': date, 'entry_price': float(price), 'shares': capital / float(price)}
        
        elif signal_type == 'sell' and position is not None:
            exit_price = float(price)
            pnl = (exit_price - position['entry_price']) * position['shares']
            pnl_total += pnl

            trades.append({
                'entry': position['entry_date'],
                'exit': date,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl
            })

            position = None

    # -------------------------------------------------------------------------
    # 🔵 COURBE DU CAPITAL
    # -------------------------------------------------------------------------
    capital_series = []
    current_capital = capital

    # Liste chronologique des trades
    trade_events = [(t['exit'], t['pnl']) for t in trades]
    trade_events = sorted(trade_events, key=lambda x: x[0])

    last_capital = capital
    for date in df.index:
        for d, pnl in trade_events:
            if date == d:
                last_capital += pnl
        capital_series.append(last_capital)

    df["Capital"] = capital_series
    # -------------------------------------------------------------------------

    # FIGURE PRIX + RSI
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Prix", line=dict(color="#1f77b4")), row=1, col=1)
    if buy: fig.add_trace(go.Scatter(x=[b[0] for b in buy], y=[b[1] for b in buy], mode='markers', name='Achat', marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
    if sell: fig.add_trace(go.Scatter(x=[s[0] for s in sell], y=[s[1] for s in sell], mode='markers', name='Vente', marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
    if bull: fig.add_trace(go.Scatter(x=[b[0] for b in bull], y=[b[1] for b in bull], mode='markers', name='Div+', marker=dict(color='lime', size=12, symbol='star')), row=1, col=1)
    if bear: fig.add_trace(go.Scatter(x=[b[0] for b in bear], y=[b[1] for b in bear], mode='markers', name='Div-', marker=dict(color='orange', size=12, symbol='star')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
    
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(height=800, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    # -------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}")
    col2.metric("Prix", f"${float(df['Close'].iloc[-1]):.2f}")
    col3.metric("Trades", f"{len(trades)} complétés")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Capital Initial", f"${capital:,.0f}")
    pnl_pct = (pnl_total / capital) * 100 if capital > 0 else 0
    col5.metric("PnL Total", f"${pnl_total:,.2f}", f"{pnl_pct:+.2f}%")
    col6.metric("Capital Final", f"${capital + pnl_total:,.2f}")

    # -------------------------------------------------------------------------
    # 📈 FIGURE CAPITAL
    # -------------------------------------------------------------------------
    st.subheader("📈 Évolution du Capital")

    fig_cap = go.Figure()
    fig_cap.add_trace(go.Scatter(
        x=df.index,
        y=df["Capital"],
        mode="lines",
        line=dict(width=3),
        name="Capital ($)"
    ))

    fig_cap.update_layout(
        height=350,
        hovermode='x unified',
        margin=dict(l=30, r=30, t=40, b=30)
    )

    st.plotly_chart(fig_cap, use_container_width=True)
    
    st.subheader("📋 Historique des Signaux")
    
    signals = []
    for date, price in buy:
        signals.append({"Date": pd.Timestamp(date).strftime("%Y-%m-%d"), "Type": "🟢 Achat", "Prix": f"${float(price):.2f}", "Note": "RSI > 30"})
    for date, price in sell:
        signals.append({"Date": pd.Timestamp(date).strftime("%Y-%m-%d"), "Type": "🔴 Vente", "Prix": f"${float(price):.2f}", "Note": "RSI < 70"})
    for date, price in bull:
        signals.append({"Date": pd.Timestamp(date).strftime("%Y-%m-%d"), "Type": "⭐ Div Haussière", "Prix": f"${float(price):.2f}", "Note": "Prix↓ RSI↑"})
    for date, price in bear:
        signals.append({"Date": pd.Timestamp(date).strftime("%Y-%m-%d"), "Type": "⭐ Div Baissière", "Prix": f"${float(price):.2f}", "Note": "Prix↑ RSI↓"})
    
    if signals:
        signals_df = pd.DataFrame(signals).sort_values("Date", ascending=False)
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun signal détecté sur cette période")
    
    if trades:
        st.subheader("💰 Détail des Trades")
        trades_df = pd.DataFrame(trades)
        trades_df['entry'] = pd.to_datetime(trades_df['entry']).dt.strftime("%Y-%m-%d")
        trades_df['exit'] = pd.to_datetime(trades_df['exit']).dt.strftime("%Y-%m-%d")
        trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
        trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
        trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:,.2f}")
        trades_df.columns = ['Entrée', 'Sortie', 'Prix Achat', 'Prix Vente', 'PnL']
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

else:
    st.subheader("📘 Présentation du Modèle RSI & Détection des Divergences")

    st.markdown("""
### ▶️ Objectif du modèle  
L’application analyse l’action **{ticker}** en utilisant l'indicateur RSI pour identifier :  
- les zones de **survente (RSI < 30)** et **surachat (RSI > 70)**,  
- les **divergences haussières** et **baissières**,  
- la **succession des signaux** pour simuler automatiquement des trades.

---

### 📊 1. RSI — Relative Strength Index  
Le RSI mesure la dynamique du prix en comparant la force des hausses et des baisses récentes.

- **RSI > 70 → surachat → pression vendeuse potentielle**  
- **RSI < 30 → survente → pression acheteuse potentielle**  
- Les retournements du RSI autour de ces seuils génèrent des signaux de **buy/sell**.

---

### 🔍 2. Divergences Prix / RSI  
Une divergence apparaît quand **le prix** et **le RSI** évoluent en sens opposé.

#### 🔹 Divergence Haussière  
- Le prix **fait un nouveau plus bas**  
- Le RSI **fait un plus haut**  
→ Suggestion d’un affaiblissement de la pression vendeuse  
→ Signal d’achat potentiel

#### 🔹 Divergence Baissière  
- Le prix **fait un nouveau plus haut**  
- Le RSI **fait un plus bas**  
→ Essoufflement de la dynamique haussière  
→ Signal de vente potentiel

Ces signaux permettent d’anticiper des retournements locaux.

---

### 📈 3. Stratégie utilisée dans le modèle  
Le modèle :  
1. Analyse le prix et calcule le RSI  
2. Repère les **extrema** du prix et du RSI  
3. Détecte si des divergences sont alignées  
4. Génère automatiquement des signaux Buy/Sell  
5. Simule des trades pour afficher :  
   - le **PnL**,  
   - la courbe du **capital**,  
   - les signaux détectés,  
   - l’historique des opérations.

Cliquez sur **Calculer** dans la barre latérale pour lancer l’analyse.
""")
    
    st.info("➡️ Configurez vos paramètres dans la barre de gauche puis cliquez sur **Calculer**.")