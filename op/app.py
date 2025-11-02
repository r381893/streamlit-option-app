import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import json
import os
import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="選擇權 / 微台策略比較", layout="wide")
st.markdown("""
<style>
body { font-family: 'Microsoft JhengHei', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- Black-Scholes 公式 ---
def black_scholes(S, K, T, r, sigma, option_type):
    S = float(S); K = float(K)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type=="call" else max(K - S, 0.0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# --- 選擇權損益 ---
def option_payoff(S_range, K, option_type, position, entry_price, qty, multiplier):
    intrinsic = np.maximum(S_range-K,0) if option_type=="call" else np.maximum(K-S_range,0)
    if position=="buy":
        return (intrinsic-float(entry_price))*float(multiplier)*int(qty)
    else:
        return (float(entry_price)-intrinsic)*float(multiplier)*int(qty)

# --- 期貨 / 微台損益 ---
def future_payoff(S_range, position, entry_price, qty, multiplier):
    entry_price = float(entry_price)
    qty = int(qty)
    multiplier = float(multiplier)
    if position=="buy":
        return (S_range-entry_price)*multiplier*qty
    else:
        return (entry_price-S_range)*multiplier*qty

# --- JSON 儲存 ---
def save_positions():
    safe_A = []
    safe_B = []
    for pos in st.session_state.strategy_A:
        safe_A.append({
            "asset_type": str(pos.get("asset_type")),
            "option_type": None if pos.get("option_type") is None else str(pos.get("option_type")),
            "position": str(pos.get("position")),
            "K": None if pos.get("K") is None else float(pos.get("K")),
            "entry_price": float(pos.get("entry_price")),
            "qty": int(pos.get("qty")),
            "multiplier": float(pos.get("multiplier"))
        })
    for pos in st.session_state.strategy_B:
        safe_B.append({
            "asset_type": str(pos.get("asset_type")),
            "option_type": None if pos.get("option_type") is None else str(pos.get("option_type")),
            "position": str(pos.get("position")),
            "K": None if pos.get("K") is None else float(pos.get("K")),
            "entry_price": float(pos.get("entry_price")),
            "qty": int(pos.get("qty")),
            "multiplier": float(pos.get("multiplier"))
        })
    data = {
        "strategy_A": safe_A,
        "strategy_B": safe_B,
        "S0": float(st.session_state.S0)
    }
    with open("positions.json","w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

# --- 載入策略 ---
def load_positions():
    if os.path.exists("positions.json"):
        with open("positions.json","r",encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.strategy_A = data.get("strategy_A",[])
            st.session_state.strategy_B = data.get("strategy_B",[])
            st.session_state.S0 = data.get("S0",16000.0)

# --- 初始化 session_state ---
if "strategy_A" not in st.session_state: st.session_state.strategy_A=[]
if "strategy_B" not in st.session_state: st.session_state.strategy_B=[]
if "S0" not in st.session_state: st.session_state.S0=16000.0

load_positions()

# --- 標題 ---
st.title("📊 多倉位選擇權 / 微台策略比較工具")

# --- 側欄參數 ---
st.sidebar.header("⚙ 全域參數設定")

# 到期日
today = datetime.date.today()
expiry_date = st.sidebar.date_input("到期日", value=today + datetime.timedelta(days=30))
days_to_expiry = (expiry_date-today).days
T = max(days_to_expiry/365.0,0.0)
st.sidebar.write(f"🕒 距離到期：{days_to_expiry} 天（約 {T:.3f} 年）")

# 標的現價
st.session_state.S0 = st.sidebar.number_input("標的現價", value=float(st.session_state.S0), step=10.0)
if st.sidebar.button("💾 儲存目前標的價"):
    save_positions()
    st.sidebar.success("✅ 已儲存現價，下次自動載入！")

# 其他參數
r = st.sidebar.number_input("無風險利率", value=0.01, format="%.4f")
sigma = st.sidebar.number_input("波動率 (Volatility)", value=0.2, format="%.4f")
range_points = st.sidebar.number_input("區間範圍 (點)", value=1500, step=100)
step = st.sidebar.number_input("步長 (點)", value=100, step=10)

# --- 新增倉位函式 ---
def add_position(strategy_name):
    with st.expander(f"➕ 新增倉位到 {strategy_name}"):
        asset_type = st.selectbox("標的類型", ["選擇權","微台"], key=f"{strategy_name}_asset")
        position = st.selectbox("買賣方向", ["buy","sell"], key=f"{strategy_name}_pos")
        qty = st.number_input("口數", value=1, step=1, key=f"{strategy_name}_qty")
        if asset_type=="選擇權":
            option_type = st.selectbox("選擇權類型", ["call","put"], key=f"{strategy_name}_opt")
            K = st.number_input("履約價", value=float(st.session_state.S0), key=f"{strategy_name}_K")
            entry_price = st.number_input("成交價 (權利金)", value=100.0, key=f"{strategy_name}_price")
            theoretical_price = black_scholes(float(st.session_state.S0), float(K), float(T), float(r), float(sigma), option_type)
            st.write(f"理論權利金: {theoretical_price:.2f}")
            multiplier=50
        else:
            option_type=None; K=None
            entry_price = st.number_input("成交價 (微台)", value=float(st.session_state.S0), key=f"{strategy_name}_price")
            multiplier=10
        if st.button(f"新增到 {strategy_name}", key=f"add_{strategy_name}"):
            position_data = {
                "asset_type": asset_type,
                "option_type": option_type,
                "position": position,
                "K": None if K is None else float(K),
                "entry_price": float(entry_price),
                "qty": int(qty),
                "multiplier": float(multiplier)
            }
            if strategy_name=="策略 A": st.session_state.strategy_A.append(position_data)
            else: st.session_state.strategy_B.append(position_data)
            save_positions(); st.rerun()  # 修正為新的指令

# --- 刪除倉位 ---
def delete_position(strategy_name):
    if strategy_name=="策略 A":
        for i,pos in enumerate(list(st.session_state.strategy_A)):
            if st.button(f"刪除 A 倉位 {i+1}", key=f"del_A_{i}"):
                st.session_state.strategy_A.pop(i); save_positions(); st.experimental_rerun()
    else:
        for i,pos in enumerate(list(st.session_state.strategy_B)):
            if st.button(f"刪除 B 倉位 {i+1}", key=f"del_B_{i}"):
                st.session_state.strategy_B.pop(i); save_positions(); st.experimental_rerun()

# --- 顯示策略區塊 ---
col1,col2 = st.columns(2)
with col1:
    st.header("策略 A"); add_position("策略 A")
    st.dataframe(pd.DataFrame(st.session_state.strategy_A), use_container_width=True)
    delete_position("策略 A")
with col2:
    st.header("策略 B"); add_position("策略 B")
    st.dataframe(pd.DataFrame(st.session_state.strategy_B), use_container_width=True)
    delete_position("策略 B")

# --- 計算策略總損益 ---
S0 = float(st.session_state.S0)
S_range = np.arange(S0-range_points, S0+range_points+step, step)
def calc_strategy(strategy_positions, S_range):
    total = np.zeros_like(S_range, dtype=float)
    for pos in strategy_positions:
        if pos["asset_type"]=="選擇權":
            total += option_payoff(S_range,pos["K"],pos["option_type"],pos["position"],
                                   pos["entry_price"],pos["qty"],pos["multiplier"])
        else:
            total += future_payoff(S_range,pos["position"],pos["entry_price"],pos["qty"],pos["multiplier"])
    return total
payoff_A = calc_strategy(st.session_state.strategy_A, S_range)
payoff_B = calc_strategy(st.session_state.strategy_B, S_range)

# --- 即時計算現價損益 ---
current_A = np.interp(S0, S_range, payoff_A)
current_B = np.interp(S0, S_range, payoff_B)
st.markdown(f"""
### 💰 即時損益  
- 策略 A：<span style="color:deepskyblue; font-size:20px;">{current_A:,.0f}</span>  
- 策略 B：<span style="color:violet; font-size:20px;">{current_B:,.0f}</span>  
""", unsafe_allow_html=True)

# --- Plotly 圖表 ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=S_range, y=payoff_A, mode="lines", name="策略 A", line=dict(color="deepskyblue", width=3)))
fig.add_trace(go.Scatter(x=S_range, y=payoff_B, mode="lines", name="策略 B", line=dict(color="violet", width=3)))
# 標示現價損益
fig.add_trace(go.Scatter(x=[S0], y=[current_A], mode="markers+text", name="A 現價損益",
                         text=[f"A：{current_A:.0f}"], textposition="top left", marker=dict(color="deepskyblue", size=10)))
fig.add_trace(go.Scatter(x=[S0], y=[current_B], mode="markers+text", name="B 現價損益",
                         text=[f"B：{current_B:.0f}"], textposition="top right", marker=dict(color="violet", size=10)))
fig.update_layout(title="策略損益比較", xaxis_title="標的價格", yaxis_title="損益 (元)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- 損益表格 + 顏色 ---
table_data = pd.DataFrame({"價格":S_range,"策略 A 損益":payoff_A,"策略 B 損益":payoff_B})
def color_negative(val):
    try: return 'background-color: pink' if float(val)<0 else 'background-color: lightblue'
    except: return ''
styled_table = table_data.head(30).style.applymap(color_negative, subset=["策略 A 損益","策略 B 損益"])
st.subheader("📋 損益比較表 (顯示 30 筆)")
st.markdown(styled_table.to_html(), unsafe_allow_html=True)

# --- 匯出 CSV ---
csv = table_data.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇ 下載損益表 (CSV)", data=csv, file_name="損益比較.csv", mime="text/csv")

