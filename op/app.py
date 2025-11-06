import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import requests
import time
import yfinance as yf
from datetime import date, timedelta
from scipy.stats import norm

# ======== 修正中文亂碼 (設置 Matplotlib 字體) ========
# 優先使用 SimHei (常用於Matplotlib的中文簡體) 或 Microsoft JhengHei (繁體 Windows)
chinese_fonts = ['Microsoft JhengHei', 'SimHei', 'DFKai-SB', 'BiauKai', 'Arial Unicode MS']
font_found = False
for font in chinese_fonts:
    # 檢查字體是否存在於系統中
    try:
        if font_manager.findfont(font, fallback_to_default=False):
            rcParams['font.sans-serif'] = [font]
            font_found = True
            break
    except:
        pass
        
if not font_found:
    # 如果都找不到，使用預設列表，讓Matplotlib嘗試 fallback
    rcParams['font.sans-serif'] = chinese_fonts

rcParams['axes.unicode_minus'] = False # 正常顯示負號

# 策略顏色定義
STRATEGY_COLORS = {
    "策略 A": '#a7d9f7',
    "策略 B": '#c0f2c0'
}

# 策略顏色函數 (用於 Pandas Styler)
def color_strategy(val):
    """根據策略名稱返回 CSS 樣式字符串"""
    color = STRATEGY_COLORS.get(val, '#8c8c8c')
    return f'background-color: {color}; font-weight: bold; color: #04335a;'
    
# ======== 頁面設定 ========
st.set_page_config(page_title="選擇權與微台損益模擬（即時指數版）", layout="wide")

# ======== CSS 樣式（🎯 核心修正：隱藏 Expander 圖標名稱洩露） ========
st.markdown(
    """
    <style>
    /* 基礎字體設定 */
    html, body, .stApp, .stApp * {
        font-family: 'DFKai-SB', 'BiauKai', 'Microsoft JhengHei', sans-serif !important;
        font-size: 15px;
    }
    
    :root {
        --card-bg: #ffffff;
        --page-bg: #f3f6fb;
        --accent: #0b5cff;
        --muted: #6b7280;
    }
    /* ... 保持您的其他 CSS 樣式 ... */
    
    .title { font-size: 30px; font-weight: 800; color: #04335a; margin-bottom: 4px; padding-top: 10px; }
    .subtitle { color: var(--muted); margin-top: -8px; margin-bottom: 20px; font-size: 16px; }
    .card { background: var(--card-bg); padding: 18px 22px; border-radius: 12px; box-shadow: 0 8px 30px rgba(11,92,255,0.08); margin-bottom: 25px; }
    .card .section-title { font-size: 20px; font-weight: 700; color: #04335a; margin-bottom: 15px; border-bottom: 2px solid #eaeef7; padding-bottom: 5px; }
    .stButton>button { border-radius: 8px; height: 38px; font-size: 15px; }
    .small-muted { color: var(--muted); font-size: 14px; }
    hr { border: 0; height: 1px; background: #eaeef7; margin: 14px 0; }
    .position-row-text { font-size: 16px; padding: 5px 0; }
    .position-nowrap { white-space: nowrap; }
    .buy-color { color: #0b5cff; font-weight: bold; }
    .sell-color { color: #cf1322; font-weight: bold; }
    .strategy-a-bg { background-color: #a7d9f7; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .strategy-b-bg { background-color: #c0f2c0; padding: 0 4px; border-radius: 4px; font-weight: bold; }

    /* 🎯 核心修正：針對 st.expander 內的圖標名稱重疊問題 (問題 1 & 2) */
    /* 目標是隱藏 Streamlit 內部用來顯示圖標的文字組件（即洩露的 keyboard_arrow_...） */
    /* 這組規則針對所有 st.expander 標籤內的第一個子元素（標題列），並找到其中包含圖標文字的部分 */
    div[data-testid="stExpander"] div[data-testid="stText"] {
        white-space: nowrap !important;
        overflow: hidden !important;
        /* 增加以下規則以確保洩露的文字被推到視野外或完全隱藏 */
        display: none !important; 
    }
    
    /* 確保 Expander 標題容器本身不會因為內容溢出而變形 */
    div[data-testid="stExpanderToggle"] {
        overflow: hidden !important;
        white-space: nowrap !important;
        line-height: 1.2;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📈 選擇權與微台損益模擬（即時指數版）</div>'
            '<div class="subtitle">自動抓取加權指數，作為價平中心點進行模擬</div>', unsafe_allow_html=True)

# ---
## ⚙️ 系統設定與資料獲取
# ---

# ======== 設定常數 ========
POSITIONS_FILE = "positions_store.json"
MULTIPLIER_MICRO = 10.0
MULTIPLIER_OPTION = 50.0
PRICE_STEP = 100.0

# ======== 網路資料抓取函式 (使用 yfinance) ========
@st.cache_data(ttl=600)
def get_tse_index_price(ticker="^TWII"):
    """從 Yahoo Finance 獲取加權指數的最新價格"""
    try:
        tse_ticker = yf.Ticker(ticker)
        info = tse_ticker.info
        price = info.get('regularMarketPrice')
        
        if price is None or price == 0:
            price = info.get('regularMarketPreviousClose')

        if price and price > 1000:
            return float(price)
            
        st.warning(f"⚠️ 無法從 {ticker} 獲取有效價格，將使用備用值。", icon="⚠️")
        return None
        
    except Exception as e:
        st.error(f"❌ 透過 yfinance 抓取指數價格失敗：{e}", icon="❌")
        return None

# ======== Black-Scholes 模型函式 (新增/修正) ========
def safe_log(x):
    return np.log(np.maximum(x, 1e-10))
def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 1e-10))
    
def black_scholes_model(S, K, T_years, r, sigma, option_type):
    """
    Black-Scholes 模型計算選擇權理論價格, Delta, Gamma
    S: 現貨價, K: 履約價, T_years: 剩餘年數 (T/365), r: 無風險利率, sigma: 波動率
    """
    if T_years <= 1e-6 or sigma <= 1e-6:
        # 臨近到期或波動率為零時，近似於內含價值
        intrinsic = 0
        if option_type == '買權':
            intrinsic = max(0, S - K)
        elif option_type == '賣權':
            intrinsic = max(0, K - S)
        return intrinsic, intrinsic, 0.0 # 理論價, 內含價值, 時間價值

    # 確保 S 和 K 為正
    S = max(1e-6, S)
    K = max(1e-6, K)
    
    d1 = (safe_log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * safe_sqrt(T_years))
    d2 = d1 - sigma * safe_sqrt(T_years)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    
    price = 0.0

    if option_type == '買權':
        price = S * N_d1 - K * np.exp(-r * T_years) * N_d2
        intrinsic = max(0.0, S - K)
        
    elif option_type == '賣權':
        price = K * np.exp(-r * T_years) * N_neg_d2 - S * N_neg_d1
        intrinsic = max(0.0, K - S)
        
    price = max(0.0, price) # 價格不能為負
    
    time_value = max(0.0, price - intrinsic)
    
    return price, intrinsic, time_value # 理論價, 內含價值, 時間價值

# ======== 載入與儲存函式 (維持不變) ========
def load_positions(fname=POSITIONS_FILE):
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                loaded_center = None
                # 新增讀取舊格式的 BS 參數，如果沒有則使用預設
                loaded_t = 6 
                loaded_r = 0.015
            elif isinstance(data, dict) and "positions" in data:
                df = pd.DataFrame(data["positions"])
                loaded_center = data.get("center_price")
                loaded_t = data.get("days_to_expiry", 6)
                loaded_r = data.get("risk_free_rate", 0.015)
            else:
                st.error("讀取儲存檔格式錯誤。", icon="❌")
                return None, None, None, None
            
            required_cols = {
                "策略": str, "商品": str, "選擇權類型": str, "履約價": object,
                "方向": str, "口數": int, "成交價": float
            }
            for c, dtype in required_cols.items():
                if c not in df.columns:
                    df[c] = ""
            
            df["口數"] = df["口數"].fillna(0).astype(int)
            df["成交價"] = df["成交價"].fillna(0.0).astype(float)
            
            def norm_strike(v):
                if v == "" or pd.isna(v): return ""
                try: return float(v)
                except: return ""
            df["履約價"] = df["履約價"].apply(norm_strike)

            return df, loaded_center, loaded_t, loaded_r
        except Exception as e:
            st.error(f"讀取儲存檔失敗: {e}", icon="❌")
            return None, None, None, None
    return None, None, None, None

def save_positions(df, center_price, days_to_expiry, risk_free_rate, fname=POSITIONS_FILE):
    try:
        data = {
            "center_price": center_price,
            "days_to_expiry": days_to_expiry,
            "risk_free_rate": risk_free_rate,
            "positions": df.to_dict(orient="records")
        }
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"儲存失敗: {e}", icon="❌")
        return False
        
# ======== 初始化 session state (新增 BS 模型參數) ========
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=[
        "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價"
    ])
if "target_prices" not in st.session_state:
    st.session_state.target_prices = []
if "_edit_index" not in st.session_state:
    st.session_state._edit_index = -1
if "tse_index_price" not in st.session_state:
    st.session_state.tse_index_price = None
if "center_price" not in st.session_state:
    st.session_state.center_price = None
# 新增 BS 模型參數預設值
if "days_to_expiry" not in st.session_state:
    st.session_state.days_to_expiry = 6 
if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 0.015 # 1.5%

# ********* 獲取並設定中心價 *********
if st.session_state.tse_index_price is None:
    tse_price = get_tse_index_price()
    if tse_price and tse_price > 1000:
        st.session_state.tse_index_price = tse_price
        st.sidebar.success(f"🌐 最新加權指數：{tse_price:,.2f}。", icon="✅")
    else:
        st.session_state.tse_index_price = 10000.0
        st.sidebar.info("🌐 無法獲取即時指數，使用備用中心價 10,000.0。", icon="ℹ️")

if st.session_state.center_price is None:
    st.session_state.center_price = st.session_state.tse_index_price
        
# ---
## 🗃️ 倉位管理與檔案操作
# ---

# ======== 檔案操作區 ========
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 檔案操作與清理</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔄 載入倉位", use_container_width=True):
            df, loaded_center, loaded_t, loaded_r = load_positions()
            if df is not None:
                st.session_state.positions = df
                if loaded_center is not None:
                    st.session_state.center_price = loaded_center
                    st.session_state.days_to_expiry = loaded_t
                    st.session_state.risk_free_rate = loaded_r
                    st.success(f"✅ 已從檔案載入倉位、中心價 {loaded_center:,.1f} 及 BS 參數。")
                else:
                    st.success("✅ 已從檔案載入倉位，中心價及 BS 參數使用預設值")
            else:
                st.info("找不到儲存檔或檔案為空。")
    with col2:
        if st.button("💾 儲存倉位", use_container_width=True):
            if not st.session_state.positions.empty:
                current_center = st.session_state.get("simulation_center_price_input")
                center_to_save = current_center if current_center is not None else st.session_state.center_price
                
                # 抓取目前的 BS 參數 (即使在側邊欄變動過)
                t_to_save = st.session_state.days_to_expiry
                r_to_save = st.session_state.risk_free_rate
                
                ok = save_positions(st.session_state.positions, center_to_save, t_to_save, r_to_save)
                if ok:
                    st.session_state.center_price = center_to_save
                    st.success(f"✅ 已儲存到 {POSITIONS_FILE}，中心價 {center_to_save:,.1f} 及 BS 參數已記錄")
                else:
                    st.info("目前沒有倉位可儲存。")
    with col3:
        if st.button("🧹 清空所有倉位", use_container_width=True):
            st.session_state.positions = pd.DataFrame(columns=[
                "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價"
            ])
            st.session_state._edit_index = -1
            st.session_state.target_prices = []
            st.session_state.center_price = st.session_state.tse_index_price
            st.session_state.days_to_expiry = 6 # 清空時重設 BS 參數
            st.session_state.risk_free_rate = 0.015
            st.success("已清空所有倉位與狀態。")
    st.markdown("</div>", unsafe_allow_html=True)

# ======== 新增倉位 (維持不變) ========
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">➕ 新增倉位 (建立持倉)</div>', unsafe_allow_html=True)

col_strat, col_prod = st.columns(2)
with col_strat:
    new_strategy = st.selectbox("策略", ["策略 A", "策略 B"], key="new_strategy_outside")
with col_prod:
    new_product = st.selectbox("商品", ["微台", "選擇權"], key="new_product_outside")

strike_default = round(st.session_state.center_price / 100) * 100
new_opt_type = ""
new_strike = ""

if st.session_state.new_product_outside == "選擇權":
    st.markdown("---") 
    st.markdown("##### 選擇權細節")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        new_opt_type = st.selectbox("選擇權類型", ["買權", "賣權"], key="new_opt_type_outside")
    with opt_col2:
        new_strike = st.number_input("履約價", min_value=0.0, step=0.5, value=float(strike_default), key="new_strike_outside")
    st.markdown("---")

with st.form(key="add_position_form"):
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        strategy_style = "strategy-a-bg" if st.session_state.new_strategy_outside == "策略 A" else "strategy-b-bg"
        st.markdown(f"**策略：** <span class='{strategy_style}'>{st.session_state.new_strategy_outside}</span>", unsafe_allow_html=True)
        new_direction = st.radio("方向", ["買進", "賣出"], horizontal=True, key="new_direction_inside")
        
    with c2:
        st.markdown(f"**商品：** `{st.session_state.new_product_outside}`")
        new_lots = st.number_input("口數", min_value=1, step=1, value=1, key="new_lots_inside")
        
    with c3:
        if st.session_state.new_product_outside == "選擇權":
              strike_val = st.session_state.new_strike_outside
              st.markdown(f"**類型：** `{st.session_state.new_opt_type_outside}` / **履約價：** `{strike_val:,.1f}`")
        else:
              st.markdown(f"**<div style='height: 19.5px;'></div>**", unsafe_allow_html=True)
              
        new_entry = st.number_input("成交價（權利金或口數成交價）", min_value=0.0, step=0.5, value=0.0, key="new_entry_inside")
        
    submitted = st.form_submit_button("✅ 新增倉位 (加入持倉)", use_container_width=True)
    
    if submitted:
        
        product_value = st.session_state.new_product_outside
        strategy_value = st.session_state.new_strategy_outside
        
        if product_value == "選擇權":
              strike_value = float(st.session_state.new_strike_outside)
              opt_type_value = st.session_state.new_opt_type_outside
        else:
              strike_value = ""
              opt_type_value = ""
        
        rec = {
            "策略": strategy_value,
            "商品": product_value,
            "選擇權類型": opt_type_value,
            "履約價": strike_value,
            "方向": st.session_state.new_direction_inside,
            "口數": int(st.session_state.new_lots_inside),
            "成交價": float(st.session_state.new_entry_inside)
        }
        st.session_state.positions = pd.concat([st.session_state.positions, pd.DataFrame([rec])], ignore_index=True)
        st.success("已新增倉位，請在下方持倉明細確認。")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ======== 持倉明細 & 編輯/刪除 (維持不變) ========
positions_df = st.session_state.positions.copy()
if positions_df.empty:
    st.info("尚無任何倉位資料，請先新增或從檔案載入。")
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 現有持倉明細與快速移除</div>', unsafe_allow_html=True)
    
    c_strat_h, c_details_h, c_lots_h, c_entry_h, c_delete_h = st.columns([1, 5.5, 1.5, 1.5, 1])
    c_strat_h.markdown("策略", unsafe_allow_html=True)
    c_details_h.markdown("細節 (索引/商品/類型/履約價)", unsafe_allow_html=True)
    c_lots_h.markdown("方向/口數", unsafe_allow_html=True)
    c_entry_h.markdown("<div style='text-align: right;'>成交價</div>", unsafe_allow_html=True)
    c_delete_h.markdown("<div style='text-align: right;'>操作</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    
    for index, row in positions_df.iterrows():
        
        details = f"({index}) {row['商品']} / "
        if row['商品'] == "選擇權":
            strike_val = row['履約價']
            details += f"{row['選擇權類型']} @ {strike_val:,.1f}" if strike_val != "" else f"{row['選擇權類型']} @ ---"
        else:
            details += f"---"
        
        direction_style = "buy-color" if row['方向'] == "買進" else "sell-color"
        strategy_style = "strategy-a-bg" if row['策略'] == "策略 A" else "strategy-b-bg"
        
        c_strat, c_details, c_lots, c_entry, c_delete = st.columns([1, 5.5, 1.5, 1.5, 1])

        with c_strat:
            st.markdown(f'<div class="position-row-text"><span class="{strategy_style}">{row["策略"]}</span></div>', unsafe_allow_html=True)

        with c_details:
            st.markdown(f'<div class="position-row-text">{details}</div>', unsafe_allow_html=True)
            
        with c_lots:
            st.markdown(f'<div class="position-row-text position-nowrap {direction_style}">{row["方向"]} {row["口數"]} 口</div>', unsafe_allow_html=True)
            
        with c_entry:
            st.markdown(f'<div class="position-row-text position-nowrap" style="text-align: right;">{row["成交價"]:,.2f}</div>', unsafe_allow_html=True)

        with c_delete:
            if st.button("移除", key=f"delete_btn_{index}", type="secondary", use_container_width=True):
                st.session_state.positions = st.session_state.positions.drop(index).reset_index(drop=True)
                st.toast(f"✅ 已移除 (索引 {index}) 倉位！")
                st.rerun()
        
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)

    # 編輯功能 (Expander 修正: CSS 處理圖示文字洩露)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ 編輯倉位 (索引式)</div>', unsafe_allow_html=True)
    
    current_indices = positions_df.index.tolist()
    
    with st.expander("編輯單列倉位"):
        
        col_idx, col_load = st.columns([1,2])
        
        if current_indices:
            if st.session_state._edit_index == -1 and current_indices:
                st.session_state._edit_index = current_indices[0]
                
            with col_idx:
                selected_index = st.selectbox(
                    "選擇要編輯的索引",
                    options=current_indices,
                    index=current_indices.index(st.session_state._edit_index) if st.session_state._edit_index in current_indices else 0,
                    key="edit_select_index"
                )
            
            with col_load:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"載入索引 {selected_index} 到編輯表單", use_container_width=True):
                    st.session_state._edit_index = int(selected_index)
                    st.toast(f"已載入索引 {selected_index} 的資料。")

            idx = st.session_state._edit_index
            
            if idx in positions_df.index:
                st.markdown(f"**👉 編輯索引 {idx} 的倉位（修改後按 儲存修改）**")
                row = positions_df.loc[idx]
                
                with st.form(key=f"edit_form_{idx}"):
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        f_strategy = st.selectbox("策略", ["策略 A", "策略 B"], index=0 if row["策略"] == "策略 A" else 1, key=f"e_strat_{idx}")
                        f_product = st.selectbox("商品", ["微台", "選擇權"], index=0 if row["商品"] == "微台" else 1, key=f"e_prod_{idx}")
                    with f_col2:
                        f_direction = st.selectbox("方向", ["買進", "賣出"], index=0 if row["方向"] == "買進" else 1, key=f"e_dir_{idx}")
                        f_lots = st.number_input("口數", value=int(row["口數"]), step=1, min_value=1, key=f"e_lots_{idx}")
                    with f_col3:
                        f_entry = st.number_input("成交價", value=float(row["成交價"]), step=0.1, key=f"e_entry_{idx}")

                    if f_product == "選擇權":
                        opt_options = ["買權", "賣權"]
                        default_opt_idx = 0 if row["選擇權類型"] == "買權" else 1
                        f_opt_type = st.selectbox("選擇權類型", opt_options, index=default_opt_idx, key=f"e_opttype_{idx}")
                        strike_val = float(row["履約價"]) if row["履約價"] != "" else st.session_state.center_price
                        f_strike = st.number_input("履約價", value=strike_val, step=0.5, key=f"e_strike_{idx}")
                    else:
                        f_opt_type = ""
                        f_strike = ""
                    
                    submitted = st.form_submit_button("💾 儲存修改", use_container_width=True)
                    if submitted:
                        st.session_state.positions.loc[idx, ["策略","商品","選擇權類型","履約價","方向","口數","成交價"]] = [
                            f_strategy, f_product, f_opt_type, float(f_strike) if f_product=="選擇權" else "",
                            f_direction, int(f_lots), float(f_entry)
                        ]
                        st.session_state._edit_index = -1
                        st.success("✅ 倉位已更新，請查看上方明細。")
                        st.rerun()
            else:
                st.info("請先載入要編輯的倉位索引。")
        else:
            st.info("目前無倉位可編輯。")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
# ---
## 📈 損益計算與模擬
# ---
    
if not positions_df.empty:

    # ======== 損益計算基礎（側邊欄）(新增 BS 參數輸入) ========
    
    st.sidebar.markdown('## 🛠️ 損益模擬設定')
    center = st.sidebar.number_input(
        "價平中心價 (Center)",
        value=st.session_state.center_price,
        key="simulation_center_price_input",
        step=1.0,
        help="損益曲線圖的中心點價格，預設為最新加權指數/上次儲存值"
    )
    
    PRICE_RANGE = st.sidebar.number_input(
        "模擬範圍 (±點數)",
        value=1500,
        step=100,
        min_value=100,
        help="價格範圍為 [Center - Range, Center + Range]"
    )
    
    st.sidebar.markdown('### Black-Scholes 模型參數')
    
    # 設置 T 和 R (根據您的截圖 image_d0d7dd.png)
    col_t, col_r = st.sidebar.columns(2)
    with col_t:
        days_to_expiry = st.number_input(
            "到期剩餘天數 (T, 天)", 
            min_value=1, 
            value=st.session_state.days_to_expiry, 
            step=1, 
            key="days_to_expiry_input",
            help="選擇權距離到期日的天數。"
        )
        st.session_state.days_to_expiry = days_to_expiry
        
    with col_r:
        risk_free_rate_percent = st.number_input(
            "無風險利率 (R, %)", 
            min_value=0.0, 
            value=st.session_state.risk_free_rate * 100, 
            step=0.1, 
            format="%.2f",
            key="risk_free_rate_input",
            help="例如：1.5% 請輸入 1.5。"
        )
        st.session_state.risk_free_rate = risk_free_rate_percent / 100
        
    # 新增波動率輸入
    volatility = st.sidebar.number_input(
        "波動率 (Sigma, %)",
        min_value=1.0,
        value=20.0, # 預設值
        step=1.0,
        format="%.1f",
        key="volatility_input",
        help="輸入年化波動率百分比 (例如：20%)。"
    )
    sigma = volatility / 100.0 # 轉換為小數
    
    # 顯示確認的參數值
    st.sidebar.markdown(f"""
    <div style='font-size:14px; margin-top: 15px;'>
        <p><b>中心價:</b> <span style="color:#04335a; font-weight:700;">{center:,.1f}</span></p>
        <p><b>模擬範圍:</b> <span style="color:#04335a; font-weight:700;">±{PRICE_RANGE} 點</span></p>
        <p><b>BS T:</b> <span style="color:#cf1322; font-weight:700;">{days_to_expiry} 天 ({days_to_expiry/365:.4f} 年)</span></p>
        <p><b>BS R:</b> <span style="color:#cf1322; font-weight:700;">{st.session_state.risk_free_rate*100:.2f}%</span></p>
        <p><b>BS Sigma:</b> <span style="color:#cf1322; font-weight:700;">{volatility:.1f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

    offsets = np.arange(-PRICE_RANGE, PRICE_RANGE + 1e-6, PRICE_STEP)
    prices = [center + float(off) for off in offsets]

    def profit_for_row_at_price(row, price):
        prod = row["商品"]
        direction = row["方向"]
        lots = float(row["口數"])
        entry = float(row["成交價"]) if row["成交價"] != "" else 0.0
        
        multiplier = MULTIPLIER_MICRO if prod == "微台" else MULTIPLIER_OPTION
        
        if prod == "微台":
            # 微台損益 = (結算價 - 成交價) * 口數 * 乘數
            return (price - entry) * lots * multiplier if direction == "買進" else (entry - price) * lots * multiplier
        else:
            # 選擇權損益 = (內含價值 @ 結算價 - 成交價) * 口數 * 乘數
            strike = float(row["履約價"]) if row["履約價"] != "" else 0.0
            opt_type = row.get("選擇權類型", "")
            
            if opt_type == "買權":
                intrinsic = max(0.0, price - strike)
            elif opt_type == "賣權":
                intrinsic = max(0.0, strike - price)
            else:
                intrinsic = 0.0
                
            return (intrinsic - entry) * lots * multiplier if direction == "買進" else (entry - intrinsic) * lots * multiplier

    a_profits, b_profits = [], []
    for p in prices:
        a_df = positions_df[positions_df["策略"]=="策略 A"]
        b_df = positions_df[positions_df["策略"]=="策略 B"]
        a_val = a_df.apply(lambda r: profit_for_row_at_price(r,p), axis=1).sum()
        b_val = b_df.apply(lambda r: profit_for_row_at_price(r,p), axis=1).sum()
        a_profits.append(a_val)
        b_profits.append(b_val)

    # ======== 損益曲線圖 & 表格 (亂碼問題已在上方字體設定修正) ========
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 損益曲線與詳表</div>', unsafe_allow_html=True)

    col_chart, col_download = st.columns([3,1])
    with col_chart:
        st.subheader("📈 損益曲線（策略 A vs 策略 B）")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(prices, a_profits, label="策略 A", linewidth=2, color="#0b5cff")
        ax.plot(prices, b_profits, label="策略 B", linewidth=2, color="#2aa84f")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(center, color="gray", linestyle=":", linewidth=1)
        ax.set_xlim(center-PRICE_RANGE, center+PRICE_RANGE)
        
        ax.set_xlabel("結算價", fontsize=12)
        ax.set_ylabel("損益金額", fontsize=12)
        
        # 確保標題和圖例使用正確的字體
        title_font = font_manager.FontProperties(family=rcParams['font.sans-serif'][0], size=14)
        ax.set_title(f"策略 A / 策略 B 損益曲線（價平 {center:.1f} ±{int(PRICE_RANGE)}）", fontproperties=title_font)
        
        # 讓圖例也使用中文字體
        legend = ax.legend(prop=font_manager.FontProperties(family=rcParams['font.sans-serif'][0], size=10))
        
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    table_df = pd.DataFrame({
        "價格": prices,
        "相對於價平(點)": [int(p-center) for p in prices],
        "策略 A 損益": a_profits,
        "策略 B 損益": b_profits
    }).sort_values(by="價格", ascending=False).reset_index(drop=True)

    def color_profit(val):
        try: f=float(val)
        except: return ''
        if f>0: return 'background-color: #d8f5e2; color: #008000;'
        elif f<0: return 'background-color: #ffe6e8; color: #cf1322;'
        return ''
        
    styled_table = table_df.style.format({
        "價格": "{:,.1f}",
        "相對於價平(點)": "{:+d}",
        "策略 A 損益": "{:,.0f}",
        "策略 B 損益": "{:,.0f}"
    }).applymap(color_profit, subset=["策略 A 損益","策略 B 損益"])
    
    st.markdown(f"<div class='small-muted'>每 {int(PRICE_STEP)} 點損益表（價平 {center:,.1f} ±{int(PRICE_RANGE)}）</div>", unsafe_allow_html=True)
    st.table(styled_table)

    with col_download:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        csv = table_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 模擬損益 CSV", data=csv, file_name="profit_table.csv", mime="text/csv", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================
    # 💰 選擇權理論平倉損益列表 (新增功能)
    # ==========================================================
    opt_positions_df = positions_df[positions_df["商品"] == "選擇權"].copy().reset_index(drop=True)
    
    if not opt_positions_df.empty:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 選擇權理論平倉損益列表 (Black-Scholes 模型)</div>', unsafe_allow_html=True)
        
        current_price_rounded = round(center, 2)
        
        st.markdown(f"**模型假設：** 目前的股價指數為 <span style='color:#0b5cff; font-weight:bold;'>{current_price_rounded:,.2f}</span>，並使用 Black-Scholes 模型計算理論平倉時的損益。", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px; color:#cf1322;'>**BS 參數：** T = {days_to_expiry} 天, R = {st.session_state.risk_free_rate*100:.2f}%, Sigma = {volatility:.1f}%</div>", unsafe_allow_html=True)
        
        # 1. 計算理論價
        T_years = days_to_expiry / 365.0
        results = []
        for index, row in opt_positions_df.iterrows():
            strike = float(row["履約價"])
            opt_type = row["選擇權類型"]
            entry = float(row["成交價"])
            lots = float(row["口數"])
            direction = row["方向"]
            
            # 進行 BS 計算
            # S: 現價 (center), K: 履約價, T_years, r, sigma, option_type
            theoretical_price, intrinsic_value, time_value = black_scholes_model(
                current_price_rounded, strike, T_years, st.session_state.risk_free_rate, sigma, opt_type
            )
            
            # 理論平倉價 - 成交價
            price_difference = theoretical_price - entry
            
            # 理論平倉損益 = (理論價 - 成交價) * 乘數 * 口數 * 買賣方向
            multiplier = MULTIPLIER_OPTION
            sign = 1 if direction == "買進" else -1
            theoretical_profit = price_difference * multiplier * lots * sign
            
            results.append({
                "策略": row["策略"],
                "選擇權類型": opt_type,
                "履約價": strike,
                "方向": direction,
                "口數": lots,
                "成交價": entry,
                "內含價值(IV)": intrinsic_value,
                "理論價(BS Price)": theoretical_price,
                "理論時間價值(TV)": time_value,
                "理論平倉損益": theoretical_profit
            })
            
        bs_df = pd.DataFrame(results)

        # 2. 應用樣式
        def color_bs_profit(val):
            try: f=float(val)
            except: return ''
            if f > 0: return 'color: #0b5cff; font-weight: 700;'
            elif f < 0: return 'color: #cf1322; font-weight: 700;'
            return ''

        styled_bs_table = bs_df.style.format({
            "履約價": "{:,.1f}",
            "口數": "{:d}",
            "成交價": "{:,.2f}",
            "內含價值(IV)": "{:,.2f}",
            "理論價(BS Price)": "{:,.2f}",
            "理論時間價值(TV)": "{:,.2f}",
            "理論平倉損益": "{:,.0f}"
        }).applymap(color_bs_profit, subset=["理論平倉損益"]).apply(lambda x: [color_strategy(v) for v in x], subset=['策略'])
        
        st.dataframe(styled_bs_table, use_container_width=True)
        
        # 3. 彙總數據
        total_theo_profit = bs_df["理論平倉損益"].sum()
        total_theo_tv_loss = bs_df.apply(lambda r: r['理論時間價值(TV)'] * r['口數'] * MULTIPLIER_OPTION * (-1 if r['方向'] == '賣出' else 1), axis=1).sum()
        
        total_profit_style = "color: #0b5cff;" if total_theo_profit > 0 else "color: #cf1322;"
        total_tv_style = "color: #cf1322;" if total_theo_tv_loss < 0 else "color: #0b5cff;" # 權利金損失用紅色

        st.markdown("---")
        st.subheader("彙總數據")
        
        col_tv, col_profit = st.columns(2)
        
        with col_tv:
            st.markdown(f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; background-color: #f7f7f7;'>
                <span class='small-muted'>理論總時間價值損益 (金額)</span><br>
                <span style='font-size: 24px; font-weight: bold; {total_tv_style}'>NT$ {total_theo_tv_loss:,.0f}</span>
                <span class='small-muted'> (理論平倉時的 TV 總和 * 口數 * 乘數)</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col_profit:
            st.markdown(f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; background-color: #f7f7f7;'>
                <span class='small-muted'>理論總平倉損益 (金額)</span><br>
                <span style='font-size: 24px; font-weight: bold; {total_profit_style}'>NT$ {total_theo_profit:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

        csv_bs = bs_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 理論價損益 CSV", data=csv_bs, file_name="theoretical_profit_table.csv", mime="text/csv", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ==========================================================
    # 💵 最終結算損益分析 (修正 Expander 洩露問題 2)
    # ==========================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💵 假設結算損益分析 (微台+選擇權)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:14px; margin-bottom: 10px; color:#cf1322;'>
        此計算假設**目標到價**即為**最終結算價** (時間價值歸零)，並計算所有部位的損益。
        **這就是您的每個倉位到期結算時的最終損益預期**。
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_add, col_remove = st.columns([2,1,2])
    with col_input:
        add_price = st.number_input("輸入目標結算價", value=float(center), step=0.5, key="add_price_input")
    with col_add:
        if st.button("➕ 加入目標結算價", use_container_width=True):
            v = float(add_price)
            if v not in st.session_state.target_prices:
                st.session_state.target_prices.append(v)
                st.session_state.target_prices.sort(reverse=True)
            st.toast(f"已加入目標結算價: {v:.1f}")
    with col_remove:
        if st.session_state.target_prices:
            to_remove = st.selectbox("選擇要移除的結算價", options=["無"] + [f"{p:,.1f}" for p in st.session_state.target_prices])
            if st.button("🗑️ 移除選定結算價", type="secondary", use_container_width=True):
                if to_remove != "無":
                    val = float(to_remove.replace(',', ''))
                    st.session_state.target_prices = [p for p in st.session_state.target_prices if p != val]
                    st.toast(f"已移除結算價 {val:,.1f}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.session_state.target_prices:
        rows = []
        per_position_details = {}
        
        for tp in st.session_state.target_prices:
            a_df = positions_df[positions_df["策略"]=="策略 A"]
            b_df = positions_df[positions_df["策略"]=="策略 B"]
            a_val = a_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1).sum()
            b_val = b_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1).sum()
            total_val = a_val + b_val
            rows.append({"結算價": tp, "相對於價平(點)": int(tp-center), "策略 A 損益": a_val, "策略 B 損益": b_val, "總損益": total_val})
            
            combined_df = positions_df.copy() 
            combined_df["結算損益"] = combined_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1)
            per_position_details[tp] = combined_df

        target_df = pd.DataFrame(rows).sort_values(by="結算價", ascending=False).reset_index(drop=True)

        def color_target_profit(val):
            try: f=float(val)
            except: return ''
            if f>0: return 'background-color: #e6faff'
            elif f<0: return 'background-color: #fff0f0'
            return ''

        styled_target = target_df.style.format({
            "結算價": "{:,.1f}",
            "相對於價平(點)": "{:+d}",
            "策略 A 損益": "{:,.0f}",
            "策略 B 損益": "{:,.0f}",
            "總損益": "**{:,.0f}**"
        }).applymap(color_target_profit, subset=["總損益"]).applymap(color_profit, subset=["策略 A 損益","策略 B 損益"])
        
        st.subheader("🎯 目標結算價總損益一覽")
        st.dataframe(styled_target, use_container_width=True)

        csv2 = target_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 結算損益 CSV", data=csv2, file_name="settlement_profit.csv", mime="text/csv", key="download_target_csv")

        st.markdown("---")
        st.subheader("📝 **每筆倉位**在目標結算價下的損益明細")
        
        if not positions_df.empty:
            for tp in st.session_state.target_prices:
                total_profit_tp = target_df[target_df['結算價']==tp]['總損益'].iloc[0]
                st_class = "color: #0b5cff;" if total_profit_tp > 0 else "color: #cf1322;"
                
                # Expander 修正: CSS 處理圖示文字洩露
                expander_label = f"🔍 結算價 {tp:,.1f} — 總損益：{total_profit_tp:,.0f} (點擊展開)"
                
                with st.expander(expander_label, expanded=False):
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 10px; padding: 5px 10px; background-color: #f0f8ff; border-radius: 6px; border-left: 5px solid #0b5cff;'>
                        <b>目標結算價: {tp:,.1f}</b> / 
                        <b>總損益: <span style='{st_class}'>{total_profit_tp:,.0f}</span></b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    df_detail = per_position_details[tp].copy()
                    df_detail_display = df_detail.reset_index(drop=True)
                    
                    df_detail_display = df_detail_display[[
                        "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價", "結算損益"
                    ]]

                    def color_detail_profit(val):
                        try: f=float(val)
                        except: return ''
                        if f>0: return 'color: #0b5cff; font-weight: 700;'
                        elif f<0: return 'color: #cf1322; font-weight: 700;'
                        return ''

                    styled_detail = df_detail_display.style.format({
                        "履約價": lambda v: f"{v:,.1f}" if v != "" else "",
                        "成交價": "{:,.2f}",
                        "口數": "{:d}",
                        "結算損益": "{:,.0f}" 
                    }).applymap(color_detail_profit, subset=["結算損益"])

                    def color_strategy_detail(val):
                        if val == "策略 A": return 'background-color: #a7d9f7;'
                        elif val == "策略 B": return 'background-color: #c0f2c0;'
                        return ''
                    styled_detail = styled_detail.applymap(color_strategy_detail, subset=["策略"])


                    st.dataframe(styled_detail, use_container_width=True)
        else:
            st.info("目前沒有倉位可以計算明細損益。")
    else:
        st.markdown("<div class='small-muted' style='margin-top:8px'>尚未設定目標結算價，請新增結算價以查看損益。</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
