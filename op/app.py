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

# ======== 修正中文亂碼 (設置 Matplotlib 字體，包含標楷體備用) ========
chinese_fonts = ['Microsoft JhengHei', 'DFKai-SB', 'BiauKai', 'Arial Unicode MS']
font_found = False
for font in chinese_fonts:
    if font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        rcParams['font.sans-serif'] = [font]
        font_found = True
        break
        
if not font_found:
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

# ======== CSS 樣式（維持不變） ========
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
    body { background-color: var(--page-bg); }
    /* 主標題 */
    .title {
        font-size: 30px;
        font-weight: 800;
        color: #04335a;
        margin-bottom: 4px;
        padding-top: 10px;
    }
    .subtitle {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 20px;
        font-size: 16px;
    }
    /* 卡片樣式 */
    .card {
        background: var(--card-bg);
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(11,92,255,0.08);
        margin-bottom: 25px;
    }
    /* 區塊標題 */
    .card .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #04335a;
        margin-bottom: 15px;
        border-bottom: 2px solid #eaeef7;
        padding-bottom: 5px;
    }
    /* 按鈕樣式 */
    .stButton>button {
        border-radius: 8px;
        height: 38px;
        font-size: 15px;
    }
    .small-muted { color: var(--muted); font-size: 14px; }
    hr { border: 0; height: 1px; background: #eaeef7; margin: 14px 0; }
    
    /* 列表式倉位顯示的樣式 */
    .position-row-text {
        font-size: 16px;
        padding: 5px 0;
    }
    .position-nowrap {
        white-space: nowrap;
    }
    .buy-color { color: #0b5cff; font-weight: bold; }
    .sell-color { color: #cf1322; font-weight: bold; }
    
    /* 策略 A/B 顏色加深 */
    .strategy-a-bg { background-color: #a7d9f7; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    .strategy-b-bg { background-color: #c0f2c0; padding: 0 4px; border-radius: 4px; font-weight: bold; }
    
    /* 核心修正：針對 st.expander 內的元素進行精確間距調整，解決重疊問題 */
    div[data-testid="stExpander"] {
        margin-top: 5px; 
    }
    div[data-testid="stExpander"] > div:nth-child(2) {
        padding-top: 10px;
    }
    /* 更穩定地修正 Expander 內文字體重疊 */
    div[data-testid="stExpander"] > div:first-child {
        margin-bottom: 5px; 
    }
    .stMarkdown {
        margin-top: 0;
        margin-bottom: 0;
    }
    /* 確保標題和副標題不被其他元件擠壓 */
    .title, .subtitle {
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
RISK_FREE_RATE = 0.015 

# ======== 網路資料抓取函式 (使用 yfinance) ========
@st.cache_data(ttl=600)
def get_tse_index_price(ticker="^TWII"):
    """
    從 Yahoo Finance 獲取加權指數的最新價格 (透過 yfinance 函式庫)
    """
    try:
        tse_ticker = yf.Ticker(ticker)
        info = tse_ticker.info
        
        price = info.get('regularMarketPrice')
        
        if price is None:
            price = info.get('regularMarketPreviousClose')

        if price and price > 1000:
            return float(price)
            
        st.warning(f"⚠️ 無法從 {ticker} 獲取有效價格，將使用備用值。", icon="⚠️")
        return None
        
    except Exception as e:
        st.error(f"❌ 透過 yfinance 抓取指數價格失敗：{e}", icon="❌")
        return None

# ======== Black-Scholes 模型函式 (保留但已不再使用於頁面顯示) ========
def black_scholes_model(S, K, T, r, sigma, option_type):
    """
    Black-Scholes 模型計算選擇權理論價格
    """
    if T <= 0 or sigma == 0:
        if option_type == 'C':
            return max(0, S - K)
        else: # P
            return max(0, K - S)
    
    S = max(1e-6, S)
    K = max(1e-6, K)
    T = max(1e-6, T)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return 0.0
    
    return price

# ======== 載入與儲存函式 (修正邏輯，不儲存/不載入 center_price) ========
def load_positions(fname=POSITIONS_FILE):
    # 此函式被修改，當讀取到 center_price 時，會忽略它，始終返回 None 作為 loaded_center
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                loaded_center = None # 舊格式或列表格式，忽略中心價
            elif isinstance(data, dict) and "positions" in data:
                df = pd.DataFrame(data["positions"])
                loaded_center = None # <--- 修正: 即使檔案中存在 center_price，我們也忽略它，確保使用最新大盤指數
            else:
                st.error("讀取儲存檔格式錯誤。", icon="❌")
                return None, None
            
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

            return df, loaded_center
        except Exception as e:
            st.error(f"讀取儲存檔失敗: {e}", icon="❌")
            return None, None
    return None, None

def save_positions(df, center_price_to_ignore, fname=POSITIONS_FILE):
    # 此函式被修改，不再將中心價寫入檔案
    try:
        data = {
            "positions": df.to_dict(orient="records")
            # 刪除 "center_price" 的寫入
        }
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"儲存失敗: {e}", icon="❌")
        return False
        
# ======== 初始化 session state (維持不變) ========
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
    
# ********* 初始化補償值 *********
if "compensation_a" not in st.session_state:
    st.session_state.compensation_a = 0.0
if "compensation_b" not in st.session_state:
    st.session_state.compensation_b = 0.0

# ********* 獲取並設定中心價 *********
if st.session_state.tse_index_price is None:
    tse_price = get_tse_index_price()
    if tse_price and tse_price > 1000:
        st.session_state.tse_index_price = tse_price
        st.sidebar.success(f"🌐 最新加權指數：{tse_price:,.2f}。", icon="✅")
    else:
        st.session_state.tse_index_price = 10000.0
        st.sidebar.info("🌐 無法獲取即時指數，使用備用中心價 10,000.0。", icon="ℹ️")

# 始終使用最新的大盤指數作為預設中心價，除非 session_state 中已經有值
if st.session_state.center_price is None:
    st.session_state.center_price = st.session_state.tse_index_price
        
# ---
## 🗃️ 倉位管理與檔案操作
# ---

# ======== 檔案操作區 (修正儲存/載入邏輯) ========
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 檔案操作與清理</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔄 載入倉位", use_container_width=True):
            df, loaded_center = load_positions()
            if df is not None:
                st.session_state.positions = df
                # loaded_center 會是 None，不會覆蓋最新的大盤指數
                st.success(f"✅ 已從檔案載入倉位，中心價使用最新大盤指數")
            else:
                st.info("找不到儲存檔或檔案為空。")
    with col2:
        if st.button("💾 儲存倉位", use_container_width=True):
            if not st.session_state.positions.empty:
                # 即使這裡傳入 center，save_positions 也不會將其寫入檔案
                current_center = st.session_state.get("simulation_center_price_input")
                center_to_save = current_center if current_center is not None else st.session_state.center_price
                
                ok = save_positions(st.session_state.positions, center_to_save) # center_to_save 將被忽略
                if ok:
                    # 關鍵修正: 移除這行，不覆蓋 st.session_state.center_price
                    # st.session_state.center_price = center_to_save
                    st.success(f"✅ 已儲存到 {POSITIONS_FILE}，中心價將保持為最新大盤指數。")
                else:
                    st.info("目前沒有倉位可儲存。")
    with col3:
        if st.button("🧹 清空所有倉位", use_container_width=True):
            st.session_state.positions = pd.DataFrame(columns=[
                "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價"
            ])
            st.session_state._edit_index = -1
            st.session_state.target_prices = []
            # 清空後，將中心價重設為最新的大盤指數
            st.session_state.center_price = st.session_state.tse_index_price
            st.session_state.compensation_a = 0.0 # 清空補償值
            st.session_state.compensation_b = 0.0 # 清空補償值
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
            details += f"{strike_val:,.1f} {row['選擇權類型']}" if strike_val != "" else f"--- {row['選擇權類型']}"
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

    # 編輯功能 (維持不變)
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

    # ======== 損益計算基礎（側邊欄）(新增補償值) ========
    
    st.sidebar.markdown('## 🛠️ 損益模擬設定')
    center = st.sidebar.number_input(
        "價平中心價 (Center)",
        # 這裡從 session_state.center_price 獲取，該值在啟動時會被最新的大盤指數初始化
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
    
    # ********* 新增補償值輸入框 *********
    st.sidebar.markdown("---")
    st.sidebar.markdown('### 💰 已實現損益 (補償值)')
    compensation_a = st.sidebar.number_input(
        "策略 A 補償值 (已平倉損益)",
        value=st.session_state.compensation_a,
        step=100.0,
        key="compensation_a_input",
        help="加入已實現的獲利/虧損，讓曲線圖平移。"
    )
    compensation_b = st.sidebar.number_input(
        "策略 B 補償值 (已平倉損益)",
        value=st.session_state.compensation_b,
        step=100.0,
        key="compensation_b_input",
        help="加入已實現的獲利/虧損，讓曲線圖平移。"
    )
    st.session_state.compensation_a = compensation_a
    st.session_state.compensation_b = compensation_b
    # ********* 補償值輸入框結束 *********
    
    
    st.sidebar.markdown(f"""
    <div style='font-size:14px; margin-top: 15px;'>
        <p><b>中心價:</b> <span style="color:#04335a; font-weight:700;">{center:,.1f}</span></p>
        <p><b>模擬範圍:</b> <span style="color:#04335a; font-weight:700;">±{PRICE_RANGE} 點</span></p>
        <hr>
        <p><b>策略 A 補償:</b> <span style="color:#0b5cff; font-weight:700;">{compensation_a:,.0f}</span></p>
        <p><b>策略 B 補償:</b> <span style="color:#2aa84f; font-weight:700;">{compensation_b:,.0f}</span></p>
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

    a_profits_raw, b_profits_raw = [], []
    for p in prices:
        a_df = positions_df[positions_df["策略"]=="策略 A"]
        b_df = positions_df[positions_df["策略"]=="策略 B"]
        a_val = a_df.apply(lambda r: profit_for_row_at_price(r,p), axis=1).sum()
        b_val = b_df.apply(lambda r: profit_for_row_at_price(r,p), axis=1).sum()
        a_profits_raw.append(a_val)
        b_profits_raw.append(b_val)
        
    # ********* 加入補償值 *********
    a_profits_total = [p + compensation_a for p in a_profits_raw]
    b_profits_total = [p + compensation_b for p in b_profits_raw]
    # ********* 加入補償值結束 *********

    # ======== 損益曲線圖 & 表格 (更新為總損益) ========
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 損益曲線與詳表</div>', unsafe_allow_html=True)

    col_chart, col_download = st.columns([3,1])
    with col_chart:
        st.subheader("📈 損益曲線（含補償值）")
        fig, ax = plt.subplots(figsize=(10,5))
        # 繪製總損益曲線
        ax.plot(prices, a_profits_total, label=f"策略 A 總損益 (含 {compensation_a:,.0f} 補償)", linewidth=2, color="#0b5cff")
        ax.plot(prices, b_profits_total, label=f"策略 B 總損益 (含 {compensation_b:,.0f} 補償)", linewidth=2, color="#2aa84f")
        
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(center, color="gray", linestyle=":", linewidth=1)
        ax.set_xlim(center-PRICE_RANGE, center+PRICE_RANGE)
        
        ax.set_xlabel("結算價", fontsize=12)
        ax.set_ylabel("損益金額", fontsize=12)
        ax.set_title(f"策略 A / 策略 B 總損益曲線（價平 {center:.1f} ±{int(PRICE_RANGE)}）", fontsize=14)
        
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    # ********* 更新表格欄位 *********
    table_df = pd.DataFrame({
        "價格": prices,
        "相對於價平(點)": [int(p-center) for p in prices],
        "策略 A 原始損益": a_profits_raw,
        "策略 B 原始損益": b_profits_raw,
        "策略 A 總損益": a_profits_total,
        "策略 B 總損益": b_profits_total
    }).sort_values(by="價格", ascending=False).reset_index(drop=True)

    def color_profit(val):
        try: f=float(val)
        except: return ''
        if f>0: return 'color: #008000; font-weight: 600;'
        elif f<0: return 'color: #cf1322; font-weight: 600;'
        return ''
        
    def color_total_profit(val):
        try: f=float(val)
        except: return ''
        if f>0: return 'background-color: #d8f5e2; color: #008000; font-weight: 700;'
        elif f<0: return 'background-color: #ffe6e8; color: #cf1322; font-weight: 700;'
        return ''
        
    styled_table = table_df.style.format({
        "價格": "{:,.1f}",
        "相對於價平(點)": "{:+d}",
        "策略 A 原始損益": "{:,.0f}",
        "策略 B 原始損益": "{:,.0f}",
        "策略 A 總損益": "**{:,.0f}**",
        "策略 B 總損益": "**{:,.0f}**"
    }).applymap(color_profit, subset=["策略 A 原始損益","策略 B 原始損益"]) \
      .applymap(color_total_profit, subset=["策略 A 總損益","策略 B 總損益"])
    
    st.markdown(f"<div class='small-muted'>每 {int(PRICE_STEP)} 點損益表（價平 {center:,.1f} ±{int(PRICE_RANGE)}）</div>", unsafe_allow_html=True)
    st.table(styled_table)

    with col_download:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        # 匯出所有欄位
        csv = table_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 模擬損益 CSV", data=csv, file_name="profit_table_with_compensation.csv", mime="text/csv", use_container_width=True)
    # ********* 更新表格欄位結束 *********
    
    st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================
    # 💵 最終結算損益分析 (更新為總損益)
    # ==========================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💵 假設結算損益分析 (微台+選擇權)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:14px; margin-bottom: 10px; color:#cf1322;'>
        此計算假設**目標到價**即為**最終結算價** (時間價值歸零)，並計算所有部位的損益。
        <b><span style='color:#0b5cff;'>策略 A 補償值: {compensation_a:,.0f}</span></b> / 
        <b><span style='color:#2aa84f;'>策略 B 補償值: {compensation_b:,.0f}</span></b>
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
        
        # 進行計算 (使用原有的 profit_for_row_at_price 函數，它計算的就是到期結算損益)
        for tp in st.session_state.target_prices:
            a_df = positions_df[positions_df["策略"]=="策略 A"]
            b_df = positions_df[positions_df["策略"]=="策略 B"]
            a_raw_val = a_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1).sum()
            b_raw_val = b_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1).sum()
            
            a_total_val = a_raw_val + compensation_a
            b_total_val = b_raw_val + compensation_b
            total_sum = a_total_val + b_total_val
            
            rows.append({
                "結算價": tp, 
                "相對於價平(點)": int(tp-center), 
                "策略 A 原始損益": a_raw_val, 
                "策略 B 原始損益": b_raw_val,
                "策略 A 總損益": a_total_val, 
                "策略 B 總損益": b_total_val,
                "總計 (A+B)": total_sum
            })
            
            # 詳情計算
            combined_df = positions_df.copy() # 使用全部部位
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
            "策略 A 原始損益": "{:,.0f}",
            "策略 B 原始損益": "{:,.0f}",
            "策略 A 總損益": "{:,.0f}", 
            "策略 B 總損益": "{:,.0f}",
            "總計 (A+B)": "**{:,.0f}**"
        }).applymap(color_target_profit, subset=["總計 (A+B)"]) \
          .applymap(color_profit, subset=["策略 A 原始損益","策略 B 原始損益", "策略 A 總損益", "策略 B 總損益"])
        
        st.subheader("🎯 目標結算價總損益一覽")
        st.dataframe(styled_target, use_container_width=True)

        csv2 = target_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 結算損益 CSV", data=csv2, file_name="settlement_profit_with_compensation.csv", mime="text/csv", key="download_target_csv")

        st.markdown("---")
        st.subheader("📝 **每筆倉位**在目標結算價下的損益明細")
        
        # 檢查是否有任何部位
        if not positions_df.empty:
            for tp in st.session_state.target_prices:
                total_profit_tp = target_df[target_df['結算價']==tp]['總計 (A+B)'].iloc[0]
                st_class = "color: #0b5cff;" if total_profit_tp > 0 else "color: #cf1322;"
                
                expander_label = f"🔍 結算價 {tp:,.1f} — 總損益 (含補償值)：{total_profit_tp:,.0f} (點擊展開)"
                
                with st.expander(expander_label, expanded=False):
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 10px; padding: 5px 10px; background-color: #f0f8ff; border-radius: 6px; border-left: 5px solid #0b5cff;'>
                        <b>目標結算價: {tp:,.1f}</b> / 
                        <b>總損益 (含補償): <span style='{st_class}'>{total_profit_tp:,.0f}</span></b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    df_detail = per_position_details[tp].copy()
                    df_detail_display = df_detail.reset_index(drop=True)
                    # 調整欄位名稱以符合結算邏輯
                    df_detail_display = df_detail_display[[
                        "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價", "結算損益"
                    ]]

                    # 新增補償值列
                    compensation_rows = []
                    if compensation_a != 0:
                        compensation_rows.append({
                            "策略": "策略 A", "商品": "已實現", "選擇權類型": "---", "履約價": "", 
                            "方向": "---", "口數": 1, "成交價": 0.0, "結算損益": compensation_a
                        })
                    if compensation_b != 0:
                         compensation_rows.append({
                            "策略": "策略 B", "商品": "已實現", "選擇權類型": "---", "履約價": "", 
                            "方向": "---", "口數": 1, "成交價": 0.0, "結算損益": compensation_b
                        })
                    
                    if compensation_rows:
                        df_comp = pd.DataFrame(compensation_rows)
                        df_detail_display = pd.concat([df_detail_display, df_comp], ignore_index=True)
                        df_detail_display = df_detail_display.sort_values(by=['策略', '商品'], ascending=[True, False]).reset_index(drop=True)

                    def color_detail_profit(val):
                        try: f=float(val)
                        except: return ''
                        if f>0: return 'color: #0b5cff; font-weight: 700;'
                        elif f<0: return 'color: #cf1322; font-weight: 700;'
                        return ''

                    styled_detail = df_detail_display.style.format({
                        "履約價": lambda v: f"{v:,.1f}" if v != "" else "",
                        "成交價": "{:,.2f}",
                        "口數": lambda v: f"{int(v)}" if v > 1 else "",
                        "結算損益": "{:,.0f}" # 單位是金額
                    }).applymap(color_detail_profit, subset=["結算損益"])

                    def color_strategy_detail(val):
                        if val == "策略 A": return 'background-color: #a7d9f7;'
                        elif val == "策略 B": return 'background-color: #c0f2c0;'
                        return ''
                    styled_detail = styled_detail.applymap(color_strategy_detail, subset=["策略"])


                    st.dataframe(styled_detail, use_container_width=True)
        else:
            st.info("目前無倉位，僅顯示補償值。")

    st.markdown("</div>", unsafe_allow_html=True)
