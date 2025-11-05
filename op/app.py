import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import requests 
import time 
import yfinance as yf 
from datetime import date, timedelta
from scipy.stats import norm 

# ======== 修正中文亂碼 ========
rcParams['font.sans-serif'] = ['Microsoft JhengHei']
rcParams['axes.unicode_minus'] = False

# ======== 頁面設定 ========
st.set_page_config(page_title="選擇權與微台損益模擬（即時指數版）", layout="wide")

# ======== CSS 樣式（美化） ========
st.markdown(
    """
    <style>
    :root {
        --card-bg: #ffffff;
        --page-bg: #f3f6fb;
        --accent: #0b5cff;
        --muted: #6b7280;
    }
    body { background-color: var(--page-bg); }
    /* 主標題 */
    .title {
        font-size: 28px;
        font-weight: 800;
        color: #04335a;
        margin-bottom: 4px;
        padding-top: 10px;
    }
    .subtitle {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 20px;
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
        font-size: 18px;
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
    }
    .small-muted { color: var(--muted); font-size: 13px; }
    hr { border: 0; height: 1px; background: #eaeef7; margin: 14px 0; }
    /* 讓 Streamlit Table/DataFrame 內部的顏色應用更明顯 */
    .css-1r6wy5w, .css-e370h9 {
        border-radius: 12px;
        overflow: hidden;
    }
    /* 確保 Sidebar 中的警告/成功訊息顯示更清晰 */
    .stSidebar .stAlert {
        padding: 5px 10px;
        margin-bottom: 10px;
    }
    
    /* ***** 修正後的自定義列表式倉位顯示的樣式 ***** */
    .position-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px dashed #e0e0e0;
        font-size: 14px;
    }
    /* 調整寬度比例，給細節和口數更多空間 */
    .col-strategy { width: 10%; font-weight: bold; color: #04335a; padding-left: 5px; }
    .col-details { width: 50%; } /* 增加細節空間 */
    .col-lots { width: 15%; text-align: left; font-weight: bold; } /* 增加方向/口數空間 */
    .col-entry { width: 15%; text-align: right; color: #555; } /* 增加成交價空間 */
    .col-delete { width: 10%; text-align: right; }
    .buy-color { color: #0b5cff; }
    .sell-color { color: #cf1322; }
    
    /* 確保文字在 st.columns 內垂直居中 */
    .st-emotion-cache-p5msec {
        display: flex;
        flex-direction: column;
        justify-content: center;
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
RISK_FREE_RATE = 0.015 # 預設無風險利率 (年化 1.5%)

# ======== 網路資料抓取函式 (使用 yfinance) ========
@st.cache_data(ttl=600) # 緩存 10 分鐘，避免頻繁請求
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

# ======== Black-Scholes 模型函式 ========
def black_scholes_model(S, K, T, r, sigma, option_type):
    """
    Black-Scholes 模型計算選擇權理論價格
    S: 標的物價格 (Center Price)
    K: 履約價
    T: 剩餘時間 (年化, 例如 5/365)
    r: 無風險利率 (年化)
    sigma: 波動率 (年化)
    option_type: 'C' (Call 買權) 或 'P' (Put 賣權)
    """
    # 確保 T 不為零或負數，否則直接返回內含價值
    if T <= 0 or sigma == 0:
        if option_type == 'C':
            return max(0, S - K)
        else: # P
            return max(0, K - S)
    
    # 避免 log(0) 或 sqrt(0)
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

# ======== 載入與儲存函式 (維持不變) ========
def load_positions(fname=POSITIONS_FILE):
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                loaded_center = None 
            elif isinstance(data, dict) and "positions" in data:
                df = pd.DataFrame(data["positions"])
                loaded_center = data.get("center_price")
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

def save_positions(df, center_price, fname=POSITIONS_FILE):
    try:
        data = {
            "center_price": center_price, 
            "positions": df.to_dict(orient="records")
        }
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"儲存失敗: {e}", icon="❌")
        return False
        
# ======== 初始化 session state ========
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
            df, loaded_center = load_positions() 
            if df is not None:
                st.session_state.positions = df
                if loaded_center is not None:
                    st.session_state.center_price = loaded_center 
                    st.success(f"✅ 已從檔案載入倉位及中心價 {loaded_center:,.1f}")
                else:
                    st.success("✅ 已從檔案載入倉位，中心價使用預設值")
            else:
                st.info("找不到儲存檔或檔案為空。")
    with col2:
        if st.button("💾 儲存倉位", use_container_width=True):
            if not st.session_state.positions.empty:
                current_center = st.session_state.get("simulation_center_price_input")
                center_to_save = current_center if current_center is not None else st.session_state.center_price
                
                ok = save_positions(st.session_state.positions, center_to_save)
                if ok:
                    st.session_state.center_price = center_to_save 
                    st.success(f"✅ 已儲存到 {POSITIONS_FILE}，中心價 {center_to_save:,.1f} 已記錄")
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
            st.success("已清空所有倉位與狀態。")
    st.markdown("</div>", unsafe_allow_html=True)

# ======== 新增倉位 (使用 session state center_price) ========
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">➕ 新增倉位 (建立持倉)</div>', unsafe_allow_html=True)

# 1. 策略和商品必須在 form 之外，才能讓商品選擇即時更新
col_strat, col_prod = st.columns(2)
with col_strat:
    new_strategy = st.selectbox("策略", ["策略 A", "策略 B"], key="new_strategy_outside") 
with col_prod:
    new_product = st.selectbox("商品", ["微台", "選擇權"], key="new_product_outside") 

# 2. 選擇權類型和履約價的條件式渲染 (依然在 form 之外)
strike_default = round(st.session_state.center_price / 100) * 100 
new_opt_type = ""
new_strike = "" 

if st.session_state.new_product_outside == "選擇權":
    st.markdown("---") # 分隔線讓選擇權欄位更清晰
    st.markdown("##### 選擇權細節")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        new_opt_type = st.selectbox("選擇權類型", ["買權", "賣權"], key="new_opt_type_outside")
    with opt_col2:
        new_strike = st.number_input("履約價", min_value=0.0, step=0.5, value=float(strike_default), key="new_strike_outside") 
    st.markdown("---") 

# 3. 將其餘輸入放入 st.form，並使用 form key 確保數據在提交時被收集
with st.form(key="add_position_form"):
    
    # 調整：將方向、口數、成交價放在三欄
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"**策略：** `{st.session_state.new_strategy_outside}`")
        new_direction = st.radio("方向", ["買進", "賣出"], horizontal=True, key="new_direction_inside")
        
    with c2:
        st.markdown(f"**商品：** `{st.session_state.new_product_outside}`")
        new_lots = st.number_input("口數", min_value=1, step=1, value=1, key="new_lots_inside")
        
    with c3:
        if st.session_state.new_product_outside == "選擇權":
             st.markdown(f"**類型：** `{st.session_state.new_opt_type_outside}` / **履約價：** `{st.session_state.new_strike_outside:,.1f}`")
        else:
             st.markdown(f"**<div style='height: 19.5px;'></div>**", unsafe_allow_html=True) # 調整間距
             
        new_entry = st.number_input("成交價（權利金或口數成交價）", min_value=0.0, step=0.5, value=0.0, key="new_entry_inside")
        
    # 提交按鈕
    submitted = st.form_submit_button("✅ 新增倉位 (加入持倉)", use_container_width=True)
    
    if submitted:
        
        # 從 form 外的 session_state 獲取條件式的值
        product_value = st.session_state.new_product_outside
        strategy_value = st.session_state.new_strategy_outside
        
        if product_value == "選擇權":
             # 從 form 外的 key 獲取值
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
        st.rerun() # 新增後刷新，確保列表立即更新

st.markdown("</div>", unsafe_allow_html=True)

# ======== 持倉明細 & 編輯/刪除 (列表式顯示和行旁按鈕) ========
positions_df = st.session_state.positions.copy()
if positions_df.empty:
    st.info("尚無任何倉位資料，請先新增或從檔案載入。")
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 現有持倉明細與快速移除</div>', unsafe_allow_html=True)
    
    # 標題行 (使用 st.columns 模擬標題，與下方內容對齊)
    c_strat_h, c_details_h, c_lots_h, c_entry_h, c_delete_h = st.columns([1, 5, 1.5, 1.5, 1])
    c_strat_h.markdown("策略", unsafe_allow_html=True)
    c_details_h.markdown("細節 (索引/商品/類型/履約價)", unsafe_allow_html=True)
    c_lots_h.markdown("方向/口數", unsafe_allow_html=True)
    c_entry_h.markdown("<div style='text-align: right;'>成交價</div>", unsafe_allow_html=True)
    c_delete_h.markdown("<div style='text-align: right;'>操作</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    
    # 使用迴圈遍歷 DataFrame 的每一行 (iterrows 包含 index)
    for index, row in positions_df.iterrows():
        
        # 1. 組裝詳細資訊字串
        details = f"({index}) {row['商品']} / "
        if row['商品'] == "選擇權":
            strike_val = row['履約價']
            details += f"{row['選擇權類型']} @ {strike_val:,.1f}" if strike_val != "" else f"{row['選擇權類型']} @ ---"
        else:
            details += f"---"
        
        # 2. 決定方向顏色
        direction_style = "buy-color" if row['方向'] == "買進" else "sell-color"
        
        # 3. 使用 st.columns 創建互動式佈局 (與標題行比例保持一致)
        c_strat, c_details, c_lots, c_entry, c_delete = st.columns([1, 5, 1.5, 1.5, 1])

        # 使用自定義的 CSS class 來控制垂直對齊
        with c_strat:
            st.markdown(f'<div class="col-strategy">{row["策略"]}</div>', unsafe_allow_html=True)

        with c_details:
            st.markdown(f'<div class="col-details">{details}</div>', unsafe_allow_html=True)
            
        with c_lots:
            # 使用 HTML 標記來控制方向和口數的樣式
            st.markdown(f'<div class="col-lots {direction_style}">{row["方向"]} {row["口數"]} 口</div>', unsafe_allow_html=True)
            
        with c_entry:
            st.markdown(f'<div class="col-entry" style="text-align: right;">{row["成交價"]:,.2f}</div>', unsafe_allow_html=True)

        with c_delete:
            # 💥 關鍵：使用唯一的 key，點擊後觸發刪除操作
            if st.button("移除", key=f"delete_btn_{index}", type="secondary", use_container_width=True):
                # 執行刪除操作 (使用索引刪除，不會錯亂)
                st.session_state.positions = st.session_state.positions.drop(index).reset_index(drop=True)
                st.toast(f"✅ 已移除 (索引 {index}) 倉位！")
                st.rerun() # 刪除後立即刷新頁面以更新列表
        
        # 模擬分隔線
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)

    # 編輯功能 (改為使用 Selectbox 選擇索引)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ 編輯倉位 (索引式)</div>', unsafe_allow_html=True)
    
    current_indices = positions_df.index.tolist()
    
    with st.expander("✏️ 編輯單列倉位"):
        
        col_idx, col_load = st.columns([1,2])
        
        if current_indices:
            # 確保 _edit_index 初始值在有效範圍內
            if st.session_state._edit_index == -1 and current_indices:
                 st.session_state._edit_index = current_indices[0]
                 
            with col_idx:
                # 使用 selectbox 確保用戶選擇的是有效的現有索引
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
            
            # 檢查索引是否有效
            if idx in positions_df.index:
                st.markdown(f"**👉 編輯索引 {idx} 的倉位（修改後按 儲存修改）**")
                # 由於 st.session_state.positions 已經被 drop 掉，這裡需要從原始的 positions_df 獲取行
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

                    # 條件式渲染選擇權欄位
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
                        # 直接修改該索引的行
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
    
# 損益計算僅在有倉位時進行
if not positions_df.empty:

    # ======== 損益計算基礎（側邊欄）========
    
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
    
    st.sidebar.markdown(f"""
    <div style='font-size:14px; margin-top: 15px;'>
        <p><b>中心價:</b> <span style="color:#04335a; font-weight:700;">{center:,.1f}</span></p>
        <p><b>模擬範圍:</b> <span style="color:#04335a; font-weight:700;">±{PRICE_RANGE} 點</span></p>
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
            return (price - entry) * lots * multiplier if direction == "買進" else (entry - price) * lots * multiplier
        else:
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

    # ======== 損益曲線圖 & 表格 ========
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
        ax.set_xlabel("結算價")
        ax.set_ylabel("損益金額")
        ax.set_title(f"策略 A / 策略 B 損益曲線（價平 {center:.1f} ±{int(PRICE_RANGE)}）")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    # ======== 損益表 (使用 st.table 確保完全展開) ========
    table_df = pd.DataFrame({
        "價格": prices,
        "相對於價平(點)": [int(p-center) for p in prices],
        "策略 A 損益": a_profits,
        "策略 B 損益": b_profits
    }).sort_values(by="價格", ascending=False).reset_index(drop=True)

    def color_profit(val):
        try: f=float(val)
        except: return ''
        if f>0: return 'background-color: #d8f5e2'
        elif f<0: return 'background-color: #ffe6e8'
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


    # ======== 到價損益 (維持不變) ========
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 到價損益分析</div>', unsafe_allow_html=True)
    
    col_input, col_add, col_remove = st.columns([2,1,2])
    with col_input:
        add_price = st.number_input("輸入目標到價", value=float(center), step=0.5, key="add_price_input")
    with col_add:
        if st.button("➕ 加入到價", use_container_width=True):
            v = float(add_price)
            if v not in st.session_state.target_prices:
                st.session_state.target_prices.append(v)
                st.session_state.target_prices.sort(reverse=True)
            st.toast(f"已加入到價: {v:.1f}")
    with col_remove:
        if st.session_state.target_prices:
            to_remove = st.selectbox("選擇要移除的到價", options=["無"] + [f"{p:,.1f}" for p in st.session_state.target_prices])
            if st.button("🗑️ 移除選定到價", type="secondary", use_container_width=True):
                if to_remove != "無":
                    val = float(to_remove.replace(',', ''))
                    st.session_state.target_prices = [p for p in st.session_state.target_prices if p != val]
                    st.toast(f"已移除到價 {val:,.1f}")
    
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
            rows.append({"到價": tp, "相對於價平(點)": int(tp-center), "策略 A 損益": a_val, "策略 B 損益": b_val, "總損益": total_val})
            
            combined_df = pd.concat([a_df, b_df], ignore_index=True).reset_index(drop=True)
            combined_df["到價損益"] = combined_df.apply(lambda r: profit_for_row_at_price(r, tp), axis=1)
            per_position_details[tp] = combined_df

        target_df = pd.DataFrame(rows).sort_values(by="到價", ascending=False).reset_index(drop=True)

        def color_target_profit(val):
            try: f=float(val)
            except: return ''
            if f>0: return 'background-color: #e6faff'
            elif f<0: return 'background-color: #fff0f0'
            return ''

        styled_target = target_df.style.format({
            "到價": "{:,.1f}",
            "相對於價平(點)": "{:+d}",
            "策略 A 損益": "{:,.0f}",
            "策略 B 損益": "{:,.0f}",
            "總損益": "**{:,.0f}**"
        }).applymap(color_target_profit, subset=["總損益"]).applymap(color_profit, subset=["策略 A 損益","策略 B 損益"])
        
        st.subheader("到價總損益一覽")
        st.dataframe(styled_target, use_container_width=True) 

        csv2 = target_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 匯出 到價損益 CSV", data=csv2, file_name="target_profit.csv", mime="text/csv", key="download_target_csv")

        st.markdown("---")
        st.subheader("每筆倉位在目標價的損益明細")
        for tp in st.session_state.target_prices:
            total_profit_tp = target_df[target_df['到價']==tp]['總損益'].iloc[0]
            st_class = "color: #0b5cff;" if total_profit_tp > 0 else "color: #cf1322;"
            
            # 使用純文字作為 st.expander 標籤，避免 TypeError
            expander_label = f"🔍 到價 {tp:,.1f} — 總損益：{total_profit_tp:,.0f} (點擊展開)"
            
            with st.expander(expander_label, expanded=False): 
                
                # 在展開區塊內，使用 st.markdown 顯示美化後的標題
                st.markdown(f"""
                <div style='margin-bottom: 10px; padding: 5px 10px; background-color: #f0f8ff; border-radius: 6px; border-left: 5px solid #0b5cff;'>
                    <b>目標到價: {tp:,.1f}</b> / 
                    <b>總損益: <span style='{st_class}'>{total_profit_tp:,.0f}</span></b>
                </div>
                """, unsafe_allow_html=True)
                
                df_detail = per_position_details[tp].copy()
                df_detail_display = df_detail.reset_index(drop=True)
                df_detail_display = df_detail_display[[
                    "策略", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價", "到價損益"
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
                    "到價損益": "{:,.0f}"
                }).applymap(color_detail_profit, subset=["到價損益"])

                st.dataframe(styled_detail, use_container_width=True)
    else:
        st.markdown("<div class='small-muted' style='margin-top:8px'>尚未設定到價，請新增到價以查看到價損益。</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
    
    # ---
    ## ⏳ 選擇權時間價值分析 (逐日遞減)
    # ---

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏳ 選擇權時間價值分析 (逐日遞減)</div>', unsafe_allow_html=True)
    
    # 篩選出所有選擇權倉位
    options_df = positions_df[positions_df["商品"] == "選擇權"].copy().reset_index()
    
    if options_df.empty:
        st.info("目前無選擇權倉位，此功能僅適用於選擇權。")
    else:
        st.sidebar.markdown('---')
        st.sidebar.markdown('## ⏳ 選擇權估值')
        
        # 1. 波動率輸入
        volatility = st.sidebar.number_input(
            "假設年化波動率 (IV, %)", 
            value=15.0, 
            min_value=1.0, 
            max_value=100.0, 
            step=1.0,
            format="%.1f",
            key="iv_input",
            help="請輸入您對市場預期的波動率百分比 (例如 15 表示 15%)"
        ) / 100.0 # 轉換為小數
        
        # 2. 結算日期輸入
        # 確保預設值不會導致 initial_days <= 0
        default_settle_date = date.today() + timedelta(days=5)
        
        settle_date = st.sidebar.date_input(
            "預計結算日期 (到期日)",
            value=default_settle_date,
            min_value=date.today() + timedelta(days=1),
            key="settle_date_input",
            help="選擇您想模擬的結算日期，必須晚於今天"
        )
        
        # 3. 模擬天數範圍
        initial_days = (settle_date - date.today()).days
        days_to_simulate = st.number_input(
            "模擬天數 (N 天內，從結算日前 N 天開始)",
            value=min(5, initial_days) if initial_days > 0 else 1,
            min_value=1,
            max_value=initial_days if initial_days > 0 else 1,
            step=1
        )
        
        # 4. 剩餘天數計算
        days_to_expiry_start = initial_days
        
        st.sidebar.markdown(f"""
        <div style='font-size:14px; margin-top: 15px;'>
            <p><b>當前剩餘:</b> <span style="color:#cf1322; font-weight:700;">{days_to_expiry_start} 天</span></p>
            <p><b>模擬 IV (σ):</b> <span style="color:#0b5cff; font-weight:700;">{volatility*100:.1f} %</span></p>
        </div>
        """, unsafe_allow_html=True)

        if initial_days <= 0:
            st.warning("⚠️ 預計結算日期必須晚於今天。", icon="⚠️")
        else:
            
            # --- 開始逐日疊代計算 ---
            daily_results = []
            
            # 模擬從結算日倒數 N 天開始
            simulation_start_date = settle_date - timedelta(days=days_to_simulate - 1)
            
            # 確保模擬開始日不早於今天
            if simulation_start_date < date.today():
                 simulation_start_date = date.today()
            
            # 建立要模擬的日期範圍
            sim_dates = []
            temp_date = simulation_start_date
            while temp_date <= settle_date:
                # 排除週末，但若結算日剛好是週末，則保留結算日
                if temp_date.weekday() < 5 or temp_date == settle_date: # 0-4 是週一到週五
                    sim_dates.append(temp_date)
                temp_date += timedelta(days=1)
            
            # 初始化前一天的理論價值，用於計算每日 Theta 損耗
            previous_total_value = None

            for sim_date in sim_dates:
                
                # 計算當前模擬日期距離結算日的剩餘天數
                days_left = (settle_date - sim_date).days
                time_to_expiry = days_left / 365.0
                
                # 排除 T < 0 的情況
                if time_to_expiry < 0:
                    continue
                
                current_total_profit_loss = 0.0
                current_total_value_sum = 0.0 # 總持倉的理論市值

                for index, row in options_df.iterrows():
                    
                    K = float(row["履約價"]) if row["履約價"] != "" else center # 確保履約價是 float
                    opt_code = 'C' if row["選擇權類型"] == '買權' else 'P'
                    is_buy = row["方向"] == "買進"
                    
                    # 1. 計算理論價 (在該天結束時的價值)
                    # 標的物價格 S 假設不變，為 center
                    theo_price = black_scholes_model(center, K, time_to_expiry, RISK_FREE_RATE, volatility, opt_code)
                    
                    # 2. 累加倉位市值 (用來計算 Theta 每日價值差異)
                    current_value_row = theo_price * row["口數"] * MULTIPLIER_OPTION
                    current_total_value_sum += current_value_row # 總市值

                    # 3. 計算總持倉的損益 (現值 vs 成本)
                    cost_value = row["成交價"] * row["口數"] * MULTIPLIER_OPTION
                    
                    if is_buy:
                        profit_loss = current_value_row - cost_value
                    else:
                        profit_loss = cost_value - current_value_row
                    
                    current_total_profit_loss += profit_loss
                
                # 計算每日 Theta 效應 (每日市值變化)
                daily_theta_effect = 0.0
                if previous_total_value is not None:
                    # Theta 損耗 = 前一天的理論市值 - 當天的理論市值
                    daily_theta_effect = previous_total_value - current_total_value_sum

                # 該行資料 (每日彙總)
                daily_results.append({
                    "模擬日期": sim_date.strftime("%Y-%m-%d"),
                    "剩餘天數": days_left,
                    "持倉總市值": current_total_value_sum,
                    "當日時間損耗(Theta)": daily_theta_effect,
                    "當前總損益(現值-成本)": current_total_profit_loss
                })
                
                # 更新前一天的總市值
                previous_total_value = current_total_value_sum
                

            results_df = pd.DataFrame(daily_results)
            
            st.markdown(f"#### 結算日：{settle_date.strftime('%Y-%m-%d')} (模擬結算價: {center:,.1f})")
            
            # 格式化表格
            def color_pl(val):
                try: f=float(val)
                except: return ''
                if f>0: return 'background-color: #d8f5e2; color: #0b5cff; font-weight: 700;'
                elif f<0: return 'background-color: #ffe6e8; color: #cf1322; font-weight: 700;'
                return ''
            
            # 突出顯示 Theta 每日損耗的列
            def color_theta(val):
                if val > 0: return 'background-color: #fffbe6; font-weight: 700;' # 正值 (買方虧損/賣方獲利)
                if val < 0: return 'background-color: #e6f7ff; font-weight: 700;' # 負值 (買方獲利/賣方虧損 - 除非賣方付錢)
                return ''

            styled_results = results_df.style.format({
                "持倉總市值": "{:,.0f}",
                "當日時間損耗(Theta)": "{:,.0f}",
                "當前總損益(現值-成本)": "{:,.0f}",
            }).applymap(color_pl, subset=["當前總損益(現值-成本)"]).applymap(color_theta, subset=["當日時間損耗(Theta)"])
            
            st.dataframe(styled_results, use_container_width=True)
            
            st.caption(f"""
            **說明：**
            * **模擬結算價**：假設指數每天都停留在 **{center:,.1f}**。
            * **當前總損益**：該日持倉的**理論現值**與您**原始成本**的損益差異。
            * **當日時間損耗(Theta)**：前一天總市值減去當天總市值。這個值通常為正值，代表時間經過一天，持倉價值（對買方不利，對賣方有利）的自然損耗。
            """)

    st.markdown("</div>", unsafe_allow_html=True)
