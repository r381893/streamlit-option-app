import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import requests 
import time 

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📈 選擇權與微台損益模擬（即時指數版）</div>'
            '<div class="subtitle">自動抓取加權指數，作為價平中心點進行模擬</div>', unsafe_allow_html=True)

# ======== 設定常數 ========
POSITIONS_FILE = "positions_store.json"
MULTIPLIER_MICRO = 10.0
MULTIPLIER_OPTION = 50.0
PRICE_STEP = 100.0

# ======== 網路資料抓取函式 ========
@st.cache_data(ttl=600) # 緩存 10 分鐘，避免頻繁請求
def get_tse_index_price(ticker="^TWII"):
    """
    從 Yahoo Finance 獲取加權指數的最新價格
    """
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        data = response.json()
        
        if data and 'quoteResponse' in data and 'result' in data['quoteResponse']:
            result = data['quoteResponse']['result']
            if result:
                price = result[0].get('regularMarketPrice')
                if price is not None:
                    return float(price)
                return float(result[0].get('regularMarketPreviousClose', 0.0))
        
        st.warning(f"⚠️ 無法解析 {ticker} 價格資料，將使用備用值。", icon="⚠️")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 網路請求失敗，無法抓取指數價格：{e}", icon="❌")
        return None
    except Exception as e:
        st.error(f"❌ 處理指數價格時發生錯誤：{e}", icon="❌")
        return None

# ======== 載入與儲存函式 (支援儲存 Center Price) ========
def load_positions(fname=POSITIONS_FILE):
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 判斷是新格式 (dict) 還是舊格式 (list)
            if isinstance(data, list):
                df = pd.DataFrame(data)
                loaded_center = None # 舊格式無儲存中心價
            elif isinstance(data, dict) and "positions" in data:
                df = pd.DataFrame(data["positions"])
                loaded_center = data.get("center_price")
            else:
                st.error("讀取儲存檔格式錯誤。", icon="❌")
                return None, None
            
            # 確保所有欄位都存在且型別正確
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
            "center_price": center_price, # <-- 新增儲存中心價
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
if "center_price" not in st.session_state: # <-- 追蹤並儲存價平中心價
    st.session_state.center_price = None

# ********* 獲取並設定中心價 (優先使用儲存的值，其次使用即時指數) *********
if st.session_state.tse_index_price is None:
    tse_price = get_tse_index_price()
    if tse_price and tse_price > 1000:
        st.session_state.tse_index_price = tse_price
        st.sidebar.success(f"🌐 最新加權指數：{tse_price:,.2f}。", icon="✅")
    else:
        st.session_state.tse_index_price = 10000.0 # 備用值
        st.sidebar.info("🌐 無法獲取即時指數，使用備用中心價 10,000.0。", icon="ℹ️")

# 如果 session state 中沒有儲存的中心價，則使用 TSE 指數作為首次預設值
if st.session_state.center_price is None:
    st.session_state.center_price = st.session_state.tse_index_price
        
# ======== 檔案操作區 ========
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 檔案操作與清理</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔄 載入倉位", use_container_width=True):
            df, loaded_center = load_positions() # <-- 接收載入的中心價
            if df is not None:
                st.session_state.positions = df
                if loaded_center is not None:
                    st.session_state.center_price = loaded_center # <-- 更新 session state 中的中心價
                    st.success(f"✅ 已從檔案載入倉位及中心價 {loaded_center:,.1f}")
                else:
                    st.success("✅ 已從檔案載入倉位，中心價使用預設值")
            else:
                st.info("找不到儲存檔或檔案為空。")
    with col2:
        if st.button("💾 儲存倉位", use_container_width=True):
            if not st.session_state.positions.empty:
                # 獲取目前 sidebar 輸入框的值作為要儲存的中心價
                current_center = st.session_state.get("simulation_center_price_input")
                
                # 如果 sidebar 輸入框有值，就用它；否則使用 session state 的當前值
                center_to_save = current_center if current_center is not None else st.session_state.center_price
                
                ok = save_positions(st.session_state.positions, center_to_save)
                if ok:
                    st.session_state.center_price = center_to_save # 更新 session state
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
            st.session_state.center_price = st.session_state.tse_index_price # 清空後中心價回到即時指數
            st.success("已清空所有倉位與狀態。")
    st.markdown("</div>", unsafe_allow_html=True)

# ======== 新增倉位 (不變) ========
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown('<div class="section-title">➕ 新增倉位 (建立持倉)</div>', unsafe_allow_html=True)

with st.form(key="add_position_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        new_strategy = st.selectbox("策略", ["策略 A", "策略 B"], key="new_strategy")
        new_product = st.selectbox("商品", ["微台", "選擇權"], key="new_product")
    with c2:
        new_direction = st.radio("方向", ["買進", "賣出"], horizontal=True, key="new_direction")
        new_lots = st.number_input("口數", min_value=1, step=1, value=1, key="new_lots")
    with c3:
        new_entry = st.number_input("成交價（權利金或口數成交價）", min_value=0.0, step=0.5, value=0.0, key="new_entry")
    
    if new_product == "選擇權":
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            new_opt_type = st.selectbox("選擇權類型", ["買權", "賣權"], key="new_opt_type")
        with opt_col2:
            # 履約價預設值使用當前中心價的百位數整數
            strike_default = round(st.session_state.center_price / 100) * 100 
            new_strike = st.number_input("履約價", min_value=0.0, step=0.5, value=strike_default, key="new_strike") 
    else:
        new_opt_type = ""
        new_strike = ""

    submitted = st.form_submit_button("✅ 新增倉位 (加入持倉)", use_container_width=True)
    if submitted:
        rec = {
            "策略": new_strategy,
            "商品": new_product,
            "選擇權類型": new_opt_type if new_product == "選擇權" else "",
            "履約價": float(new_strike) if new_product == "選擇權" else "",
            "方向": new_direction,
            "口數": int(new_lots),
            "成交價": float(new_entry)
        }
        st.session_state.positions = pd.concat([st.session_state.positions, pd.DataFrame([rec])], ignore_index=True)
        st.success("已新增倉位，請在下方持倉明細確認。")

st.markdown("</div>", unsafe_allow_html=True)

# ======== 持倉明細 & 編輯/刪除 (不變) ========
positions_df = st.session_state.positions.copy()
if positions_df.empty:
    st.info("尚無任何倉位資料，請先新增或從檔案載入。")
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 現有持倉明細</div>', unsafe_allow_html=True)
    
    display_df = positions_df.reset_index().rename(columns={"index": "索引"})
    
    def row_color_by_strategy(row):
        if row["策略"] == "策略 A": return ['background-color: #e6f7ff'] * len(row)
        elif row["策略"] == "策略 B": return ['background-color: #e8fff5'] * len(row)
        return [''] * len(row)

    styled_display = display_df.style.format({
        "履約價": lambda v: f"{v:,.1f}" if v != "" else "",
        "成交價": "{:,.2f}",
        "口數": "{:d}"
    }).apply(row_color_by_strategy, axis=1)

    st.dataframe(styled_display, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ 編輯與刪除倉位</div>', unsafe_allow_html=True)
    
    max_index = len(display_df) - 1
    
    with st.expander("✏️ 編輯單列倉位"):
        if max_index >= 0:
            col_idx, col_load = st.columns([1,2])
            with col_idx:
                row_to_edit = st.number_input("要編輯的索引 (0 開始)", min_value=0, max_value=max_index, value=0, step=1, key="edit_idx_input")
            with col_load:
                if st.button(f"載入索引 {int(row_to_edit)} 到編輯表單", use_container_width=True):
                    st.session_state._edit_index = int(row_to_edit)
                    st.toast(f"已載入索引 {row_to_edit} 的資料。")

            idx = st.session_state._edit_index
            if 0 <= idx <= max_index:
                st.markdown(f"**👉 編輯索引 {idx} 的倉位（修改後按 儲存修改）**")
                row = display_df.loc[idx]
                with st.form(key=f"edit_form_{idx}"):
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        f_strategy = st.selectbox("策略", ["策略 A", "策略 B"], index=0 if row["策略"] == "策略 A" else 1)
                        f_product = st.selectbox("商品", ["微台", "選擇權"], index=0 if row["商品"] == "微台" else 1)
                    with f_col2:
                        f_direction = st.selectbox("方向", ["買進", "賣出"], index=0 if row["方向"] == "買進" else 1)
                        f_lots = st.number_input("口數", value=int(row["口數"]), step=1, min_value=1)
                    with f_col3:
                        f_entry = st.number_input("成交價", value=float(row["成交價"]), step=0.1)

                    if f_product == "選擇權":
                        opt_options = ["買權", "賣權"]
                        default_opt_idx = 0 if row["選擇權類型"] == "買權" else 1
                        f_opt_type = st.selectbox("選擇權類型", opt_options, index=default_opt_idx)
                        f_strike = st.number_input("履約價", value=float(row["履約價"]) if row["履約價"] != "" else st.session_state.center_price, step=0.5)
                    else:
                        f_opt_type = ""
                        f_strike = ""
                    
                    submitted = st.form_submit_button("💾 儲存修改", use_container_width=True)
                    if submitted:
                        updated = st.session_state.positions.copy().reset_index(drop=True)
                        updated.loc[idx, ["策略","商品","選擇權類型","履約價","方向","口數","成交價"]] = [
                            f_strategy, f_product, f_opt_type, float(f_strike) if f_product=="選擇權" else "",
                            f_direction, int(f_lots), float(f_entry)
                        ]
                        st.session_state.positions = updated
                        st.session_state._edit_index = -1
                        st.success("✅ 倉位已更新，請查看上方明細。")
            else:
                st.info("請先載入要編輯的索引。")
        else:
            st.info("目前無倉位可編輯。")

    with st.expander("🗑️ 刪除單列倉位"):
        if max_index >= 0:
            del_col1, del_col2 = st.columns([1,2])
            with del_col1:
                del_index = st.number_input("輸入要刪除的索引", min_value=0, max_value=len(positions_df)-1, step=1, key="del_idx_input")
            with del_col2:
                if st.button("🗑️ 確認刪除該倉位", type="primary", use_container_width=True):
                    st.session_state.positions = positions_df.drop(int(del_index)).reset_index(drop=True)
                    st.session_state._edit_index = -1
                    st.success(f"✅ 已刪除索引 {int(del_index)} 的倉位。")
        else:
            st.info("目前無倉位可刪除。")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
# 損益計算僅在有倉位時進行
if not positions_df.empty:

    # ======== 損益計算基礎（側邊欄）========
    
    st.sidebar.markdown('## 🛠️ 損益模擬設定')
    center = st.sidebar.number_input(
        "價平中心價 (Center)", 
        value=st.session_state.center_price, # <-- 使用 session state 儲存的值作為預設值
        key="simulation_center_price_input", # <-- 設置 key 以便儲存按鈕讀取
        step=1.0, 
        help="損益曲線圖的中心點價格，預設為最新加權指數/上次儲存值"
    )
    
    PRICE_RANGE = st.sidebar.number_input(
        "模擬範圍 (±點數)", 
        value=1500, # 維持 1500 點預設
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


    # ======== 到價損益 ========
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
            
            # 修正後的 expader 標題 (使用 HTML <b> 代替 Markdown **)
            expander_title = f"🔍 <b>到價 {tp:,.1f}</b> — 總損益：<span style='{st_class}'>{total_profit_tp:,.0f}</span> (點擊展開)"
            
            with st.expander(expander_title, expanded=False, unsafe_allow_html=True): 
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
