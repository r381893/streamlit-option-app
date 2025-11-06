import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.stats import norm

# ====================================================================
# 1. 常數與設定 (Constants and Setup)
# ====================================================================

# 選擇權乘數 (台指選擇權)
MULTIPLIER_OPTION = 50 
# 無風險利率 (R) - 年化百分比，這裡假設 1.5%
RISK_FREE_RATE = 0.015 

# Streamlit 頁面設定
st.set_page_config(
    page_title="多策略選擇權倉位回測分析",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 導入自訂 CSS 樣式
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 由於我們沒有檔案，直接在代碼中定義 CSS
st.markdown("""
<style>
    /* 基礎排版優化 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* 標題與副標題 */
    .title {
        font-size: 2em;
        font-weight: bold;
        color: #04335a;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #6c757d;
        margin-bottom: 1em;
    }
    
    /* 💥 新增：修正 Streamlit 標題和 Expander 內容的間距，解決文字重疊 */
    div[data-testid="stExpander"] > div:first-child {
        margin-bottom: 5px; /* 確保 Expander 標題和下方內容有間距 */
    }
    /* 修正：消除 '編輯單列倉位' 下拉選單中可能出現的重疊 */
    div[data-testid="stExpander"] div[data-testid="stForm"] {
        padding-top: 5px; 
    }
    /* 修正：確保 st.markdown 標籤在 Expander 內有正確的邊距 */
    .stMarkdown {
        margin-top: 0;
        margin-bottom: 0;
    }
    /* 確保標題和副標題不被其他元件擠壓 */
    .title, .subtitle {
        line-height: 1.2;
    }

    /* Streamlit 訊息框調整 */
    div[data-testid="stAlert"] {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Metric 元件優化 */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* 側邊欄調整 */
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 350px;
    }

    /* 確保表格內容文字清晰 */
    .dataframe th, .dataframe td {
        white-space: nowrap !important; /* 防止文字換行 */
    }

</style>
""", unsafe_allow_html=True)

# 策略顏色定義
STRATEGY_COLORS = {
    "單買": '#1890ff',  # 藍色
    "單賣": '#fa541c',  # 紅橙色
    "多頭價差": '#7cb305',  # 綠色
    "空頭價差": '#ffc53d',  # 黃色
    "勒式": '#eb2f96',  # 粉色
    "跨式": '#597ef7',  # 淺藍色
    "其他組合": '#8c8c8c'  # 灰色
}

# 策略顏色函數 (用於 Pandas Styler)
def color_strategy(val):
    color = STRATEGY_COLORS.get(val, '#8c8c8c')
    return f'background-color: {color}; color: white; font-weight: bold;'


# ====================================================================
# 2. Black-Scholes 模型 (Black-Scholes Model)
# ====================================================================

def black_scholes_model(S, K, T, r, sigma, option_type):
    """
    Black-Scholes 選擇權定價模型
    
    參數:
    S (float): 標的資產現價 (Strike Price)
    K (float): 履約價格 (Strike Price)
    T (float): 到期時間 (年化)
    r (float): 無風險利率 (年化)
    sigma (float): 波動率 (年化)
    option_type (str): 'C' (Call 買權) 或 'P' (Put 賣權)
    
    回傳:
    float: 選擇權理論價格
    """
    if T <= 0:
        # 如果時間已到期，價格就是內含價值 (Intrinsic Value)
        if option_type == 'C':
            return max(0, S - K)
        else: # 'P'
            return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type 必須是 'C' 或 'P'")
        
    return price

# ====================================================================
# 3. 初始化與數據管理 (Initialization and Data Management)
# ====================================================================

# 初始化 Session State
if 'positions_df' not in st.session_state:
    st.session_state.positions_df = pd.DataFrame(columns=[
        "ID", "商品", "選擇權類型", "履約價", "方向", "口數", "成交價", "策略"
    ])
    st.session_state.positions_df['ID'] = st.session_state.positions_df['ID'].astype(int)

if 'next_id' not in st.session_state:
    st.session_state.next_id = 1

if 'simulation_center_price_input' not in st.session_state:
    st.session_state.simulation_center_price_input = 20000

# 策略推導函數 (簡化版)
def infer_strategy(df):
    if df.empty:
        return ""
    
    calls = df[df['選擇權類型'] == '買權']
    puts = df[df['選擇權類型'] == '賣權']
    
    # 單一倉位
    if len(df) == 1:
        return "單買" if df.iloc[0]['方向'] == '買進' else "單賣"

    # 簡單組合
    if len(df) == 2:
        # 價差組合 (Spread) - 買權或賣權
        if len(calls) == 2 or len(puts) == 2:
            strike1, strike2 = sorted(df['履約價'].unique())
            df_sorted = df.sort_values(by='履約價')
            
            # 買進低履約價，賣出高履約價
            if df_sorted.iloc[0]['方向'] == '買進' and df_sorted.iloc[1]['方向'] == '賣出':
                return "多頭價差" # Bull Call Spread 或 Bear Put Spread
            # 賣出低履約價，買進高履約價
            elif df_sorted.iloc[0]['方向'] == '賣出' and df_sorted.iloc[1]['方向'] == '買進':
                return "空頭價差" # Bear Call Spread 或 Bull Put Spread
        
        # 跨/勒式 (Strangle/Straddle) - 一買權一賣權，同到期日
        if len(calls) == 1 and len(puts) == 1:
            c_dir = calls.iloc[0]['方向']
            p_dir = puts.iloc[0]['方向']
            c_k = calls.iloc[0]['履約價']
            p_k = puts.iloc[0]['履約價']
            
            if c_dir == p_dir:
                if c_k == p_k:
                    return "跨式" # Long/Short Straddle
                else:
                    return "勒式" # Long/Short Strangle
                    
    return "其他組合" # 複雜組合或無法簡單判斷

# ====================================================================
# 4. 倉位管理函數 (Position Management Functions)
# ====================================================================

def add_position(commodity, opt_type, strike, direction, quantity, price, strategy):
    """新增單一倉位"""
    new_data = {
        "ID": st.session_state.next_id,
        "商品": commodity,
        "選擇權類型": opt_type,
        "履約價": float(strike),
        "方向": direction,
        "口數": int(quantity),
        "成交價": float(price),
        "策略": strategy
    }
    st.session_state.positions_df = pd.concat(
        [st.session_state.positions_df, pd.DataFrame([new_data])],
        ignore_index=True
    )
    st.session_state.next_id += 1

def remove_position(position_id):
    """移除單一倉位"""
    st.session_state.positions_df = st.session_state.positions_df[
        st.session_state.positions_df['ID'] != position_id
    ].reset_index(drop=True)

def edit_position_form(position_id):
    """編輯單一倉位表單"""
    with st.form(key=f'edit_form_{position_id}'):
        current_data = st.session_state.positions_df[st.session_state.positions_df['ID'] == position_id].iloc[0]
        
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1.5])
        
        new_commodity = col1.selectbox("商品", options=["選擇權", "期貨"], index=0 if current_data['商品'] == "選擇權" else 1, key=f'edit_commodity_{position_id}')
        new_opt_type = col2.selectbox("類型", options=["買權", "賣權"], index=0 if current_data['選擇權類型'] == "買權" else 1, key=f'edit_opt_type_{position_id}')
        new_strike = col3.number_input("履約價", value=current_data['履約價'], min_value=1.0, step=50.0, format="%.0f", key=f'edit_strike_{position_id}')
        new_direction = col4.selectbox("方向", options=["買進", "賣出"], index=0 if current_data['方向'] == "買進" else 1, key=f'edit_direction_{position_id}')
        new_quantity = col5.number_input("口數", value=current_data['口數'], min_value=1, step=1, key=f'edit_quantity_{position_id}')
        
        new_price = st.number_input("成交價", value=current_data['成交價'], min_value=0.01, step=0.5, format="%.2f", key=f'edit_price_{position_id}')
        new_strategy = st.text_input("策略標籤", value=current_data['策略'], key=f'edit_strategy_{position_id}')

        col_b1, col_b2 = st.columns([1, 4])
        if col_b1.form_submit_button("💾 更新"):
            update_position(position_id, new_commodity, new_opt_type, new_strike, new_direction, new_quantity, new_price, new_strategy)
            st.rerun()
        col_b2.markdown(f'<span style="color: #6c757d; font-size: 0.8em; margin-left: 10px;">ID: {position_id}</span>', unsafe_allow_html=True)


def update_position(position_id, commodity, opt_type, strike, direction, quantity, price, strategy):
    """執行更新單一倉位"""
    idx = st.session_state.positions_df[st.session_state.positions_df['ID'] == position_id].index
    if not idx.empty:
        st.session_state.positions_df.loc[idx, "商品"] = commodity
        st.session_state.positions_df.loc[idx, "選擇權類型"] = opt_type
        st.session_state.positions_df.loc[idx, "履約價"] = float(strike)
        st.session_state.positions_df.loc[idx, "方向"] = direction
        st.session_state.positions_df.loc[idx, "口數"] = int(quantity)
        st.session_state.positions_df.loc[idx, "成交價"] = float(price)
        st.session_state.positions_df.loc[idx, "策略"] = strategy


# ====================================================================
# 5. 主應用程式邏輯 (Main Application Logic)
# ====================================================================

# === 標題 ===
st.markdown('<p class="title">📈 多策略選擇權倉位回測分析</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">模擬市場價格波動，分析多策略選擇權組合的損益變化。</p>', unsafe_allow_html=True)


# === 側邊欄：倉位新增表單 ===
with st.sidebar:
    st.header("➕ 新增交易倉位")
    with st.form("new_position_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        commodity = col1.selectbox("商品", options=["選擇權", "期貨"], index=0)
        
        # 選擇權特有欄位
        if commodity == "選擇權":
            opt_type = col2.selectbox("選擇權類型", options=["買權", "賣權"])
            strike = st.number_input("履約價", min_value=1.0, step=50.0, format="%.0f", value=st.session_state.simulation_center_price_input)
        else:
            # 期貨
            opt_type = np.nan # 暫時不用
            strike = np.nan
            col2.selectbox("期貨類型", options=["台指期"], index=0) # 佔位
            
        col3, col4 = st.columns(2)
        direction = col3.selectbox("方向", options=["買進", "賣出"])
        quantity = col4.number_input("口數", min_value=1, step=1, value=1)
        
        price = st.number_input("成交價", min_value=0.01, step=0.5, format="%.2f", value=50.0)
        strategy = st.text_input("策略標籤 (Ex: 鐵兀鷹)", value=f"策略 {st.session_state.next_id}")
        
        if st.form_submit_button("✅ 新增倉位"):
            add_position(commodity, opt_type, strike, direction, quantity, price, strategy)
            st.success("倉位已新增！")
            st.rerun()

    # === 側邊欄：模擬中心價設定 ===
    st.markdown("---")
    st.header("⚙️ 模擬參數設定")
    
    st.number_input(
        "價平中心價 (S)",
        min_value=1.0,
        step=100.0,
        format="%.0f",
        key="simulation_center_price_input",
        help="市場當前價位，用於計算損益和 Black-Scholes 模型。"
    )
    
    st.number_input(
        "波動範圍 (± 點)",
        min_value=100,
        step=100,
        value=1500,
        key="simulation_range_input",
        help="以中心價為基準，上下各延伸多少點作為模擬區間。"
    )
    
    # 模擬點數密度
    st.slider(
        "模擬點數密度 (步進)",
        min_value=10,
        max_value=100,
        step=10,
        value=50,
        key="simulation_step_input",
        help="繪圖時價格間隔的點數。"
    )
    
    # === 側邊欄：Black-Scholes 模型參數 (已修改為日期輸入) ===
    
    st.markdown('---')
    st.markdown('## ⏳ 選擇權估值')

    # 1. 波動率輸入
    volatility = st.number_input(
        "假設年化波動率 (IV, %)",
        value=25.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        format="%.1f",
        key="volatility_input",
        help="用於 Black-Scholes 模型計算的年化波動率 (Sigma)"
    )
    sigma = volatility / 100.0

    # 2. **將天數改為日期輸入**
    default_expiry_date = date.today() + timedelta(days=7)
    expiry_date = st.date_input(
        "選擇權到期日 (T)",
        value=default_expiry_date,
        min_value=date.today() + timedelta(days=1),
        key="expiry_date_input",
        help="計算剩餘天數，用於 Black-Scholes 模型。"
    )

    # 根據日期計算天數
    today = date.today()
    days_to_expiry_raw = (expiry_date - today).days
    days_to_expiry = max(1, days_to_expiry_raw) # 確保至少為 1 天
    T = days_to_expiry / 365.0 # 年化時間

    st.markdown(f"""
    <div style='font-size:14px; margin-top: 8px;'>
        <b>到期剩餘:</b> <span style="color:#04335a; font-weight:700;">{days_to_expiry_raw} 天</span>
        (年化 $T={T:.4f}$)
    </div>
    """, unsafe_allow_html=True)

    # 3. 無風險利率
    st.markdown(f"**無風險利率 (R):** <span style='color:green; font-weight:700;'>{RISK_FREE_RATE*100:.1f}%</span>", unsafe_allow_html=True)


# === 主要區塊：倉位列表與編輯 ===
st.header("📋 目前倉位列表")

if st.session_state.positions_df.empty:
    st.warning("請在左側側邊欄新增交易倉位。")
else:
    # 應用策略顏色樣式
    styled_df = st.session_state.positions_df.style.format({
        "履約價": "{:,.0f}",
        "口數": "{:,.0f}",
        "成交價": "{:,.2f}"
    }).applymap(color_strategy, subset=["策略"])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("🛠️ 編輯或移除倉位")

    # 遍歷 DataFrame，為每個倉位創建編輯/移除按鈕
    for _, row in st.session_state.positions_df.iterrows():
        position_id = row['ID']
        
        # 使用 expander 來包裹編輯表單
        with st.expander(f"編輯單列倉位: ID {position_id} | {row['策略']} - {row['選擇權類型']} {row['履約價']:.0f}"):
            edit_position_form(position_id) # 顯示編輯表單
            
            # 移除按鈕
            if st.button(f"🗑️ 移除此倉位 (ID: {position_id})", key=f'remove_btn_{position_id}'):
                remove_position(position_id)
                st.success(f"倉位 ID {position_id} 已移除。")
                st.rerun()

# ====================================================================
# 6. 時間價值分析 (Time Value Analysis)
# ====================================================================

st.header("---")
st.header("⏳ 選擇權時間價值分析")

# 過濾出選擇權部位
options_df = st.session_state.positions_df[st.session_state.positions_df["商品"] == "選擇權"].copy().reset_index(drop=True)
current_center_price = st.session_state.simulation_center_price_input # 使用模擬中心價作為 Black-Scholes 的 S

if options_df.empty:
    st.info("請新增選擇權部位以進行時間價值分析。")
else:
    # 1. 計算時間價值
    def calculate_time_value(row):
        strike = float(row['履約價'])
        opt_type_bs = 'C' if row['選擇權類型'] == '買權' else 'P'
        entry_price = float(row['成交價'])
        
        # 1. 內含價值 (Intrinsic Value, IV)
        intrinsic_value = max(0.0, current_center_price - strike) if opt_type_bs == 'C' else max(0.0, strike - current_center_price)
        
        # 2. 目前時間價值 (Time Value, TV) = 成交價 - 內含價值
        time_value = entry_price - intrinsic_value
        
        # 3. Black-Scholes 理論價格 (僅供參考)
        bs_price = black_scholes_model(
            S=current_center_price, 
            K=strike, 
            T=T, # 年化天數
            r=RISK_FREE_RATE, 
            sigma=sigma, 
            option_type=opt_type_bs
        )
        
        # 4. Black-Scholes 理論時間價值 (理論價格 - 內含價值)
        bs_time_value = bs_price - intrinsic_value
        
        return pd.Series({
            '內含價值': intrinsic_value,
            '目前時間價值': time_value,
            'BS理論價格': bs_price,
            'BS理論時間價值': bs_time_value
        })

    # 將計算結果加入 DataFrame
    options_tv_df = options_df.apply(calculate_time_value, axis=1)
    options_tv_df = pd.concat([options_df, options_tv_df], axis=1)

    # 2. 顯示時間價值表格
    st.subheader("⏱️ 選擇權持倉時間價值列表")
    st.markdown(f"""
    <div style='font-size:14px; margin-bottom: 10px;'>
        基於目前的 <b>價平中心價 {current_center_price:,.1f}</b>，計算每個選擇權部位的**內含價值**與**時間價值**。
    </div>
    """, unsafe_allow_html=True)
    
    display_cols = [
        "策略", "選擇權類型", "履約價", "方向", "口數", "成交價",
        "內含價值", "目前時間價值", "BS理論時間價值"
    ]
    
    # 定義時間價值顏色
    def color_tv(val):
        try: f=float(val)
        except: return ''
        if f > 0: return 'color: #0b5cff; font-weight: 700;' # 藍色 (有時間價值)
        elif f < 0: return 'color: #cf1322; font-weight: 700;' # 紅色 (負時間價值)
        return ''
        
    styled_tv_df = options_tv_df[display_cols].style.format({
        "履約價": "{:,.1f}",
        "成交價": "{:,.2f}",
        "內含價值": "{:,.2f}",
        "目前時間價值": "{:,.2f}",
        "BS理論時間價值": "{:,.2f}"
    }).applymap(color_strategy, subset=["策略"]).applymap(color_tv, subset=["目前時間價值", "BS理論時間價值"])

    st.dataframe(styled_tv_df, use_container_width=True, hide_index=True)

    # 3. 彙總資訊 (總時間價值損益)
    
    # 計算每個部位的時間價值金額
    options_tv_df["時間價值金額"] = options_tv_df["目前時間價值"] * options_tv_df["口數"] * MULTIPLIER_OPTION
    
    # 時間價值損益貢獻：買進部位(-)，賣出部位(+)
    def time_decay_impact(row):
        tv_amount = row["時間價值金額"]
        if row["方向"] == "買進":
            return -tv_amount
        else: # "賣出"
            return tv_amount
            
    options_tv_df["時間價值損益貢獻"] = options_tv_df.apply(time_decay_impact, axis=1)

    # 彙總計算
    total_time_decay_impact = options_tv_df["時間價值損益貢獻"].sum()

    st.markdown("#### 彙總數據")
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.metric(
            label="所有持倉總時間價值金額 (NT$)",
            value=f"NT$ {options_tv_df['時間價值金額'].abs().sum():,.0f}",
            help="將所有倉位的權利金中的時間價值部分乘以口數和乘數的總和 (絕對值)。"
        )
    with col_sum2:
        st.metric(
            label="倉位整體時間價值損益影響 (金額)",
            value=f"NT$ {total_time_decay_impact:,.0f}",
            delta=f"NT$ {total_time_decay_impact:,.0f}",
            delta_color="normal",
            help="整體倉位因時間流逝而獲得 (正)/損失 (負) 的潛在總權利金。賣方為正，買方為負。"
        )
    st.markdown("---")

# ====================================================================
# 7. 損益分析 (P&L Analysis)
# ====================================================================

st.header("📊 損益曲線分析")

# ... 接下來是損益分析和圖表的代碼 (如果您有這部分代碼，請在此處貼上) ...
# 由於原始請求沒有提供損益計算的代碼，這裡假設您會接續加入。

# 這裡只是一個佔位符，用於指示損益分析區塊
st.info("請接續貼上您原有的『損益曲線分析』代碼，以完成應用程式。")


# === 額外：原始的 Black-Scholes Theta 模擬 ===
# 如果您需要 Black-Scholes 價格隨時間衰減的圖表，可以在此處實現。
# ...
