import streamlit as st
import requests
import time
import json
import os
import random
from datetime import datetime
import plotly.graph_objects as go 
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 1. Configuration & Localization =================
st.set_page_config(page_title="CF Rating Predictor", layout="wide")

# Translation Dictionary
TRANSLATIONS = {
    "en": {
        "title": "Codeforces Rating Simulator",
        "subtitle": "Visualize your potential rating trajectory by integrating Virtual Participation (VP) history.",
        "sidebar_header": "Configuration",
        "input_handle": "CF Handle",
        "input_init_rating": "Initial Rating",
        "note": "**Note:** When analyzing a specific Handle for the first time, the system needs to download its historical data. This may take a few minutes. Subsequent runs for the same Handle will be instant due to caching.",
        "checkbox_show_real": "Show Official Rating Comparison",
        "checkbox_help": "Overlays your actual official rating curve (dashed blue line) for comparison.",
        "btn_clear_cache": "Clear Cache",
        "msg_cache_cleared": "Cache cleared successfully.",
        "btn_start": "Start Analysis",
        "msg_init": "Initializing analysis engine...",
        "warn_no_data": "No data found. The Handle might be invalid or has no contest history.",
        "warn_no_vp": "No Virtual Participation records found. Please check if the user has participated virtually.",
        "metric_pred": "Predicted Rating (w/ VP)",
        "metric_real": "Official Rating",
        "metric_gap": "Gap (VP Effect)",
        "chart_title": "Rating Trajectory",
        "legend_sim": "Simulated Path (Real + VP)",
        "legend_real": "Official Rating Curve",
        "tooltip_real": "REAL (Official)",
        "tooltip_vp": "VP (Virtual)",
        "tooltip_init": "Initial",
        "tooltip_rank": "Rank",
        "tooltip_rating": "Rating",
        "tooltip_date": "Date"
    },
    "zh": {
        "title": "Codeforces Rating 模拟器",
        "subtitle": "通过整合虚拟参赛 (VP) 记录，可视化您的潜在 Rating 走势。",
        "sidebar_header": "配置面板",
        "input_handle": "输入 CF Handle",
        "input_init_rating": "初始 Rating",
        "note": "**提示：** 每个 Handle 首次进行分析时，都需要下载该用户的历史比赛数据，耗时可能较长（几分钟）。该 Handle 后续再次运行时将使用缓存，速度会显著加快。",
        "checkbox_show_real": "显示官方 Rating 对比",
        "checkbox_help": "勾选后，将在图中叠加显示你实际的官方 CF Rating 曲线（蓝色虚线）以作对比。",
        "btn_clear_cache": "清除缓存",
        "msg_cache_cleared": "缓存已清除。",
        "btn_start": "开始分析",
        "msg_init": "正在初始化分析引擎...",
        "warn_no_data": "未找到数据。可能是 Handle 输入错误，或该用户没有任何比赛记录。",
        "warn_no_vp": "未检测到虚拟参赛 (VP) 记录。请确认该用户是否有过 VP。",
        "metric_pred": "当前预测分 (含VP)",
        "metric_real": "官方实际分",
        "metric_gap": "VP 带来的差距",
        "chart_title": "Rating 走势图",
        "legend_sim": "模拟走势 (实战 + VP)",
        "legend_real": "官方实际曲线",
        "tooltip_real": "实战 (Official)",
        "tooltip_vp": "虚拟参赛 (VP)",
        "tooltip_init": "初始分",
        "tooltip_rank": "排名",
        "tooltip_rating": "分数",
        "tooltip_date": "日期"
    }
}

# Language Selector
lang_option = st.sidebar.selectbox("Language / 语言", ["English", "中文"])
LANG = "en" if lang_option == "English" else "zh"
T = TRANSLATIONS[LANG]

# ================= 2. Interface Rendering =================
st.title(T["title"])
st.markdown(T["subtitle"])

st.sidebar.header(T["sidebar_header"])

# Input fields
HANDLE = st.sidebar.text_input(T["input_handle"], value="vivid_stareium")
INITIAL_RATING = st.sidebar.number_input(T["input_init_rating"], value=1400, step=100)

# Professional note
st.sidebar.info(T["note"])

SHOW_REAL_CURVE = st.sidebar.checkbox(
    T["checkbox_show_real"], 
    value=False, 
    help=T["checkbox_help"]
)

if st.sidebar.button(T["btn_clear_cache"]):
    st.cache_data.clear()
    if 'sim_data' in st.session_state:
        del st.session_state['sim_data']
    st.sidebar.success(T["msg_cache_cleared"])

# ================= 3. Data Processing Logic =================
CACHE_DIR = "data"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_json(url, params=None, use_cache=True):
    params_str = ""
    if params:
        sorted_keys = sorted(params.keys())
        params_str = "_" + "_".join([f"{k}_{params[k]}" for k in sorted_keys])
    
    method_name = url.split('/')[-1]
    filename = f"{method_name}{params_str}.json".replace(":", "").replace("/", "")
    filepath = os.path.join(CACHE_DIR, filename)

    if use_cache and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass

    try:
        if not use_cache: 
            time.sleep(random.uniform(0.05, 0.1))
        
        resp = requests.get(url, params=params, timeout=15)
        
        if resp.status_code == 429:
            time.sleep(random.uniform(1.5, 2.5))
            return get_json(url, params, use_cache)
        
        resp.raise_for_status()
        data = resp.json()
        if data['status'] != 'OK': return None
        
        if use_cache:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data['result'], f, ensure_ascii=False)
        return data['result']
    except Exception:
        return None

def fetch_vp_rank(cid, handle, sub_ts):
    """Worker function to fetch rank from contest standings."""
    standings = get_json("https://codeforces.com/api/contest.standings", {
        "contestId": cid, "handles": handle, "showUnofficial": True
    })
    
    contest_name = f"Contest {cid}"
    user_rows = []
    if standings:
        contest_name = standings['contest']['name']
        try:
            for row in standings['rows']:
                if row['party']['participantType'] == 'VIRTUAL': user_rows.append(row)
        except: pass
        
    matched_rank = 0
    if user_rows:
        user_rows.sort(key=lambda r: r['party']['startTimeSeconds'])
        best_row = None
        for row in user_rows:
            if row['party']['startTimeSeconds'] <= sub_ts + 48 * 3600: 
                best_row = row
            else: break
        if not best_row: best_row = user_rows[-1]
        matched_rank = best_row['rank']
        
    return cid, matched_rank, contest_name, sub_ts

def fetch_rating_changes(cid):
    """Worker function to fetch rating changes for a specific contest."""
    data = get_json("https://codeforces.com/api/contest.ratingChanges", {"contestId": cid})
    return cid, data

@st.cache_data(ttl=3600, show_spinner=False)
def get_processed_data(handle, init_rating):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Fetch Official History
    status_text.text("Fetching official contest history...")
    real_data = get_json("https://codeforces.com/api/user.rating", {"handle": handle}, use_cache=False)
    
    real_history = []
    if real_data:
        for row in real_data:
            real_history.append({
                "date": datetime.fromtimestamp(row['ratingUpdateTimeSeconds']),
                "rating": row['newRating'],
                "name": row['contestName'],
                "rank": row['rank']
            })

    events = []
    if real_data:
        for row in real_data:
            events.append({
                "cid": row['contestId'], "rank": row['rank'],
                "ts": row['ratingUpdateTimeSeconds'], "type": "REAL",
                "name": row['contestName']
            })

    # 2. Fetch VP History
    status_text.text("Scanning submission history for VPs...")
    progress_bar.progress(5)
    subs = get_json("https://codeforces.com/api/user.status", {"handle": handle}, use_cache=False)
    
    final_vp_list = []
    
    if subs:
        vp_timestamps = {} 
        for s in subs:
            if s.get('author', {}).get('participantType') == 'VIRTUAL':
                cid = s['contestId']
                ts = s['creationTimeSeconds']
                if cid not in vp_timestamps: vp_timestamps[cid] = []
                vp_timestamps[cid].append(ts)
        
        for cid, times in vp_timestamps.items():
            times.sort()
            curr_start = times[0]
            final_vp_list.append((cid, curr_start)) 
            for k in range(1, len(times)):
                if times[k] - times[k-1] > 2 * 24 * 3600:
                    curr_start = times[k]
                    final_vp_list.append((cid, curr_start))

    # 3. Parallel Fetch: VP Ranks
    valid_vp_list = [x for x in final_vp_list if x[0] < 100000]
    total_vp = len(valid_vp_list)
    
    if total_vp > 0:
        status_text.text(f"Retrieving ranks for {total_vp} virtual contests...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_vp_rank, cid, handle, ts): cid for cid, ts in valid_vp_list}
            
            completed_count = 0
            for future in as_completed(futures):
                cid, rank, cname, ts = future.result()
                
                if rank > 0:
                    events.append({
                        "cid": cid, "rank": rank, "ts": ts, "type": "VP", "name": cname
                    })
                
                completed_count += 1
                prog = 5 + int((completed_count / total_vp) * 45)
                progress_bar.progress(prog)
                status_text.text(f"Analyzing VP: {completed_count}/{total_vp} (Contest {cid})")

    events.sort(key=lambda x: x['ts'])
    
    # 4. Parallel Fetch: Rating Changes
    target_cids = [e['cid'] for e in events if e['cid'] < 100000]
    total_downloads = len(target_cids)
    changes_cache = {}
    
    if total_downloads > 0:
        status_text.text(f"Downloading rating change data for {total_downloads} contests...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_cid = {executor.submit(fetch_rating_changes, cid): cid for cid in target_cids}
            
            completed_count = 0
            for future in as_completed(future_to_cid):
                cid, data = future.result()
                if data:
                    changes_cache[cid] = data
                
                completed_count += 1
                prog = 50 + int((completed_count / total_downloads) * 45)
                progress_bar.progress(prog)
                status_text.text(f"Fetching Rating Data: {completed_count}/{total_downloads}")

    # 5. Calculate Ratings
    status_text.text("Finalizing rating simulation...")
    progress_bar.progress(98)
    
    curr = init_rating
    sim_history = []
    
    if events:
        sim_history.append({
            "date": datetime.fromtimestamp(events[0]['ts'] - 86400),
            "rating": curr, "type": "INIT", "name": "Start", "rank": 0, "delta": 0
        })

    for event in events:
        cid = event['cid']
        changes = changes_cache.get(cid)
        if not changes: continue

        candidates = [p for p in changes if abs(p['oldRating'] - curr) < 300] or changes
        best_proxy = min(candidates, key=lambda p: abs(p['oldRating']-curr)*0.8 + abs(p['rank']-event['rank'])*1.5)
        
        delta = best_proxy['newRating'] - best_proxy['oldRating']
        if event['rank'] < best_proxy['rank']: delta += 1
        elif event['rank'] > best_proxy['rank']: delta -= 1
        
        curr += delta
        sim_history.append({
            "date": datetime.fromtimestamp(event['ts']),
            "rating": curr,
            "type": event['type'],
            "name": event['name'],
            "rank": event['rank'],
            "delta": delta
        })
    
    progress_bar.empty()
    status_text.empty()
    return sim_history, real_history

# ================= 4. Visualization =================
def plot_plotly(sim_history, real_history, show_real, local_t):
    fig = go.Figure()

    sim_dates = [h['date'] for h in sim_history]
    sim_ratings = [h['rating'] for h in sim_history]
    
    sim_hover = []
    for h in sim_history:
        type_str = local_t["tooltip_real"] if h['type'] == 'REAL' else local_t["tooltip_vp"]
        if h['type'] == 'INIT': type_str = local_t["tooltip_init"]
        
        sim_hover.append(
            f"<b>{h['name']}</b><br>{type_str}<br>{local_t['tooltip_rank']}: {h['rank']}<br>"
            f"{local_t['tooltip_rating']}: {h['rating']} ({h['delta']:+d})<br>{local_t['tooltip_date']}: {h['date'].strftime('%Y-%m-%d')}"
        )

    # Line for simulated path
    fig.add_trace(go.Scatter(
        x=sim_dates, y=sim_ratings, mode='lines',
        line=dict(color='gray', width=1.5), 
        name=local_t["legend_sim"], 
        hoverinfo='skip'
    ))
    
    # Markers for simulated path
    colors = ['#FF0000' if h['type'] == 'VP' else '#000000' for h in sim_history]
    sizes = [12 if h['type'] == 'VP' else 8 for h in sim_history]
    symbols = ['star' if h['type'] == 'VP' else 'circle' for h in sim_history]

    fig.add_trace(go.Scatter(
        x=sim_dates, y=sim_ratings, mode='markers',
        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color='white')),
        text=sim_hover, hoverinfo='text', showlegend=False
    ))

    # Optional: Official Rating Curve
    if show_real and real_history:
        real_dates = [h['date'] for h in real_history]
        real_vals = [h['rating'] for h in real_history]
        real_hover = [f"<b>{h['name']}</b><br>{local_t['metric_real']}: {h['rating']}<br>{local_t['tooltip_rank']}: {h['rank']}" for h in real_history]
        
        fig.add_trace(go.Scatter(
            x=real_dates, y=real_vals, mode='lines+markers',
            line=dict(color='#1E90FF', width=2, dash='dash'),
            marker=dict(size=6, color='#1E90FF'),
            text=real_hover, hoverinfo='text',
            name=local_t["legend_real"],
            opacity=0.7
        ))

    # Background Color Bands (Codeforces Tiers)
    all_ratings = sim_ratings + ([h['rating'] for h in real_history] if show_real else [])
    min_r, max_r = min(all_ratings), max(all_ratings)
    y_lower = max(0, min_r - 200)
    y_upper = max_r + 200
    
    color_bands = [
        (0, 1199, '#CCCCCC'), (1200, 1399, '#77FF77'), (1400, 1599, '#77DDFF'),
        (1600, 1899, '#AAAAFF'), (1900, 2099, '#FF88FF'), (2100, 2299, '#FFCC88'),
        (2300, 2399, '#FFBB55'), (2400, 2999, '#FF7777'), (3000, 4000, '#AA0000')
    ]
    shapes = []
    for low, high, color in color_bands:
        if high < y_lower or low > y_upper: continue
        shapes.append(dict(type="rect", xref="paper", yref="y", x0=0, y0=low, x1=1, y1=high, fillcolor=color, opacity=0.15, layer="below", line_width=0))

    fig.update_layout(
        title=f"{local_t['chart_title']} ({HANDLE})",
        yaxis_title="Rating", yaxis=dict(range=[y_lower, y_upper]),
        shapes=shapes, hovermode="closest", height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= 5. Main Execution =================

if st.button(T["btn_start"], type="primary"):
    with st.spinner(T["msg_init"]):
        sim_data, real_data = get_processed_data(HANDLE, INITIAL_RATING)
        st.session_state['sim_data'] = sim_data
        st.session_state['real_data'] = real_data

if 'sim_data' in st.session_state:
    sim_data = st.session_state['sim_data']
    real_data = st.session_state['real_data']
    
    if not sim_data:
        st.error(T["warn_no_data"])
    
    else:
        vp_count = sum(1 for d in sim_data if d.get('type') == 'VP')
        if vp_count == 0:
            st.warning(T["warn_no_vp"])

        last_sim_rating = sim_data[-1]['rating']
        last_sim_delta = sim_data[-1]['delta']
        last_real_rating = real_data[-1]['rating'] if real_data else INITIAL_RATING
        gap = last_sim_rating - last_real_rating
        
        col1, col2, col3 = st.columns(3)
        col1.metric(T["metric_pred"], f"{last_sim_rating}", f"{last_sim_delta:+d}")
        col2.metric(T["metric_real"], f"{last_real_rating}")
        col3.metric(T["metric_gap"], f"{gap:+d}", delta_color="normal")
        
        st.divider()
        plot_plotly(sim_data, real_data, SHOW_REAL_CURVE, T)