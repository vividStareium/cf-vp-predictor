import streamlit as st
import requests
import time
import json
import os
import random
from datetime import datetime
import plotly.graph_objects as go 
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc 
import threading
import shutil

api_lock = threading.Semaphore(2)
file_lock = threading.Lock()

st.set_page_config(page_title="CF Rating Predictor", layout="wide")

TRANSLATIONS = {
    "en": {
        "title": "Codeforces Rating Simulator",
        "subtitle": "Visualize your potential rating trajectory by integrating Virtual Participation (VP) history.",
        "sidebar_header": "Configuration",
        "input_handle": "CF Handle",
        "input_init_rating": "Initial Rating",
        "note": "**Note:** When analyzing a specific Handle for the first time, the system needs to download its historical data. This may take a few minutes. Be patient pls.",
        "checkbox_show_real": "Show Official Rating Comparison",
        "checkbox_help": "Overlays your actual official rating curve (dashed blue line) for comparison.",
        "checkbox_unrated": "Show Unrated Contests",
        "btn_clear_cache": "Refresh Personal Data",
        "help_btn_refresh": "Deletes local data for the current handle.\nFixes issues where recent VP records are not updating.\n\nNote: Re-downloading data will take time. Use with caution.",
        "msg_refresh_success": "Memory cleared. Deleted {count} local cache files for **{handle}**.",
        "msg_refresh_none": "Memory cleared. No local cache files found for **{handle}**.",
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
        "note": "**提示：** 每个 Handle 首次进行分析时，都需要下载该用户的历史比赛数据，耗时可能较长（几分钟），请耐心等待。",
        "checkbox_show_real": "显示官方 Rating 对比",
        "checkbox_help": "勾选后，将在图中叠加显示你实际的官方 CF Rating 曲线（蓝色虚线）以作对比。",
        "checkbox_unrated": "显示 Unrated 比赛",
        "btn_clear_cache": "刷新个人数据",
        "help_btn_refresh": "按下后将删除当前 Handle 的本地数据。\n可解决近期 VP 后数据未更新的问题。\n\n注意：再次查询需重新等待较长时间下载数据，请谨慎使用。",
        "msg_refresh_success": "内存已清理。已删除用户 **{handle}** 的 {count} 个本地缓存文件。",
        "msg_refresh_none": "内存已清理。未找到用户 **{handle}** 的本地缓存。",
        "msg_cache_cleared": "缓存已清除。",
        "btn_start": "开始分析",
        "msg_init": "正在初始化分析引擎...",
        "warn_no_data": "未找到数据。该 Handle 可能无效或没有任何参赛记录。",
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

lang_option = st.sidebar.selectbox("Language / 语言", ["English", "中文"])
LANG = "en" if lang_option == "English" else "zh"
T = TRANSLATIONS[LANG]

st.title(T["title"])
st.markdown(T["subtitle"])

st.sidebar.header(T["sidebar_header"])

HANDLE = st.sidebar.text_input(T["input_handle"], value="vivid_stareium")
INITIAL_RATING = st.sidebar.number_input(T["input_init_rating"], value=1400, step=100)

st.sidebar.info(T["note"])

SHOW_REAL_CURVE = st.sidebar.checkbox(
    T["checkbox_show_real"], 
    value=False, 
    help=T["checkbox_help"]
)

SHOW_UNRATED = st.sidebar.checkbox(
    T["checkbox_unrated"], 
    value=False
)

CACHE_DIR = "data"  
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if st.sidebar.button(T["btn_clear_cache"], help=T["help_btn_refresh"]):
    st.cache_data.clear()
    
    files_deleted = 0
    if os.path.exists(CACHE_DIR):
        target_handle_lower = HANDLE.lower()
        for fname in os.listdir(CACHE_DIR):
            fname_lower = fname.lower()
            patterns = [
                f"handle_{target_handle_lower}_",
                f"handle_{target_handle_lower}.json",
                f"handles_{target_handle_lower}_",
                f"handles_{target_handle_lower}.json"
            ]
            
            if any(p in fname_lower for p in patterns):
                try:
                    file_path = os.path.join(CACHE_DIR, fname)
                    os.remove(file_path)
                    files_deleted += 1
                except Exception as e:
                    print(f"Error deleting {fname}: {e}")

    if files_deleted > 0:
        st.success(T["msg_refresh_success"].format(count=files_deleted, handle=HANDLE))
    else:
        st.warning(T["msg_refresh_none"].format(handle=HANDLE))

def handle_430_cooldown(seconds=180):
    cooldown_placeholder = st.sidebar.empty() 
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        cooldown_placeholder.error(
            f"**Codeforces Rate Limit**\n\n"
            f"The server is temporarily throttled due to high traffic.\n\n"
            f"**Cooling down: {mins:02d}:{secs:02d} remaining**\n\n"
            "Please do not refresh the page. The request will automatically resume after the countdown."
        )
        time.sleep(1)
    cooldown_placeholder.empty() 

def get_json(url, params=None, use_cache=True):
    params_str = ""
    if params:
        sorted_keys = sorted(params.keys())
        params_str = "_" + "_".join([f"{k}_{params[k]}" for k in sorted_keys])
    
    method_name = url.split('/')[-1]
    filename = f"{method_name}{params_str}.json".replace(":", "").replace("/", "")
    filepath = os.path.join(CACHE_DIR, filename)

    def parse_cache_data(data, fname):
        if "ratingChanges" in fname and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return [{"rank": r[0], "oldRating": r[1], "newRating": r[2]} for r in data]
        if "contest.standings" in fname and isinstance(data, list):
            return data
        return data

    if use_cache and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return parse_cache_data(json.load(f), filename)
        except Exception:
            pass 

    with api_lock:
        if use_cache and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return parse_cache_data(json.load(f), filename)
            except Exception: pass

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(1.1) 
                else:
                    time.sleep(random.uniform(0.2, 0.4))
                
                resp = requests.get(url, params=params, timeout=15)

                if resp.status_code == 430:
                    handle_430_cooldown(200) 
                    continue
                if resp.status_code == 429:
                    time.sleep(2) 
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                if data['status'] != 'OK': 
                    raise ValueError(f"API Error: {data.get('comment')}")
                
                result = data['result']
                save_data = result

                if "ratingChanges" in url:
                    save_data = [[row['rank'], row['oldRating'], row['newRating']] for row in result]
                elif "contest.standings" in url:
                    rows = result.get('rows', []) if isinstance(result, dict) else []
                    save_data = [[row['rank'], row['party']['startTimeSeconds'], 1 if row['party']['participantType'] == 'VIRTUAL' else 0] for row in rows]
                
                if use_cache:
                    if not os.path.exists(CACHE_DIR): 
                        os.makedirs(CACHE_DIR)
                    with file_lock: 
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, separators=(',', ':'))
                
                if "ratingChanges" in url:
                    return [{"rank": r[0], "oldRating": r[1], "newRating": r[2]} for r in save_data]
                return save_data if "contest.standings" in url else result

            except Exception as e:
                print(f"Fetch error {url} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1: return None

    return None

def fetch_vp_rank(cid, handle, sub_ts):
    standings = get_json("https://codeforces.com/api/contest.standings", {
        "contestId": cid, "handles": handle, "showUnofficial": True
    }, use_cache=True)
    
    contest_name = f"Contest {cid}"
    
    user_rows = []
    if standings and isinstance(standings, list):
        for row in standings:
            if isinstance(row, list) and len(row) >= 3:
                if row[2] == 1: 
                    user_rows.append(row)
        
    matched_rank = 0
    if user_rows:
        user_rows.sort(key=lambda r: r[1])
        best_row = None
        for row in user_rows:
            if row[1] <= sub_ts + 48 * 3600: 
                best_row = row
            else: break
        if not best_row: best_row = user_rows[-1]
        matched_rank = best_row[0]
        
    return cid, matched_rank, contest_name, sub_ts

def fetch_rating_changes_to_disk(cid):
    get_json("https://codeforces.com/api/contest.ratingChanges", {"contestId": cid}, use_cache=True)
    return cid

def process_batches(task_list, worker_func, max_workers=10, update_callback=None):
    results = []
    total = len(task_list)
    chunks = [task_list[i:i + max_workers] for i in range(0, total, max_workers)]
    
    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in chunk:
                if isinstance(item, tuple):
                    future = executor.submit(worker_func, *item)
                else:
                    future = executor.submit(worker_func, item)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Batch execution error: {e}")
                if update_callback:
                    update_callback()
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def get_processed_data(handle, init_rating):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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

    valid_vp_list = [x for x in final_vp_list if x[0] < 100000]
    total_vp = len(valid_vp_list)
    
    if total_vp > 0:
        status_text.text(f"Retrieving ranks for {total_vp} virtual contests...")
        tasks = [(cid, handle, ts) for cid, ts in valid_vp_list]
        progress_tracker = {"count": 0}
        def vp_progress_cb():
            progress_tracker["count"] += 1
            completed = progress_tracker["count"]
            prog = 5 + int((completed / total_vp) * 45)
            progress_bar.progress(prog)
            status_text.text(f"Analyzing VP: {completed}/{total_vp}")

        vp_results = process_batches(tasks, fetch_vp_rank, max_workers=5, update_callback=vp_progress_cb)

        for res in vp_results:
            if not res: continue
            cid, rank, cname, ts = res
            if rank > 0:
                events.append({
                    "cid": cid, "rank": rank, "ts": ts, "type": "VP", "name": cname
                })

    events.sort(key=lambda x: x['ts'])
    
    target_cids = [e['cid'] for e in events if e['cid'] < 100000]
    total_downloads = len(target_cids)
    
    if total_downloads > 0:
        status_text.text(f"Syncing contest data to disk (Lazy Loading)...")
        progress_tracker_dl = {"count": 0}
        def dl_progress_cb():
            progress_tracker_dl["count"] += 1
            completed = progress_tracker_dl["count"]
            prog = 50 + int((completed / total_downloads) * 45)
            progress_bar.progress(prog)
            status_text.text(f"Syncing Data: {completed}/{total_downloads}")
            
        process_batches(target_cids, fetch_rating_changes_to_disk, max_workers=1, update_callback=dl_progress_cb)

    status_text.text("Calculating rating simulation...")
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
        cname = event['name']
        
        changes = get_json("https://codeforces.com/api/contest.ratingChanges", {"contestId": cid}, use_cache=True)
        
        if not changes:
            days_diff = (time.time() - event['ts']) / (24 * 3600)
            
            if days_diff < 10:
                fname = f"ratingChanges_contestId_{cid}.json"
                fpath = os.path.join(CACHE_DIR, fname)
                
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception: 
                        pass
                
                time.sleep(0.2) 
                changes = get_json("https://codeforces.com/api/contest.ratingChanges", {"contestId": cid}, use_cache=True)

        if not changes: continue

        active_changes = [p for p in changes if p['newRating'] != p['oldRating']]
        
        max_official_rating = 0
        if active_changes:
            max_official_rating = max(p['oldRating'] for p in active_changes)
        elif changes:
            max_official_rating = max(p['oldRating'] for p in changes)
        
        threshold = int((max_official_rating + 100) / 100) * 100
            
        if curr >= threshold:
            sim_history.append({
                "date": datetime.fromtimestamp(event['ts']),
                "rating": curr, "type": event['type'], "name": f"{cname} (Unrated)",
                "rank": event['rank'], "delta": 0
            })
            continue 

        candidates = [p for p in changes if abs(p['oldRating'] - curr) < 300]
        if not candidates: candidates = changes

        scored_candidates = []
        for p in candidates:
            cost = abs(p['oldRating'] - curr) * 0.8 + abs(p['rank'] - event['rank']) * 1.5
            scored_candidates.append((cost, p))
        
        scored_candidates.sort(key=lambda x: x[0])
        top_proxies = [x[1] for x in scored_candidates[:5]] 
        
        if not top_proxies: continue

        raw_deltas = []
        for p in top_proxies:
            d = p['newRating'] - p['oldRating']
            if event['rank'] < p['rank']: d += 1
            elif event['rank'] > p['rank']: d -= 1
            raw_deltas.append(d)
        
        final_delta = 0
        if len(raw_deltas) >= 3:
            raw_deltas.sort()
            trimmed_deltas = raw_deltas[1:-1]
            final_delta = sum(trimmed_deltas) / len(trimmed_deltas)
        else:
            final_delta = sum(raw_deltas) / len(raw_deltas)
            
        delta_int = int(round(final_delta))
        
        curr += delta_int
        sim_history.append({
            "date": datetime.fromtimestamp(event['ts']),
            "rating": curr, "type": event['type'], "name": cname,
            "rank": event['rank'], "delta": delta_int
        })
    
    gc.collect()
    progress_bar.empty()
    status_text.empty()
    return sim_history, real_history

def plot_plotly(sim_history, real_history, show_real, local_t):
    from datetime import timedelta
    fig = go.Figure()

    plot_dates = []
    date_counter = {}

    for h in sim_history:
        date_key = h['date'].strftime('%Y-%m-%d')
        if date_key in date_counter:
            date_counter[date_key] += 1
        else:
            date_counter[date_key] = 0
        
        offset = timedelta(hours=6 * date_counter[date_key])
        plot_dates.append(h['date'] + offset)

    sim_ratings = [h['rating'] for h in sim_history]
    sim_hover = []
    for h in sim_history:
        type_str = local_t["tooltip_real"] if h['type'] == 'REAL' else local_t["tooltip_vp"]
        if h['type'] == 'INIT': type_str = local_t["tooltip_init"]
        
        sim_hover.append(
            f"<b>{h['name']}</b><br>{type_str}<br>{local_t['tooltip_rank']}: {h['rank']}<br>"
            f"{local_t['tooltip_rating']}: {h['rating']} ({h['delta']:+d})<br>{local_t['tooltip_date']}: {h['date'].strftime('%Y-%m-%d')}"
        )

    fig.add_trace(go.Scatter(
        x=plot_dates, y=sim_ratings, mode='lines',
        line=dict(color='gray', width=1.5), 
        name=local_t["legend_sim"], hoverinfo='skip'
    ))
    
    colors = ['#FF0000' if h['type'] == 'VP' else '#000000' for h in sim_history]
    sizes = [12 if h['type'] == 'VP' else 8 for h in sim_history]
    symbols = ['star' if h['type'] == 'VP' else 'circle' for h in sim_history]

    fig.add_trace(go.Scatter(
        x=plot_dates, y=sim_ratings, mode='markers',
        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color='white')),
        text=sim_hover, hoverinfo='text', showlegend=False
    ))

    if show_real and real_history:
        real_dates = [h['date'] for h in real_history]
        real_vals = [h['rating'] for h in real_history]
        real_hover = [f"<b>{h['name']}</b><br>{local_t['metric_real']}: {h['rating']}<br>{local_t['tooltip_rank']}: {h['rank']}" for h in real_history]
        
        fig.add_trace(go.Scatter(
            x=real_dates, y=real_vals, mode='lines+markers',
            line=dict(color='#1E90FF', width=2, dash='dash'),
            marker=dict(size=6, color='#1E90FF'),
            text=real_hover, hoverinfo='text',
            name=local_t["legend_real"], opacity=0.7
        ))

    all_ratings = sim_ratings + ([h['rating'] for h in real_history] if show_real else [])
    if all_ratings:
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

if st.button(T["btn_start"], type="primary"):
    
    if HANDLE == "ALL_HISTORY":
        st.info("Downloading all contest data (ID 1 -> Last)...")
        contests_data = get_json("https://codeforces.com/api/contest.list", {"gym": "false"}, use_cache=False)
        if not contests_data:
            st.error("Failed to fetch contest list")
            st.stop()
            
        target_contests = [c['id'] for c in contests_data if c['phase'] == 'FINISHED' and c['id'] < 10000]
        target_contests.sort()
        total_c = len(target_contests)
        st.write(f"Found {total_c} contests. Starting sequential download...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        progress_tracker_all = {"count": 0}
        def all_progress_cb():
            progress_tracker_all["count"] += 1
            completed = progress_tracker_all["count"]
            if completed % 10 == 0 or completed == total_c: 
                prog = int((completed / total_c) * 100)
                progress_bar.progress(min(prog, 100))
                status_text.text(f"Processing: {completed}/{total_c}")

        process_batches(target_contests, fetch_rating_changes_to_disk, max_workers=5, update_callback=all_progress_cb)
        progress_bar.progress(100)
        st.success(f"Done. {total_c} contests cached.")

    else:
        with st.spinner(T["msg_init"]):
            sim_data, real_data = get_processed_data(HANDLE, INITIAL_RATING)
            st.session_state['sim_data'] = sim_data
            st.session_state['real_data'] = real_data

if 'sim_data' in st.session_state and HANDLE != "ALL_HISTORY":

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

        plot_sim_data = sim_data
        if not SHOW_UNRATED:
            plot_sim_data = [d for d in sim_data if d['type'] == 'INIT' or "(Unrated)" not in d['name']]

        plot_plotly(plot_sim_data, real_data, SHOW_REAL_CURVE, T)
