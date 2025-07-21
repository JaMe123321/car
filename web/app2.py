
import threading, time, cv2, base64, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime
import pymysql.cursors
import subprocess
import shlex
import threading, time, cv2
from queue import Queue

# ─── 建立 Flask App 並啟用 CORS ──────────────────────────────────────────
app = Flask(__name__)
CORS(app)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

# ─── 全域狀態變數 ────────────────────────────────────────────────────────
level_result       = ""      # 前端「垃圾：…」文字顯示
recognition_active = True    # 控制 YOLO 偵測是否繼續
category_counts    = {"Tissue":0,"Bottle":0,"Plastic":0,"Total":0}# 今日分類計數
latest_crop_b64    = None    # 最近一次碰線截圖（Base64）
latest_crop_label  = ""      # 截圖對應的分類標籤
sensor_data        = {'data': ''}  # 存放最新的感測器原始 JSON
last_db_write_time = 0.0     # 上次寫入感測器 log 的時間戳，用於節流
saved_ids = set()           #存圖用
# ─── MySQL 資料庫設定 ──────────────────────────────────────────────────
DB_CONFIG = {
    'host':     '127.0.0.1',
    'port':     3306,
    'user':     'root',
    'password': 'james9344',
    'db':       'sales_db',
    'charset':  'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
"""取得新的資料庫連線"""
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)
"""將目前的垃圾分類統計寫入 sale 表
    仿照：日期、時間、總數、各類別數"""
def save_classification_to_db(counts):
    now = datetime.now()
    date_str, time_str = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
    total, t, b, p = counts['Total'], counts['Tissue'], counts['Bottle'], counts['Plastic']
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO `sale`(`date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`)"
                " VALUES(%s,%s,%s,%s,%s,%s)",
                (date_str, time_str, total, t, b, p)
            )
            conn.commit()
    finally:
        conn.close()

"""查詢今日已存入的首筆 sale 記錄
    返回 dict：{'date', 'Total', 'Tissue', 'Bottle', 'Plastic'}"""
def get_today_sales():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {'date': today, 'Total': 0, 'Tissue': 0, 'Bottle': 0, 'Plastic': 0}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM `sale` WHERE `date`=%s", (today,))
            row = cur.fetchone()
            if row:
                data.update(row)
    finally:
        conn.close()
    return data

# ─── YOLO 模型與影像串流設定 ───────────────────────────────────────────────
# 載入自訓練權重，並啟用物件追踪
model = YOLO(r"D:\圖\garbage\all3\train18\weights\best.pt", 'track')
# 使用 IP Webcam Android 應用的串流位址
#stream_url = 'http://192.168.52.70:8080/video'
#cap1 = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
#cap1 = cv2.VideoCapture(r"D:\圖\garbage\影二\1.mp4")

rtsp_url = "rtsp://192.168.52.138:8554/live"
cap1 = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
# 等待 5 秒再超时
cap1.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
cap1.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
if not cap1.isOpened():
    raise RuntimeError("❌ 無法打開攝影機，請確認連接")
size = (int(cap1.get(3)), int(cap1.get(4)))
"""準備錄影器（XVID 可換成 MP4V、H264 需要系統裝解碼）"""
# 讀一次取得實際解析度
width  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 實際攝影機解析度: ({width}, {height})")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = cv2.VideoWriter('record.avi', fourcc, 30.0, (width, height))
"""在影像上畫出綠色方框與置中文字標籤
    box: [x1,y1,x2,y2]"""
def box_label(image, box, label='', color=(0,255,0), txt_color=(0,0,0)):
    p1 = (int(box[0]-15), int(box[1]-25))
    p2 = (int(box[2]+15), int(box[3]+25))
    cv2.rectangle(image, p1, p2, color, 4, cv2.LINE_AA)
    if label:
        center_x = (p1[0] + p2[0]) // 2
        label_y  = p1[1] - 10
        w, h      = cv2.getTextSize(label, 0, 1, 2)[0]
        lx        = center_x - w // 2
        # 背板
        cv2.rectangle(
            image,
            (lx-10,   label_y-h-10),
            (lx+w+10, label_y+10),
            color, -1, cv2.LINE_AA
        )
        cv2.putText(image, label, (lx, label_y),
                    0, 1, txt_color, 2, cv2.LINE_AA)

"""影像串流路由：回傳 multipart/x-mixed-replace 連續 jpeg
    YOLO 物件追蹤、分級計數、紅線判斷、FPS 疊加"""
@app.route('/video_feed')
def video_feed():
    def gen():
        # 使用全域變數來儲存即時辨識結果與統計資訊
        global level_result, category_counts, latest_crop_b64, latest_crop_label, saved_ids

        # 初始化追蹤用集合與變數
        crossed_ids, touched_ids = set(), set()         # crossed_ids：已穿越紅線的 ID；touched_ids：已觸線擷取圖的 ID
        last_seen, prev_y2 = {}, {}                      # last_seen：記錄每個 ID 最後出現的時間；prev_y2：上次出現的 y2 座標
        start, frame_count = time.time(), 0              # 用於 FPS 計算的起始時間與幀數計數器
        threshold_y = size[1] - 350                      # 紅線的 y 座標位置（畫面底部往上 350 像素）
        label_map = {0: "Bottle", 1: "Tissue", 2: "Plastic"}  # 類別編號與文字標籤對照表

        while recognition_active:  # 若辨識開啟，則持續進行影像讀取與分析
            ret, frame = cap1.read()                    # 讀取攝影機畫面
            if not ret:
                continue

            disp = frame.copy()                         # 建立一份畫面副本，用來畫框與顯示
            results = model.track(frame, iou=0.3, conf=0.5, persist=True, device="cuda")  # 使用 YOLO 模型進行物件追蹤
            now = time.time()

            # 移除超過 5 秒未出現的 tracking ID
            for tid, t0 in list(last_seen.items()):
                if now - t0 > 5:
                    last_seen.pop(tid)
                    crossed_ids.discard(tid)
                    touched_ids.discard(tid)
                    prev_y2.pop(tid, None)

            for box in results[0].boxes:  # 遍歷 YOLO 模型偵測出的每個物件框
                cls = int(box.cls[0])                         # 取得類別編號
                label = label_map.get(cls, "未知")             # 查表轉成文字標籤
                level_result = label                          # 更新目前全域辨識結果

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())           # 解析出 bounding box 座標
                tid = int(box.id.item()) if box.id is not None else None  # 若有 ID，則轉為整數

                if tid is not None:
                    last_seen[tid] = last_seen.get(tid, now)

                    # 判斷是否觸線，且該 ID 尚未被擷取過圖
                if tid and tid not in touched_ids and y1 <= threshold_y <= y2:
                    appear_duration = now - last_seen.get(tid, now)
                    if appear_duration >= 1.5:  # 若存在超過 1.5 秒才儲存
                        crop = frame[y1:y2, x1:x2]
                        ok, buf = cv2.imencode('.jpg', crop)
                        if ok:
                            # 產生唯一 hash key 避免重複
                            crop_hash = f"{label}_{x1}_{y1}_{x2}_{y2}"
                            if crop_hash not in saved_ids:
                                touched_ids.add(tid)
                                saved_ids.add(crop_hash)

                                latest_crop_b64 = base64.b64encode(buf).decode()
                                latest_crop_label = label

                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                os.makedirs('captures', exist_ok=True)
                                fn = f"captures/{label}_id{tid}_{ts}.jpg"
                                with open(fn, 'wb') as f:
                                    f.write(buf.tobytes())
                                print(f"[SAVE] 已存檔：{fn}")

                # 判斷物件是否從紅線上方穿越到下方
                if tid is not None:
                    prev = prev_y2.get(tid, 0)
                    if prev < threshold_y <= y2 and tid not in crossed_ids:
                        crossed_ids.add(tid)                 # 標記已穿越
                        category_counts[label] += 1          # 該類別累加計數
                        category_counts["Total"] += 1        # 總數加 1
                    prev_y2[tid] = y2                        # 更新該 ID 的 y2 座標

                # 在畫面中畫出 bounding box 與標籤
                box_label(disp, box.xyxy[0], label)

            # 畫出紅線（判斷基準線）與即時計算 FPS 顯示
            cv2.line(disp, (0, threshold_y), (disp.shape[1], threshold_y),
                     (0, 0, 255), 2, cv2.LINE_AA)
            frame_count += 1
            elapsed = now - start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(disp, f"FPS: {fps:.2f}",
                            (disp.shape[1] - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

            # 將畫面壓縮為 JPEG 並透過 yield 回傳給瀏覽器串流顯示
            ok2, buf2 = cv2.imencode('.jpg', disp)
            if ok2:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buf2.tobytes() + b'\r\n')

    # 使用 multipart 格式回傳 JPEG 影像流，讓前端 <img> 可以即時顯示
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/latest_snapshot')
def latest_snapshot():
    return jsonify({
        'label': latest_crop_label,
        'image': latest_crop_b64
    })

# ─── 自動錄影程式 ─────────────────────────────────────────────────────────
"""獨立執行：FFmpeg 分段录制 RTSP 流：
      - 用 TCP 拉流（更稳定）每隔 幾 秒分新檔錄影"""
import subprocess
import shlex

def ffmpeg_record_loop(rtsp_url: str):
    # 确保 records 目录存在
    os.makedirs("records", exist_ok=True)

    # FFmpeg 命令：
    cmd = (
        f"ffmpeg "
        f"-rtsp_transport tcp "    # 强制 TCP，较少丢包
        f"-i {shlex.quote(rtsp_url)} "
        f"-c copy "               # 不转码，直接拷贝容器/码流
        f"-f segment "            # 使用 segment 复用器
        f"-segment_time 60 "    # 每小时一个段
        f"-reset_timestamps 1 "   # 每个文件时间戳从 0 开始
        f"-strftime 1 "           # 启用 %Y%m%d_%H%M 变量
        f"records/rec_%Y%m%d_%H%M.mp4"
    )

    print("🔴 啟動 FFmpeg 錄影：" , cmd)
    # 用 Popen 不阻塞主线程
    subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ─── 首頁 & 統計 API ─────────────────────────────────────────────────────
"""首頁：渲染 index.html 並帶入今日統計"""
@app.route('/')
def index():
    sales = get_today_sales()
    return render_template('index.html',
        level_result  = level_result,
        total_gherkin = sales['Total'],
        level_S       = sales['Tissue'],
        level_A       = sales['Bottle'],
        level_B       = sales['Plastic']
    )

"""AJAX 拿前端顯示文字「垃圾：…」"""
@app.route('/get_level_result')
def get_level_result():
    return jsonify({'level_result': level_result})

"""歷史查詢頁面"""
@app.route('/time_search1')
def time_search1():
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('time_search1.html', today=today)

"""AJAX 拿到即時分類統計，用於首頁長條圖更新"""
@app.route('/get_statistics')
def get_statistics():
    return jsonify({
        'total_gherkin': category_counts['Total'],
        'level_S':       category_counts['Tissue'],
        'level_A':       category_counts['Bottle'],
        'level_B':       category_counts['Plastic']
    })

"""點擊『儲存』，把目前統計寫入 DB 並清零"""
@app.route('/save', methods=['POST'])
def save_data():
    global category_counts
    if category_counts['Total'] > 0:
        save_classification_to_db(category_counts)
        category_counts = {k:0 for k in category_counts}
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'no_data'})

"""歷史查詢 AJAX：依日期回傳所有紀錄"""
@app.route('/select_date', methods=['POST'])
def select_date():
    sel = request.json.get('date')
    if not sel:
        return jsonify([]), 200

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT `date`,`time`,`Total`,`Tissue`,`Bottle`,`Plastic`
                FROM `sale`
                WHERE `date` = %s
                ORDER BY `time`
            """, (sel,))
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        t = r['time'].strftime('%H:%M:%S')   if hasattr(r['time'], 'strftime') else str(r['time'])
        out.append({
            'date':    d,
            'time':    t,
            'Total':   int(r['Total']),
            'Tissue':  int(r['Tissue']),
            'Bottle':  int(r['Bottle']),
            'Plastic': int(r['Plastic'])
        })
    return jsonify(out), 200

# ─── 感測器頁面 & 資料 API ───────────────────────────────────────────────
"""接收外部感測器 POST JSON：
    - 永遠存到 sensor_data 讓前端 /sensor_data 可讀
    - 拆解成 5 組距離，若有小於 30 且距離上次寫入 >1s，寫入 sensor_log"""
@app.route('/data', methods=['POST'])
def receive_data():
    global sensor_data, last_db_write_time

    sensor_data = request.get_json(force=True) or {}
    #print("收到感測器資料：", sensor_data)

    raw   = sensor_data.get('data','')
    parts = raw.split(',')
    if len(parts) == 5:
        try:
            left, fl, f, fr, right = map(float, parts)
        except ValueError:
            print("[DATA ERROR] parse float fail:", parts)
        else:
            now_ts = time.time()
            # 條件：任一 < 30 且已隔 1 秒
            if any(d < 30.0 for d in (left, fl, f, fr, right)) \
               and (now_ts - last_db_write_time) > 1.0:

                conn = get_db_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                          INSERT INTO `sensor_log`
                            (`timestamp`,`left_dist`,`front_left_dist`,
                             `front_dist`,`front_right_dist`,`right_dist`)
                          VALUES (NOW(), %s, %s, %s, %s, %s)
                        """, (left, fl, f, fr, right))
                    conn.commit()
                    print(f"[DB] 記錄感測器：{parts}")
                except Exception as e:
                    print("[DB ERROR]", e)
                finally:
                    conn.close()

                last_db_write_time = now_ts
            else:
                print(f"[SKIP] 條件不符或 1s 未到：{parts}")
    else:
        print("[DATA ERROR] 格式錯誤，parts長度 != 5:", parts)

    # 永遠回 200，讓前端可取得最新 sensor_data
    return '', 200

"""前端不斷輪詢此路由，取得最新 sensor_data JSON"""
@app.route('/sensor_data')
def get_sensor_data():
    return jsonify(sensor_data or {}), 200

"""渲染即時感測器頁面模板"""
@app.route('/sensor_page')
def sensor_page():
    return render_template('sensor_page.html')

# ─── 啟動 App ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 如果還要背景錄影，就保留這行
    threading.Thread(target=lambda: ffmpeg_record_loop(rtsp_url), daemon=True).start()

    # 用 Flask 內建啟動（只給 Waitress 來 serve）
    app.run(host='0.0.0.0', port=7777, threaded=True)

