from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_mysqldb import MySQL
from flask_session import Session
from datetime import datetime
from datetime import timedelta
import os
import re
import json
import fitz
import ollama
import secrets
import random
import subprocess
import requests
import MySQLdb.cursors


app = Flask(__name__)

# MySQL連接
app.secret_key = secrets.token_hex(16)
app.config["SESSION_TYPE"] = "filesystem"   
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'csie'
app.config['MYSQL_DB'] = 'pythonlogin'

# 閒置登出
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10) #閒置時間設定

mysql=MySQL(app)
Session(app)
CORS(app)  

# -------------------------------------------------------------
#  SoVITS  語音模型設定與 TTS 工具函式
# -------------------------------------------------------------
VOICE_MODELS = {
    "mann": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mann_e2_s50_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mann-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mann.wav",
        "prompt_text": "作者蘆園老師因為當時連載尚未結束，所以一直不想要改編成其他載體，可是製作方不斷釋出他們的誠意，他終究還是開口答應了"
    },
    "mrd": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mrd_e2_s102_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mrd-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mrd.wav",
        "prompt_text": "開始審理案件，被告我看看，在二零二三年接了太多鯊魚廣告，受到觀眾舉發控告。"
    },
    "andy": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/andy_e2_s98_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/andy-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\andy.wav",
        "prompt_text": "好玩有趣的創意，用一支手機拍攝起來，開心的分享在社群媒體上面，剛開始都沒有人看"
    }
}
TTS_HOST = "http://127.0.0.1:9880"
_current_sovits = [None]
_current_gpt    = [None]

def _switch_if_needed(cache, new_path, ep, tag):
    if cache[0] == new_path:
        return
    r = requests.get(f"{TTS_HOST}/{ep}", params={"weights_path": new_path}, timeout=120)
    r.raise_for_status()
    cache[0] = new_path

def tts(text: str, model_key: str, out_path="static/output.wav"):
    m = VOICE_MODELS[model_key]
    _switch_if_needed(_current_sovits, m["sovits"], "set_sovits_weights", "SoVITS")
    _switch_if_needed(_current_gpt,    m["gpt"],    "set_gpt_weights",   "GPT")
    params = {
        "text": text, "text_lang": "zh",
        "ref_audio_path": m["ref_audio_path"],
        "prompt_lang": "zh", "prompt_text": m["prompt_text"],
        "text_split_method": "cut1", "batch_size": 2, "sample_steps": 16,
        "media_type": "wav", "streaming_mode": "false"
    }
    r = requests.get(f"{TTS_HOST}/tts", params=params, timeout=240 )
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)

# -------------------------------------------------------------
#  其餘輔助函式：GPU 檢查、PDF 讀取、成績解析、評分、綜合評語
# -------------------------------------------------------------

def check_gpu():
    try:
        out = subprocess.check_output("nvidia-smi", shell=True).decode()
        return "GPU is available", out
    except subprocess.CalledProcessError:
        return "GPU is not available", ""

print(*check_gpu())

# 讀取PDF
def load_reference_answers_from_pdf(pdf_path):
    """
    從PDF中讀取Q:C:A:格式的題目
    返回格式: [(question, answer, chinese_translation), ...]
    如果找不到Q:C:A:格式，會回退到原始格式
    """
    doc = fitz.open(pdf_path)
    lines = []
    # 收集所有pdf之行
    for page in doc:
        for line in page.get_text().split("\n"):
            line = line.strip()
            if line and line.startswith(("Q:", "C:", "A:")):
                lines.append(line)
    
    doc.close()
    # 找出開頭為Q:C:A:
    qa_pairs = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and (
            lines[i].startswith("Q:") and 
            lines[i+1].startswith("C:") and 
            lines[i+2].startswith("A:")):
            
            question = lines[i][2:].strip()
            chinese = lines[i+1][2:].strip()
            answer = lines[i+2][2:].strip()
            
            qa_pairs.append((question, answer, chinese))
            i += 3
        else:
            i += 1
    
    if not qa_pairs:
        print("警告：沒有找到Q:C:A:格式，使用原始格式解析")
        doc = fitz.open(pdf_path)
        qas = []
        for page in doc:
            for line in page.get_text().split("\n"):
                if line.startswith(("Q:", "A:")):
                    qas.append(line.strip())
        doc.close()
        
        qs, ans = [], []
        for i in range(0, len(qas)-1, 2):
            if qas[i].startswith("Q:") and qas[i+1].startswith("A:"):
                qs.append(qas[i][2:].strip())
                ans.append(qas[i+1][2:].strip())
        qa_pairs = [(q, a, "") for q, a in zip(qs, ans)]   # 中文翻譯為空(有無都行)
    
    return qa_pairs

# 單題評分
score_re = re.compile(r"整體表現評分[:：]\s?(\d(?:\.\d)?)\s?分")

def parse_result(txt):
    m = score_re.search(txt)
    return float(m.group(1)) if m else 0

def evaluate_single_answer(ans, q, ref):
    prompt = f"""
你是一位英文老師，請依下列「評分標準」對學生的口說回答進行評分並給予回饋。  
學生的英文程度皆是初級，評分無需過多嚴格。
請用繁體中文回答，格式一定要包含：  
1. 整體表現評分（0-5 分，必須是整數）  
2. 錯誤說明與改善建議  
3. 參考答案（可簡短列出重點）

【評分標準】
5分:發音清晰、正確；語調自然。內容切題，表達流暢；語法與字彙偶有小錯誤但不影響溝通。  
4分:發音、語調大致正確；少數錯誤。內容切題；語法、字彙偶有錯誤但不影響溝通。  
3分:發音/語調時有錯誤，略影響理解；基本句型可用，但語法、字彙不足以完整表達。  
2分:發音/語調錯誤偏多；朗讀時常跳過難字；語法、字彙錯誤造成溝通困難。  
1分:發音/語調嚴重錯誤；句構錯亂，單字量嚴重不足，幾乎無法溝通。  
0分:未答或內容與題目無關。

【題目】{q}
【參考答案】{ref}
【學生回答】{ans}
"""
    res = ollama.chat(
        model="gemma3:27b",
        messages=[{"role": "user", "content": prompt}],
    )
    return res["message"]["content"]

# 綜合評語
def overall_comment(records):
    total = sum(r["score"] for r in records)
    avg   = total / len(records)
    detail = "\n\n".join([f"題目{i+1}:\n{r['result']}" for i,r in enumerate(records)])
    prompt = f"""
你是英文老師，學生平均 {avg:.2f}/5 分，
請用 80 字內給中文鼓勵式綜合評語。(請勿將提示語或字詞顯示出來)
評語的開頭請用:同學好，
細節：\n{detail}"""
    res = ollama.chat(model="gemma3:27b", messages=[{"role":"user","content":prompt}])
    return avg, res["message"]["content"]


# JSON(存所有歷史紀錄)
SCORES_FILE = 'test_scores.json'

def load_scores():
    """
    從 JSON 檔案載入分數紀錄
    """
    try:
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"載入分數檔案時發生錯誤: {e}")
        return []

def save_scores(scores):
    """
    將分數紀錄儲存到 JSON 檔案
    """
    try:
        with open(SCORES_FILE, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存分數檔案時發生錯誤: {e}")

def add_test_score(score):
    """
    新增測驗分數到紀錄中
    """
    scores = load_scores()
    score_record = {
        'score': round(score, 2),
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    scores.append(score_record)
    
    # 保留最近100筆記錄
    if len(scores) > 100:
        scores = scores[-100:]
    
    save_scores(scores)

# 取得最近的記錄
def get_recent_records(limit=5):
    """
    取得最近的測驗記錄
    """
    try:
        scores = load_scores()
        
        if not scores:
            return []
        
        recent_records = scores[-limit:] if len(scores) >= limit else scores
        return list(reversed(recent_records))  # 顯示在最上面為最新紀錄
        
    except Exception as e:
        print(f"取得最近記錄時發生錯誤: {e}")
        return []

def get_recent_average_score():
    """
    取得最近五筆紀錄的平均分數
    """
    try:
        scores = load_scores()
        
        if not scores:
            return None
        
        recent_scores = scores[-5:] if len(scores) >= 5 else scores  # 取得近5筆紀錄
        
        if recent_scores:
            total_score = sum(record['score'] for record in recent_scores)
            average = round(total_score / len(recent_scores), 1)
            return average
        
        return None
        
    except Exception as e:
        print(f"取得平均分數時發生錯誤: {e}")
        return None
    
def add_test_record(user_id, score):
    """將測驗記錄加入到 scores 表"""
    try:
        cursor = mysql.connection.cursor()
        
        # 插入測驗記錄到scores表，taken使用當前時間
        cursor.execute('''
            INSERT INTO scores (user_id, taken, score) 
            VALUES (%s, NOW(), %s)
        ''', (user_id, score))
        
        mysql.connection.commit()
        cursor.close()
        
        print(f"測驗記錄已保存：用戶ID {user_id}，分數 {score}")
        
    except Exception as e:
        print(f"保存測驗記錄時發生錯誤：{e}")
        if mysql.connection:
            mysql.connection.rollback()

# -------------------------------------------------------------
#  Flask Routes
# -------------------------------------------------------------
QUIZ_QUESTION_COUNT = 5 # 測驗題數

@app.route("/")
def home():
    # session.clear()
    recent_records = get_recent_records(5)
    return render_template("index.html", recent_records=recent_records)

@app.route("/feature")
def feature():
    return render_template("feature.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/example")
def example():
    return render_template("example.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    # 如果出現問題，輸出一條消息
    msg = ''

    # 檢查 POST 請求中是否存在username, password, email之必填項目
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # 使用MySQL檢查帳戶是否存在
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()

        # 驗證檢查
        if account:
            msg = '帳戶已存在！'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = '無效的電子郵件地址！'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = '用戶名只能包含字母和數字！'
        elif not username or not password or not email:
            msg = '請填寫表單！'
        else:  # 將帳戶新增到accounts中
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = '您已成功註冊！'
    elif request.method == 'POST':  # 表單為空 沒有數據
        msg = '請填寫表單！'
    return render_template('register.html', msg=msg)

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'loggedin' in session and session['loggedin']:
        return redirect(url_for('voice_select'))
    msg = ''
    
    # 檢查 POST 請求中是否存在email, password之必填項目
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        # 檢查帳戶是否存在
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s AND password = %s', (email, password,))
        account = cursor.fetchone()
        if account:  # 帳戶存在且密碼正確
            session.permanent = False
            session['loggedin'] = True
            session['id'] = account['id']
            session['email'] = account['email']  # 儲存email
            cursor.execute('UPDATE accounts SET login_count = login_count + 1 WHERE id = %s', [session['id']]) # 記錄登入次數
            mysql.connection.commit()
            cursor.close()
            return redirect(url_for('home'))
        else:  # 帳戶不存在或email/密碼不正確
            msg = 'e-mail或密碼不正確！'
            cursor.close()
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
   session.clear()
#    session.pop('loggedin', None)
#    session.pop('id', None)
#    session.pop('email', None)
   return redirect(url_for('login'))

@app.route('/profile')
def profile():
    # 檢查用戶是否已登入
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        cursor.close()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))

@app.route('/get-test-records', methods=['GET'])
def get_test_records():
    try:
        # 檢查用戶是否已登入
        if 'loggedin' not in session:
            return jsonify({'success': False, 'message': '請先登入'}), 401
            
        user_id = session['id']
        
        # 從資料庫查詢測驗記錄 (按時間降序排列)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''
            SELECT taken, score 
            FROM scores 
            WHERE user_id = %s 
            ORDER BY taken DESC
        ''', (user_id,))
        
        records = cursor.fetchall()
        cursor.close()
        
        # 轉換資料格式
        formatted_records = []
        for record in records:
            formatted_records.append({
                'taken': record['taken'].isoformat() if record['taken'] else None,
                'score': float(record['score'])
            })
        
        return jsonify({
            'success': True,
            'records': formatted_records
        })
        
    except Exception as e:
        print(f"Error getting test records: {e}")
        return jsonify({'success': False, 'message': '獲取記錄失敗'}), 500

@app.route('/change-password', methods=['POST']) 
def change_password():
    """處理密碼修改請求"""
    
    # 檢查用戶是否已登入
    if 'loggedin' not in session or not session['loggedin']:
        return jsonify({
            'success': False,
            'message': '請先登入'
        }), 401
    
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        if not current_password or not new_password:
            return jsonify({
                'success': False,
                'message': '請填寫所有必要欄位'
            })
        
        # 密碼長度驗證
        if len(new_password) < 3:
            return jsonify({
                'success': False,
                'message': '新密碼長度至少需要3個字符'
            })
        
        # 獲取舊密碼
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT password FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        
        if not account:
            cursor.close()
            return jsonify({
                'success': False,
                'message': '用戶不存在'
            })
        stored_password = account['password']
        
        # 舊密碼驗證
        if stored_password != current_password:
            cursor.close()
            return jsonify({
                'success': False,
                'message': '舊密碼錯誤'
            })
        
        # 更新新密碼
        cursor.execute('UPDATE accounts SET password = %s WHERE id = %s', (new_password, session['id']))
        mysql.connection.commit()
        
        # 檢查密碼是否成功更新
        if cursor.rowcount > 0:
            cursor.close()
            return jsonify({
                'success': True,
                'message': '密碼修改成功！'
            })
        else:
            cursor.close()
            return jsonify({
                'success': False,
                'message': '密碼修改失敗'
            })   
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'服務器錯誤: {str(e)}'
        })

@app.route("/voice_select", methods=["GET", "POST"])
def voice_select():
    # 檢查用戶是否已登入
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    if request.method == "GET":
        return render_template("voice_select.html")

    voice = request.form.get("voice_model", "mann")
    session["voice_model"] = voice
    return redirect(url_for("english"))

@app.route("/english", methods=["GET", "POST"])
def english():
    pdf_path = "GEPT_Complete.pdf"
    if not os.path.exists(pdf_path):
        return "找不到教材 PDF"
    qa_pairs = load_reference_answers_from_pdf(pdf_path)

    # "GET"表示新的測驗 清空session之數據(免得記錄到上次測驗的題目)
    if request.method == "GET":
        session.pop("records", None)
        session.pop("used_idx", None)
        session.pop("current_q", None)
        session.pop("q_no", None)
    
    # 初始化session
    session.setdefault("records", [])
    session.setdefault("used_idx", [])
    session.setdefault("q_no", 1)

    if "current_q" not in session:
        idx = random.randrange(len(qa_pairs))
        session["current_q"] = qa_pairs[idx]  # question, answer, chinese
        session["used_idx"].append(idx)

    if request.method == "POST":
        user_ans = request.form.get("user_answer", "").strip()
        current_q = session["current_q"]
        if len(current_q) == 3:
            q, ref, chinese = current_q
        else:
            q, ref = current_q
            chinese = ""
            
        if user_ans:
            result = evaluate_single_answer(user_ans, q, ref)
            score  = parse_result(result)
            session["records"].append({"question":q,"answer":user_ans,"result":result,"score":score})
        
        if request.form.get("action") == "finish" or session["q_no"] >= QUIZ_QUESTION_COUNT:
            return redirect(url_for("eng_result"))
        
        session["q_no"] += 1
        remain = [i for i in range(len(qa_pairs)) if i not in session["used_idx"]]
        if remain:  # 確保還有剩餘的題目
            idx = random.choice(remain)
            session["current_q"] = qa_pairs[idx]
            session["used_idx"].append(idx)

    current_q = session["current_q"]
    if len(current_q) == 3:
        q, ref, chinese_translation = current_q
    else:
        q, ref = current_q
        chinese_translation = ""
        
    last = session["q_no"] >= QUIZ_QUESTION_COUNT
    
    return render_template("english.html", 
                         question=q, 
                         chinese_translation=chinese_translation,
                         show_next=not last, 
                         show_end=last)

@app.route("/eng_result")
def eng_result():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('UPDATE accounts SET practice_count = practice_count + 1 WHERE id = %s', [session['id']])  # 紀錄練習次數
    cursor.execute('UPDATE scores SET score_id = score_id + 1 WHERE user_id = %s', [session['id']])
    mysql.connection.commit()
    cursor.close()

    if "records" not in session:
        return redirect(url_for("home"))

    while len(session["records"]) < QUIZ_QUESTION_COUNT:
        session["records"].append({"question":"未作答","answer":"未作答","result":"無資料","score":0})

    avg, comment = overall_comment(session["records"])
    add_test_record(session['id'], avg)
    add_test_score(avg)

    # 語音只朗讀平均分與綜合評語
    try:
        tts(f"你的平均分數是 {avg:.2f} 分。{comment}", session.get("voice_model", "mann"),out_path="static/output.wav")
    except Exception as e:
        print("TTS 失敗", e)
    return render_template("eng_result.html", records=session["records"], avg_score=avg, overall_evaluation=comment)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)  #"120.105.129.156:5050"
