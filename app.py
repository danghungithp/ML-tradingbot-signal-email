import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import mplfinance as mpf
from pybitget import Client
from flask import Flask, render_template, request, jsonify
import os
from io import BytesIO
import base64
import threading

# Flask app và danh sách tín hiệu
app = Flask(__name__)
signals_data = []
chart_data = None  # Biểu đồ giao dịch

# Hàm vẽ biểu đồ giao dịch
def generate_chart(df):
    if df is None or df.empty:
        print("DataFrame để vẽ biểu đồ rỗng.")
        return None

    df_plot = df.copy()
    df_plot['t'] = pd.to_datetime(df_plot['t'])
    df_plot.set_index('t', inplace=True)
    df_plot = df_plot.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})

    # Tạo Series cho tín hiệu mua, bán và giữ
    buy_markers = pd.Series(index=df_plot.index, data=np.nan)
    sell_markers = pd.Series(index=df_plot.index, data=np.nan)
    hold_markers = pd.Series(index=df_plot.index, data=np.nan)

    # Gán giá trị tại các điểm có tín hiệu
    buy_markers[df_plot['ml_signal'] == 'BUY'] = df_plot['Close'][df_plot['ml_signal'] == 'BUY']
    sell_markers[df_plot['ml_signal'] == 'SELL'] = df_plot['Close'][df_plot['ml_signal'] == 'SELL']
    hold_markers[df_plot['ml_signal'] == 'HOLD'] = df_plot['Close'][df_plot['ml_signal'] == 'HOLD']

    # Tạo addplot cho tín hiệu
    ap_buy = mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='green', ylabel='BUY')
    ap_sell = mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='red', ylabel='SELL')
    ap_hold = mpf.make_addplot(hold_markers, type='scatter', markersize=50, marker='o', color='blue', ylabel='HOLD')

    # Tạo biểu đồ
    fig, ax = mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        volume=True,
        addplot=[ap_buy, ap_sell, ap_hold],
        returnfig=True,
        figsize=(10, 6),
        title="Candlestick Chart with Trading Signals"
    )

    # Lưu biểu đồ vào buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return chart_data

# Route chính hiển thị giao diện
@app.route("/")
def index():
    global chart_data
    return render_template("index.html", chart_data=chart_data, signals=signals_data)

# API để nhận email từ người dùng
@app.route("/subscribe", methods=["POST"])
def subscribe():
    email = request.form.get("email")
    if email:
        # Kiểm tra và thêm email vào file emails.txt
        file_path = "emails.txt"
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass  # Tạo file nếu chưa tồn tại

        with open(file_path, "r") as f:
            existing_emails = f.read().splitlines()

        if email not in existing_emails:
            with open(file_path, "a") as f:
                f.write(email + "\n")
            return jsonify({"message": f"Email {email} đã được đăng ký nhận tín hiệu!"}), 200
        else:
            return jsonify({"message": f"Email {email} đã tồn tại trong danh sách!"}), 200

    return jsonify({"message": "Vui lòng nhập email hợp lệ!"}), 400

class TradingSignalGenerator:
    def __init__(self):
        self.model = LinearRegression()
        # Cấu hình email
        self.sender_email = "mcpsoftware@gmail.com"  # Thay bằng email của bạn
        self.email_password = "jziw httm tesa gsfw"  # Thay bằng mật khẩu email hoặc app password
        self.receiver_email = ""
        # Thông tin xác thực API
        self.api_key = "bg_86188284c2aaccc2213329c20d721586"
        self.api_secret = "5775476af556ac9b7b1b579fa371f753939eafa2704aa766f41185ad4d78c9ee"
        self.api_passphrase = "Danghungit@85"

        # Khởi tạo client
        self.client = Client(self.api_key, self.api_secret, self.api_passphrase, use_server_time=False)

        # Thông tin trading
        self.symbol = "BTCUSDT_UMCBL"
        self.interval = "15m"
    def get_historical_data(self, from_time, to_time):
        try:
            candles = self.client.mix_get_candles(
                self.symbol,
                self.interval,
                from_time,
                to_time
            )

            if candles:
                print(f"Số lượng nến nhận được: {len(candles)}")
                clean_data = []
                for candle in candles:
                    try:
                        clean_candle = {
                            't': int(float(candle[0])),  # Timestamp
                            'o': float(candle[1]),       # Open
                            'h': float(candle[2]),       # High
                            'l': float(candle[3]),       # Low
                            'c': float(candle[4]),       # Close
                            'v': float(candle[5]),       # Volume (baseVol)
                        }
                        clean_data.append(clean_candle)
                    except (ValueError, IndexError) as e:
                        print(f"Bỏ qua nến không hợp lệ: {str(e)}")
                        continue

                if not clean_data:
                    print("Không có dữ liệu hợp lệ sau khi làm sạch")
                    return None

                df = pd.DataFrame(clean_data)
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                df = df.sort_values('t')
                return df
            else:
                print("Không nhận được dữ liệu từ API")
                return None

        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu: {str(e)}")
            return None

    def predict_signal(self, features, current_price):
        """Dự đoán tín hiệu mua/bán/giữ bằng mô hình Linear Regression"""
        if not hasattr(self.model, 'coef_'):
            return ['HOLD'] * len(features)
        
        predictions = self.model.predict(features)
        signals = []
        for pred, current in zip(predictions, current_price):
            if pred > current * 1.01:  # Giá dự đoán cao hơn 1%
                signals.append('BUY')
            elif pred < current * 0.99:  # Giá dự đoán thấp hơn 1%
                signals.append('SELL')
            else:
                signals.append('HOLD')
        return signals

    def analyze_price_action(self, df):
        df = df.copy()
        
        # Tính các chỉ báo kỹ thuật
        #df['SMA_9'] = df['c'].rolling(window=9).mean()
        #df['SMA_20'] = df['c'].rolling(window=20).mean()
        #df['SMA_50'] = df['c'].rolling(window=50).mean()
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        tr = df[['h', 'l', 'c']].apply(lambda x: max(x['h'] - x['l'], abs(x['h'] - x['c']), abs(x['l'] - x['c'])), axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        df['price_change_pct'] = df['c'].pct_change() * 100
        df['volume_change_pct'] = df['v'].pct_change() * 100
        
        # Nhận diện mô hình nến
        df['is_doji'] = (abs(df['o'] - df['c']) <= (df['h'] - df['l']) * 0.1)
        df['is_hammer'] = ((df['h'] - df['c'] <= (df['h'] - df['l']) * 0.1) & 
                           (df['c'] - df['o'] >= (df['h'] - df['l']) * 0.6))
        df['is_shooting_star'] = ((df['c'] - df['l'] <= (df['h'] - df['l']) * 0.1) & 
                                  (df['h'] - df['c'] >= (df['h'] - df['l']) * 0.6))
        df['is_bullish_engulfing'] = ((df['o'] < df['c'].shift(1)) & 
                                      (df['c'] > df['o'].shift(1)) & 
                                      (df['c'] - df['o'] > df['o'].shift(1) - df['c'].shift(1)))
        df['is_bearish_engulfing'] = ((df['o'] > df['c'].shift(1)) & 
                                      (df['c'] < df['o'].shift(1)) & 
                                      (df['o'] - df['c'] > df['c'].shift(1) - df['o'].shift(1)))
        
        # Tạo các cột tính năng
        feature_cols = ['price_change_pct', 'volume_change_pct',
                        'is_doji', 'is_hammer', 'is_shooting_star', 'is_bullish_engulfing', 'is_bearish_engulfing']
        valid_data = df[feature_cols].dropna()
        
        df['ml_signal'] = pd.Series([None] * len(df), dtype='object')
        
        if not valid_data.empty:
            current_prices = df.loc[valid_data.index, 'c']
            signals = self.predict_signal(valid_data, current_prices)
            df.loc[valid_data.index, 'ml_signal'] = signals
        
        return df

    def generate_signals(self, df):
        df = self.analyze_price_action(df)
        signals = df['ml_signal'].tolist()
        return signals, df

    def plot_chart(self, df):
        """Vẽ biểu đồ nến với tín hiệu mua và bán"""
        df_plot = df.copy()
        
        # Chuyển cột 't' thành datetime và đặt làm index
        df_plot['t'] = pd.to_datetime(df_plot['t'])
        df_plot.set_index('t', inplace=True)
        
        # Đổi tên cột theo chuẩn của mplfinance
        df_plot = df_plot.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        
        # Tạo Series cho tín hiệu mua và bán
        buy_markers = pd.Series(index=df_plot.index, data=np.nan)
        sell_markers = pd.Series(index=df_plot.index, data=np.nan)
        
        # Gán giá trị tại các điểm có tín hiệu
        buy_markers[df_plot['ml_signal'] == 'BUY'] = df_plot['Close'][df_plot['ml_signal'] == 'BUY']
        sell_markers[df_plot['ml_signal'] == 'SELL'] = df_plot['Close'][df_plot['ml_signal'] == 'SELL']
        
        # Tạo addplot
        ap_buy = mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='green')
        ap_sell = mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='red')
        
        # Vẽ biểu đồ nến với tín hiệu
        mpf.plot(df_plot, type='candle', style='charles', volume=True, addplot=[ap_buy, ap_sell],
                 title='Candlestick Chart with Trading Signals', figsize=(12, 8))

    def send_email(self, signal, symbol, price, time):
        """Gửi email thông báo tín hiệu đến tất cả email trong emails.txt"""
        file_path = "emails.txt"
        if not os.path.exists(file_path):
            print("File emails.txt không tồn tại. Không có email để gửi.")
            return

        with open(file_path, "r") as f:
            email_list = f.read().splitlines()

        if not email_list:
            print("Danh sách email rỗng. Không có email để gửi.")
            return

        # Nội dung email
        subject = f"Trading Signal: {signal}"
        text = f"Trading {symbol} Signal: {signal}\nPrice: {price}\nTime: {time}"

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.sender_email

        part = MIMEText(text, "plain")
        message.attach(part)

        # Gửi email đến từng người nhận
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.email_password)
                for recipient in email_list:
                    message["To"] = recipient
                    server.sendmail(self.sender_email, recipient, message.as_string())
                    print(f"Email sent successfully to {recipient} for {signal} at {time}")
        except Exception as e:
            print(f"Failed to send email: {e}")

# Bot chạy định kỳ
def run_bot():
    global signals_data, chart_data
    bot = TradingSignalGenerator()

    # File CSV để lưu dữ liệu
    data_file = "historical_data.csv"

    # Kiểm tra nếu file CSV tồn tại
    if os.path.exists(data_file):
        print("Đang tải dữ liệu từ file CSV...")
        df = pd.read_csv(data_file, parse_dates=['t'])
    else:
        print("Đang tải dữ liệu lịch sử 3 tháng từ API...")
        current_time = datetime.now()
        to_time = int(current_time.timestamp() * 1000)
        from_time = to_time - (3 * 30 * 24 * 60 * 60 * 1000)  # Lùi về 3 tháng (3 * 30 ngày)
        df = bot.get_historical_data(from_time, to_time)

        if df is None or df.empty:
            print("Không thể tải dữ liệu lịch sử. Dừng bot.")
            return

        # Lưu dữ liệu vào file CSV
        df.to_csv(data_file, index=False)
        print(f"Dữ liệu lịch sử đã được lưu vào {data_file}")

    # Tạo các cột tính năng và huấn luyện mô hình lần đầu
    print("Phân tích dữ liệu và huấn luyện mô hình lần đầu...")
    df = bot.analyze_price_action(df)
    df['future_close'] = df['c'].shift(-1)
    train_data = df.dropna(subset=['future_close'])

    if not train_data.empty:
        feature_cols = ['price_change_pct', 'volume_change_pct',
                        'is_doji', 'is_hammer', 'is_shooting_star', 'is_bullish_engulfing', 'is_bearish_engulfing']
        X_train = train_data[feature_cols].dropna()
        if not X_train.empty and X_train.shape[0] > 0:
            y_train = train_data.loc[X_train.index, 'future_close']
            bot.model.fit(X_train, y_train)  # Huấn luyện mô hình

    # Vòng lặp cập nhật dữ liệu mỗi 15 phút
    while True:
        try:
            print("Đang tải dữ liệu mới nhất (15 phút gần nhất)...")
            current_time = datetime.now()
            to_time = int(current_time.timestamp() * 1000)
            from_time = to_time - (15 * 60 * 1000)  # Lấy dữ liệu 15 phút gần nhất

            # Lấy dữ liệu mới
            new_data = bot.get_historical_data(from_time, to_time)
            if new_data is not None and not new_data.empty:
                # Kết hợp dữ liệu mới vào dữ liệu cũ
                df = pd.concat([df, new_data]).drop_duplicates(subset=['t']).reset_index(drop=True)

                # Lưu dữ liệu cập nhật vào file CSV
                df.to_csv(data_file, index=False)
                print(f"Dữ liệu đã được cập nhật và lưu vào {data_file}")

                # Phân tích dữ liệu và huấn luyện lại mô hình
                print("Phân tích dữ liệu và huấn luyện lại mô hình...")
                df = bot.analyze_price_action(df)
                df['future_close'] = df['c'].shift(-1)
                train_data = df.dropna(subset=['future_close'])

                if not train_data.empty:
                    X_train = train_data[feature_cols].dropna()
                    if not X_train.empty and X_train.shape[0] > 0:
                        y_train = train_data.loc[X_train.index, 'future_close']
                        bot.model.fit(X_train, y_train)  # Huấn luyện lại mô hình

                # Phát tín hiệu
                print("Tạo tín hiệu giao dịch...")
                signals, df = bot.generate_signals(df)
                latest_signal = signals[-1]
                latest_time = df['t'].iloc[-1]
                latest_price = df['c'].iloc[-1]

                # Cập nhật danh sách tín hiệu
                signals_data.append({
                    "time": latest_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": latest_signal,
                    "price": latest_price
                })

                # Cập nhật biểu đồ
                chart_data = generate_chart(df)

                print("Tín hiệu mới nhất:", latest_signal)
                print(df[['t', 'c', 'ml_signal']].tail())

                if latest_signal in ['BUY', 'SELL', 'HOLD']:
                    bot.send_email(latest_signal, "BTC", latest_price, latest_time)

            # Đợi 15 phút trước khi lặp lại
            time.sleep(15 * 60)
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(60)

# Chạy Flask và bot song song
if __name__ == "__main__":
    # Chạy bot trong một luồng riêng
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Chạy Flask
    app.run(debug=True, port=5001)