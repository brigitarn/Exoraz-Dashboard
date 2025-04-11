# == IMPORT ==
import streamlit as st
import pandas as pd
import altair as alt
import re
import io
from wordcloud import WordCloud
from collections import Counter
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import subprocess
if "setup_ran" not in st.session_state:
    subprocess.run(["python", "setup.py"], check=True)
    st.session_state.setup_ran = True

def predict_future_months(model, df, feature_scaler, target_scaler, features, n_months=6):
    from datetime import datetime
    future_predictions = []

    # Ambil data terakhir
    last_row = df[features].iloc[-1].copy()
    last_published = pd.to_datetime(df['published']).max()

    for i in range(n_months):
        next_published = last_published + pd.DateOffset(months=1)

        # Update fitur waktu
        next_input = last_row.copy()
        next_input['published_day'] = next_published.dayofweek
        next_input['published_hour'] = 12
        next_input['description_length'] = last_row['description_length']
        next_input['engagement_rate'] = last_row['engagement_rate']

        for col in ['Duration', 'Likes', 'Comments', 'Positive', 'Neutral', 'Negative']:
            next_input[col] = df[col].median()

        normalized_input = feature_scaler.transform([next_input.values])
        input_reshaped = normalized_input.reshape((1, 1, len(features)))
        pred = model.predict(input_reshaped)
        pred_inversed = target_scaler.inverse_transform(pred)
        pred_final = np.expm1(pred_inversed)[0][0]

        future_predictions.append({
            'Published': next_published,
            'Actual': np.nan,
            'Predicted': round(pred_final, 2),
            'PublishedLabel': next_published.strftime('%b %Y')
        })

        last_published = next_published
        last_row = next_input

    return pd.DataFrame(future_predictions)


# == CONFIG ==
st.set_page_config(page_title="EXORAZ Dashboard", layout="wide")

# == STYLE ==
st.markdown("""
    <style>
        .block-container {
            padding: 1rem;
        }
        .bordered-box {
            border: 2px solid #ffffff22;
            padding: 20px;
            border-radius: 10px;
            background-color: #0e1117;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# == LOAD DATA ==
df = pd.read_excel("Dataset.xlsx")
sentiment = pd.read_excel("Sentiment.xlsx")
df['published'] = pd.to_datetime(df['published'], utc=True)

# == WORDCLOUD ==
def generate_wordcloud_image(dataframe, text_column):
    text_data = dataframe[text_column].dropna().astype(str)
    if text_data.empty:
        raise ValueError("Tidak ada komentar yang tersedia untuk Word Cloud.")
    text = ' '.join(text_data.tolist())
    text = re.sub(r'[^\w\s]', '', text)
    word_counts = Counter(text.lower().split())
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2')
    return wordcloud.generate_from_frequencies(word_counts).to_image()

# == SIDEBAR FILTER ==
st.sidebar.header("🔍 Filter Data")
show_all_data = st.sidebar.checkbox("Tampilkan Seluruh Data", value=False)
future_toggle = st.sidebar.checkbox("Tampilkan Future Prediction", value=False)
month_map = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

if not show_all_data:
    years = df['published'].dt.year.sort_values().unique()
    selected_year = st.sidebar.selectbox("Tahun", years, index=len(years)-1)
    months = df['published'].dt.month.unique()
    selected_month = st.sidebar.selectbox("Bulan", sorted(months), format_func=lambda x: month_map[x])
    period = st.sidebar.radio("Periode", options=["MTD", "YTD"], index=0)

    # Tambahkan ini
from datetime import datetime
import pytz

if not show_all_data:
    selected_date = datetime(selected_year, selected_month, 1).replace(tzinfo=pytz.UTC)
else:
    selected_date = df['published'].max().to_pydatetime()

filtered_df = df.copy()
if not show_all_data:
    filtered_df = filtered_df[filtered_df['published'].dt.year == selected_year]
    if period == "MTD":
        filtered_df = filtered_df[filtered_df['published'].dt.month == selected_month]
    elif period == "YTD":
        start_date = pd.Timestamp(year=selected_year - 1, month=selected_month, day=1, tz="UTC")
        end_date = pd.Timestamp(year=selected_year, month=selected_month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
        filtered_df = filtered_df[(filtered_df['published'] >= start_date) & (filtered_df['published'] <= end_date)]
if not future_toggle:
    filtered_df = filtered_df[filtered_df['published'] <= pd.Timestamp.now(tz="UTC")]
filtered_sentiment = sentiment[sentiment['video_id'].isin(filtered_df['video_id'])]

# == HEADER ==
st.markdown("<h1 style='text-align: center;'>EXORA<span style='color: red;'>Z</span> Channel</h1>", unsafe_allow_html=True)

# == METRICS ==
col1, col2, col3 = st.columns(3)
def metric_box(title, value):
    return f"""
        <div style='border: 2px solid white; padding: 20px; border-radius: 10px; text-align: center;
        background-color: #0e1117; color: white; font-size: 20px; margin-bottom: 10px;'>
            <div style='font-weight: bold;'>{title}</div>
            <div style='font-size: 26px;'>{value}</div>
        </div>
    """
with col1:
    st.markdown(metric_box("Total Videos", filtered_df["video_id"].nunique()), unsafe_allow_html=True)
with col2:
    st.markdown(metric_box("Total Comments", int(filtered_df["Comments"].sum())), unsafe_allow_html=True)
with col3:
    st.markdown(metric_box("Total Viewers", int(filtered_df["Viewers"].sum())), unsafe_allow_html=True)

# == LSTM: PREDIKSI VIEWERS ==
import isodate
df['Duration'] = df['Duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
df['published_day'] = df['published'].dt.dayofweek
df['published_hour'] = df['published'].dt.hour
df['description_length'] = df['description'].apply(lambda x: len(str(x)))
df['engagement_rate'] = (df['Likes'] + df['Comments']) / (df['Viewers'] + 1)

features = ['Duration', 'Likes', 'Comments', 'Positive', 'Neutral', 'Negative',
            'published_day', 'published_hour', 'description_length', 'engagement_rate']
target = 'Viewers'

# Remove outliers & log transform
Q1, Q3 = df[target].quantile(0.25), df[target].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[target] < (Q1 - 1.5 * IQR)) | (df[target] > (Q3 + 1.5 * IQR)))]
df[target] = np.log1p(df[target])

# Scaling
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
df[features] = feature_scaler.fit_transform(df[features])
df[target] = target_scaler.fit_transform(df[[target]])

# Split
X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, test_size=0.4, random_state=32)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Model
from pathlib import Path
from tensorflow.keras.models import load_model

model_path = "lstm_model.h5"

if Path(model_path).exists():
    model = load_model(model_path, compile=False)  # 👈 Tambahkan compile=False
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse', metrics=['mae'])  # Kompilasi ulang
else:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(1, len(features))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse', metrics=['mae'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]
    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
    model.save(model_path)

# Predict
# Make sure video_id is still in the cleaned df
assert 'video_id' in df.columns, "video_id missing from df after cleaning"

# Reset index before use (important after filtering rows)
df = df.reset_index(drop=True)

# Predict on all data (same as before)
X_all = df[features].values
X_all_reshaped = X_all.reshape((X_all.shape[0], 1, X_all.shape[1]))
y_all = df[target].values.reshape(-1, 1)

y_all_pred = model.predict(X_all_reshaped)

# Inverse transform (MinMaxScaler + log1p)
y_all_actual = np.expm1(target_scaler.inverse_transform(y_all))
y_all_predicted = np.expm1(target_scaler.inverse_transform(y_all_pred))

# Create result_df with all matching columns from filtered df
result_df = pd.DataFrame({
    'video_id': df['video_id'],
    'Published': df['published'],
    'PublishedLabel': df['published'].dt.strftime('%b %Y'),
    'Actual': y_all_actual.flatten(),
    'Predicted': y_all_predicted.flatten()
})

assert len(df) == len(y_all_predicted), "Mismatch between df and prediction results!"


# Sort result
result_df['PublishedLabel'] = result_df['Published'].dt.strftime('%b %Y')
result_df = result_df.sort_values(by='Published').reset_index(drop=True)
# Tandai baris aktual dan prediksi historis
result_df['Type'] = 'Actual'
result_df['Predicted_only'] = result_df['Predicted']
result_df.loc[result_df['Actual'].isna(), 'Type'] = 'Future'  # jaga-jaga

if show_all_data:
    filtered_result = result_df.copy()
else:
    filtered_result = result_df[result_df['video_id'].isin(filtered_df['video_id'])]

# Simpan state prediksi future agar tidak dihitung ulang
if 'last_future_df' not in st.session_state:
    st.session_state.last_future_df = pd.DataFrame()
    st.session_state.last_future_date = None

# Cek apakah perlu prediksi future
if future_toggle:
    max_actual_date = pd.to_datetime(result_df[~result_df['Actual'].isna()]['Published']).max()

    if selected_date > max_actual_date:
        delta_months = (selected_date.year - max_actual_date.year) * 12 + (selected_date.month - max_actual_date.month)

        # Cek apakah perlu prediksi ulang
        if st.session_state.last_future_date != selected_date or st.session_state.last_future_df.empty:
            future_df = predict_future_months(model, df, feature_scaler, target_scaler, features, n_months=delta_months)
            future_df['Type'] = 'Future'
            future_df['Actual'] = np.nan
            st.session_state.last_future_df = future_df
            st.session_state.last_future_date = selected_date
        else:
            future_df = st.session_state.last_future_df

        filtered_result = pd.concat([filtered_result, future_df], ignore_index=True)

# Tambahkan LineType (untuk analisis lanjutan kalau mau pakai)
filtered_result['LineType'] = filtered_result.apply(
    lambda row: 'Actual' if not np.isnan(row['Actual']) else (
        'Future' if row['Type'] == 'Future' else 'Predicted'
    ),
    axis=1
)


# Cek apakah user pilih bulan setelah data aktual berakhir
max_actual_date = pd.to_datetime(filtered_result['Published']).max()
if selected_date > max_actual_date:
    delta_months = (selected_date.year - max_actual_date.year) * 12 + (selected_date.month - max_actual_date.month)

    if st.session_state.last_future_date != selected_date or st.session_state.last_future_df.empty:
        future_df = predict_future_months(model, df, feature_scaler, target_scaler, features, n_months=delta_months)
        future_df['Type'] = 'Future'
        future_df['Actual'] = np.nan
        st.session_state.last_future_df = future_df
        st.session_state.last_future_date = selected_date
    else:
        future_df = st.session_state.last_future_df

    filtered_result = pd.concat([filtered_result, future_df], ignore_index=True)

    filtered_result['LineType'] = filtered_result.apply(
        lambda row: 'Actual' if not np.isnan(row['Actual']) else (
            'Future' if row['Type'] == 'Future' else 'Predicted'
        ),
        axis=1
    )


# Display
st.markdown("#### 📈 Grafik Total Viewers per Tanggal Publish")

viewer_chart_df = filtered_result.copy()
viewer_chart_df['Published'] = pd.to_datetime(viewer_chart_df['Published'])
viewer_chart_df = viewer_chart_df.sort_values('Published')

# Titik transisi: terakhir ada data aktual
last_actual_date = viewer_chart_df[~viewer_chart_df['Actual'].isna()]['Published'].max()

# === Garis Actual ===
actual_chart = alt.Chart(viewer_chart_df[~viewer_chart_df['Actual'].isna()]).mark_line(
    color='blue', point=alt.OverlayMarkDef(color='yellow')
).encode(
    x=alt.X('Published:T', title='Tanggal Publish', axis=alt.Axis(format='%b %Y')),
    y=alt.Y('Actual:Q', title='Total Viewers'),
    tooltip=[
        alt.Tooltip('Published:T', title='Tanggal'),
        alt.Tooltip('Actual:Q', title='Viewers', format=',')
    ]
)

# === Garis Predicted (semua: historis dan future) ===
predicted_chart = alt.Chart(viewer_chart_df).mark_line(
    strokeDash=[4, 4], color='white'
).encode(
    x='Published:T',
    y='Predicted:Q',
    tooltip=[
        alt.Tooltip('Published:T', title='Tanggal'),
        alt.Tooltip('Predicted:Q', title='Prediksi', format=',')
    ]
)

# === Garis vertikal pemisah ===
rule_chart = alt.Chart(pd.DataFrame({
    'Published': [last_actual_date]
})).mark_rule(color='red', strokeDash=[4, 4], size=2).encode(
    x='Published:T'
)

# === Label angka Viewers (untuk titik-titik prediksi & actual) ===
valid_labels_df = viewer_chart_df[~viewer_chart_df['Predicted'].isna()]
label_chart = alt.Chart(valid_labels_df).mark_text(
    align='left', baseline='middle', dx=5, dy=-10, fontSize=11,
    color='white'
).encode(
    x='Published:T',
    y='Predicted:Q',
    text=alt.Text('Predicted:Q', format=',.0f')
)

# === Label angka Actual (untuk titik-titik actual) ===
actual_label_chart = alt.Chart(viewer_chart_df[~viewer_chart_df['Actual'].isna()]).mark_text(
    align='right', baseline='middle', dx=-8, dy=-10, fontSize=11,
    color='white', opacity=0.85
).encode(
    x='Published:T',
    y='Actual:Q',
    text=alt.Text('Actual:Q', format=',.0f')
)


# Gabungkan chart
final_chart = (actual_chart + predicted_chart + rule_chart + label_chart + actual_label_chart).properties(
    height=400
).interactive(bind_y=False)

st.altair_chart(final_chart, use_container_width=True)

st.write(f"YTD rows: {len(filtered_df)}")
if len(filtered_df) > 1000:
    st.warning("Terlalu banyak data, chart mungkin berat!")
    filtered_df = filtered_df.iloc[::5]


# == SENTIMEN & WORD CLOUD ==
st.markdown("### 📊 Distribusi Sentimen & Word Cloud")
col_sentiment, col_wordcloud = st.columns(2)

with col_sentiment:
    st.markdown("#### 📊 Distribusi Sentimen")
    sentiment_counts = filtered_sentiment['classification'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'Jumlah']
    sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x='Sentimen:N', y='Jumlah:Q',
        color=alt.Color('Sentimen:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                                      range=['green', 'orange', 'red']))
    ).properties(width=300, height=300)
    st.altair_chart(sentiment_chart, use_container_width=True)

with col_wordcloud:
    st.markdown("#### ☁️ Word Cloud Komentar")
    try:
        filtered_comments = filtered_sentiment[filtered_sentiment['text'].notnull()]
        wordcloud_image = generate_wordcloud_image(filtered_comments, 'text')
        st.image(wordcloud_image, use_container_width=True)
    except ValueError as e:
        st.warning(str(e))
