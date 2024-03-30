###############################################################################
# Họ tên:   Nguyễn Minh Thức
# Lớp:      LDS0_k292 – ONLINE
# Email:    nguyenminhthuc1987@gmail.com
# Link:     https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
###############################################################################

import streamlit as st
import pandas as pd
import numpy as np
import pickle

import findspark
findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler

conf = SparkConf()\
            .setMaster("local")\
            .setAppName("03_predict")
SparkContext.setSystemProperty('spark.executor.memory', '4g')
sc = SparkContext.getOrCreate(conf = conf)
spark = SparkSession(sc)


st.set_page_config(page_title="Dự đoán")
# sample data
df_sample = pd.read_csv("artifact/sample.csv")
# pyspark model
model = KMeansModel.load("artifact/model_KMeans_pyspark")
# Yeo-Johnson transformer
with open("artifact/yeojohnson_transformer", 'rb') as f:
  transformer = pickle.load(f)
# StandardScaler
with open("artifact/StandardScaler", 'rb') as f:
  scaler = pickle.load(f)




st.markdown("# <center>Project 3:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Customer Segmentation</span><p>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span></p></center>", unsafe_allow_html=True)
_, col = st.columns((1,2))
with col:
  col.image("images/predict.gif")




if "RFM" not in st.session_state:
    st.session_state.RFM = []

if "cluster_RFM" not in st.session_state:
    st.session_state.cluster_RFM = []

def new_RFM():
    st.session_state.RFM.append(
        {
            "Recency": st.session_state.Recency,
            "Frequency": st.session_state.Frequency,
            "Monetary": st.session_state.Monetary,
        }
    )

def predict_RFM():
    df = pd.DataFrame(st.session_state.RFM)
    transformed_df = transformer.transform(df.copy())
    transformed_df.columns = ['YJ_Recency', 'YJ_Frequency', 'YJ_Monetary']
    scaled_df = scaler.transform(transformed_df)
    scaled_df = pd.DataFrame(scaled_df, columns=['Recency', 'Frequency', 'Monetary'])
    # https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems
    pd.DataFrame.iteritems = pd.DataFrame.items
    ########### pandas DF -> pyspark DF
    sparkDF = spark.createDataFrame(scaled_df) 
    vec_assembly = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol='features')
    final_data = vec_assembly.transform(sparkDF)
    predictions = model.transform(final_data)
    ########### pyspark DF -> pandas DF
    df_predict = predictions.toPandas()
    df = df.assign(prediction=df_predict['prediction'].values)
    df['Cluster'] = df.prediction.map({
       0: "AVERAGE",
       1: "BEST",
       2: "RISK",
       3: "ABOVE_RISK"
    })
    st.session_state.cluster_RFM = df[['Cluster', 'Recency', 'Frequency', 'Monetary']]


st.subheader("Nhập thông tin RFM của khách hàng")
with st.expander("**Giải thích RFM như sau:**"):
  col1, padding, col2 = st.columns((150,10,200))
  with col1:
    col1.markdown("""
* **Recency (R)**: đo lường **số ngày kể từ lần mua hàng cuối cùng (lần truy cập gần đây nhất) đến ngày giả định chung** để tính toán (ví dụ: ngày hiện tại, hoặc ngày max trong danh sách giao dịch).
* **Frequency (F)**: đo lường **số lượng giao dịch** (tổng số lần mua hàng) được thực hiện trong thời gian nghiên cứu.
* **Monetary Value (M)**: đo lường **số tiền** mà mỗi khách hàng đã chi tiêu trong thời gian nghiên cứu
                """)
  with col2:
    col2.image("images/RFMcube.png")
    col2.image("images/rfm_highest_value_customer.png")
# Cho người dùng chọn nhập dữ liệu hoặc upload file
type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu trực tiếp", "Upload file"])

if type == "Nhập dữ liệu trực tiếp":
  with st.form("new_RFM", clear_on_submit=True):
    Recency = st.slider("Recency", 1, 365, 100, key="Recency")
    Frequency = st.slider("Frequency", 1, 1000, 5, key="Frequency")
    Monetary = st.slider("Monetary", 1, 1000000, 100, key="Monetary")
    st.form_submit_button("Thêm", on_click=new_RFM)
elif type == "Upload file":
    st.subheader("Upload file")
    # Upload file
    uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv"])
    if uploaded_file is not None:
        # Đọc file dữ liệu
        df = pd.read_csv(uploaded_file)
        print(df.columns.tolist())
        if len(df.columns) == 3 \
          and all(col in ['Recency', 'Frequency', 'Monetary'] for col in df.columns.tolist()) \
          and all(df[col].dtype in [np.dtype('int64'), np.dtype('float64')] for col in df.columns):
            st.session_state.RFM = df
        else:
            st.warning('file upload chứa dữ liệu không hợp lệ!', icon="⚠️")
        
        # st.write(df)
    with st.expander("**Yêu cầu file upload như sau:**"):
      st.markdown("""
* File có định dạng **.csv**
* Có 3 cột: **Recency**, **Frequency**, **Monetary** 
* Dữ liệu cho mỗi cột là kiểu **số nguyên hoặc số thập phân**
* Ví dụ:

| Recency  | Frequency  | Monetary  | 
|---|---|---| 
| 100  | 5  | 100.5  | 
| 100  | 5  | 8924  |
| 100  | 5  | 49076  |
                """)
      st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
      st.markdown("* Hoặc download **file mẫu** như sau:")    
      st.dataframe(df_sample, hide_index=True)
    

st.write("# Danh sách RFM")
with st.expander("**Giải thích cluster như sau:**"):
  st.image("images/KMeans_LDS9_SnakePlot_analysis_label.png")
  st.image("images/KMeans_LDS9_SnakePlot_edited.png")

_, col1, _, col2 = st.columns((30, 100, 50, 150))
with col1:
  if col1.button("Dự đoán"):
      if len(st.session_state['RFM']) > 0:
        predict_RFM()
      else:
         col1.warning('Vui lòng cung cấp thông tin RFM của khách hàng', icon="⚠️")
with col2:
  if col2.button("Tạo mới"):
    st.session_state['RFM'] = []
    st.session_state['cluster_RFM'] = []

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)

col1, _, col2 = st.columns((250, 50, 450))
with col1:
  RFM_df = pd.DataFrame(st.session_state.RFM)
  col1.dataframe(RFM_df, hide_index=True)
with col2:
  cluster_RFM_df = pd.DataFrame(st.session_state.cluster_RFM)
  col2.dataframe(cluster_RFM_df, hide_index=True)