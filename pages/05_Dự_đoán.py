############################################################################################
# Lớp:                  LDS0_k292 – ONLINE
# Final project:        Customer Segmentation
# Link:                 https://lds0k292project3-aumweaa9bnsappppx4gj9pzv.streamlit.app/
# Giáo viên:            Khuất Thùy Phương -   Email: tubirona@gmail.com
# Học viên:             Nguyễn Minh Thức  -   Email: nguyenminhthuc1987@gmail.com
# Học viên cùng nhóm:   Trần Hạnh Triết   -   Email: trietcv.colab@gmail.com
############################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import pickle

import findspark
findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler




# st.set_page_config(page_title="Dự đoán", layout="wide")
st.set_page_config(page_title="Dự đoán")




conf = SparkConf()\
            .setMaster("local")\
            .setAppName("03_predict")
SparkContext.setSystemProperty('spark.executor.memory', '4g')
sc = SparkContext.getOrCreate(conf = conf)
spark = SparkSession(sc)



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
cluster_map = {
  0: 'AVERAGE',
  1: 'BEST',
  2: 'RISK',
  3: 'ABOVE_RISK'
}





# bug -> https://discuss.streamlit.io/t/anchor-tag/43688
# # https://discuss.streamlit.io/t/need-to-automatically-go-at-the-top-of-the-page/34728
# st.markdown("<div id='top'></div>", unsafe_allow_html=True)
# # https://www.linkedin.com/pulse/creating-floating-button-css-javascript-step-by-step-chowdhury-proma
# st.markdown(
#     """
#     <style>
#     .floating-button-div {
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#     }

#     .fb {
#         background-color: #4CAF50;
#         color: white;
#         border: none;

#         padding: 20px;
#         font-size: 16px;
#         cursor: pointer;
#         box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.5);
#     }

#     #myBtn:hover {
#         background-color: #555;
#     }
#     </style>
#     <script type="text/javascript">
#         var floatingButtonContainer = document.querySelector('.floating-button-div');
#         var scrollY = window.scrollY;


#         window.addEventListener('scroll', function() {
#             scrollY = window.scrollY;
#             floatingButtonContainer.style.top = scrollY + window.innerHeight - 150 + 'px';
#         });
 
#     </script>
#     <div class="floating-button-div">
#         <a target="_self" href="#top">
#             <button class="fb" id="myBtn" title="Đầu trang">Top</button>
#         </a>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )




st.markdown("# <center><span style='font-size:32px'>Final project<span>:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;font-size:46px'> Customer Segmentation</span><p>Giáo viên: <span style='font-weight:bold;'>Khuất Thùy Phương</span><br/>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span><br/>Học viên cùng nhóm: <span style='font-weight:bold;'>Trần Hạnh Triết</span></p></center>", unsafe_allow_html=True)
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
    df['Cluster'] = df.prediction.map(cluster_map)
    st.session_state.cluster_RFM = df[['Cluster', 'Recency', 'Frequency', 'Monetary']]

def plot_count(data, title, explode=None):
    # https://www.pythoncharts.com/matplotlib/pie-chart-matplotlib/

    fig, ax = plt.subplots(figsize=(6, 6))

    # https://stackoverflow.com/questions/59644751/show-both-value-and-percentage-on-a-pie-chart
    total = sum(i for i in data.values())
    my_fmt = lambda x: '{:.1f}%\n({:.0f})'.format(x, total*x/100)

    patches, texts, pcts = ax.pie(
                                    data.values(), labels=data.keys(), autopct=my_fmt,
                                    wedgeprops={'linewidth': 2.0, 'edgecolor': 'white'},
                                    textprops={'fontsize':12},
                                    # textprops={'size': 'x-large'},
                                    # startangle=90,
                                    explode=explode)
    # For each wedge, set the corresponding text label color to the wedge's face color.
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())
    plt.setp(pcts, color='black')
    plt.setp(texts, fontweight=400)
    ax.set_title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    # st.pyplot(fig)
    return fig

st.header("Nhập thông tin RFM của khách hàng", divider='gray')
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
  with st.expander("**Hướng dẫn nhập liệu và dự đoán**"):
     st.markdown('''
1. Kéo thả slider chọn số liệu phù hợp cho mỗi khách hàng
2. Nhấn "**Thêm**" để chuyển số liệu vào bảng RFM bên dưới
3. Nhấn "**Dự đoán**" để thực hiện việc dự đoán sau khi đã thêm dữ liệu RFM
4. Nhấn "**Tạo mới**" để xóa các dữ liệu RFM đã thêm/dự đoán
                 ''')
  with st.form("new_RFM", clear_on_submit=True):
    Recency = st.slider("Recency", 1, 365, 100, key="Recency")
    Frequency = st.slider("Frequency", 1, 1000, 5, key="Frequency")
    Monetary = st.slider("Monetary", 1, 350000, 100, key="Monetary")
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
            st.session_state.RFM = df.to_dict('records')
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
    

col1, col2 = st.columns((100, 100))
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


st.header("Danh sách RFM", divider='gray')
with st.expander("**Giải thích cluster như sau:**"):
  st.image("images/KMeans_LDS9_SnakePlot_analysis_label.png")
  st.image("images/KMeans_LDS9_SnakePlot_edited.png")



st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)

col1, _, col2 = st.columns((250, 50, 450))
with col1:
  RFM_df = pd.DataFrame(st.session_state.RFM)
  col1.dataframe(RFM_df, hide_index=True)
with col2:
  cluster_RFM_df = pd.DataFrame(st.session_state.cluster_RFM)
  col2.dataframe(cluster_RFM_df, 
                 hide_index=True,
                 width=550)

st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)

### Report & Visualization

if cluster_RFM_df.shape[0] > 0:
  # print(cluster_RFM_df)
  rfm_agg = cluster_RFM_df.groupby('Cluster').agg({
      'Recency':'mean',
      'Frequency':'mean',
      'Monetary':['mean', 'count']
  }).round(0)
  rfm_agg.columns = rfm_agg.columns.droplevel()
  rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean','Count']
  rfm_agg['Precent'] = np.round( (rfm_agg['Count']/rfm_agg.Count.sum())*100, 2 )
  rfm_agg = rfm_agg.reset_index()
  data = {}
  for idx, row in rfm_agg.iterrows():
    data[row['Cluster']] = row['Count']
  fig = plot_count(data, 'Tỉ lệ RFM giữa các cluster')
  st.pyplot(fig)

  st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)

  groupby_cluster =  cluster_RFM_df.groupby('Cluster', as_index=False)['Monetary'].sum()
  # print(groupby_cluster)
  fig, ax = plt.subplots(figsize=(14, 10))
  squarify.plot(
    sizes=groupby_cluster['Monetary'],
    text_kwargs={'fontsize':14, 'weight':'bold'},
    label=['{} \n{:0,.2f} $ \n ({}%)'.format(*groupby_cluster.iloc[i], np.round( (groupby_cluster.iloc[i, 1]/groupby_cluster.Monetary.sum())*100, 2 )) for i in range(0, len(groupby_cluster))], 
    alpha=0.5)
  plt.title('Tỉ lệ doanh thu giữa các cluster', fontsize=26, fontweight='bold')
  plt.axis('off')
  st.pyplot(fig)

  fig, ax = plt.subplots(figsize=(6, 6))
  sns.barplot(x=rfm_agg['Cluster'], y=rfm_agg['MonetaryMean'], ax=ax)
  ax.bar_label(ax.containers[0], label_type='edge')
  ax.set_title('Trung bình doanh thu trong từng cluster', fontsize=18)
  st.pyplot(fig)

  st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)

  groupby_cluster = cluster_RFM_df.groupby('Cluster', as_index=False)['Frequency'].sum()
  fig, ax = plt.subplots(figsize=(6, 6))
  sns.barplot(x=groupby_cluster['Cluster'], y=groupby_cluster['Frequency'], ax=ax)
  ax.bar_label(ax.containers[0], label_type='edge')
  ax.set_title('Tần suất mua hàng trong từng cluster', fontsize=18)
  st.pyplot(fig)

  fig, ax = plt.subplots(figsize=(6, 6))
  sns.barplot(x=rfm_agg['Cluster'], y=rfm_agg['FrequencyMean'], ax=ax)
  ax.bar_label(ax.containers[0], label_type='edge')
  ax.set_title('Tần suất trung bình mua hàng trong từng cluster', fontsize=18)
  st.pyplot(fig)

  st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)

  groupby_cluster = cluster_RFM_df.groupby('Cluster', as_index=False)['Recency'].sum()
  fig, ax = plt.subplots(figsize=(6, 6))
  sns.barplot(x=groupby_cluster['Cluster'], y=groupby_cluster['Recency'], ax=ax)
  ax.bar_label(ax.containers[0], label_type='edge')
  ax.set_title('Recency mua hàng trong từng cluster', fontsize=18)
  st.pyplot(fig)

  fig, ax = plt.subplots(figsize=(6, 6))
  sns.barplot(x=rfm_agg['Cluster'], y=rfm_agg['RecencyMean'], ax=ax)
  ax.bar_label(ax.containers[0], label_type='edge')
  ax.set_title('Trung bình Recency mua hàng trong từng cluster', fontsize=18)
  st.pyplot(fig)

for v in cluster_RFM_df.Cluster.unique().tolist():
    st.subheader(f"Danh sách các RFM thuộc cluster {v}", divider='gray')
    df_temp = cluster_RFM_df[cluster_RFM_df['Cluster']==v]
    st.dataframe(df_temp, hide_index=True)