###############################################################################
# Họ tên:   Nguyễn Minh Thức
# Lớp:      LDS0_k292 – ONLINE
# Email:    nguyenminhthuc1987@gmail.com
# Link:     https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
###############################################################################

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




df_rfm_agg = pd.read_csv("artifact/rfm_agg_KMeans_pyspark.csv")
clustered_df = pd.read_csv("artifact/df_RFM_clusters_pyspark.csv")
df = pd.read_csv("data/OnlineRetail_cleaned.csv")




st.set_page_config(page_title="Thống kê")




st.markdown("# <center>Project 3:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Customer Segmentation</span><p>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span></p></center>", unsafe_allow_html=True)
_, col = st.columns((1,3))
with col:
  col.image("images/report.jpg")

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

data = {}
for idx, row in df_rfm_agg.iterrows():
  data[row['Cluster']] = row['Count']

col1, col2 = st.columns(2)
with col1:
   fig = plot_count(data, 'Tỉ lệ Customer giữa các cluster')
   col1.pyplot(fig)
with col2:
   col2.markdown("<center><span style='font-weight:600;'>SnakePlot thể hiện RFM của các cluster</span></center>", unsafe_allow_html=True)
   col2.image("images/KMeans_LDS9_SnakePlot_edited.png")

st.markdown("""### Các cluster không có sự chênh lệch lớn về số lượng customer, nhưng có sự khác biệt về RFM 
* Có thể gán label mỗi cluster như sau:
  - **Cluster0: AVERAGE**
  - **Cluster1: BEST**
  - **Cluster2: RISK**
  - **Cluster3: ABOVE_RISK**
""")
st.image("images/KMeans_LDS9_SnakePlot_analysis_label.png")


st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)


cluster_map = {
  0: 'AVERAGE',
  1: 'BEST',
  2: 'RISK',
  3: 'ABOVE_RISK'
}
groupby_cluster =  clustered_df.groupby('prediction', as_index=False)['Monetary'].sum()
data = {}
for idx, row in groupby_cluster.iterrows():
  data[cluster_map[row['prediction']]] = row['Monetary']
fig = plot_count(data, "Tỉ lệ doanh thu giữa các cluster", explode=[0.2, 0, 0.3, 0])
st.pyplot(fig)

cluster_map2 = {
  'Cluster0': 'AVERAGE',
  'Cluster1': 'BEST',
  'Cluster2': 'RISK',
  'Cluster3': 'ABOVE_RISK'
}
fig, ax = plt.subplots(figsize=(6, 6))
sns.barplot(x=df_rfm_agg['Cluster'].map(cluster_map2), y=df_rfm_agg['MonetaryMean'], ax=ax)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Trung bình doanh thu của mỗi Customer trong từng cluster', fontsize=18)
st.pyplot(fig)


st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)


groupby_cluster = clustered_df.groupby('prediction', as_index=False)['Frequency'].sum()
fig, ax = plt.subplots(figsize=(6, 6))
sns.barplot(x=groupby_cluster['prediction'].map(cluster_map), y=groupby_cluster['Frequency'], ax=ax)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Tần suất mua hàng của từng cluster', fontsize=18)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 6))
sns.barplot(x=df_rfm_agg['Cluster'].map(cluster_map2), y=df_rfm_agg['FrequencyMean'], ax=ax)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title('Tần suất trung bình mua hàng của mỗi customer trong từng cluster', fontsize=18)
st.pyplot(fig)

combined_df = clustered_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'prediction']].copy()
combined_df['Cluster'] = combined_df['prediction'].map(cluster_map)
combined_df['Country'] = combined_df['CustomerID'].map(lambda id: df[df['CustomerID']==id]['Country'].tolist()[0])
combined_df['FromDate'] = combined_df['CustomerID'].map(lambda id: df[df['CustomerID']==id]['Date'].min())
combined_df['ToDate'] = combined_df['CustomerID'].map(lambda id: df[df['CustomerID']==id]['Date'].max())
# st.dataframe(combined_df, hide_index=True)
for k,v in cluster_map.items():
  st.markdown(f"### Danh sách các customer thuộc cluster {v}({k})")
  df_temp = combined_df[combined_df['prediction']==k]
  # print(df_temp.head(2))
  st.dataframe(df_temp, hide_index=True)