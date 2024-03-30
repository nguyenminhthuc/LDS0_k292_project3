###############################################################################
# Họ tên:   Nguyễn Minh Thức
# Lớp:      LDS0_k292 – ONLINE
# Email:    nguyenminhthuc1987@gmail.com
# Link:     https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
###############################################################################

import streamlit as st




st.set_page_config(page_title="Build Model")




st.markdown("# <center>Project 3:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Customer Segmentation</span><p>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span></p></center>", unsafe_allow_html=True)
st.image("images/Unsupervised-Learning-Clustering.png")




st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""# Build Model
* ScikitLearn - LDS6:
    - RFM + KMeans
    - RFM + Hierachical Clustering
* Pyspark - LDS9:
    - RFM + KMeans
            """)

# LDS6 - RFM + KMeans
st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown(""" ## 1. RFM + KMeans - LDS6""")
st.markdown("### Xác định k-clusters")
col1, padding, col2 = st.columns((100, 5, 200))
with col1:
    # col1.write('\n'*500)
    col1.markdown("Sử dụng **Elbow method => k=5**")
with col2:
    col2.image("images/KMeans_LDS6_Elbow.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("### Build KMeans model với k=5")
st.markdown("""
**ScatterPlot**  => các cluster tạo thành **các cluster tách biệt nhau hoàn toàn**
            """)
st.image("images/KMeans_LDS6_ScatterPlot.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
**HeatMap** => các cluster khác nhau **không chênh lệch đáng kể về số lượng customer trong mỗi cluster**
* cao nhất là cluster 0 có 938 customer (~21.62%), 
* thấp nhất là cluster 3 có 808 customer (~18.62%)
""")
st.image("images/KMeans_LDS6_HeatMap.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""**SnakePlot**""")
col1, col2 = st.columns(2)
with col1:
    st.image("images/KMeans_LDS6_snakeplot_analysis.png")
with col2:
    st.image("images/KMeans_LDS6_SnakePlot.png")




# LDS6 - RFM + Hierarchical Clustering
st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown(""" ## 2. RFM + Hierarchical Clustering - LDS6""")
st.markdown("""### Xác định k-clusters
* Tính toán **linkage** và sử dụng **dendrogram** => **k=5**
""")
st.image("images/Hierarchical_LDS6_Dendrogram.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("### Build Hierarchical model với k=5")
st.markdown("""
**ScatterPlot**  => các cluster tạo thành **các cluster chưa hoàn toàn tách biệt nhau**, vẫn có sự **chồng lấp giữa các cluster 1,2,3,4**
            """)
st.image("images/Hierarchical_LDS6_ScatterPlot.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
**HeatMap** => các cluster khác nhau **có sự chênh lệch đáng kể về số lượng customer trong mỗi cluster**
* cao nhất là cluster 0 có 4279 customer (~98.62%)
* thấp nhất là cluster 3 có 2 customer (~0.05%)
""")
st.image("images/Hierarchical_LDS6_HeatMap.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""**SnakePlot**""")
col1, col2 = st.columns(2)
with col1:
    st.image("images/Hierarchical_LDS6_SnakePlot_analysis.png")
with col2:
    st.image("images/Hierarchical_LDS6_SnakePlot.png")




# LDS9 - RFM + KMeans
st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown(""" ## 3. RFM + KMeans - LDS9""")
st.markdown("""### Xác định k-clusters
* Sử dụng cả Silhouette Score và Elbow Method để lựa chọn k-clusters
* **=> k=4**
""")
col1, col2 = st.columns(2)
with col1:
    col1.markdown("<center><span style='font-weight:bold;'>Silhouette Score</span></center>", unsafe_allow_html=True)
    col1.image("images/KMeans_LDS9_Silhouette.png")
with col2:
    col2.markdown("<center><span style='font-weight:bold;'>Elbow Method</span></center>", unsafe_allow_html=True)
    col2.image("images/KMeans_LDS9_Elbow.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("### Build KMeans model với k=4")
st.markdown("""
**ScatterPlot**  => các cluster tạo thành **các cluster tách biệt nhau hoàn toàn**
            """)
st.image("images/KMeans_LDS9_ScatterPlot.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
**HeatMap** => **các cluster khác nhau không chênh lệch đáng kể về số lượng customer trong mỗi cluster**, 
* cao nhất là cluster 0 có 1250 customer (~28.81%), 
* thấp nhất là cluster 3 có 962 customer (~22.17%)
""")
st.image("images/KMeans_LDS9_HeatMap.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""**SnakePlot**""")
col1, col2 = st.columns(2)
with col1:
    st.image("images/KMeans_LDS9_SnakePlot_analysis.png")
with col2:
    st.image("images/KMeans_LDS9_SnakePlot_edited.png")




st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""# Kết luận:
* Đối với RFM + Hierarchical Clustering cho kết quả phân cụm không tốt, các cluster không tách biệt hoàn toàn với nhau
* Đối với RFM + KMeans cho kết quả tốt hơn so với RFM thủ công vì tỉ lệ về số lượng customer trong mỗi nhóm không chênh lệch nhiều
* => **Chọn RFM + KMeans với model của LDS9**
            """)