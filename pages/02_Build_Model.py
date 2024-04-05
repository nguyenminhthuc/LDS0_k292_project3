###############################################################################
# Họ tên:   Nguyễn Minh Thức
# Lớp:      LDS0_k292 – ONLINE
# Email:    nguyenminhthuc1987@gmail.com
# Link:     https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
###############################################################################

import streamlit as st




# st.set_page_config(page_title="Build Model", layout="wide")
st.set_page_config(page_title="Build Model")




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
st.image("images/Unsupervised-Learning-Clustering.png")




st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.header('Build Model', divider='gray')
st.markdown("""
* ScikitLearn - LDS6:
    - RFM + KMeans
    - RFM + Hierachical Clustering
* Pyspark - LDS9:
    - RFM + KMeans
            """)

# LDS6 - RFM + KMeans
st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.header("1. RFM + KMeans - LDS6", divider='gray')
st.subheader("Xác định k-clusters")
col1, padding, col2 = st.columns((100, 5, 200))
with col1:
    # col1.write('\n'*500)
    col1.markdown("Sử dụng **Elbow method => k=5**")
with col2:
    col2.image("images/KMeans_LDS6_Elbow.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.subheader("Build KMeans model với k=5")
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
st.header("2. RFM + Hierarchical Clustering - LDS6", divider='gray')
st.subheader("Xác định k-clusters")
st.markdown("""
* Tính toán **linkage** và sử dụng **dendrogram** => **k=5**
""")
st.image("images/Hierarchical_LDS6_Dendrogram.png")

st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
st.subheader("Build Hierarchical model với k=5")
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
st.header("3. RFM + KMeans - LDS9", divider='gray')
st.subheader('Xác định k-clusters')
st.markdown("""
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
st.subheader("Build KMeans model với k=4")
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
st.header('Kết luận', divider='gray')
st.image('images/Comparison_Table.jpg')
st.markdown("""
* Đối với RFM + Hierarchical Clustering cho kết quả phân cụm không tốt, các cluster không tách biệt hoàn toàn với nhau
* => **Chọn phân cụm khách hàng k = 4 với RFM & Kmeans - LDS9** vì số phân cụm nhỏ, đồng đều, dễ quản lý và phù hợp với 4 đặc tính mua hàng chủ yếu của khách hàng.
            """)