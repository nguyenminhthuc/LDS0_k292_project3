###############################################################################
# GUI Project:  Customer Segmentation
# Link:         https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
# Lớp:          LDS0_k292 – ONLINE
# Họ tên:       Nguyễn Minh Thức
# Email:        nguyenminhthuc1987@gmail.com
###############################################################################

import streamlit as st
import pandas as pd
import numpy as np




# st.set_page_config(page_title="Tìm kiếm", layout="wide")
st.set_page_config(page_title="Tìm kiếm")




df = pd.read_csv('data/OnlineRetail_cleaned.csv', index_col='Unnamed: 0')
# print(df.head(2))
clustered_df = pd.read_csv('artifact/df_RFM_clusters_pyspark.csv', index_col='Unnamed: 0')
# print(clustered_df.head(2))
cluster_map = {
    0: "AVERAGE",
    1: "BEST",
    2: "RISK",
    3: "ABOVE_RISK"
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




st.markdown("# <center>Project 3:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Customer Segmentation</span><p>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span></p></center>", unsafe_allow_html=True)
_, col = st.columns((1,2))
with col:
  col.image("images/Search.png")




def print_data_2_cols(title, value):
    col1, _, col2 = st.columns((20, 5, 20))
    with col1:
      col1.markdown(title)
    with col2:
      col2.write(value)

st.subheader("Nhập mã khách hàng", divider='gray')
# customer_id = st.text_input("")
customer_id = st.selectbox(label=" ", placeholder="Nhập mã khác hàng, ví dụ: 14646", options=clustered_df.CustomerID.unique().tolist())
if customer_id != "":
    try:
        customer_id = int(customer_id)
        if customer_id not in clustered_df['CustomerID'].values:
            st.warning(f'Không tìm thấy khách hàng với id={customer_id}', icon="⚠️")
        else:
            st.subheader("Thông tin khách hàng", divider='gray')
            print_data_2_cols('* **CustomerID**', customer_id)
            print_data_2_cols('* **Quốc gia**', df[df.CustomerID==customer_id]['Country'].tolist()[0])

            cluster_id = clustered_df[clustered_df.CustomerID==customer_id]['prediction'].tolist()[0]
            print_data_2_cols('* **Cluster**', cluster_map[cluster_id] + f" ({cluster_id})")
            with st.expander("**Giải thích cluster như sau:**"):
              st.image("images/KMeans_LDS9_SnakePlot_analysis_label.png")
              st.image("images/KMeans_LDS9_SnakePlot_edited.png")

            print_data_2_cols('* **Ngày giao dịch lần đầu**', df[df.CustomerID==customer_id]['Date'].min())
            print_data_2_cols('* **Ngày giao dịch gần nhất**', df[df.CustomerID==customer_id]['Date'].max())

            print_data_2_cols('* **Recency**', clustered_df[clustered_df.CustomerID==customer_id]['Recency'].tolist()[0])
            print_data_2_cols('* **Frequency**', clustered_df[clustered_df.CustomerID==customer_id]['Frequency'].tolist()[0])
            print_data_2_cols('* **Monetary**', '$ {:,.2f}'.format(clustered_df[clustered_df.CustomerID==customer_id]['Monetary'].tolist()[0]))
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

            st.markdown("* **Chi tiết các giao dịch**")
            st.dataframe(df[df.CustomerID==customer_id], hide_index=True)
    except:
        st.warning(f'CustomerID={customer_id} không đúng', icon="⚠️")
        