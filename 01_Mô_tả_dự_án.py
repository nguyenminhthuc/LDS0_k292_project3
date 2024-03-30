###############################################################################
# Họ tên:   Nguyễn Minh Thức
# Lớp:      LDS0_k292 – ONLINE
# Email:    nguyenminhthuc1987@gmail.com
# Link:     https://lds0k292project3-fjkgkvwzxabyzuptxsnzkm.streamlit.app/
###############################################################################

import streamlit as st
import pandas as pd




st.set_page_config(page_title="Mô tả dự án")
df = pd.read_csv("data/OnlineRetail_cleaned.csv", index_col="Unnamed: 0")




st.markdown("# <center>Project 3:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Customer Segmentation</span><p>Học viên: <span style='font-weight:bold;'>Nguyễn Minh Thức</span></p></center>", unsafe_allow_html=True)
col1, padding1, col2, padding2, col3 = st.columns((8,2,100,2,8))
with col1:
    col1.write(' '*10)
with col2:
    col2.image("images/segmentation2.jpg")
with col3:
    col3.write(' '*10)




st.markdown("### Mục tiêu:")
st.markdown(""" Xây dựng mô hình nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung, điều này giúp cho doanh nghiệp:
- Xây dựng các chiến dịch tiếp thị tốt hơn
- Giữ chân nhiều khách hàng hơn
- Cải tiến dịch vụ
- Tăng khả năng mở rộng
- Tối ưu hóa giá
- Tăng doanh thu
""")




st.markdown("### Customer Segmetation sử dụng RFM")
col1, padding, col2 = st.columns((150,10,200))
with col1:
    col1.markdown("""
RFM nghiên cứu hành vi của khách hàng và phân nhóm dựa trên ba yếu tố đo lường:
- **Recency (R)**: đo lường số ngày kể từ lần mua hàng cuối
cùng (lần truy cập gần đây nhất) đến ngày giả định chung
để tính toán (ví dụ: ngày hiện tại, hoặc ngày max trong
danh sách giao dịch).
- **Frequency (F)**: đo lường số lượng giao dịch (tổng số lần
mua hàng) được thực hiện trong thời gian nghiên cứu.
- **Monetary Value (M)**: đo lường số tiền mà mỗi khách hàng
đã chi tiêu trong thời gian nghiên cứu
""")
with col2:
    col2.image("images/RFMcube.png")
    col2.image("images/rfm_highest_value_customer.png")




st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("### Dataset (ví dụ 10 dòng dữ liệu)")
st.dataframe(df.sample(10), hide_index=True)




st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("### Tiền xử lý dữ liệu")
st.image('images/Preprocessing.png')
st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.image('images/Preprocessing2.png')




st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
### Phân cụm RFM thủ công
* Dựa theo rule đơn giản như sau:
    - Chia theo tiêu chí về doanh thu (Monetary) và thời gian mua (Recency) => xác định nhóm mang lại doanh thu nhiều và nhóm có tiềm năng mang lại doanh thu
    - Frequency được sử dụng ít quan trọng hơn 2 tiêu chí trên
            """)
col1, padding, col2 = st.columns((200, 5, 200))
with col1:
    st.image("images/RFM_manual_rules.png")
with col2:
    st.image("images/RFM_manual_snakeplot.png")

st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
**ScatterPlot**  => các cluster tạo thành **các cluster tách biệt nhau hoàn toàn**
            """)
st.image("images/RFM_manual_scatterplot.png")

st.markdown('<div style="padding: 50px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""
**HeatMap** => các cluster khác nhau **có sự chênh lệch đáng kể về số lượng customer trong mỗi cluster**
* cao nhất là cluster OTHER có 1707 customer (~39.34%), 
* thấp nhất là cluster NEW có 216 customer (~4.98%)
            """)
st.image("images/RFM_manual_heatmap.png")

st.markdown('<div style="padding: 20px 5px;"></div>', unsafe_allow_html=True)
st.markdown("""### Nhận xét:
=> **cần build với các model + RFM**, sau đó so sánh kết quả từ model với RFM thủ công
""")