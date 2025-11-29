import streamlit as st
import pandas as pd
import utils as ut
import datetime



# THANH MENU
st.title("CUSTOMER SEGEGMENTATION")
menu = ["Problem Statement",
        "Analysis & Results",
        "Customer Segmentation System 1",
        "Customer Segmentation System 2"]
choice = st.sidebar.selectbox('Menu', menu)

list_products=ut.products_list(100)

# NỘI DUNG PHẦN 'ĐẶT VẤN ĐỀ'
if choice == "Problem Statement":
    # NỘI DUNG 'VẤN ĐỀ CỦA DOANH NGHIỆP'    
    st.subheader('BUSINESS PROBLEM')
    text_business_problem=ut.readtxt('SEGMENTATION_BUSINESS_PROBLEM.txt')[0]
    st.write(text_business_problem)
    st.image('https://media.istockphoto.com/id/1864935946/vector/people-shopping-supermarket-female-and-male-characters-choose-products-in-store-consumers.jpg?s=612x612&w=0&k=20&c=vQQ_UXzY09N1qzlR__Nwa-04jjYTIuS1TjQu_dTMP0Y=',
              caption='Grocery store')
    # NỘI DUNG 'MỤC TIÊU CỦA DỰ ÁN'
    st.subheader('OBJECTIVE')
    text_objective=ut.readtxt('SEGMENTATION_OBJECTIVE.txt')[0]
    st.write(text_objective)
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6wAEdCVsMmgrLF_hbE4PanfLt7OrzWGYOwg&shttps://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6wAEdCVsMmgrLF_hbE4PanfLt7OrzWGYOwg&s',
              caption='RFM method')

# NỘI DUNG 'PHÂN TÍCH VÀ KẾT QUẢ'
elif choice == "Analysis & Results":
    # NỘI DUNG EDA 
    st.subheader("Exploratory data analysis")
    text_eda_intro=ut.readtxt('SEGMENTATION_EDA_INTRODUCTION.txt')[0]
    st.write(text_eda_intro)
    # SELECTBOX EDA
    selectbox_eda=['Total Amount',
                   'Product',
                   'Product Category',
                   'Recency',
                   'Frequency',
                   'Monetary']
    choice_eda=st.selectbox(label='EDA',options=selectbox_eda,label_visibility='collapsed')
    index_choice_eda=selectbox_eda.index(choice_eda)
    with st.container():
        col1,col2=st.columns(2)
        with col1:
            # hiển thị các hình ảnh biểu đồ tại đây
            st.image(f"{index_choice_eda}_eda.PNG", width=340, caption=f'Biểu đồ mô tả {choice_eda}')
        with col2:
            # hiển thị nội dung mô tả biểu đồ tại đây
            comment_content=ut.readtxt('SEGMENTATION_EDA_GRAPH_COMMENTS.txt')[1][index_choice_eda].strip()
            st.write(comment_content)

    # NỘI DUNG 'ĐÁNH GIÁ CÁC PHƯƠNG PHÁP CUSTOMER SEGMENTATION':
    st.subheader("CUSTOMER SEGMENTATION METHOD EVALUATION")
    text_eda_intro=ut.readtxt('SEGMENTATION_MODEL_ASSESSMENT.txt')[0]
    st.write(text_eda_intro)
    # SELECTBOX 'ĐÁNH GIÁ CÁC PHƯƠNG PHÁP CUSTOMER SEGMENTATION':
    selectbox_model=['Mannual method',
                     'Machine learning method']
    choice_model=st.selectbox(label='Method',options=selectbox_model,label_visibility='collapsed')
    index_choice_model=selectbox_model.index(choice_model)
    assessment_content=ut.readtxt('SEGMENTATION_MODEL_ASSESSMENT_CONTENT.txt')[1][index_choice_model]
    if choice_model=='Mannual methodmethod':
        with st.container():
            col1,col2=st.columns(2)
            with col1:
                #hiển thị bảng kết quả khi sử dụng tập luật
                df_rule_set=pd.read_csv('RULE_SET.csv',delimiter=',')
                st.table(df_rule_set)
            with col2:
                # hiển thị nội dung mô tả phần tập luật
                final_assessment_content=assessment_content.strip()
                st.write(assessment_content)
    elif choice_model=='Machine learning method':
        # hiển thị nội dung mô tả phần content-based recommender system
        final_assessment_content=assessment_content.strip()
        st.write(assessment_content)
        with st.container():
            col1,col2=st.columns(2)
            # hiển thị các hình ảnh biểu đồ mô tả kết quả clustering bằng sklearn tại đây
            st.write('The use of traditional machine learning')
            with col1:
                st.write('Outliers')
                st.image('0_model_assessment_sklearn.PNG',
                         width=340,
                         caption='Evaluation of silhouette score with outliers')
                st.image('1_model_assessment_sklearn.PNG',
                         width=340,
                         caption='Evaluation SSE score with outliers')
                df_sklearn_with_outlier=pd.read_csv('sklearn_with_outliers.csv',delimiter=',')
                st.dataframe(df_sklearn_with_outlier)
            with col2:
                st.write('None-outliers')
                st.image('2_model_assessment_sklearn.PNG',
                         width=315,
                         caption='Evaluation of silhouette score without outliers')
                st.image('3_model_assessment_sklearn.PNG',
                         width=315,
                         caption='Evaluation SSE score without outliers')
                df_sklearn_without_outlier=pd.read_csv('sklearn_without_outliers.csv',delimiter=',')
                st.dataframe(df_sklearn_without_outlier)
        with st.container():
            col3,col4=st.columns(2)
            # hiển thị các hình ảnh biểu đồ mô tả kết quả clustering bằng pyspark tại đây
            st.write('Sử dụng mô hình học máy dữ liệu lớn')
            with col3:
                st.write('Outliers')
                st.image('0_model_assessment_pyspark.PNG',
                         width=340,
                         caption='Evaluation of silhouette score with outliers')
                st.image('1_model_assessment_pyspark.PNG',
                         width=340,
                         caption='Evaluation SSE score with outliers')
            with col4:
                st.write('None-outliers')
                st.image('2_model_assessment_pyspark.PNG',
                         width=340,
                         caption='Evaluation of silhouette score without outliers')
                st.image('3_model_assessment_pyspark.PNG',
                         width=340,
                         caption='Evaluation SSE score without outliers')
   


elif choice=='Customer Segmentation System 1':
    # Sử dụng các điều khiển nhập
    # Nhập tên khách hàng tại đây
    st.subheader("CUSTOMER SEGMENTATION")
    name = st.text_input(label='Name',placeholder='Type your name here')
    # Nhập tuổi khách hàng
    age = st.number_input(label='Age',
                        min_value=1,
                        max_value=150,
                        step=1)
    # Nhập nghề khách hàng tại đây
    occupation = st.text_input(label='Job',placeholder='Type your job here')
    #Thông tin hóa đơn
    
    if "invoices" not in st.session_state:
        st.session_state.invoices = []
    if "invoice_index" not in st.session_state:
        st.session_state.invoice_index = 1
    
    st.write("Type the information of invoice here")
    with st.expander("Add new invoice"):
        with st.form(key=f"form_invoice_{st.session_state.invoice_index}"):
            min_date=datetime.datetime(2014,1,1)
            max_date=datetime.datetime(2015,12,30)
            invoice_date = st.date_input("Invoice date",value=min_date,min_value=min_date,max_value=max_date)
            products = st.multiselect("Select Purchased Product", options=list_products)
            quantity_dictionary = dict()
            for idx, p in enumerate(products):
                key = f"qty_{st.session_state.invoice_index}_{idx}"
                qty = st.number_input(f"Quantity {p}", min_value=1, step=1, value=1, key=key)
                quantity_dictionary[p] = int(qty)
            submitted = st.form_submit_button("Save the ")
        if submitted:
            if not products:
               st.warning("Please select at least one product")
            else:
                st.session_state.invoices.append({"Date": invoice_date,"Product": quantity_dictionary})
                st.success("Invoice savedsaved!")
                st.session_state.invoice_index += 1
        # Xuất DataFrame chỉ có 2 cột: Date, Product (dạng dictionary)
    if st.session_state.invoices:
        df = pd.DataFrame(st.session_state.invoices)
    
    if st.button('Customer segmentation'):
        if name and age and st.session_state.invoices:
            frequency=ut.F(df)
            recency=ut.R(df)
            monetary=ut.M(df)
            df_RCFM=ut.df_RFM(recency,frequency,monetary)
            customer_label=ut.customer_segmentation(df_RCFM)
            recommendation=ut.readtxt('SEGMENTATION_CUSTOMER_DESCRITION_ML.txt')[1][customer_label]
            customer=ut.customer_group(customer_label)
            st.write(f'{name} là **{customer}**')
            st.write(recommendation)
        else:
            st.warning('Please enter valid information')
        
elif choice == 'Customer Segmentation System 2':
    # Trường hợp 1: Nhập 1 khách hàng
    st.subheader('CUSTOMER SEGMENTATION')
    # Cho người dùng chọn giá trị R, F, M 1..4  (lưu ý đây chỉ là ví dụ đơn giản, trong thực tế R, F, M có thể là giá trị khác)      
    R_value = slider = st.slider("Select recency",0,600,0,1)
    F_value = slider = st.slider("Select frequency",0,100,0,1)
    M_value = slider = st.slider("Select monetary",0,2000,0,1)
    # Dựa trên 3 giá trị R, F, M để phân loại khách hàng vào các nhóm    
    if st.button("Customer segmentation"):
        customer=ut.assign_segment(R_value,F_value,M_value)
        list_customer_segments =["Main Customers",
                                 "Dormant Customers",
                                 "Promising Customers",
                                 "Potential New Customers",
                                 "Potential Customers",
                                 "Other"]

        customer_index=list_customer_segments.index(customer)
        recommendation=ut.readtxt('SEGMENTATION_CUSTOMER_-DESCRIPTION_RFM.txt')[1][customer_index]
        st.write(f'This customer belongs to **{customer}**. {recommendation}')



    # Trường hợp 2: Đọc dữ liệu từ file csv
    st.write("### Read csv file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Entered Data:")
        st.dataframe(df)
        if st.button("Segment Customers from File"):
            # Hiển thị kết quả ra dataframe
            st.write("Customers segmentation result:")
            result = []
            for index, row in df.iterrows():
                R_value = row['R']
                F_value = row['F']
                M_value = row['M']
                if R_value >= 3 and F_value >= 3 and M_value >= 3:
                    result.append("VIP")
                elif R_value >= 3 and F_value >= 2 and M_value >= 2:
                    result.append("Loyal")
                elif R_value >= 2 and F_value >= 2 and M_value >= 2:
                    result.append("Potential")
                elif R_value <=1 and F_value <= 1 and M_value <= 1:
                    result.append("Lost")
                else:
                    result.append("New")
            df['Customer Segment'] = result
            st.dataframe(df)

            st.write("Row-based Segmentation Results:")
            for index, row in df.iterrows():
                R_value = row['R']
                F_value = row['F']
                M_value = row['M']
                st.write(f"Customer {index+1}: R={R_value}, F={F_value}, M={M_value} --> ", end="")

                customer_clustering(R_value, F_value, M_value)


