
import streamlit as st
import pandas as pd
import datetime

# ============ تحميل البيانات ============
@st.cache_data
def load_data():
    return pd.read_csv("products_with_clusters.csv")

products_df = load_data()

# ============ إعداد الصفحة ============
st.set_page_config(page_title="🛍️ نظام توصية المنتجات", layout="wide")
st.title("🛍️ نظام توصية المنتجات باستخدام العنقدة و Apriori")
st.markdown("### منصة تفاعلية لاستكشاف توصيات المنتجات، تحليل العناقيد، وتقييم التشابه.")

# ============ القائمة الجانبية ============
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3595/3595455.png", width=60)
st.sidebar.title("📂 القائمة الرئيسية")
section = st.sidebar.radio("اختر القسم:", [
    "📋 تفاصيل منتج",
    "🤝 توصيات من نفس العنقود",
    "📊 مقارنة العنقدة",
    "🔗 توصيات Apriori",
    "📝 تقييم تشابه",
    "📈 إحصائيات"
])

# ============ تفاصيل المنتج ============
if section == "📋 تفاصيل منتج":
    st.header("📋 عرض تفاصيل منتج")
    product_names = sorted(products_df['ProductName'].unique())
    selected_product = st.selectbox("🔍 اختر منتج:", product_names)
    product_info = products_df[products_df['ProductName'] == selected_product]
    st.dataframe(product_info, use_container_width=True)

# ============ توصيات من نفس العنقود ============
elif section == "🤝 توصيات من نفس العنقود":
    st.header("🤝 توصيات من نفس العنقود")
    selected_product = st.selectbox("🔍 اختر منتج:", sorted(products_df['ProductName'].unique()))
    cluster_type = st.radio("🧠 اختر نوع العنقدة:", ['مع السعر', 'بدون السعر'], horizontal=True)
    cluster_col = 'Cluster_With_Price' if cluster_type == 'مع السعر' else 'Cluster_Without_Price'
    product_info = products_df[products_df['ProductName'] == selected_product]

    if not product_info.empty:
        product_cluster = product_info[cluster_col].values[0]
        recommendations = products_df[
            (products_df[cluster_col] == product_cluster) &
            (products_df['ProductName'] != selected_product)
        ]
        if not recommendations.empty:
            st.success(f"✅ تم العثور على {len(recommendations)} منتج مشابه.")
            st.dataframe(recommendations[['ProductName', 'Price', 'Brand', 'Category']], use_container_width=True)
        else:
            st.info("🚫 لا توجد منتجات مشابهة لهذا المنتج في نفس العنقود.")

# ============ مقارنة العنقدة ============
elif section == "📊 مقارنة العنقدة":
    st.header("📊 مقارنة نتائج العنقدة")
    try:
        comp_df = pd.read_csv("comparison_results.csv")
        st.dataframe(comp_df, use_container_width=True)
        st.bar_chart(comp_df.set_index("Cluster Type")["Average Score"])
    except FileNotFoundError:
        st.warning("⚠️ لم يتم العثور على ملف المقارنة 'similarity_comparison.csv'.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل المقارنة: {e}")

# ============ توصيات Apriori ============
elif section == "🔗 توصيات Apriori":
    st.header("🔗 توصيات Apriori")
    try:
        apriori_df = pd.read_csv("apriori_rules.csv")
        apriori_df['antecedents'] = apriori_df['antecedents'].astype(str)
        apriori_df['consequents'] = apriori_df['consequents'].astype(str)

        selected_product = st.selectbox("اختر منتج لرؤية توصيات Apriori:", sorted(products_df['ProductName'].unique()))
        related_rules = apriori_df[apriori_df['antecedents'].str.contains(selected_product, case=False)]

        if not related_rules.empty:
            st.success(f"✅ تم العثور على {len(related_rules)} توصية لـ {selected_product}")
            st.dataframe(related_rules[['antecedents', 'consequents', 'confidence', 'lift']], use_container_width=True)
        else:
            st.info("🚫 لا توجد توصيات Apriori لهذا المنتج.")
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على apriori_rules.csv. تأكد من وجود الملف في نفس مجلد التطبيق.")
    except Exception as e:
        st.error(f"❌ حدث خطأ: {e}")

# ============ تقييم تشابه ============
elif section == "📝 تقييم تشابه":
    st.header("📝 تقييم يدوي لتشابه المنتجات")
    cluster_type = st.radio("نوع العنقدة:", ['مع السعر', 'بدون السعر'], horizontal=True)
    cluster_col = 'Cluster_With_Price' if cluster_type == 'مع السعر' else 'Cluster_Without_Price'
    sample_size = st.slider("🔢 عدد المنتجات لتقييمها:", min_value=1, max_value=10, value=3)
    start_button = st.button("🚀 ابدأ التقييم")

    if start_button:
        results = []
        samples = products_df.sample(n=sample_size, random_state=42)

        for _, product in samples.iterrows():
            st.markdown(f"### 🟦 المنتج الأساسي: {product['ProductName']}")
            st.dataframe(product.to_frame().T)

            cluster_val = product[cluster_col]
            recommended = products_df[(products_df[cluster_col] == cluster_val) &
                                      (products_df['ProductID'] != product['ProductID'])]

            if recommended.empty:
                st.info("⚠️ لا يوجد منتجات مشابهة لهذا المنتج.")
                continue

            for _, rec in recommended.iterrows():
                with st.form(key=f"form_{product['ProductID']}_{rec['ProductID']}"):
                    st.markdown(f"#### ➡️ مقترح: {rec['ProductName']}")
                    st.dataframe(rec.to_frame().T)

                    score = st.slider("💬 أدخل تقييم التشابه (0-100):", 0, 100, 50)
                    submitted = st.form_submit_button("💾 حفظ التقييم")

                    if submitted:
                        results.append({
                            'BaseProduct': product['ProductName'],
                            'SuggestedProduct': rec['ProductName'],
                            'Score': score,
                            'ClusterType': cluster_type
                        })
                        st.success("✅ تم حفظ التقييم!")

        if results:
            result_df = pd.DataFrame(results)
            filename = f"manual_eval_{cluster_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_df.to_csv(filename, index=False)
            st.success(f"📁 تم حفظ التقييمات في الملف: {filename}")
            st.dataframe(result_df)

# ============ الإحصائيات ============

elif section == "📈 إحصائيات":
    st.header("📈 إحصائيات تفاعلية وديناميكية")

    with st.expander("📊 المقاييس العامة"):
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 عدد المنتجات", len(products_df))
        col2.metric("🧠 العناقيد (مع السعر)", products_df['Cluster_With_Price'].nunique())
        col3.metric("🧠 العناقيد (بدون السعر)", products_df['Cluster_Without_Price'].nunique())

        col4, col5, col6 = st.columns(3)
        col4.metric("💰 متوسط السعر", f"{products_df['Price'].mean():.2f}")
        col5.metric("⭐ متوسط التقييم", f"{products_df['Rating'].mean():.2f}")
        col6.metric("💲 أعلى سعر", f"{products_df['Price'].max():.2f}")

    chart_option = st.selectbox("📈 اختر نوع الإحصائية التي تريد عرضها:", [
        "📊 توزيع عدد المنتجات في كل عنقود (مع السعر)",
        "📂 توزيع المنتجات حسب الفئة (Category)",
        "🏷️ توزيع حسب الماركات (Brand)",
        "📦 Boxplot: السعر مقابل العناقيد",
        "💵 Histogram لتوزيع الأسعار",
        "⭐ خطي لتوزيع التقييمات"
    ])

    import matplotlib.pyplot as plt
    import seaborn as sns

    if chart_option == "📊 توزيع عدد المنتجات في كل عنقود (مع السعر)":
        st.bar_chart(products_df['Cluster_With_Price'].value_counts().sort_index())

    elif chart_option == "📂 توزيع المنتجات حسب الفئة (Category)":
        st.bar_chart(products_df['Category'].value_counts())

    elif chart_option == "🏷️ توزيع حسب الماركات (Brand)":
        brand_counts = products_df['Brand'].value_counts().head(10)
        fig, ax = plt.subplots()
        brand_counts.plot.pie(autopct='%1.1f%%', ax=ax, figsize=(6,6), ylabel="")
        st.pyplot(fig)

    elif chart_option == "📦 Boxplot: السعر مقابل العناقيد":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=products_df, x="Cluster_With_Price", y="Price", ax=ax)
        ax.set_title("Boxplot للسعر حسب العنقود")
        st.pyplot(fig)

    elif chart_option == "💵 Histogram لتوزيع الأسعار":
        fig, ax = plt.subplots()
        products_df['Price'].hist(bins=20, ax=ax)
        ax.set_xlabel("السعر")
        ax.set_ylabel("عدد المنتجات")
        ax.set_title("توزيع الأسعار")
        st.pyplot(fig)

    elif chart_option == "⭐ خطي لتوزيع التقييمات":
        fig, ax = plt.subplots()
        products_df['Rating'].plot(kind='line', title="توزيع التقييمات", ax=ax)
        st.pyplot(fig)


