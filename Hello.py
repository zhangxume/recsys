import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import math
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

# 设置页面为宽模式
st.set_page_config(layout="wide")


@st.cache_data
def load_movie_info_dict():
    with open('movie_info_dict.json', 'r', encoding='utf-8') as fp:
        return json.load(fp)


# 获取热门电影
@st.cache_data
def get_top_movies(movie_info_dict, top_n=10):
    # 将电影信息转换为DataFrame
    movies_df = pd.DataFrame.from_dict(movie_info_dict, orient='index')
    # 确保DOUBAN_VOTES是数值类型，并填充缺失值
    movies_df['DOUBAN_VOTES'] = pd.to_numeric(movies_df['DOUBAN_VOTES'], errors='coerce').fillna(0)
    # 根据DOUBAN_VOTES降序排列，取前top_n部电影
    top_movies_df = movies_df.sort_values(by='DOUBAN_VOTES', ascending=False).head(top_n)
    # 转换回字典，并确保包含ID
    top_movies_dict = top_movies_df.to_dict(orient='index')
    for mid in top_movies_dict.keys():
        top_movies_dict[mid]["ID"] = mid  # 将电影ID作为信息的一部分加入
    return top_movies_dict


# 定义展示单个电影信息的函数
def display_single_movie_info(movie_info):
    # 创建一个容器
    with st.container(height=500):
        st.markdown(f"<h2 style='font-weight:bold; font-size:18px'>{movie_info['NAME']}</h2>", unsafe_allow_html=True)
        # 创建两列布局
        cols = st.columns(2)
        # 左侧是电影海报
        movie_id = movie_info["ID"]
        file_path = f'moviedata-10m/cover_images/{movie_id}.webp'
        # 如果电影海报存在，则使用该海报，否则使用默认海报
        if os.path.exists(file_path):
            cover_url = file_path
        else:
            cover_url = "./moviedata-10m/cover_default.webp"
        with cols[0]:
            with st.container(height=300):
                st.image(cover_url)

        # 右侧是电影信息
        with cols[1]:
            st.caption(f"**类型:** {movie_info['GENRES']}")
            st.caption(f"**导演:** {movie_info['DIRECTORS']}")
            st.caption(f"**上映日期:** {movie_info['RELEASE_DATE']}")
            st.caption(f"**制片国家/地区:** {movie_info['REGIONS']}")
            st.caption(f"**语言:** {movie_info['LANGUAGES']}")
            st.caption(f"**豆瓣评分:** {movie_info['DOUBAN_SCORE']}")
            # 在 Streamlit 中使用 caption 来显示标题
            st.caption("**故事梗概:**")
            # 使用 Streamlit 的 markdown 函数和 CSS 来实现内容的截断和显示
            st.caption(
                f"""
                <style>
                .storyline {{
                    display: -webkit-box;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    -webkit-line-clamp: 3;  # 限制在3行内显示文本
                    line-clamp: 3;  # 同样的效果，对于支持该属性的浏览器
                    max-width: 100%;
                }}
                </style>
                <div class="storyline" title="{movie_info['STORYLINE']}">
                    {movie_info['STORYLINE']}
                </div>
                """,
                unsafe_allow_html=True
            )

            # 如果您希望内容显示多行，可以调整 CSS 中的 white-space 属性


# 定义展示多个电影信息的函数
def display_multiple_movie_info(movie_info_list):
    num_cols = 3  # 一行显示三个电影卡片
    num_movies = len(movie_info_list)
    num_rows = math.ceil(num_movies / num_cols)
    for row in range(num_rows):
        # st.write(f"第 {row + 1} 行:")
        cols = st.columns(num_cols)  # 创建固定数量的列
        start_index = row * num_cols
        end_index = min((row + 1) * num_cols, num_movies)
        for i in range(num_cols):
            col_index = start_index + i
            if col_index < end_index:
                # 如果当前索引有电影信息，则显示
                movie_info = movie_info_list[col_index]
                with cols[i]:
                    display_single_movie_info(movie_info)
            else:
                # 否则，在列中放置一个空的占位符
                with cols[i]:
                    st.empty()


# 定义根据用户描述推荐电影的函数
def recommend_movies_based_on_description(user_description, top_n=5):
    # 将用户描述转换为TF-IDF向量
    user_tfidf = vectorizer.transform([user_description])
    # 计算与所有电影描述的余弦相似度
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    # 获取相似度最高的N个电影的索引
    most_similar_movie_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # 使用电影ID作为键获取推荐电影的详细信息
    recommended_movies_ids = [list(movie_info_dict.keys())[index] for index in most_similar_movie_indices]
    recommended_movies_info = [movie_info_dict[mid] for mid in recommended_movies_ids]

    # 包装推荐的电影信息，确保每个推荐包含电影ID
    recommended_movies_with_id = [{"ID": mid, **info} for mid, info in
                                  zip(recommended_movies_ids, recommended_movies_info)]

    return recommended_movies_with_id


# 定义一个函数来处理可能的NaN值和将信息转化为字符串
def safe_str(obj):
    return '' if obj is np.nan else str(obj)


# 假设这些是允许的用户ID
allowed_user_ids = ['9866', '6866', '6055', '5980', '12034', '227']

# Streamlit 应用界面
st.title("🎞️RecSys")
with st.spinner('Loading...'):
    # 初始化 Spark
    spark = SparkSession.builder \
        .appName("MovieRecommendation") \
        .master("local[*]") \
        .getOrCreate()
    # 加载模型
    model = ALSModel.load("./model")
    # 加载电影信息
    movie_info_dict = load_movie_info_dict()
    # 在初始化TF-IDF Vectorizer之前对corpus进行分词处理
    corpus = []
    for mid, info in movie_info_dict.items():
        # 使用safe_str函数确保NaN值被适当处理
        text_parts = [
            safe_str(info.get('STORYLINE', '')),
            safe_str(info.get('NAME', '')),
            safe_str(info.get('ALIAS', '')),
            safe_str(info.get('ACTORS', '')),
            safe_str(info.get('DIRECTORS', '')),
            safe_str(info.get('GENRES', '')),
            safe_str(info.get('LANGUAGES', '')),
            safe_str(info.get('REGIONS', '')),
            safe_str(info.get('RELEASE_DATE', '')),
            safe_str(info.get('TAGS', '')),
            safe_str(info.get('YEAR', '')),
            # 根据需要添加其他字段
        ]

        text = " ".join(text_parts)
        segmented_text = " ".join(jieba.cut(text))
        corpus.append(segmented_text)
    # 初始化TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

st.markdown("请输入您的用户ID：(Examples `9866` `6866` `6055` `5980` `12034` `227`)")
user_id_input = st.text_input("请输入您的用户ID：", label_visibility="collapsed")
st.divider()
# 用户描述输入
st.markdown(
    "忘记电影名了吗？试试提供类型、导演、剧情或关键元素描述来寻找。无论是特殊的情节、导演特色，还是电影的某个场景，只需一些线索，我们可以帮您找回那部遗失的电影。")
st.markdown(
    """
    - `斯皮尔伯格导演的一部关于虚拟现实世界的科幻电影，里面有一个名为绿洲的游戏，玩家在游戏中寻找彩蛋，故事发生在未来世界。`
    - `主角伪装成医生、律师甚至是飞行员，全美国各地都留下了他的足迹。记得他用这些身份骗了很多钱，FBI都在追捕他。`
    """
)

user_description_input = st.text_area("描述你想看的电影类型或内容：", height=100, label_visibility="collapsed")
if st.button("推荐"):
    if user_id_input and not user_description_input:
        if user_id_input not in allowed_user_ids:
            st.write("欢迎新用户，为您推荐热门电影：")
            top_movies_dict = get_top_movies(movie_info_dict)
            # 展示热门电影信息
            display_multiple_movie_info(list(top_movies_dict.values()))
        else:
            user_id = int(user_id_input)
            # 为该用户生成推荐
            with st.spinner("In progress..."):
                userRecs = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["user_id"]), 5)
                recs_df = userRecs.toPandas()
                recommendations = recs_df['recommendations'][0]
            st.write("为您推荐的电影：")
            # 在循环中调用展示电影信息的函数
            movie_info_list = []

            for rec in recommendations:
                movie_id = rec['MOVIE_ID']  # 获取电影 ID
                movie_info = movie_info_dict[str(movie_id)]
                movie_info['ID'] = movie_id  # 确保电影信息中包含ID
                movie_info_list.append(movie_info)
            # 调用展示多个电影信息的函数
            # st.write(movie_info_list)
            display_multiple_movie_info(movie_info_list)
    elif user_description_input and not user_id_input:
        user_description_processed = " ".join(jieba.cut(user_description_input))
        # 用户描述输入推荐逻辑
        recommended_movies = recommend_movies_based_on_description(user_description_processed, top_n=5)  # 假设推荐前5部
        # 展示推荐电影信息
        st.write("根据您的描述，我们为您推荐以下电影：")
        display_multiple_movie_info(recommended_movies)
    else:
        st.write("请选择一种方式进行输入。")

spark.stop()
