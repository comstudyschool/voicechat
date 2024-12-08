import os
import openai
import streamlit as st
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

# 환경 변수 로드
#load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 텍스트를 벡터로 변환
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]

# 코사인 유사도 계산
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 유사도 기반 문서 검색
def search_similar_documents(df, query_embedding):
    df["similarity"] = df["embedding"].apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    return df.sort_values("similarity", ascending=False).head(3)

# GPT 응답 생성
def generate_response(documents, query):
    system_prompt = f"""
    You are an AI chatbot that answers questions based on the provided documents.
    Use the following documents to answer the user's query in detail:
    Document 1: {documents.iloc[0]["text"]}
    Document 2: {documents.iloc[1]["text"]}
    Document 3: {documents.iloc[2]["text"]}
    Respond in Korean.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500,
    )
    return response["choices"][0]["message"]["content"]

# Streamlit 애플리케이션
def main():
    st.title("PDF 기반 RAG 챗봇")

    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    query = st.text_input("질문을 입력하세요:")  # 항상 표시   

    if uploaded_file:
        # PDF에서 텍스트 추출
        with st.spinner("PDF에서 텍스트 추출 중..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("텍스트 추출 완료!")

        # 텍스트를 분할하여 벡터화
        with st.spinner("텍스트 임베딩 생성 중..."):
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            embeddings = [get_embedding(chunk) for chunk in chunks]
        st.success("임베딩 생성 완료!")

        # 데이터프레임 생성
        df = pd.DataFrame({"text": chunks, "embedding": embeddings})

        # 사용자 질문 처리
        # query = st.text_input("질문을 입력하세요:")
        if query:
            with st.spinner("답변 생성 중..."):
                query_embedding = get_embedding(query)
                top_docs = search_similar_documents(df, query_embedding)
                response = generate_response(top_docs, query)
            st.success("답변 생성 완료!")
            st.write("### 답변")
            st.write(response)

if __name__ == "__main__":
    main()
