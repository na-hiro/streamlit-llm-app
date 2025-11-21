import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .env の読み込み（OPENAI_API_KEY を環境変数として読み込む）
load_dotenv()


def generate_answer(user_input: str, expert_type: str) -> str:
    """
    入力テキスト（user_input）と
    ラジオボタンの選択値（expert_type）を受け取り、
    LLMからの回答テキストを返す関数。
    """

    if not user_input.strip():
        return "質問が空です。テキストを入力してください。"

    # 選択された専門家の種類に応じてシステムメッセージを切り替え
    if expert_type == "Pythonエンジニア":
        system_content = (
            "あなたは経験豊富なPythonエンジニアです。"
            "初心者にも分かるように、具体例を交えながら丁寧に解説してください。"
        )
    elif expert_type == "Webエンジニア（フロントエンド）":
        system_content = (
            "あなたはWebフロントエンドに詳しいエンジニアです。"
            "HTML/CSS/JavaScriptやフレームワークについて、実践的な観点から分かりやすく説明してください。"
        )
    else:
        # 万が一想定外の値が来た場合のフォールバック
        system_content = (
            "あなたはユーザーを丁寧にサポートするAIアシスタントです。"
            "相手のレベルに合わせて、分かりやすく回答してください。"
        )

    # LangChain の ChatOpenAI を利用して LLM を呼び出す
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 利用可能なモデル名を指定
        temperature=0.7,
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)

    # response は AIMessage オブジェクトなので、content を返す
    return response.content


def main():
    st.set_page_config(
        page_title="LLM専門家相談アプリ",
        page_icon="🤖",
    )

    # アプリ概要・操作説明
    st.title("LLM専門家相談アプリ 🤖")
    st.write(
        """
このアプリは、OpenAIのLLMとLangChainを利用して、
選択した「専門家」と対話できる学習用デモアプリです。

**使い方**

1. 下のラジオボタンから相談したい「専門家の種類」を選びます  
2. テキストエリアに質問や相談内容を入力します  
3. 「送信」ボタンを押すと、LLMが専門家として回答を返します  

※このアプリは学習目的のサンプルです。実務で利用する場合は内容を十分に確認してください。
        """
    )

    # 専門家の種類選択（ラジオボタン）
    expert_type = st.radio(
        "相談したい専門家を選択してください：",
        ("Pythonエンジニア", "Webエンジニア（フロントエンド）"),
        horizontal=True,
    )

    # 入力フォーム（1つ）
    user_input = st.text_area(
        "質問内容を入力してください：",
        height=150,
        placeholder="例）PythonでWebスクレイピングを始めたいのですが、何から勉強すれば良いですか？",
    )

    # 送信ボタン
    if st.button("送信"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error(
                "OPENAI_API_KEY が設定されていません。.env または Streamlit のシークレット設定を確認してください。"
            )
        elif not user_input.strip():
            st.warning("質問内容を入力してください。")
        else:
            with st.spinner("LLMが回答を生成しています..."):
                answer = generate_answer(user_input, expert_type)

            st.subheader("回答")
            st.write(answer)


if __name__ == "__main__":
    main()
