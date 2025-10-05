import streamlit as st
import requests
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from config.settings import APP_TITLE, LOGO_PATH, APP_URL

#========================= Load vectorstore ===========================
load_dotenv()

save_path="Acidizing_faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE", "https://api.avalai.ir/v1"))

def load_faiss_index(save_path: str = "faiss_index"):
    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

faiss_vectorstore = load_faiss_index(save_path)

st.sidebar.markdown("### تهیه کننده: مجید مظلومی \n \n")
st.sidebar.markdown("### اداره آموزش و تجهیز نیروی انسانی شرکت ملی حفاری ایران \n \n")
st.sidebar.write("مرجع مطالب این چت بات فقط بخش اسیدکاری مجموعه مقررات کاری شرکت ملی حفاری ایران (NIDC Policy) است. \n \n")
sk = st.sidebar.number_input("تعداد متن استخراج شده:", min_value=3, max_value=10, value=3)
MAX_HISTORY = st.sidebar.number_input("تعداد چت ذخیره شده:", min_value=3, max_value=15, value=6)

#======================= Use LLM to prepare final response ====================
faiss_retriever = faiss_vectorstore.as_retriever(scearch_type="mmr",search_kwargs={"k": sk})

llm = ChatOpenAI(model="gpt-4o-mini",base_url=os.getenv("OPENAI_API_BASE", "https://api.avalai.ir/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),temperature=0)

chain_typ = st.sidebar.selectbox("انتخاب استراتژی ترکیب اسناد و آماده سازی خروجی",
                    ("stuff","refine"))
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_retriever,
    chain_type=chain_typ,
    return_source_documents=True
)

st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
        font-family: "Vazir", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.image(LOGO_PATH, use_container_width=False)
st.title(APP_TITLE)
st.title("🤖 چت‌بات متخصص اسید کاری حفاری")
st.title(" شما می توانید سوال در زمینه :red[اسیدکاری حفاری] بپرسید")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
# ساخت session state برای ورودی کاربر
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

query = st.chat_input("👤 پیام شما:")
st.session_state.user_input = query
if query:
    enhanced_query = f"""
    {query}
    
    لطفاً پاسخ را کامل و جامع از اسناد استخراج شده تهیه کنید و به منبع وفادار باشید.
    پاسخ ارائه شده اطلاعات کاملتری را نسبت به سوال مطرح شده ارائه کند و مطالبی که در اطراف موضوع اصلی وجود دارد را هم شامل بشود.
    فقط از اطلاعات موجود در اسناد استفاده کنید.
    """

if st.session_state.user_input:
    st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})

    messages = st.session_state.messages
    if len(messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
        messages = st.session_state.messages

    with st.spinner("در حال پاسخ دادن..."):
        try:
            docs = faiss_vectorstore.similarity_search(enhanced_query, k=sk)
            result = qa_chain.invoke({"query": enhanced_query})  

            output_text = ""
            for doc in result["source_documents"]:
                output_text += f"بخش: {doc.metadata['header']} - عنوان بخش: {doc.metadata['section_title'][:100]} - فصل: {doc.metadata['main_no']} - عنوان فصل: {doc.metadata['main_title']}\n\n"
            
            docs_list = []
            for doc in docs:
                doc_item = {
                    'metadata': f"بخش: {doc.metadata['header']} - فصل: {doc.metadata['main_no']} - عنوان فصل: {doc.metadata['main_title']}",
                    'content': doc.page_content
                }
                docs_list.append(doc_item)


            st.session_state["messages"].append({
                "role": "assistant", 
                "content":result['result'], 
                "documents_list":docs_list,
                "metadata": output_text.strip()
            })
                
        except requests.exceptions.RequestException as e:
            # خطای شبکه یا timeout
            st.error(f"⛔ خطای ارتباط با سرور: {e}")
        except Exception as e:
            # هر خطای غیرمنتظره‌ی دیگر
            st.error(f"⛔ خطای داخلی در برنامه: {e}")
                        
    # نمایش تاریخچه گفتگو
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧍‍♂️ شما:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"#### 🤖 مشاور:\n {msg['content']}")
    
        if "documents_list" in msg and msg["documents_list"]:
            st.markdown("**📚 متن مربوط به موضوع:**")
            for doc_item in msg["documents_list"]:
                st.markdown(f"**{doc_item['metadata']}**")
                st.markdown(f"<div style='color: blue; background-color: #fff0f0; padding: 10px; border-radius: 5px; border-right: 3px solid #ff4b4b;'> متن: {doc_item['content']}</div>", unsafe_allow_html=True)
                st.markdown("-------------------------------")
    
    
        if "metadata" in msg and msg["metadata"]:
            st.info("📜 **پیشنهاد عناوین برای جستجوی هدفمند**")
            st.write(msg['metadata'])


        st.markdown(50 * "=")

if st.sidebar.button("Clear"):
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.rerun()
