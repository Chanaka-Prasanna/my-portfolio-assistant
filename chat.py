
from add_to_vector_store import vector_store
from add_to_vector_store import llm

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


retriever = vector_store.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an personal assistant for Chanaka."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know and Tell me about my website https://chanakaprasanna.com."
    " keep the answer concise."
    "When someone asked who are you, you can say that you are a personal assistant for Chanaka."
    "\n\n"
    "{context}"
)

# ### Answer question ###
# system_prompt = (
#     "You are a personal assistant for Chanaka. "
#     "Your main task is to help with Chanaka's work and provide relevant information. "
#     "Use the following pieces of retrieved context to answer the questions. If you don't know the answer, say that you don't know and tell them about Chanaka's website at https://chanakaprasanna.com. "
#     "Keep the answer concise and friendly. "
#     "If the user asks for updates on projects, mention the most recent developments related to Chanaka's work, including blog posts, internships, research, and AI/ML projects. "
#     "If the question is about a specific technical aspect (e.g., ML models, coding issues, tools), refer to relevant context like the use of Scikit-Learn, cuml, AWS, React Native, or any other tools Chanaka is working with. "
#     "When someone asks, 'Who are you?' you can say, 'I am a personal assistant for Chanaka.' "
#     "If the question is related to Chanaka's personal skills or projects, provide a brief overview of his experience in AI/ML, coding, and other relevant areas. "
#     "If asked about Chanaka's internship, include details such as the training with Infinity Innovators, challenges faced, and tools used. "
#     "For questions regarding Chanaka's blog or LinkedIn posts, you should summarize recent topics and provide insights into how they align with Chanaka's interests in AI and ML. "
#     "When it comes to problem-solving, whether it's with machine learning models or coding challenges, you can reference specific examples and techniques that Chanaka is using, like SMOTE for balancing datasets or TfidfVectorizer for text data processing. "
#     "If the query is about software tools, explain Chanaka's work with tools or any other relevant technologies. "
#     "For questions about Chanaka's educational background or goals, refer to his final-year undergraduate status, career aspirations in AI/ML, and his pursuit of an R&D position. "
#     "For any questions that fall outside of the scope of Chanaka's current context, let the user know that you're unable to answer and encourage them to explore Chanaka's website for more information. "
#     "\n\n"
#     "{context}"
# )

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)