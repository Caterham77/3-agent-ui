import os
import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import dateparser
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from google_auth_oauthlib.flow import Flow


# --- Load environment variables ---
load_dotenv()

# OpenAI setup
openai_api_key = st.secrets["OPENAI_API_KEY"]
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

# Google OAuth2 setup
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CLIENT_SECRET_FILE = "GoogleCalendar_sec.json"

# --- Session Variables ---
current_credentials = None
current_calendar_id = None

# --- Authentication Functions ---
def login_user():
    global current_credentials
    global current_calendar_id

  

    flow = Flow.from_client_config(
        {
            "installed": {
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
    redirect_uri="http://localhost:8080"  # or another valid redirect for Streamlit Cloud if applicable
)



    current_credentials = flow.run_local_server(
        port=8080,
        open_browser=True,
        success_message="✅ You may now close this window.",
        authorization_prompt_message=""
    )

    print("✅ User authorized.")

    current_calendar_id = get_primary_calendar_id(current_credentials)
    if current_calendar_id:
        print(f"✅ Calendar ID for session: {current_calendar_id}")
    else:
        print("⚠️ Using 'primary' as fallback calendar ID.")
        current_calendar_id = 'primary'


def get_primary_calendar_id(credentials):
    try:
        calendar_service = build('calendar', 'v3', credentials=credentials)
        calendar_list = calendar_service.calendarList().list().execute()
        for calendar_entry in calendar_list.get('items', []):
            if calendar_entry.get('primary', False):
                return calendar_entry.get('id')
    except Exception as e:
        print(f"❌ DEBUG: Failed to retrieve primary calendar ID: {e}")
    return None

# --- Calendar Functions Wrapped as Tools ---
@tool
def create_google_calendar_event(task_name: str, due_on: str = "today", time: str = None) -> str:
    """Creates an event in Google Calendar. Can optionally include time."""
    calendar_service = build('calendar', 'v3', credentials=current_credentials)

    try:
        if due_on.lower() == "today":
            due_on = str(datetime.now().date())
        else:
            parsed = dateparser.parse(due_on, settings={"PREFER_DATES_FROM": "future"})
            if not parsed:
                return f"❌ Could not understand the date: {due_on}"
            due_on = parsed.strftime("%Y-%m-%d")

        if time:
            start_datetime = f"{due_on}T{time}:00"
            end_hour = int(time.split(':')[0]) + 1
            end_time = f"{end_hour:02}:{time.split(':')[1]}"
            end_datetime = f"{due_on}T{end_time}:00"
            event = {
                'summary': task_name,
                'start': {
                    'dateTime': start_datetime,
                    'timeZone': 'Pacific/Auckland',
                },
                'end': {
                    'dateTime': end_datetime,
                    'timeZone': 'Pacific/Auckland',
                },
            }
        else:
            event = {
                'summary': task_name,
                'start': {'date': due_on, 'timeZone': 'Pacific/Auckland'},
                'end': {'date': due_on, 'timeZone': 'Pacific/Auckland'},
            }
        created_event = calendar_service.events().insert(calendarId=current_calendar_id, body=event).execute()
        print(f"✅ Created event: {created_event['htmlLink']}")
        return f"Event created: {created_event['htmlLink']}"
    except Exception as e:
        print(f"❌ Failed to create event: {e}")
        return str(e)


@tool
def delete_google_calendar_event(event_id: str) -> str:
    """Deletes a calendar event given the event ID."""
    calendar_service = build('calendar', 'v3', credentials=current_credentials)
    try:
        calendar_service.events().delete(calendarId=current_calendar_id, eventId=event_id).execute()
        return f"Event {event_id} deleted successfully."
    except Exception as e:
        return str(e)

@tool
def list_upcoming_events(max_results: int = 5) -> str:
    """Lists upcoming events."""
    calendar_service = build('calendar', 'v3', credentials=current_credentials)
    now = datetime.utcnow().isoformat() + 'Z'
    try:
        events_result = calendar_service.events().list(
            calendarId=current_calendar_id,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        if not events:
            return "You have no upcoming events."

        event_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_list.append(f"- {event.get('summary', 'No Title')} at {start}")
        return "\n".join(event_list)

    except Exception as e:
        return str(e)

@tool
def ask_uploaded_docs(question: str) -> str:
    """Answer a question using the content of uploaded documents."""
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        return "❌ No documents have been uploaded yet."

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=model_name),
        retriever=st.session_state.retriever
    )
    return qa_chain.run(question)

# --- Streamlit UI Logic ---
def main():
    st.title("📅 Google Calendar Assistant")
    st.caption("Chat with an AI agent that can create, list, or delete your calendar events and answer questions from uploaded files.")

    if st.button("🔴 Stop Agent and Clear Session"):
        st.session_state.clear()
        st.stop()

    # --- File uploader ---
    uploaded_files = st.file_uploader("Upload TXT, PDF, or DOCX documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    all_texts = []

    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/plain":
                text = file.read().decode("utf-8")
            elif file.type == "application/pdf":
                reader = PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = DocxReader(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            else:
                text = ""

            all_texts.append(Document(page_content=text, metadata={"name": file.name}))

        vectorstore = FAISS.from_documents(all_texts, OpenAIEmbeddings())
        st.session_state.retriever = vectorstore.as_retriever()
        st.session_state.files_uploaded = True
    elif "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.files_uploaded = False

    # --- Setup Agent ---
    if "agent_executor" not in st.session_state:
        login_user()

        llm = ChatOpenAI(model_name=model_name)
        tools = [
            create_google_calendar_event,
            delete_google_calendar_event,
            list_upcoming_events,
            ask_uploaded_docs
        ]
        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant that manages Google Calendar events. The current date is: {datetime.now().date()}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        st.session_state.chat_history = [SystemMessage(content=f"You are a helpful assistant that manages Google Calendar events. The current date is: {datetime.now().date()}")]

    # --- Display chat history ---
    for msg in st.session_state.chat_history:
        if isinstance(msg, SystemMessage) and "You are a helpful assistant" in msg.content:
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # --- User input ---
    if prompt := st.chat_input("What would you like to do?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            result = st.session_state.agent_executor.invoke({"input": prompt})
            st.markdown(result["output"])
            st.session_state.chat_history.append(AIMessage(content=result["output"]))

if __name__ == "__main__":
    main()
