import streamlit as st
import pandas as pd
import re
import os
import pandas as pd
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain

df = pd.read_csv('Cleaned_Sample_OS_Reviews_sub_aspect_extracted.csv')

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = 'Surface_Analytics'


start_phrase = """

You are an AI Assistant, your task is to extract relevant sub-aspects from below mentioned list of Aspect and Sub-Aspects and their corresponding sentiment (Positive or Negative) from the user's prompt. This information will be used to forecast user sentiment accurately.

Please carefully analyze the user's prompt to identify the appropriate sub-aspects from the predefined list provided below.

### Instructions:
1. Identify only Relevant Sub-Aspects: Based on the user's prompt, select the most appropriate sub-aspects from the predefined list.
2. Assign Sentiments: Classify each sub-aspect as either Positive or Negative. Make sure:
   - Sub-aspects are unique within each sentiment category.
3. IMPORTANT: Ensure that no sub-aspect is repeated in both the Positive and Negative categories under any circumstances.
4. Handle Irrelevant Prompts: If the prompt is unrelated to any predefined sub-aspects, mark both Positive and Negative categories as "None."

*Note: Thoroughly examine each prompt for accurate extraction like for example if in user prompt he mentioned about any aspect which will affect the net sentiment less then you should mention those related keywords in Negative, as these sub-aspects are essential for reliable sentiment forecasting.*
IMPORTANT: If the prompt is not making sense return None for both Positive and Negative.
### Example Format:
User Prompt: "We have introduced an improved security patch cycle and enhanced the compatibility of the OS with older devices for the latest update. Could you generate hypothetical reviews?"

Expected Answer Format:
Positive: Security, Patch Quality, Virus Protection, Firewall, Update Frequency, Older Software Compatibility, Peripheral Support, Multi-device Sync, App Compatibility, Stability, Reliability, System Integrity, Support Duration, Privacy Controls, Bug Fixes.
Negative: Disk Space, Update Size

### Predefined List of Aspects and Sub-Aspects:
- Updates & Support: Update frequency, Bug fixes, Support duration, Patch quality, Installation ease, Feedback response, Update size, Device compatibility, Security patches, Update transparency.
- Compatibility: App compatibility, Peripheral support, Older software, File formats, Multi-device sync, Accessory support, Display resolutions, Language support, External drives, Mobile compatibility.
- Price: Affordability, Value for money, Transparent pricing, Renewal costs, Discounts, Add-on costs, Subscription model, Competitive pricing, Refund policy, Hidden fees.
- Connectivity: Wi-Fi stability, Bluetooth pairing, External displays, Hotspot support, Connection speed, Network drivers, VPN support, Seamless switching, Port compatibility, Ethernet stability
- Security: Virus protection, Firewall, Data encryption, Privacy controls, Biometric login, Security patches, Parental controls, Anti-phishing, Security alerts, Vulnerability protection
- Installation: Install ease, Install speed, Disk space, Install instructions, Hardware compatibility, Custom options, Reinstall process, Recovery options, Install support, User-friendly setup.
- User Interface (UI): Design intuitiveness, UI customization, Font clarity, Layout consistency, Navigation ease, Accessibility, Touchscreen optimization, Dark mode, Theme options, Multi-language UI
- Licensing: License cost, Terms clarity, Renewal ease, Multi-device use, License transfer, Student license, Regional license, Trial period, License management, Subscription options.
- Customization & Personalization: Theme options, Widget choices, Taskbar settings, Backgrounds, Shortcuts, Accessibility settings, Layout adjustments, Notifications, Icon visibility, Folder organization.
- System Resources: Memory usage, CPU load, Disk space, Power efficiency, Low-spec compatibility, GPU use, System load balance, Resource monitoring, Startup speed, Background processes.
- Performance: Boot speed, Responsiveness, Animation smoothness, Multitasking, Background apps, File transfer speed, Settings load time, Update speed, App launch speed, Stability.
- App Support: Productivity apps, Entertainment apps, Developer tools, App store, Native apps, Regular updates, Third-party apps, Load speed, System integration, Enterprise apps.
- Gaming: Graphics performance, Frame rate, VR support, Game mode, Network latency, Resource allocation, Controller support, Graphics drivers, Low latency, Anti-cheat tools.
- Virtualization & Cross-OS Compatibility: Dual-boot support, VM compatibility, Cross-platform apps, Remote desktop, Emulation, Linux support, Virtual memory, Network bridging, File sharing, SDK compatibility.
- Ecosystem: Device sync, Mobile OS integration, Wearable support, Cloud storage, Family sharing, App ecosystem, Home device integration, Ecosystem apps, Multi-device use, Cross-platform.
- Productivity: Productivity tools, App integration, Cloud storage, Task management, Focus mode, Document editing, Time-tracking, Screen sharing, Shortcuts, Workflow support.
- Privacy: Data tracking, Privacy controls, Data transparency, Privacy tools, Secure browsing, Ad blocking, Alerts, Encryption, Customizable settings, Privacy notifications.
- Battery: Power-saving, Battery monitoring, Fast charging, Battery health, Low-power apps, Life under load, Battery notifications, Dark mode efficiency, Battery diagnostics, Energy optimization.
- Voice & Gesture Control: Voice assistant, Voice accuracy, Gesture recognition, Command range, Custom commands, Accessibility, Device integration, Language support, Voice privacy, Response speed.
- Stability: Crash frequency, Error recovery, System uptime, Multitasking stability, Driver compatibility, Shutdown handling, System diagnostics, Restore reliability, Resilience, Update stability.
- Reliability: Consistent performance, Error handling, Support reliability, Troubleshooting, Recovery ease, System durability, Predictable updates, App consistency, System integrity, Dependability.

Here is the user prompt: 
"""


def sub_aspect_extraction(user_prompt):
    try:
        response = client.completions.create(
            model=deployment_name,
            prompt=start_phrase + user_prompt,
            max_tokens=10000,
            temperature=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error processing review: {e}")
        return "Generic"
        
def calculate_net_sentiment(group):
    total_sentiment_score = group['Sentiment_Score'].sum()
    total_review_count = group['Review_Count'].sum()

    if total_review_count == 0:
        return 0
    else:
        return (total_sentiment_score / total_review_count) * 100
        
        
def calculate_negative_review_percentage(group):
    total_reviews = group['Review_Count'].sum()
    negative_reviews = group[group['Sentiment_Score'] == -1]['Review_Count'].sum()
    
    if total_reviews == 0:
        return 0
    else:
        return (negative_reviews / total_reviews) * 100
        
def assign_sentiment(row):
    subaspect = row['Sub_Aspect']
    current_sentiment = row.get('Sentiment_Score', 0)

    if pd.isna(subaspect):
        return current_sentiment

    subaspect = subaspect.lower()

    if any(pos.lower() in subaspect for pos in positive_aspects):
        return 1
    elif any(neg.lower() in subaspect for neg in negative_aspects):
        return -1
    else:
        return current_sentiment

def get_conversational_chain_hypothetical_summary():
    global model
    global history
    try:
        prompt_template = """  
        
        As an AI assistant, your task is to provide a detailed summary based on the aspect-wise sentiment and its negative percentages, focusing on the top contributing aspects in terms of review count. Ensure that the analysis explains the percentage of negative reviews for each aspect, highlighting the reasons behind them.
        ## Instructions:
        1. Summarize the reviews for only **top 5** aspects with the highest review counts excluding Generic aspect, clearly stating the **negative percentage** for them.
        2. Explain the **reasons** for the negative feedback for these aspects, focusing on specific points mentioned by users.
        3. Provide a detailed breakdown of the aspects and ensure the negative summary is well-structured, with the main reasons for user **dissatisfaction highlighted** with negative percentages in the below format except for Generic Aspect.
            Format: 1.Aspect1:**Negative_Summary, Negative Percentage**
                    2.Aspect2:**Negative_Summary, Negative Percentage**
                    3.Aspect3:**Negative_Summary, Negative Percentage**
                    4.Aspect4:**Negative_Summary, Negative Percentage**
                    5.Aspect5:**Negative_Summary, Negative Percentage**
                    Conclusion:**    **
                    NOTE:Do not include Generic Aspect in above format, Do not mention more then 5 aspects based on highest review count
                    
        4. End with a brief conclusion summarizing the painpoints of customers in specific aspects which are having more negative percentages, indicating which areas need the most attention.
        5. When delivering the output, be confident and avoid using future tense expressions like "could be" or "may be."
         
        Context:The DataFrame contains aspect-wise review data, including sentiment scores, review counts, and calculated negative review percentages. The summary should reflect these metrics, emphasizing the negative sentiment and its causes in aspects that contribute the most to the overall review count.
        Note:**Mention the negative review percentages for top 5 aspects based on highest review count with negative summary** except Generic aspect
 
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(azure_deployment="Thruxton_R",api_version='2024-03-01-preview',temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err
# Function to handle user queries using the existing vector store
def hypothetical_summary(user_question, vector_store_path="faiss_index_OS_Data"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_hypothetical_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err

st.sidebar.title("Select Operating System")
os_options = df['OS'].unique()
selected_os = st.sidebar.selectbox('Operating System', os_options)

filtered_df_os = df[df['OS'] == selected_os]

st.header("Consumer Sentiment Simulation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat message and handle sentiment calculation
if user_prompt := st.chat_input("Enter your prompt"):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    result = sub_aspect_extraction(user_prompt)
    print(result)
    positive_match = re.search(r'Positive\s*:\s*(.*?)\s*Negative', result, re.DOTALL)
    negative_match = re.search(r'Negative\s*:\s*(.*)', result, re.DOTALL)

    positive_aspects = positive_match.group(1).split(',') if positive_match else []
    negative_aspects = negative_match.group(1).split(',') if negative_match else []
    positive_aspects = [aspect.strip() for aspect in positive_aspects]
    negative_aspects = [aspect.strip() for aspect in negative_aspects]
    
    actual_sentiment_df = filtered_df_os
    hypothetical_sentiment_df = df
    
    st.title("Operating System Consumer Sentiment Simulation")
    col1, col2 = st.columns(2)
    
    with col1:
        net_sentiment_actual = calculate_net_sentiment(actual_sentiment_df)
        st.write(f"**Actual Net Sentiment:** {net_sentiment_actual:.2f}%")
        
        aspect_wise_net_sentiment_actual = actual_sentiment_df.groupby('Aspect').apply(lambda group: pd.Series({
            'Net_Sentiment': calculate_net_sentiment(group),
            'Review_Count': group['Review_Count'].sum(),
            'Negative_Percentage': calculate_negative_review_percentage(group)
        })).reset_index()
        
        st.write("Aspect-wise Sentiment (Actual):")
        st.write(aspect_wise_net_sentiment_actual)
        
    with col2:
        actual_sentiment_df['Sentiment_Score'] = actual_sentiment_df.apply(assign_sentiment, axis=1)
        net_sentiment_hypothetical = calculate_net_sentiment(actual_sentiment_df)
        st.write(f"**Hypothetical Net Sentiment:** {net_sentiment_hypothetical:.2f}%")

        aspect_wise_net_sentiment_hypothetical = actual_sentiment_df.groupby('Aspect').apply(lambda group: pd.Series({
            'Net_Sentiment': calculate_net_sentiment(group),
            'Review_Count': group['Review_Count'].sum(),
            'Negative_Percentage': calculate_negative_review_percentage(group)
        })).reset_index()
        
        st.write("Aspect-wise Sentiment (Hypothetical):")
        st.write(aspect_wise_net_sentiment_hypothetical)
        
    forecasted_summary = hypothetical_summary(
        result + str(aspect_wise_net_sentiment_hypothetical.to_dict()) + 
        "Based on this data, provide a summary for the user question: " + user_prompt
    )
    
    st.write(forecasted_summary)

    # Append the assistant's response to the session messages
    st.session_state.messages.append({"role": "assistant", "content": forecasted_summary})
