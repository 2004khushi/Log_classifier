from dotenv import load_dotenv
from groq import Groq


load_dotenv()
groq = Groq()

def classify_with_llm(log_msg):

    prompt = f'''Classify the log message into one of these categories: 
     Workflow Error or  Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Display output as just the category name, without any other text.
    Log message: {log_msg}'''

    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    content = chat_completion.choices[0].message.content
    return content


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))
    print(classify_with_llm("Lead conversion failed for prospect ID 7842 due to missing contact information."))
    print(classify_with_llm("API endpoint 'getCustomerDetails' is deprecated and will be removed in version 3.2. Use 'fetchCustomerInfo' instead."))

