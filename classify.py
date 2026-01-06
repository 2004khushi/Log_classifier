from processor_regex import classify_with_regex
from processor_llm import classify_with_llm
from processor_bert import classify_with_bert

def classify(log_message):
    label = classify_with_regex(log_message)
    if label is None:
        return "Unknown"


if __name__ == "__main__":
    logs = [
        "User User123 logged in.",
        "Backup started at 12:00.",
        "System reboot initiated by user admin.",
        "Account with ID 123 created by admin.",
        "File backup.txt uploaded successfully by user admin.",
        "Disk cleanup completed successfully.",
    ]
    for log in logs:
        print(classify_with_regex(log))