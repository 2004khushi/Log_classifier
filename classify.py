import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from processor_regex import classify_with_regex
from processor_llm import classify_with_llm
from processor_bert import classify_with_bert

#SEE basically what we made was we need the inputs which had it target label as workflow error or deprecated we'd been removed that and used rest of the data for the regex pattern
# now the things is since we want to predict the target label itself so if we go back to data, we will see only these two belong to source-> LEGACyCRM
# so we can take the data from user by taking both log messages and it's sources too. So we can remove the LEGACyCRM ones to send it to bert or llm part.
# and the rest in regex. now for llm and regex part what we want is that the rest come in here then first sends to bert and if from there any output is UNCLASSIFIED
# then that to be sent to llm. ANd this way we achieved our hybrid model.


def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels


def classify_log(source, log_msg):
    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if not label:
            label = classify_with_bert(log_msg)
    return label


def classify_csv(input_file):
    import pandas as pd
    df = pd.read_csv(input_file)

    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Save the modified file
    output_file = "resources/output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
     classify_csv("resources/test.csv")
    # logs = [
    #     ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    #     ("BillingSystem", "User User12345 logged in."),
    #     ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
    #     ("AnalyticsEngine", "Backup completed successfully."),
    #     ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
    #     ("ModernHR", "Admin access escalation detected for user 9429"),
    #     ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
    #     ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
    #     ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
    #     ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    # ]
    # labels = classify(logs)
    # #
    # for log, label in zip(logs, labels):
    #     print(log[0], "->", label)
