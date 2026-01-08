import pandas as pd
import uuid
import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from classify import classify

app = FastAPI()


@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        # Read the uploaded CSV
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Perform classification
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        print("Dataframe:", df.to_dict())

        # Save the modified file
        unique_filename = f"resources/output_{uuid.uuid4()}.csv"
        df.to_csv(unique_filename, index=False)
        return FileResponse(
            unique_filename,
            media_type='text/csv',
            filename="classified_logs.csv"  # This is what the user's browser sees
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    finally:
        file.file.close()
        # # Clean up if the file was saved
        # if os.path.exists("output.csv"):
        #     os.remove("output.csv")