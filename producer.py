import os
from kafka import KafkaProducer
import pandas as pd
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka.errors import NoBrokersAvailable

CSV_FILE = "new_calls.csv"
seen_rows = 0

def create_kafka_producer(bootstrap_servers='localhost:9092', retries=5, delay=5):
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print(f"✅ Successfully connected to Kafka at {bootstrap_servers}")
            return producer
        except NoBrokersAvailable as e:
            print(f"⚠️ Attempt {attempt + 1}/{retries}: Kafka broker not available. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"❌ Failed to connect to Kafka after {retries} attempts")

# Initialize Kafka producer with retry logic
producer = create_kafka_producer()

class NewCallHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        global seen_rows
        if os.path.abspath(event.src_path) == os.path.abspath(CSV_FILE):
            try:
                column_names = [
                    "CallHour", "CallMinute", "CallSecond", "CallingNumber", "CalledNumber",
                    "Callduration", "callDay", "callMonth", "callYear",
                    "IntrunkTT", "OuttrunkTT", "InSwitch_IGW_TUN", "OutSwitch_IGW_TUN",
                    "Intrunk_enc", "Outtrunk_enc"
                ]
                df = pd.read_csv("new_calls.csv", names=column_names)
                new_rows = df.iloc[seen_rows:]
                if not new_rows.empty:
                    for _, row in new_rows.iterrows():
                        message = row.to_dict()
                        message["processed"] = True
                        producer.send("calls", message)
                        print(f"📤 Sent new call → {message.get('CallingNumber', 'unknown')}")
                    seen_rows = len(df)
            except Exception as e:
                print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    print("🚀 Listening new rows...")
    event_handler = NewCallHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        open("new_calls.csv", "w").close()
    observer.join()