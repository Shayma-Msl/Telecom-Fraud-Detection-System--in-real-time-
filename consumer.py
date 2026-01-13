from kafka import KafkaConsumer
import json
import pandas as pd
import mysql.connector
import torch
import joblib
from dgl.data.utils import load_graphs
from dgl import add_self_loop
from model import GAT_COBO
from utils import preprocess_call_logs_func, realtime_fraud_detection

# Load graph and model
graph, _ = load_graphs(r"graph.bin")
graph = graph[0]
graph = add_self_loop(graph)

gat_model = GAT_COBO(
    g=graph,
    num_layers=1,
    in_dim=19,
    num_hidden=64,
    num_classes=2,
    heads=[1, 1],
    activation=torch.nn.functional.elu,
    dropout=0.3,
    dropout_adj=0.3,
    feat_drop=0.1,
    attn_drop=0.1,
    negative_slope=0.2,
    residual=False
)
gat_model.load_state_dict(torch.load(r"gat_cobo.pt", map_location=torch.device("cpu")))
gat_model.eval()

lgb_model = joblib.load(r"lgbm.pkl")

# Kafka consumer
consumer = KafkaConsumer(
    'calls',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='fraud-detector',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("✅ Consumer is running and waiting for enriched calls...")

# Main loop
for message in consumer:
    call_data = message.value
    print("\n📞 Received call log.")
    try:
        # Convert the row back to a DataFrame
        new_call_df = pd.DataFrame([call_data])
        print(new_call_df)
        # Run fraud detection
        result = realtime_fraud_detection(
            new_call_csv=new_call_df,
            graph=graph,
            gat_model=gat_model,
            lgb_model=lgb_model,
            preprocess_call_logs_func=preprocess_call_logs_func,
            save_updates=True
        )

        if result is None:
            print("⚠️ Skipped: Fraud detection returned None.")
            continue

        call_data["isFraud"] = result["isFraud"]
        if result["isFraud"]:
            call_data["fraud_type"] = result["fraud_type"]
            print("🚨 FRAUD DETECTED:", result["fraud_type"])

            # Save to MySQL
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="root",
                    database="frauddetection1"
                )
                cursor = conn.cursor()

                insert_query = """
                INSERT INTO fraud_results (
                    CallHour, CallMinute, CallSecond,
                    CallingNumber, CalledNumber, Callduration,
                    callDay, callMonth, callYear,
                    IntrunkTT, OuttrunkTT, InSwitch_IGW_TUN, OutSwitch_IGW_TUN,
                    Intrunk_enc, Outtrunk_enc,
                    isFraud, fraud_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                values = (
                    int(call_data["CallHour"]),
                    int(call_data["CallMinute"]),
                    int(call_data["CallSecond"]),
                    str(call_data["CallingNumber"]),
                    str(call_data["CalledNumber"]),
                    float(call_data["Callduration"]),
                    int(call_data["callDay"]),
                    int(call_data["callMonth"]),
                    int(call_data["callYear"]),
                    int(call_data["IntrunkTT"]),
                    int(call_data["OuttrunkTT"]),
                    int(call_data["InSwitch_IGW_TUN"]),
                    int(call_data["OutSwitch_IGW_TUN"]),
                    int(call_data["Intrunk_enc"]),
                    int(call_data["Outtrunk_enc"]),
                    1,
                    call_data["fraud_type"]
                )

                cursor.execute(insert_query, values)
                conn.commit()
                print("✅ Stored in DB.")
            except Exception as db_err:
                print("❌ DB Error:", db_err)
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
        else:
            print("✅ This call is not fraud.")

    except Exception as e:
        print("❌ Processing error:", e)
