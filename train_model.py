# train_model_fixed.py - Xử lý nhãn văn bản
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import warnings
warnings.filterwarnings('ignore')
import os
import json

# Khai báo các biến toàn cục
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def convert_labels_to_numeric(labels):
    """
    Chuyển đổi nhãn văn bản thành số
    """
    label_mapping = {
        'tiêu cực': 0,
        'trung tính': 1,
        'tích cực': 2,
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    numeric_labels = []
    for label in labels:
        # Chuyển về chữ thường và strip khoảng trắng
        label_str = str(label).strip().lower()
        
        # Kiểm tra xem có phải là số không
        if label_str.isdigit():
            label_int = int(label_str)
            if label_int in [0, 1, 2]:
                numeric_labels.append(label_int)
            else:
                # Nếu là số nhưng không phải 0,1,2
                print(f"⚠️  Cảnh báo: Nhãn số {label_int} không hợp lệ, gán mặc định là trung tính (1)")
                numeric_labels.append(1)
        else:
            # Nếu là văn bản, ánh xạ
            if label_str in label_mapping:
                numeric_labels.append(label_mapping[label_str])
            else:
                # Nhãn không xác định, gán mặc định
                print(f"⚠️  Cảnh báo: Nhãn '{label}' không xác định, gán mặc định là trung tính (1)")
                numeric_labels.append(1)
    
    return numeric_labels

def load_and_prepare_data(file_path="dataset.csv"):
    """
    Tải và chuẩn bị dữ liệu từ file CSV
    """
    print("📥 Đang tải dữ liệu...")
    
    # Tải dữ liệu
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        # Thử encoding khác nếu utf-8 không hoạt động
        df = pd.read_csv(file_path, encoding='latin1')
    
    # Kiểm tra cấu trúc dữ liệu
    print(f"\n📊 Cấu trúc dữ liệu:")
    print(f"  Số hàng: {len(df)}")
    print(f"  Số cột: {len(df.columns)}")
    print(f"  Các cột: {list(df.columns)}")
    
    # Tìm cột text và label
    text_column = None
    label_column = None
    
    # Tìm cột text (có thể có tên khác)
    possible_text_columns = ['text', 'content', 'sentence', 'comment', 'review', 'văn bản', 'câu']
    for col in df.columns:
        if col.lower() in possible_text_columns:
            text_column = col
            break
    
    # Tìm cột label (có thể có tên khác)
    possible_label_columns = ['label', 'sentiment', 'emotion', 'category', 'nhãn', 'cảm xúc']
    for col in df.columns:
        if col.lower() in possible_label_columns:
            label_column = col
            break
    
    if text_column is None:
        # Lấy cột đầu tiên làm text
        text_column = df.columns[0]
    
    if label_column is None:
        # Lấy cột thứ hai làm label (nếu có)
        if len(df.columns) > 1:
            label_column = df.columns[1]
        else:
            # Nếu chỉ có một cột, tạo label mặc định
            print("⚠️  Không tìm thấy cột label, gán mặc định tất cả là trung tính (1)")
            df['label'] = 1
            label_column = 'label'
    
    print(f"  Cột text: {text_column}")
    print(f"  Cột label: {label_column}")
    
    # Làm sạch dữ liệu
    df['text'] = df[text_column].astype(str).str.strip()
    df['label'] = df[label_column]
    
    # Xóa hàng trống
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != '']
    
    # Chuyển đổi nhãn thành số
    print("\n🔄 Đang chuyển đổi nhãn...")
    df['label_numeric'] = convert_labels_to_numeric(df['label'].tolist())
    
    # Kiểm tra phân phối label
    print("\n📊 Phân phối nhãn gốc:")
    original_dist = df[label_column].value_counts()
    for label, count in original_dist.items():
        print(f"  '{label}': {count} mẫu")
    
    print("\n📊 Phân phối nhãn số hóa:")
    numeric_dist = df['label_numeric'].value_counts().sort_index()
    label_names = {0: "TIÊU CỰC", 1: "TRUNG TÍNH", 2: "TÍCH CỰC"}
    for label_num, count in numeric_dist.items():
        label_name = label_names.get(label_num, f"KHÔNG XÁC ĐỊNH ({label_num})")
        print(f"  {label_name} ({label_num}): {count} mẫu ({count/len(df)*100:.1f}%)")
    
    # Chia dữ liệu
    print(f"\n📈 Chia dữ liệu...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label_numeric']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label_numeric']
    )
    
    print(f"  Train: {len(train_df)} mẫu")
    print(f"  Validation: {len(val_df)} mẫu")
    print(f"  Test: {len(test_df)} mẫu")
    
    return train_df, val_df, test_df

def tokenize_data(tokenizer, train_df, val_df, test_df):
    """
    Tokenize dữ liệu
    """
    print("\n🔤 Đang tokenize dữ liệu...")
    
    # Lấy text và label
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label_numeric'].tolist()
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label_numeric'].tolist()
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label_numeric'].tolist()
    
    # Tokenize
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    
    # Tạo datasets
    train_dataset = SimpleDataset(train_encodings, train_labels)
    val_dataset = SimpleDataset(val_encodings, val_labels)
    test_dataset = SimpleDataset(test_encodings, test_labels)
    
    return train_dataset, val_dataset, test_dataset

def compute_metrics(p):
    """
    Tính toán metrics cho evaluation
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }

def train_sentiment_model():
    """
    Huấn luyện mô hình phân loại cảm xúc
    """
    print("🚀 Bắt đầu huấn luyện mô hình phân loại cảm xúc tiếng Việt")
    print("=" * 60)
    
    # 1. Tải và chuẩn bị dữ liệu
    try:
        train_df, val_df, test_df = load_and_prepare_data("dataset.csv")
    except FileNotFoundError:
        print("❌ Không tìm thấy file dataset.csv")
        print("💡 Tạo file dataset.csv với cấu trúc:")
        print("   text,label")
        print("   'Sản phẩm rất tốt',tích cực")
        print("   'Dịch vụ tệ',tiêu cực")
        print("   'Bình thường',trung tính")
        return
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Tải tokenizer
    print("\n🔄 Đang tải tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Tokenize dữ liệu
    train_dataset, val_dataset, test_dataset = tokenize_data(tokenizer, train_df, val_df, test_df)
    
    # 4. Tạo data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 5. Tải mô hình
    print("\n🧠 Đang tải mô hình PhoBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "TIÊU CỰC", 1: "TRUNG TÍNH", 2: "TÍCH CỰC"},
        label2id={"TIÊU CỰC": 0, "TRUNG TÍNH": 1, "TÍCH CỰC": 2},
        ignore_mismatched_sizes=True
    )
    
    # 6. Cấu hình training
    training_args = TrainingArguments(
        output_dir="./sentiment_model",
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )
    
    # 7. Tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 8. Huấn luyện
    print("\n🏋️‍♂️ Bắt đầu huấn luyện...")
    train_result = trainer.train()
    
    # 9. Lưu mô hình
    print("\n💾 Đang lưu mô hình...")
    trainer.save_model("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")
    
    # 10. Đánh giá trên tập test
    print("\n📊 Đang đánh giá trên tập test...")
    test_results = trainer.evaluate(test_dataset)
    
    print("\n" + "=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)
    
    print("\n📈 Kết quả huấn luyện:")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Test accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"  Test F1-score: {test_results.get('eval_f1', 0):.4f}")
    
    # 11. Dự đoán trên tập test
    print("\n📋 Báo cáo phân loại chi tiết:")
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=-1)
    y_true = test_predictions.label_ids
    
    # Tạo classification report
    target_names = ["TIÊU CỰC", "TRUNG TÍNH", "TÍCH CỰC"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # 12. Lưu thông tin training
    training_info = {
        "model_name": MODEL_NAME,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
        "num_test_samples": len(test_df),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "test_accuracy": float(test_results.get('eval_accuracy', 0)),
        "test_f1": float(test_results.get('eval_f1', 0)),
        "label_mapping": {
            "0": "TIÊU CỰC",
            "1": "TRUNG TÍNH", 
            "2": "TÍCH CỰC"
        }
    }
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("./sentiment_model", exist_ok=True)
    
    with open("./sentiment_model/training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    print("\n📁 Mô hình đã được lưu tại: ./sentiment_model/")
    
    # 13. Test với một số câu mẫu
    print("\n🧪 Test với câu mẫu:")
    test_sentences = [
        "Sản phẩm này rất tốt, tôi rất hài lòng",
        "Dịch vụ tệ quá, không bao giờ quay lại",
        "Cũng bình thường, không có gì đặc biệt"
    ]
    
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predicted_label = torch.argmax(predictions).item()
        label_name = {0: "TIÊU CỰC", 1: "TRUNG TÍNH", 2: "TÍCH CỰC"}.get(predicted_label, "UNKNOWN")
        confidence = torch.max(predictions).item()
        
        print(f"  '{sentence}'")
        print(f"    → {label_name} ({confidence:.2%})")
    
    print("\n🎯 Để sử dụng mô hình trong ứng dụng chính:")
    print("   Thay đổi trong main.py:")
    print("   model_path = './sentiment_model'")
    print("   model = AutoModelForSequenceClassification.from_pretrained(model_path)")
    
    return trainer, test_results

if __name__ == "__main__":
    # Kiểm tra GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Thiết bị đang sử dụng: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🐍 Phiên bản PyTorch: {torch.__version__}")
    
    # Huấn luyện mô hình
    train_sentiment_model()