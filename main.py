# main.py
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import json
import os

# Tiêu đề ứng dụng
st.set_page_config(page_title="Trợ lý phân loại cảm xúc tiếng Việt", page_icon="🇻🇳")
st.title("🇻🇳 Trợ lý phân loại cảm xúc tiếng Việt")
st.markdown("---")

# Khởi tạo model và tokenizer
@st.cache_resource
def load_model():
    model_name = "vinai/phobert-base-v2"  # Sử dụng PhoBERT tiếng Việt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tải model phân loại cảm xúc (cần train trước hoặc tải từ HuggingFace)
    # Ở đây dùng model mẫu, bạn cần thay bằng model đã train của mình
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # 3 lớp: tích cực, trung tính, tiêu cực
        ignore_mismatched_sizes=True
    )
    
    # Load weights nếu có (thay bằng đường dẫn đến model của bạn)
    # model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Dự đoán cảm xúc từ văn bản"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Nhãn cảm xúc
    labels = ["TIÊU CỰC", "TRUNG TÍNH", "TÍCH CỰC"]
    scores = predictions[0].tolist()
    
    # Lấy cảm xúc có điểm cao nhất
    predicted_label = labels[scores.index(max(scores))]
    confidence = max(scores)
    
    return predicted_label, confidence, scores

def save_result(text, sentiment, confidence, scores):
    """Lưu kết quả vào file JSON"""
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "sentiment": sentiment,
        "confidence": float(confidence),
        "scores": [float(s) for s in scores]
    }
    
    # Tạo thư mục results nếu chưa tồn tại
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Lưu vào file JSON
    filename = f"results/results_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        # Đọc dữ liệu cũ nếu có
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"results": []}
        
        # Thêm kết quả mới
        data["results"].append(result)
        
        # Lưu file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, filename
    except Exception as e:
        return False, str(e)

def main():
    # Tải model
    try:
        model, tokenizer = load_model()
        st.success("✅ Model đã sẵn sàng!")
    except Exception as e:
        st.error(f"❌ Lỗi khi tải model: {e}")
        return
    
    # Sidebar cho thông tin
    with st.sidebar:
        st.header("ℹ️ Thông tin đồ án")
        st.info("""
        **Tên đồ án:** Trợ lý phân loại cảm xúc tiếng Việt  
        **Mô hình:** PhoBERT-base-v2  
        **Lớp cảm xúc:** Tích cực, Trung tính, Tiêu cực  
        **Ngôn ngữ:** Python  
        **Thư viện:** Transformers, Streamlit
        """)
        
        st.markdown("---")
        st.header("📊 Xem kết quả đã lưu")
        if st.button("📂 Mở thư mục kết quả"):
            if os.path.exists("results"):
                st.write("Các file kết quả:")
                for file in os.listdir("results"):
                    if file.endswith(".json"):
                        st.write(f"📄 {file}")
            else:
                st.warning("Chưa có kết quả nào được lưu")
    
    # Phần chính - Nhập văn bản
    st.subheader("📝 Nhập văn bản tiếng Việt để phân tích")
    
    text_input = st.text_area(
        "Nhập văn bản của bạn:",
        height=150,
        placeholder="Ví dụ: Sản phẩm này rất tốt, tôi rất hài lòng với chất lượng dịch vụ..."
    )
    
    # Nút phân tích
    if st.button("🔍 Phân tích cảm xúc", type="primary"):
        if text_input.strip():
            with st.spinner("Đang phân tích cảm xúc..."):
                # Dự đoán cảm xúc
                sentiment, confidence, scores = predict_sentiment(text_input, model, tokenizer)
                
                # Hiển thị kết quả
                col1, col2, col3 = st.columns(3)
                
                # Hiển thị biểu tượng cảm xúc
                if sentiment == "TÍCH CỰC":
                    emoji = "😊"
                    color = "green"
                elif sentiment == "TRUNG TÍNH":
                    emoji = "😐"
                    color = "blue"
                else:
                    emoji = "😔"
                    color = "red"
                
                with col1:
                    st.metric("Cảm xúc", f"{emoji} {sentiment}")
                
                with col2:
                    st.metric("Độ tin cậy", f"{confidence:.2%}")
                
                # Hiển thị thanh điểm số
                with col3:
                    st.progress(confidence, "Mức độ")
                
                # Hiển thị chi tiết điểm số
                st.subheader("📊 Chi tiết điểm số")
                score_data = {
                    "Cảm xúc": ["TIÊU CỰC", "TRUNG TÍNH", "TÍCH CỰC"],
                    "Điểm số": [f"{s:.4f}" for s in scores]
                }
                st.dataframe(score_data, use_container_width=True)
                
                # Lưu kết quả
                success, result = save_result(text_input, sentiment, confidence, scores)
                if success:
                    st.success(f"✅ Đã lưu kết quả vào: {result}")
                else:
                    st.error(f"❌ Lỗi khi lưu kết quả: {result}")
        else:
            st.warning("⚠️ Vui lòng nhập văn bản để phân tích!")
    
    # Phần phân tích hàng loạt
    st.markdown("---")
    st.subheader("📁 Phân tích nhiều văn bản cùng lúc")
    
    uploaded_file = st.file_uploader("Tải lên file CSV/TXT (mỗi dòng một văn bản)", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Giả sử cột đầu tiên chứa văn bản
                text_column = df.columns[0]
                texts = df[text_column].astype(str).tolist()
            else:  # txt file
                texts = uploaded_file.read().decode('utf-8').splitlines()
            
            if st.button("🔍 Phân tích hàng loạt"):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    if text.strip():
                        sentiment, confidence, scores = predict_sentiment(text, model, tokenizer)
                        results.append({
                            "Văn bản": text[:100] + "..." if len(text) > 100 else text,
                            "Cảm xúc": sentiment,
                            "Độ tin cậy": f"{confidence:.2%}",
                            "Điểm tiêu cực": f"{scores[0]:.4f}",
                            "Điểm trung tính": f"{scores[1]:.4f}",
                            "Điểm tích cực": f"{scores[2]:.4f}"
                        })
                    
                    progress_bar.progress((i + 1) / len(texts))
                
                # Hiển thị kết quả
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Tải xuống kết quả
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải kết quả (CSV)",
                    data=csv,
                    file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {e}")

if __name__ == "__main__":
    main()