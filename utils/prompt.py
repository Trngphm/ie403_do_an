SYSTEM_PROMPT = "Bạn là một hệ thống tạo văn bản bằng tiếng việt."

USER_PROMPT_TEMPLATE = """Hãy dựa vào đoạn văn bản tôi gửi và tạo một đoạn văn mới giống với đoạn văn đã gửi nhưng dùng cách dùng từ và cấu trúc câu khác đoạn gốc.

Yêu cầu:
- Viết bằng TIẾNG VIỆT
- Tạo lại đoạn văn mới giữ nguyên ý chính
- Viết tự nhiên như người thật
- Độ dài đoạn mới không được khác quá (vượt quá hoặc ít hơn) 10 từ so với bản gốc
- Không thêm thông tin mới
- Không sao chép nguyên văn

Đầu vào:
{text}"""